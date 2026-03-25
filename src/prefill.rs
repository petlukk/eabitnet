//! Batched prefill: load each layer's weights once, multiply all prompt tokens.

use crate::ffi;
use crate::forward::{apply_rope, build_rope_freqs, InferenceState};
use crate::gemm_i2s::{BatchI8, i2s_gemm_mt, i2s_fused_sqrelu_gemm_mt};
use crate::matmul::{embed_f16_lookup, i8_output_matmul_mt};
#[cfg(target_arch = "aarch64")]
use crate::matmul::i8_output_matmul_speculative;
use crate::model::BitNetModel;

impl InferenceState {
    pub fn prefill(&mut self, model: &BitNetModel, tokens: &[u32]) {
        let n = tokens.len();
        let (h, hd, nh, nkv, kv, f) = (
            model.hidden_dim, model.head_dim, model.n_heads,
            model.n_kv_heads, model.kv_dim, model.ffn_dim,
        );
        let gqa_ratio = nh / nkv;

        // Embed all tokens
        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&tok| {
            let mut x = vec![0.0f32; h];
            embed_f16_lookup(model.embed_weight_f16, tok, &mut x, h);
            x
        }).collect();

        let mut batch_h = BatchI8::new(n, h);
        let mut batch_f = BatchI8::new(n, f);
        let (mut qs_all, mut ks_all, mut vs_all) = (
            vec![0.0f32; n * h], vec![0.0f32; n * kv], vec![0.0f32; n * kv],
        );
        let (mut attn_all, mut tmp_all, mut hidden_all) = (
            vec![0.0f32; n * h], vec![0.0f32; n * h], vec![0.0f32; n * f],
        );

        for layer in 0..model.n_layers {
            let lw = &model.layers[layer];

            // RMSNorm + quantize all tokens
            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(
                        xs[t].as_ptr(), lw.attn_norm,
                        self.x_norm.as_mut_ptr(), h as i32, model.rms_eps,
                    );
                }
                batch_h.quantize(t, &self.x_norm);
            }

            // Batched QKV matmul
            i2s_gemm_mt(lw.wq, lw.wq_scale, &batch_h, &mut qs_all, h, h, &self.pool);
            i2s_gemm_mt(lw.wk, lw.wk_scale, &batch_h, &mut ks_all, kv, h, &self.pool);
            i2s_gemm_mt(lw.wv, lw.wv_scale, &batch_h, &mut vs_all, kv, h, &self.pool);

            // Per-token: RoPE, cache, attention (sequential -- <1% of compute)
            for t in 0..n {
                let (q, k) = (
                    &mut qs_all[t * h..(t + 1) * h],
                    &mut ks_all[t * kv..(t + 1) * kv],
                );
                build_rope_freqs(&mut self.rope_freqs, hd, t, model.rope_theta);
                apply_rope(q, &self.rope_freqs, hd, nh);
                apply_rope(k, &self.rope_freqs, hd, nkv);
                for head in 0..nkv {
                    let off = ((layer * nkv + head) * self.max_seq_len + t) * hd;
                    self.k_cache[off..off + hd]
                        .copy_from_slice(&k[head * hd..(head + 1) * hd]);
                    self.v_cache[off..off + hd]
                        .copy_from_slice(&vs_all[t * kv + head * hd..t * kv + (head + 1) * hd]);
                }
                let scale = 1.0 / (hd as f32).sqrt();
                let attn = &mut attn_all[t * h..(t + 1) * h];
                for head in 0..nh {
                    let (kv_head, q_off) = (head / gqa_ratio, head * hd);
                    let cb = (layer * nkv + kv_head) * self.max_seq_len * hd;
                    unsafe {
                        ffi::fused_attention_f32(
                            q.as_ptr().add(q_off),
                            self.k_cache.as_ptr().add(cb),
                            self.v_cache.as_ptr().add(cb),
                            attn.as_mut_ptr().add(q_off),
                            hd as i32, (t + 1) as i32, scale,
                        );
                    }
                }
            }

            // Batched O-proj: attn_sub_norm + quantize + matmul
            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(
                        attn_all[t * h..].as_ptr(), lw.attn_sub_norm,
                        attn_all[t * h..].as_mut_ptr(), h as i32, model.rms_eps,
                    );
                }
                batch_h.quantize(t, &attn_all[t * h..(t + 1) * h]);
            }
            i2s_gemm_mt(lw.wo, lw.wo_scale, &batch_h, &mut tmp_all, h, h, &self.pool);

            // Residual
            for t in 0..n {
                unsafe {
                    ffi::vecadd_f32(
                        xs[t].as_ptr(), tmp_all[t * h..].as_ptr(),
                        xs[t].as_mut_ptr(), h as i32,
                    );
                }
            }

            // FFN: RMSNorm + quantize + fused gate+up+SquaredReLU
            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(
                        xs[t].as_ptr(), lw.ffn_norm,
                        self.x_norm.as_mut_ptr(), h as i32, model.rms_eps,
                    );
                }
                batch_h.quantize(t, &self.x_norm);
            }
            i2s_fused_sqrelu_gemm_mt(
                lw.w_gate, lw.w_gate_scale,
                lw.w_up, lw.w_up_scale,
                &batch_h, &mut hidden_all, f, h, &self.pool,
            );

            // FFN sub-norm + quantize + down proj
            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(
                        hidden_all[t * f..].as_ptr(), lw.ffn_sub_norm,
                        hidden_all[t * f..].as_mut_ptr(), f as i32, model.rms_eps,
                    );
                }
                batch_f.quantize(t, &hidden_all[t * f..(t + 1) * f]);
            }
            i2s_gemm_mt(
                lw.w_down, lw.w_down_scale, &batch_f,
                &mut tmp_all, h, f, &self.pool,
            );

            // Residual
            for t in 0..n {
                unsafe {
                    ffi::vecadd_f32(
                        xs[t].as_ptr(), tmp_all[t * h..].as_ptr(),
                        xs[t].as_mut_ptr(), h as i32,
                    );
                }
            }
        }

        // Final norm + output projection (last token only)
        self.x[..h].copy_from_slice(&xs[n - 1]);
        unsafe {
            ffi::rmsnorm_f32(
                self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(),
                h as i32, model.rms_eps,
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            if !model.embed_sketch.is_empty() {
                i8_output_matmul_speculative(
                    &model.embed_weight_i8, &model.embed_row_scales,
                    &model.embed_sketch, model.embed_sketch_dim,
                    &self.x_norm, &mut self.logits,
                    model.vocab_size, h, &self.pool,
                );
            } else {
                i8_output_matmul_mt(
                    &model.embed_weight_i8, &model.embed_row_scales,
                    &self.x_norm, &mut self.logits,
                    model.vocab_size, h, &self.pool,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        i8_output_matmul_mt(
            &model.embed_weight_i8, &model.embed_row_scales,
            &self.x_norm, &mut self.logits,
            model.vocab_size, h, &self.pool,
        );
    }
}
