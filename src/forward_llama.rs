//! Transformer forward pass for Llama models with Q4_K_M quantization.

use crate::ffi;
use crate::forward::{apply_rope, build_rope_freqs, sample};
use crate::gemm_q4k::{BatchQ8K, q4k_gemm_mt, q4k_fused_silu_gemm_mt};
use crate::gemm_q6k::q6k_gemm_mt;
use crate::matmul::embed_f16_lookup;
use crate::matmul_q4k::{q4k_embed_lookup, q4k_matmul_mt, q4k_matmul_work, q4k_fused_gate_up_silu_work};
use crate::matmul_q6k::{q6k_embed_lookup, q6k_matmul_mt, q6k_matmul_work};
use crate::model::BitNetModel;
use crate::threadpool::ThreadPool;

const Q4K_BLOCK_BYTES: usize = 144;
const Q6K_BLOCK_BYTES: usize = 210;

pub struct LlamaState {
    pool: ThreadPool,
    x: Vec<f32>,
    x_norm: Vec<f32>,
    x_q8_qs: Vec<i8>,
    x_q8_d: Vec<f32>,
    x_q8_bsums: Vec<i32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    attn_q8_qs: Vec<i8>,
    attn_q8_d: Vec<f32>,
    attn_q8_bsums: Vec<i32>,
    hidden: Vec<f32>,
    hidden_q8_qs: Vec<i8>,
    hidden_q8_d: Vec<f32>,
    hidden_q8_bsums: Vec<i32>,
    logits: Vec<f32>,
    tmp: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    rope_freqs: Vec<f32>,
    max_seq_len: usize,
}

fn q8k_blocks(dim: usize) -> usize { dim / 256 }

fn embed_token(model: &BitNetModel, token: u32, out: &mut [f32]) {
    match model.embed_dtype {
        12 => q4k_embed_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
        14 => q6k_embed_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
        _ => embed_f16_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
    }
}

impl LlamaState {
    pub fn new(model: &BitNetModel, max_seq_len: usize) -> Self {
        let h = model.hidden_dim;
        let f = model.ffn_dim;
        let v = model.vocab_size;
        let hb = q8k_blocks(h);
        let fb = q8k_blocks(f);
        let kv_cache_size = model.n_layers * model.n_kv_heads * max_seq_len * model.head_dim;
        LlamaState {
            pool: ThreadPool::new(),
            x: vec![0.0; h],
            x_norm: vec![0.0; h],
            x_q8_qs: vec![0; h + 16],
            x_q8_d: vec![0.0; hb],
            x_q8_bsums: vec![0; hb * 16],
            q: vec![0.0; h],
            k: vec![0.0; model.kv_dim],
            v: vec![0.0; model.kv_dim],
            attn_out: vec![0.0; h],
            attn_q8_qs: vec![0; h + 16],
            attn_q8_d: vec![0.0; hb],
            attn_q8_bsums: vec![0; hb * 16],
            hidden: vec![0.0; f],
            hidden_q8_qs: vec![0; f + 16],
            hidden_q8_d: vec![0.0; fb],
            hidden_q8_bsums: vec![0; fb * 16],
            logits: vec![0.0; v],
            tmp: vec![0.0; h.max(f)],
            k_cache: vec![0.0; kv_cache_size],
            v_cache: vec![0.0; kv_cache_size],
            rope_freqs: vec![0.0; model.head_dim],
            max_seq_len,
        }
    }

}

impl LlamaState {
    /// Process one token through one layer. x is modified in-place.
    fn process_layer(&mut self, model: &BitNetModel, layer: usize, x: &mut [f32], pos: usize) {
        let h = model.hidden_dim;
        let hd = model.head_dim;
        let nh = model.n_heads;
        let nkv = model.n_kv_heads;
        let kv = model.kv_dim;
        let f = model.ffn_dim;
        let gqa_ratio = nh / nkv;
        let h_nb = q8k_blocks(h);
        let f_nb = q8k_blocks(f);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        let lw = &model.q4k_layers[layer];

        // RMSNorm + quantize
        unsafe {
            ffi::rmsnorm_f32(x.as_ptr(), lw.attn_norm, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }

        // QKV — concurrent dispatch
        let total = self.pool.thread_count();
        let wv_bb = lw.wv_block_bytes;
        if total >= 3 {
            let q_t = (total / 2).max(1);
            let rem = total - q_t;
            let k_t = rem / 2;
            let v_t = rem - k_t;
            let q8p = self.x_q8_qs.as_ptr() as usize;
            let q8d = self.x_q8_d.as_ptr() as usize;
            let q8b = self.x_q8_bsums.as_ptr() as usize;
            let q_out = self.q.as_mut_ptr() as usize;
            let k_out = self.k.as_mut_ptr() as usize;
            let v_out = self.v.as_mut_ptr() as usize;
            let wq = lw.wq as usize; let wk = lw.wk as usize; let wv = lw.wv as usize;
            let h_q6k_stride = h_nb * Q6K_BLOCK_BYTES;
            self.pool.run_split3(
                q_t, |tid, nt| unsafe { q4k_matmul_work(wq as _, h_row_stride, h_nb, q8p as _, q8d as _, q8b as _, q_out as _, h, tid, nt); },
                k_t, |tid, nt| unsafe { q4k_matmul_work(wk as _, h_row_stride, h_nb, q8p as _, q8d as _, q8b as _, k_out as _, kv, tid, nt); },
                v_t, |tid, nt| unsafe {
                    if wv_bb == Q6K_BLOCK_BYTES { q6k_matmul_work(wv as _, h_q6k_stride, h_nb, q8p as _, q8d as _, q8b as _, v_out as _, kv, tid, nt); }
                    else { q4k_matmul_work(wv as _, h_row_stride, h_nb, q8p as _, q8d as _, q8b as _, v_out as _, kv, tid, nt); }
                },
            );
        } else {
            q4k_matmul_mt(lw.wq, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.q, h, &self.pool);
            q4k_matmul_mt(lw.wk, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.k, kv, &self.pool);
            if wv_bb == Q6K_BLOCK_BYTES {
                q6k_matmul_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.v, kv, &self.pool);
            } else {
                q4k_matmul_mt(lw.wv, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.v, kv, &self.pool);
            }
        }

        // RoPE + KV cache
        build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
        apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
        apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);
        for head in 0..nkv {
            let off = ((layer * nkv + head) * self.max_seq_len + pos) * hd;
            self.k_cache[off..off + hd].copy_from_slice(&self.k[head * hd..(head + 1) * hd]);
            self.v_cache[off..off + hd].copy_from_slice(&self.v[head * hd..(head + 1) * hd]);
        }

        // Attention
        let scale = 1.0 / (hd as f32).sqrt();
        let seq_len = pos + 1;
        for head in 0..nh {
            let kv_head = head / gqa_ratio;
            let q_off = head * hd;
            let cache_base = (layer * nkv + kv_head) * self.max_seq_len * hd;
            unsafe {
                ffi::fused_attention_f32(
                    self.q.as_ptr().add(q_off), self.k_cache.as_ptr().add(cache_base),
                    self.v_cache.as_ptr().add(cache_base), self.attn_out.as_mut_ptr().add(q_off),
                    hd as i32, seq_len as i32, scale,
                );
            }
        }

        // O projection + residual
        unsafe {
            ffi::quant_f32_q8k(self.attn_out.as_ptr(), self.attn_q8_qs.as_mut_ptr(),
                self.attn_q8_d.as_mut_ptr(), self.attn_q8_bsums.as_mut_ptr(), h as i32);
        }
        q4k_matmul_mt(lw.wo, h_row_stride, h_nb, self.attn_q8_qs.as_ptr(), self.attn_q8_d.as_ptr(),
            self.attn_q8_bsums.as_ptr(), &mut self.tmp, h, &self.pool);
        unsafe { ffi::vecadd_f32(x.as_ptr(), self.tmp.as_ptr(), self.attn_out.as_mut_ptr(), h as i32); }
        x[..h].copy_from_slice(&self.attn_out[..h]);

        // FFN: RMSNorm + quantize
        unsafe {
            ffi::rmsnorm_f32(x.as_ptr(), lw.ffn_norm, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }

        // Fused gate+up+SiLU
        let q8p = self.x_q8_qs.as_ptr() as usize;
        let q8d = self.x_q8_d.as_ptr() as usize;
        let q8b = self.x_q8_bsums.as_ptr() as usize;
        let h_out = self.hidden.as_mut_ptr() as usize;
        let wg = lw.w_gate as usize;
        let wu = lw.w_up as usize;
        self.pool.run(total, |tid, nt| unsafe {
            q4k_fused_gate_up_silu_work(wg as _, wu as _, h_row_stride, h_nb,
                q8p as _, q8d as _, q8b as _, h_out as _, f, tid, nt);
        });

        // Down projection + residual
        unsafe {
            ffi::quant_f32_q8k(self.hidden.as_ptr(), self.hidden_q8_qs.as_mut_ptr(),
                self.hidden_q8_d.as_mut_ptr(), self.hidden_q8_bsums.as_mut_ptr(), f as i32);
        }
        if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
            q6k_matmul_mt(lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb, self.hidden_q8_qs.as_ptr(),
                self.hidden_q8_d.as_ptr(), self.hidden_q8_bsums.as_ptr(), &mut self.tmp, h, &self.pool);
        } else {
            q4k_matmul_mt(lw.w_down, f_nb * Q4K_BLOCK_BYTES, f_nb, self.hidden_q8_qs.as_ptr(),
                self.hidden_q8_d.as_ptr(), self.hidden_q8_bsums.as_ptr(), &mut self.tmp, h, &self.pool);
        }
        unsafe { ffi::vecadd_f32(x.as_ptr(), self.tmp.as_ptr(), self.attn_out.as_mut_ptr(), h as i32); }
        x[..h].copy_from_slice(&self.attn_out[..h]);
    }

    /// Output projection: RMSNorm + quantize + matmul → logits.
    fn output_proj(&mut self, model: &BitNetModel) {
        let h = model.hidden_dim;
        let h_nb = q8k_blocks(h);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        unsafe {
            ffi::rmsnorm_f32(self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }
        if model.output_block_bytes == Q6K_BLOCK_BYTES {
            q6k_matmul_mt(model.output_weight, h_nb * Q6K_BLOCK_BYTES, h_nb, self.x_q8_qs.as_ptr(),
                self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.logits, model.vocab_size, &self.pool);
        } else {
            q4k_matmul_mt(model.output_weight, h_row_stride, h_nb, self.x_q8_qs.as_ptr(),
                self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.logits, model.vocab_size, &self.pool);
        }
    }

    /// Single-token forward pass (used for decode).
    pub fn forward(&mut self, model: &BitNetModel, token: u32, pos: usize) {
        embed_token(model, token, &mut self.x);
        let mut x = std::mem::take(&mut self.x);
        for layer in 0..model.n_layers {
            self.process_layer(model, layer, &mut x, pos);
        }
        self.x = x;
        self.output_proj(model);
    }

    /// GEMM-style batched prefill: load weight once, multiply all tokens.
    pub fn prefill(&mut self, model: &BitNetModel, tokens: &[u32]) {
        let n = tokens.len();
        let h = model.hidden_dim;
        let hd = model.head_dim;
        let nh = model.n_heads;
        let nkv = model.n_kv_heads;
        let kv = model.kv_dim;
        let f = model.ffn_dim;
        let gqa_ratio = nh / nkv;
        let h_nb = q8k_blocks(h);
        let f_nb = q8k_blocks(f);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;

        // Per-token hidden states
        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&tok| {
            let mut x = vec![0.0f32; h];
            embed_token(model, tok, &mut x);
            x
        }).collect();

        // Batch buffers
        let mut bq_h = BatchQ8K::new(n, h);
        let mut bq_f = BatchQ8K::new(n, f);
        let mut qs_all = vec![0.0f32; n * h];
        let mut ks_all = vec![0.0f32; n * kv];
        let mut vs_all = vec![0.0f32; n * kv];
        let mut attn_all = vec![0.0f32; n * h];
        let mut tmp_all = vec![0.0f32; n * h];
        let mut hidden_all = vec![0.0f32; n * f];

        for layer in 0..model.n_layers {
            let lw = &model.q4k_layers[layer];

            // Phase A: batch RMSNorm + Q8K quantize
            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(xs[t].as_ptr(), lw.attn_norm,
                        self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
                }
                bq_h.quantize(t, &self.x_norm);
            }

            // Phase B: GEMM for Q, K, V
            q4k_gemm_mt(lw.wq, h_row_stride, h_nb, &bq_h, &mut qs_all, h, &self.pool);
            q4k_gemm_mt(lw.wk, h_row_stride, h_nb, &bq_h, &mut ks_all, kv, &self.pool);
            if lw.wv_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, &bq_h, &mut vs_all, kv, &self.pool);
            } else {
                q4k_gemm_mt(lw.wv, h_row_stride, h_nb, &bq_h, &mut vs_all, kv, &self.pool);
            }

            // Phase C: RoPE + KV cache + attention (sequential per token)
            for t in 0..n {
                let q = &mut qs_all[t * h..(t + 1) * h];
                let k = &mut ks_all[t * kv..(t + 1) * kv];
                let v = &vs_all[t * kv..(t + 1) * kv];
                build_rope_freqs(&mut self.rope_freqs, hd, t, model.rope_theta);
                apply_rope(q, &self.rope_freqs, hd, nh);
                apply_rope(k, &self.rope_freqs, hd, nkv);
                for head in 0..nkv {
                    let off = ((layer * nkv + head) * self.max_seq_len + t) * hd;
                    self.k_cache[off..off + hd].copy_from_slice(&k[head * hd..(head + 1) * hd]);
                    self.v_cache[off..off + hd].copy_from_slice(&v[head * hd..(head + 1) * hd]);
                }
                let scale = 1.0 / (hd as f32).sqrt();
                let attn = &mut attn_all[t * h..(t + 1) * h];
                for head in 0..nh {
                    let kv_head = head / gqa_ratio;
                    let q_off = head * hd;
                    let cache_base = (layer * nkv + kv_head) * self.max_seq_len * hd;
                    unsafe {
                        ffi::fused_attention_f32(
                            q.as_ptr().add(q_off),
                            self.k_cache.as_ptr().add(cache_base),
                            self.v_cache.as_ptr().add(cache_base),
                            attn.as_mut_ptr().add(q_off),
                            hd as i32, (t + 1) as i32, scale,
                        );
                    }
                }
            }

            // Phase D: batch O projection
            for t in 0..n {
                let attn = &attn_all[t * h..(t + 1) * h];
                unsafe {
                    ffi::quant_f32_q8k(attn.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                        self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
                }
                bq_h.quantize(t, attn);
            }
            q4k_gemm_mt(lw.wo, h_row_stride, h_nb, &bq_h, &mut tmp_all, h, &self.pool);
            for t in 0..n {
                unsafe {
                    ffi::vecadd_f32(xs[t].as_ptr(), tmp_all[t * h..].as_ptr(),
                        xs[t].as_mut_ptr(), h as i32);
                }
            }

            // Phase E: batch FFN
            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(xs[t].as_ptr(), lw.ffn_norm,
                        self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
                }
                bq_h.quantize(t, &self.x_norm);
            }
            q4k_fused_silu_gemm_mt(lw.w_gate, lw.w_up, h_row_stride, h_nb,
                &bq_h, &mut hidden_all, f, &self.pool);
            for t in 0..n {
                let hid = &hidden_all[t * f..(t + 1) * f];
                unsafe {
                    ffi::quant_f32_q8k(hid.as_ptr(), self.hidden_q8_qs.as_mut_ptr(),
                        self.hidden_q8_d.as_mut_ptr(), self.hidden_q8_bsums.as_mut_ptr(), f as i32);
                }
                bq_f.quantize(t, hid);
            }
            if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h, &self.pool);
            } else {
                q4k_gemm_mt(lw.w_down, f_nb * Q4K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h, &self.pool);
            }
            for t in 0..n {
                unsafe {
                    ffi::vecadd_f32(xs[t].as_ptr(), tmp_all[t * h..].as_ptr(),
                        xs[t].as_mut_ptr(), h as i32);
                }
            }
        }

        self.x[..h].copy_from_slice(&xs[n - 1]);
        self.output_proj(model);
    }

    pub fn apply_repetition_penalty(&mut self, generated: &[u32], penalty: f32) {
        if penalty == 1.0 { return; }
        for &tok in generated {
            let idx = tok as usize;
            if idx < self.logits.len() {
                if self.logits[idx] > 0.0 { self.logits[idx] /= penalty; }
                else { self.logits[idx] *= penalty; }
            }
        }
    }

    pub fn sample_logits(&self, temperature: f32) -> u32 {
        sample(&self.logits, temperature)
    }
}

pub fn generate(
    model: &BitNetModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    repetition_penalty: f32,
    eos_id: u32,
    max_seq_len: usize,
    mut on_token: impl FnMut(u32),
) -> (Vec<u32>, f64, f64) {
    use std::time::Instant;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut state = LlamaState::new(model, max_seq_len);
    let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

    let prefill_start = Instant::now();
    state.prefill(model, prompt_tokens);
    output.extend_from_slice(prompt_tokens);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let first_tok_start = Instant::now();
    let mut pos = prompt_tokens.len();
    let mut n_gen = 0u32;
    let mut first_tok_ms = 0.0;

    let decode_start = Instant::now();
    for step in 0..max_tokens {
        if pos >= max_seq_len { break; }
        state.apply_repetition_penalty(&output, repetition_penalty);
        let next = state.sample_logits(temperature);
        if next == eos_id { break; }
        output.push(next);
        on_token(next);
        if step == 0 { first_tok_ms = first_tok_start.elapsed().as_secs_f64() * 1000.0; }
        state.forward(model, next, pos);
        pos += 1;
        n_gen += 1;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    let prefill_tps = prompt_tokens.len() as f64 / (prefill_ms / 1000.0);
    let decode_tps = if n_gen > 0 { n_gen as f64 / (decode_ms / 1000.0) } else { 0.0 };
    let avg_tok_ms = if n_gen > 0 { decode_ms / n_gen as f64 } else { 0.0 };

    eprintln!("\n--- perf ({} threads) ---", n_threads);
    eprintln!("prefill:    {} tokens in {:.0}ms ({:.1} tok/s)", prompt_tokens.len(), prefill_ms, prefill_tps);
    eprintln!("first tok:  {:.0}ms", first_tok_ms);
    eprintln!("decode:     {} tokens in {:.0}ms ({:.1} tok/s, {:.1}ms/tok)", n_gen, decode_ms, decode_tps, avg_tok_ms);

    (output, prefill_ms, decode_ms)
}
