//! Transformer forward pass for Llama models with Q4_K_M quantization.

use crate::ffi;
use crate::forward::{apply_rope, build_rope_freqs, sample};
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
    // Q8_K quantization buffers (reused across layers)
    x_q8_qs: Vec<i8>,
    x_q8_d: Vec<f32>,
    x_q8_bsums: Vec<i32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    // Second Q8K buffer for post-attention quantization
    attn_q8_qs: Vec<i8>,
    attn_q8_d: Vec<f32>,
    attn_q8_bsums: Vec<i32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    // Third Q8K buffer for FFN hidden
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
            gate: vec![0.0; f],
            up: vec![0.0; f],
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

    pub fn forward(&mut self, model: &BitNetModel, token: u32, pos: usize) {
        let h = model.hidden_dim;
        let hd = model.head_dim;
        let nh = model.n_heads;
        let nkv = model.n_kv_heads;
        let kv = model.kv_dim;
        let f = model.ffn_dim;
        let seq_len = pos + 1;
        let gqa_ratio = nh / nkv;
        let h_nb = q8k_blocks(h);
        let f_nb = q8k_blocks(f);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        let f_row_stride = f_nb * Q4K_BLOCK_BYTES;

        // Embedding lookup
        match model.embed_dtype {
            12 => q4k_embed_lookup(model.embed_weight_f16, token, &mut self.x, h),
            14 => q6k_embed_lookup(model.embed_weight_f16, token, &mut self.x, h),
            _ => embed_f16_lookup(model.embed_weight_f16, token, &mut self.x, h),
        }

        use std::time::Instant;
        let profile = pos == 1;
        let mut t_qkv = 0u128;
        let mut t_attn = 0u128;
        let mut t_oproj = 0u128;
        let mut t_ffn_gu = 0u128;
        let mut t_ffn_act = 0u128;
        let mut t_down = 0u128;
        let mut t_out = 0u128;
        macro_rules! prof {
            ($t:expr, $body:expr) => {
                if profile {
                    let _start = Instant::now();
                    $body;
                    $t += _start.elapsed().as_nanos();
                } else {
                    $body;
                }
            };
        }

        for layer in 0..model.n_layers {
            let lw = &model.q4k_layers[layer];

            // 1. RMSNorm(x, attn_norm) -> x_norm
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(), lw.attn_norm, self.x_norm.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }

            // 2. Quantize x_norm -> Q8K
            unsafe {
                ffi::quant_f32_q8k(
                    self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                    self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(),
                    h as i32,
                );
            }

            // 3-5. QKV concurrent dispatch (Q gets half threads, K+V split rest)
            let total = self.pool.thread_count();
            let wv_bb = lw.wv_block_bytes;
            prof!(t_qkv, {
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
                        q_t, |tid, nt| unsafe {
                            q4k_matmul_work(wq as _, h_row_stride, h_nb, q8p as _, q8d as _, q8b as _, q_out as _, h, tid, nt);
                        },
                        k_t, |tid, nt| unsafe {
                            q4k_matmul_work(wk as _, h_row_stride, h_nb, q8p as _, q8d as _, q8b as _, k_out as _, kv, tid, nt);
                        },
                        v_t, |tid, nt| unsafe {
                            if wv_bb == Q6K_BLOCK_BYTES {
                                q6k_matmul_work(wv as _, h_q6k_stride, h_nb, q8p as _, q8d as _, q8b as _, v_out as _, kv, tid, nt);
                            } else {
                                q4k_matmul_work(wv as _, h_row_stride, h_nb, q8p as _, q8d as _, q8b as _, v_out as _, kv, tid, nt);
                            }
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
            });

            // 6. RoPE
            build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
            apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
            apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);

            // 7. Store K, V in cache
            for head in 0..nkv {
                let off = ((layer * nkv + head) * self.max_seq_len + pos) * hd;
                self.k_cache[off..off + hd]
                    .copy_from_slice(&self.k[head * hd..(head + 1) * hd]);
                self.v_cache[off..off + hd]
                    .copy_from_slice(&self.v[head * hd..(head + 1) * hd]);
            }

            // 8. Fused attention
            prof!(t_attn, {
                let scale = 1.0 / (hd as f32).sqrt();
                for head in 0..nh {
                    let kv_head = head / gqa_ratio;
                    let q_off = head * hd;
                    let cache_base = (layer * nkv + kv_head) * self.max_seq_len * hd;
                    unsafe {
                        ffi::fused_attention_f32(
                            self.q.as_ptr().add(q_off),
                            self.k_cache.as_ptr().add(cache_base),
                            self.v_cache.as_ptr().add(cache_base),
                            self.attn_out.as_mut_ptr().add(q_off),
                            hd as i32, seq_len as i32, scale,
                        );
                    }
                }
            });

            // 9. Quantize attn_out -> Q8K
            prof!(t_oproj, {
                unsafe {
                    ffi::quant_f32_q8k(
                        self.attn_out.as_ptr(), self.attn_q8_qs.as_mut_ptr(),
                        self.attn_q8_d.as_mut_ptr(), self.attn_q8_bsums.as_mut_ptr(),
                        h as i32,
                    );
                }
                // 10. Q4K matmul: O = wo x attn_q8k
                q4k_matmul_mt(
                    lw.wo, h_row_stride, h_nb,
                    self.attn_q8_qs.as_ptr(), self.attn_q8_d.as_ptr(),
                    self.attn_q8_bsums.as_ptr(),
                    &mut self.tmp, h, &self.pool,
                );
            });

            // 11. Residual: x = x + O
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(), self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);

            // 12. RMSNorm(x, ffn_norm) -> x_norm
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(), lw.ffn_norm, self.x_norm.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }

            // 13. Quantize x_norm -> Q8K
            unsafe {
                ffi::quant_f32_q8k(
                    self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                    self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(),
                    h as i32,
                );
            }

            // 14-16. Fused gate+up+SiLU: dual kernel → SiLU inline → hidden directly
            prof!(t_ffn_gu, {
                let q8p = self.x_q8_qs.as_ptr() as usize;
                let q8d = self.x_q8_d.as_ptr() as usize;
                let q8b = self.x_q8_bsums.as_ptr() as usize;
                let h_out = self.hidden.as_mut_ptr() as usize;
                let wg = lw.w_gate as usize;
                let wu = lw.w_up as usize;
                self.pool.run(total, |tid, nt| unsafe {
                    q4k_fused_gate_up_silu_work(
                        wg as _, wu as _, h_row_stride, h_nb,
                        q8p as _, q8d as _, q8b as _,
                        h_out as _, f, tid, nt,
                    );
                });
            });
            let t_ffn_act = 0u128; // fused into gate+up

            // 17. Quantize hidden -> Q8K
            prof!(t_down, {
                unsafe {
                    ffi::quant_f32_q8k(
                        self.hidden.as_ptr(), self.hidden_q8_qs.as_mut_ptr(),
                        self.hidden_q8_d.as_mut_ptr(), self.hidden_q8_bsums.as_mut_ptr(),
                        f as i32,
                    );
                }
                // 18. Matmul: down = w_down x hidden_q8k (may be Q6_K)
                if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
                    q6k_matmul_mt(
                        lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb,
                        self.hidden_q8_qs.as_ptr(), self.hidden_q8_d.as_ptr(),
                        self.hidden_q8_bsums.as_ptr(),
                        &mut self.tmp, h, &self.pool,
                    );
                } else {
                    q4k_matmul_mt(
                        lw.w_down, f_row_stride, f_nb,
                        self.hidden_q8_qs.as_ptr(), self.hidden_q8_d.as_ptr(),
                        self.hidden_q8_bsums.as_ptr(),
                        &mut self.tmp, h, &self.pool,
                    );
                }
            });

            // 19. Residual: x = x + down
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(), self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);
        }

        // 20. Final RMSNorm
        unsafe {
            ffi::rmsnorm_f32(
                self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(),
                h as i32, model.rms_eps,
            );
        }

        // 21-22. Quantize and output projection
        prof!(t_out, {
            unsafe {
                ffi::quant_f32_q8k(
                    self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                    self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(),
                    h as i32,
                );
            }
            if model.output_block_bytes == Q6K_BLOCK_BYTES {
                q6k_matmul_mt(
                    model.output_weight, h_nb * Q6K_BLOCK_BYTES, h_nb,
                    self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(),
                    self.x_q8_bsums.as_ptr(),
                    &mut self.logits, model.vocab_size, &self.pool,
                );
            } else {
                q4k_matmul_mt(
                    model.output_weight, h_row_stride, h_nb,
                    self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(),
                    self.x_q8_bsums.as_ptr(),
                    &mut self.logits, model.vocab_size, &self.pool,
                );
            }
        });

        if profile {
            let ms = |ns: u128| ns as f64 / 1_000_000.0;
            let total = t_qkv + t_attn + t_oproj + t_ffn_gu + t_ffn_act + t_down + t_out;
            let pct = |ns: u128| if total > 0 { ns as f64 / total as f64 * 100.0 } else { 0.0 };
            eprintln!("\n--- profile (pos=1, {} layers) ---", model.n_layers);
            eprintln!("  QKV matmul:   {:6.1}ms  ({:.0}%)", ms(t_qkv), pct(t_qkv));
            eprintln!("  attention:    {:6.1}ms  ({:.0}%)", ms(t_attn), pct(t_attn));
            eprintln!("  O proj:       {:6.1}ms  ({:.0}%)", ms(t_oproj), pct(t_oproj));
            eprintln!("  FFN gate+up:  {:6.1}ms  ({:.0}%)", ms(t_ffn_gu), pct(t_ffn_gu));
            eprintln!("  FFN silu:     {:6.1}ms  ({:.0}%)", ms(t_ffn_act), pct(t_ffn_act));
            eprintln!("  FFN down:     {:6.1}ms  ({:.0}%)", ms(t_down), pct(t_down));
            eprintln!("  output (Q4K): {:6.1}ms  ({:.0}%)", ms(t_out), pct(t_out));
            eprintln!("  total:        {:6.1}ms", ms(total));
        }
    }

    pub fn apply_repetition_penalty(&mut self, generated: &[u32], penalty: f32) {
        if penalty == 1.0 { return; }
        for &tok in generated {
            let idx = tok as usize;
            if idx < self.logits.len() {
                if self.logits[idx] > 0.0 {
                    self.logits[idx] /= penalty;
                } else {
                    self.logits[idx] *= penalty;
                }
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
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get()).unwrap_or(1);
    let mut state = LlamaState::new(model, max_seq_len);
    let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

    let prefill_start = Instant::now();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        state.forward(model, tok, i);
        output.push(tok);
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let first_tok_start = Instant::now();
    let mut pos = prompt_tokens.len();
    let mut n_gen = 0u32;
    let mut first_tok_ms = 0.0;

    let decode_start = Instant::now();
    for step in 0..max_tokens {
        if pos >= max_seq_len {
            break;
        }
        state.apply_repetition_penalty(&output, repetition_penalty);
        let next = state.sample_logits(temperature);
        if next == eos_id {
            break;
        }
        output.push(next);
        on_token(next);
        if step == 0 {
            first_tok_ms = first_tok_start.elapsed().as_secs_f64() * 1000.0;
        }
        state.forward(model, next, pos);
        pos += 1;
        n_gen += 1;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    let prefill_tps = prompt_tokens.len() as f64 / (prefill_ms / 1000.0);
    let decode_tps = if n_gen > 0 { n_gen as f64 / (decode_ms / 1000.0) } else { 0.0 };
    let avg_tok_ms = if n_gen > 0 { decode_ms / n_gen as f64 } else { 0.0 };

    eprintln!("\n--- perf ({} threads) ---", n_threads);
    eprintln!(
        "prefill:    {} tokens in {:.0}ms ({:.1} tok/s)",
        prompt_tokens.len(), prefill_ms, prefill_tps,
    );
    eprintln!("first tok:  {:.0}ms", first_tok_ms);
    eprintln!(
        "decode:     {} tokens in {:.0}ms ({:.1} tok/s, {:.1}ms/tok)",
        n_gen, decode_ms, decode_tps, avg_tok_ms,
    );

    (output, prefill_ms, decode_ms)
}
