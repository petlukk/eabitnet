//! Transformer forward pass for BitNet b1.58 2B-4T.

use crate::ffi;
use crate::model::BitNetModel;

pub struct InferenceState {
    x: Vec<f32>,
    x_norm: Vec<f32>,
    x_quant: Vec<i8>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    attn_out_quant: Vec<i8>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    hidden_quant: Vec<i8>,
    logits: Vec<f32>,
    tmp: Vec<f32>,
    raw_scores: Vec<i32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    rope_freqs: Vec<f32>,
    max_seq_len: usize,
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3ff;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13));
    }
    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
}

fn embed_f16_lookup(embed: *const u8, token: u32, out: &mut [f32], hidden_dim: usize) {
    let row = unsafe {
        std::slice::from_raw_parts(
            embed.add(token as usize * hidden_dim * 2) as *const u16,
            hidden_dim,
        )
    };
    for i in 0..hidden_dim {
        out[i] = f16_to_f32(row[i]);
    }
}

fn f16_matmul(embed: *const u8, x: &[f32], out: &mut [f32], vocab_size: usize, hidden_dim: usize) {
    for v in 0..vocab_size {
        let row = unsafe {
            std::slice::from_raw_parts(
                embed.add(v * hidden_dim * 2) as *const u16,
                hidden_dim,
            )
        };
        let mut dot = 0.0f32;
        for d in 0..hidden_dim {
            dot += f16_to_f32(row[d]) * x[d];
        }
        out[v] = dot;
    }
}

fn build_rope_freqs(freqs: &mut [f32], head_dim: usize, pos: usize, theta: f32) {
    for i in 0..head_dim / 2 {
        let angle = pos as f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
        freqs[2 * i] = angle.cos();
        freqs[2 * i + 1] = angle.sin();
    }
}

fn apply_rope(data: &mut [f32], freqs: &[f32], head_dim: usize, n_heads: usize) {
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..head_dim / 2 {
            let cos = freqs[2 * i];
            let sin = freqs[2 * i + 1];
            let r = data[off + 2 * i];
            let im = data[off + 2 * i + 1];
            data[off + 2 * i] = r * cos - im * sin;
            data[off + 2 * i + 1] = r * sin + im * cos;
        }
    }
}

fn ternary_matmul(
    weight: *const u8,
    act: *const i8,
    act_scale: f32,
    act_sum: i32,
    weight_scale: f32,
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    raw_buf: &mut [i32],
) {
    let row_bytes = in_dim / 4;
    let mut r = 0;
    unsafe {
        while r + 4 <= out_dim {
            let w0 = weight.add(r * row_bytes);
            let w1 = weight.add((r + 1) * row_bytes);
            let w2 = weight.add((r + 2) * row_bytes);
            let w3 = weight.add((r + 3) * row_bytes);
            ffi::i2_dot_i8_4row(w0, w1, w2, w3, act, raw_buf[r..].as_mut_ptr(), in_dim as i32);
            r += 4;
        }
        while r < out_dim {
            raw_buf[r] = ffi::i2_dot_i8(weight.add(r * row_bytes), act, in_dim as i32);
            r += 1;
        }
    }
    let scale = (act_scale / 127.0) * weight_scale;
    for i in 0..out_dim {
        out[i] = (raw_buf[i] - act_sum) as f32 * scale;
    }
}

fn sample(logits: &[f32], temperature: f32) -> u32 {
    if temperature <= 0.0 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    } else {
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        scaled
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }
}

impl InferenceState {
    pub fn new(model: &BitNetModel, max_seq_len: usize) -> Self {
        let h = model.hidden_dim;
        let f = model.ffn_dim;
        let v = model.vocab_size;
        let max_raw = h.max(f).max(v);
        let kv_cache_size = model.n_layers * model.n_kv_heads * max_seq_len * model.head_dim;
        InferenceState {
            x: vec![0.0; h],
            x_norm: vec![0.0; h],
            x_quant: vec![0; h + 12],
            q: vec![0.0; h],
            k: vec![0.0; model.kv_dim],
            v: vec![0.0; model.kv_dim],
            attn_out: vec![0.0; h],
            attn_out_quant: vec![0; h + 12],
            scores: vec![0.0; max_seq_len],
            gate: vec![0.0; f],
            up: vec![0.0; f],
            hidden: vec![0.0; f],
            hidden_quant: vec![0; f + 12],
            logits: vec![0.0; v],
            tmp: vec![0.0; h.max(f)],
            raw_scores: vec![0; max_raw],
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

        // F16 embedding lookup
        embed_f16_lookup(model.embed_weight, token, &mut self.x, h);

        for layer in 0..model.n_layers {
            let lw = &model.layers[layer];

            // Attention norm
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(), lw.attn_norm, self.x_norm.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }

            // Quantize normed input
            let mut act_scale: f32 = 0.0;
            let mut act_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.x_norm.as_ptr(), self.x_quant.as_mut_ptr(),
                    &mut act_scale, &mut act_sum, h as i32,
                );
            }

            // Q, K, V projections (K/V use kv_dim for GQA)
            ternary_matmul(
                lw.wq, self.x_quant.as_ptr(), act_scale, act_sum, lw.wq_scale,
                &mut self.q, h, h, &mut self.raw_scores,
            );
            ternary_matmul(
                lw.wk, self.x_quant.as_ptr(), act_scale, act_sum, lw.wk_scale,
                &mut self.k, kv, h, &mut self.raw_scores,
            );
            ternary_matmul(
                lw.wv, self.x_quant.as_ptr(), act_scale, act_sum, lw.wv_scale,
                &mut self.v, kv, h, &mut self.raw_scores,
            );

            // RoPE — apply separately for Q (nh heads) and K (nkv heads)
            build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
            apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
            apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);

            // Store K, V in cache: layout [layer][kv_head][max_seq][head_dim]
            for head in 0..nkv {
                let off = ((layer * nkv + head) * self.max_seq_len + pos) * hd;
                self.k_cache[off..off + hd]
                    .copy_from_slice(&self.k[head * hd..(head + 1) * hd]);
                self.v_cache[off..off + hd]
                    .copy_from_slice(&self.v[head * hd..(head + 1) * hd]);
            }

            // Attention per Q head (GQA: multiple Q heads share one KV head)
            let scale = 1.0 / (hd as f32).sqrt();
            for head in 0..nh {
                let kv_head = head / gqa_ratio;
                let q_off = head * hd;
                let cache_base = (layer * nkv + kv_head) * self.max_seq_len * hd;
                unsafe {
                    ffi::attn_scores_f32(
                        self.q.as_ptr().add(q_off),
                        self.k_cache.as_ptr().add(cache_base),
                        self.scores.as_mut_ptr(),
                        hd as i32, seq_len as i32, scale,
                    );
                    ffi::softmax_f32(
                        self.scores.as_ptr(), self.scores.as_mut_ptr(), seq_len as i32,
                    );
                    ffi::attn_weighted_sum_f32(
                        self.scores.as_ptr(),
                        self.v_cache.as_ptr().add(cache_base),
                        self.attn_out.as_mut_ptr().add(q_off),
                        hd as i32, seq_len as i32,
                    );
                }
            }

            // attn_sub_norm (RMSNorm) applied to attention output BEFORE O projection
            unsafe {
                ffi::rmsnorm_f32(
                    self.attn_out.as_ptr(), lw.attn_sub_norm, self.attn_out.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }

            // Quantize sub-normed attention output, O projection
            let mut attn_scale: f32 = 0.0;
            let mut attn_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.attn_out.as_ptr(), self.attn_out_quant.as_mut_ptr(),
                    &mut attn_scale, &mut attn_sum, h as i32,
                );
            }
            ternary_matmul(
                lw.wo, self.attn_out_quant.as_ptr(), attn_scale, attn_sum, lw.wo_scale,
                &mut self.tmp, h, h, &mut self.raw_scores,
            );

            // Residual: x = x + O
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(), self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);

            // FFN norm
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(), lw.ffn_norm, self.x_norm.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }

            // Quantize for FFN
            let mut ffn_scale: f32 = 0.0;
            let mut ffn_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.x_norm.as_ptr(), self.x_quant.as_mut_ptr(),
                    &mut ffn_scale, &mut ffn_sum, h as i32,
                );
            }

            // Gate and Up projections
            ternary_matmul(
                lw.w_gate, self.x_quant.as_ptr(), ffn_scale, ffn_sum, lw.w_gate_scale,
                &mut self.gate, f, h, &mut self.raw_scores,
            );
            ternary_matmul(
                lw.w_up, self.x_quant.as_ptr(), ffn_scale, ffn_sum, lw.w_up_scale,
                &mut self.up, f, h, &mut self.raw_scores,
            );

            // Squared ReLU activation: hidden = relu²(gate) * up
            unsafe {
                ffi::squared_relu_mul_f32(
                    self.gate.as_ptr(), self.up.as_ptr(),
                    self.hidden.as_mut_ptr(), f as i32,
                );
            }

            // ffn_sub_norm (RMSNorm) on FFN intermediate before down projection
            unsafe {
                ffi::rmsnorm_f32(
                    self.hidden.as_ptr(), lw.ffn_sub_norm, self.hidden.as_mut_ptr(),
                    f as i32, model.rms_eps,
                );
            }

            // Quantize FFN hidden, down projection
            let mut down_scale: f32 = 0.0;
            let mut down_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.hidden.as_ptr(), self.hidden_quant.as_mut_ptr(),
                    &mut down_scale, &mut down_sum, f as i32,
                );
            }
            ternary_matmul(
                lw.w_down, self.hidden_quant.as_ptr(), down_scale, down_sum, lw.w_down_scale,
                &mut self.tmp, h, f, &mut self.raw_scores,
            );

            // Residual: x = x + down
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(), self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);

        }

        // Final norm
        unsafe {
            ffi::rmsnorm_f32(
                self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(),
                h as i32, model.rms_eps,
            );
        }

        // Tied embedding output: F16 matmul
        f16_matmul(
            model.embed_weight, &self.x_norm, &mut self.logits,
            model.vocab_size, h,
        );
    }

    pub fn sample_logits(&self, temperature: f32) -> u32 {
        sample(&self.logits, temperature)
    }

    pub fn generate(
        model: &BitNetModel,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        eos_id: u32,
        max_seq_len: usize,
        stream: bool,
    ) -> (Vec<u32>, f64, f64) {
        use std::time::Instant;
        let mut state = InferenceState::new(model, max_seq_len);
        let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

        let prefill_start = Instant::now();
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            state.forward(model, tok, i);
            output.push(tok);
        }
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = Instant::now();
        let mut pos = prompt_tokens.len();
        let mut n_gen = 0u32;
        for _ in 0..max_tokens {
            if pos >= max_seq_len {
                break;
            }
            let next = state.sample_logits(temperature);
            if next == eos_id {
                break;
            }
            output.push(next);
            if stream {
                // Caller handles streaming — we just track count
            }
            state.forward(model, next, pos);
            pos += 1;
            n_gen += 1;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        let prefill_tps = prompt_tokens.len() as f64 / (prefill_ms / 1000.0);
        let decode_tps = if n_gen > 0 { n_gen as f64 / (decode_ms / 1000.0) } else { 0.0 };

        eprintln!("\n--- perf ---");
        eprintln!("prefill: {} tokens in {:.0}ms ({:.1} tok/s)", prompt_tokens.len(), prefill_ms, prefill_tps);
        eprintln!("decode:  {} tokens in {:.0}ms ({:.1} tok/s)", n_gen, decode_ms, decode_tps);

        (output, prefill_ms, decode_ms)
    }
}
