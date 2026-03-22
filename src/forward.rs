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
    logits_quant: Vec<i8>,
    tmp: Vec<f32>,
    raw_scores: Vec<i32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    rope_freqs: Vec<f32>,
    max_seq_len: usize,
}

fn build_rope_freqs(freqs: &mut [f32], head_dim: usize, pos: usize, theta: f32) {
    for i in 0..head_dim / 2 {
        let angle = pos as f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
        freqs[2 * i] = angle.cos();
        freqs[2 * i + 1] = angle.sin();
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
        let cache_size = model.n_layers * max_seq_len * h;
        InferenceState {
            x: vec![0.0; h],
            x_norm: vec![0.0; h],
            x_quant: vec![0; h + 12],
            q: vec![0.0; h],
            k: vec![0.0; h],
            v: vec![0.0; h],
            attn_out: vec![0.0; h],
            attn_out_quant: vec![0; h + 12],
            scores: vec![0.0; max_seq_len],
            gate: vec![0.0; f],
            up: vec![0.0; f],
            hidden: vec![0.0; f],
            hidden_quant: vec![0; f + 12],
            logits: vec![0.0; v],
            logits_quant: vec![0; h + 12],
            tmp: vec![0.0; h.max(f)],
            raw_scores: vec![0; max_raw],
            k_cache: vec![0.0; cache_size],
            v_cache: vec![0.0; cache_size],
            rope_freqs: vec![0.0; model.head_dim],
            max_seq_len,
        }
    }

    /// Run one token through the model, update KV cache. Returns logits index via sample.
    pub fn forward(&mut self, model: &BitNetModel, token: u32, pos: usize) {
        let h = model.hidden_dim;
        let hd = model.head_dim;
        let nh = model.n_heads;
        let f = model.ffn_dim;
        let seq_len = pos + 1;

        // Embedding lookup
        unsafe {
            let embed_ptr = model.embed_weight.add(token as usize * h);
            std::ptr::copy_nonoverlapping(embed_ptr, self.x.as_mut_ptr(), h);
        }

        for layer in 0..model.n_layers {
            let lw = &model.layers[layer];

            // Attention norm
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(),
                    lw.attn_norm,
                    self.x_norm.as_mut_ptr(),
                    h as i32,
                    model.rms_eps,
                );
            }

            // Quantize normed input
            let mut act_scale: f32 = 0.0;
            let mut act_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.x_norm.as_ptr(),
                    self.x_quant.as_mut_ptr(),
                    &mut act_scale,
                    &mut act_sum,
                    h as i32,
                );
            }

            // Q, K, V projections
            ternary_matmul(
                lw.wq, self.x_quant.as_ptr(), act_scale, act_sum, 1.0,
                &mut self.q, h, h, &mut self.raw_scores,
            );
            ternary_matmul(
                lw.wk, self.x_quant.as_ptr(), act_scale, act_sum, 1.0,
                &mut self.k, h, h, &mut self.raw_scores,
            );
            ternary_matmul(
                lw.wv, self.x_quant.as_ptr(), act_scale, act_sum, 1.0,
                &mut self.v, h, h, &mut self.raw_scores,
            );

            // RoPE
            build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
            unsafe {
                ffi::rope_f32(
                    self.q.as_mut_ptr(),
                    self.k.as_mut_ptr(),
                    self.rope_freqs.as_ptr(),
                    hd as i32,
                    nh as i32,
                );
            }

            // Store K, V in cache: layout [layer][head][max_seq][head_dim]
            for head in 0..nh {
                let off = ((layer * nh + head) * self.max_seq_len + pos) * hd;
                self.k_cache[off..off + hd]
                    .copy_from_slice(&self.k[head * hd..(head + 1) * hd]);
                self.v_cache[off..off + hd]
                    .copy_from_slice(&self.v[head * hd..(head + 1) * hd]);
            }

            // Attention per head
            let scale = 1.0 / (hd as f32).sqrt();
            for head in 0..nh {
                let q_off = head * hd;
                let cache_base = (layer * nh + head) * self.max_seq_len * hd;
                unsafe {
                    ffi::attn_scores_f32(
                        self.q.as_ptr().add(q_off),
                        self.k_cache.as_ptr().add(cache_base),
                        self.scores.as_mut_ptr(),
                        hd as i32,
                        seq_len as i32,
                        scale,
                    );
                    ffi::softmax_f32(
                        self.scores.as_ptr(),
                        self.scores.as_mut_ptr(),
                        seq_len as i32,
                    );
                    ffi::attn_weighted_sum_f32(
                        self.scores.as_ptr(),
                        self.v_cache.as_ptr().add(cache_base),
                        self.attn_out.as_mut_ptr().add(q_off),
                        hd as i32,
                        seq_len as i32,
                    );
                }
            }

            // Quantize attention output
            let mut attn_scale: f32 = 0.0;
            let mut attn_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.attn_out.as_ptr(),
                    self.attn_out_quant.as_mut_ptr(),
                    &mut attn_scale,
                    &mut attn_sum,
                    h as i32,
                );
            }

            // Output projection
            ternary_matmul(
                lw.wo, self.attn_out_quant.as_ptr(), attn_scale, attn_sum, 1.0,
                &mut self.tmp, h, h, &mut self.raw_scores,
            );

            // Residual: x = x + O (use tmp, then copy back)
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(),
                    self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), // borrow attn_out as temp dest
                    h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);

            // FFN norm
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(),
                    lw.ffn_norm,
                    self.x_norm.as_mut_ptr(),
                    h as i32,
                    model.rms_eps,
                );
            }

            // Quantize for FFN
            let mut ffn_scale: f32 = 0.0;
            let mut ffn_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.x_norm.as_ptr(),
                    self.x_quant.as_mut_ptr(),
                    &mut ffn_scale,
                    &mut ffn_sum,
                    h as i32,
                );
            }

            // Gate and Up projections
            ternary_matmul(
                lw.w_gate, self.x_quant.as_ptr(), ffn_scale, ffn_sum, 1.0,
                &mut self.gate, f, h, &mut self.raw_scores,
            );
            ternary_matmul(
                lw.w_up, self.x_quant.as_ptr(), ffn_scale, ffn_sum, 1.0,
                &mut self.up, f, h, &mut self.raw_scores,
            );

            // Activation: squared_relu_mul
            unsafe {
                ffi::squared_relu_mul_f32(
                    self.gate.as_ptr(),
                    self.up.as_ptr(),
                    self.hidden.as_mut_ptr(),
                    f as i32,
                );
            }

            // Quantize FFN hidden
            let mut down_scale: f32 = 0.0;
            let mut down_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.hidden.as_ptr(),
                    self.hidden_quant.as_mut_ptr(),
                    &mut down_scale,
                    &mut down_sum,
                    f as i32,
                );
            }

            // Down projection
            ternary_matmul(
                lw.w_down, self.hidden_quant.as_ptr(), down_scale, down_sum, 1.0,
                &mut self.tmp, h, f, &mut self.raw_scores,
            );

            // Residual: x = x + down
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(),
                    self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), // borrow as temp dest
                    h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);
        }

        // Final norm
        unsafe {
            ffi::rmsnorm_f32(
                self.x.as_ptr(),
                model.norm_weight,
                self.x_norm.as_mut_ptr(),
                h as i32,
                model.rms_eps,
            );
        }

        // Quantize for output projection
        let mut out_scale: f32 = 0.0;
        let mut out_sum: i32 = 0;
        unsafe {
            ffi::quant_f32_i8(
                self.x_norm.as_ptr(),
                self.logits_quant.as_mut_ptr(),
                &mut out_scale,
                &mut out_sum,
                h as i32,
            );
        }

        // Output projection: hidden_dim → vocab_size
        ternary_matmul(
            model.output_weight, self.logits_quant.as_ptr(), out_scale, out_sum, 1.0,
            &mut self.logits, model.vocab_size, h, &mut self.raw_scores,
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
    ) -> Vec<u32> {
        let mut state = InferenceState::new(model, max_seq_len);
        let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

        // Process prompt (prefill)
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            state.forward(model, tok, i);
            output.push(tok);
        }

        // Generate
        let mut pos = prompt_tokens.len();
        for _ in 0..max_tokens {
            if pos >= max_seq_len {
                break;
            }
            let next = state.sample_logits(temperature);
            if next == eos_id {
                break;
            }
            output.push(next);
            state.forward(model, next, pos);
            pos += 1;
        }

        output
    }
}
