//! Transformer forward pass for BitNet b1.58 2B-4T.

use crate::ffi;
use crate::matmul::{embed_f16_lookup, i8_output_matmul_mt, ternary_matmul_mt, ternary_matmul_parallel_pair, ternary_matmul_qkv};
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
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    hidden_quant: Vec<i8>,
    logits: Vec<f32>,
    tmp: Vec<f32>,
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

fn argmax(s: &[f32]) -> u32 {
    s.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32).unwrap_or(0)
}

fn sample(logits: &[f32], temperature: f32) -> u32 {
    if temperature <= 0.0 { return argmax(logits); }
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    argmax(&scaled)
}

impl InferenceState {
    pub fn new(model: &BitNetModel, max_seq_len: usize) -> Self {
        let h = model.hidden_dim;
        let f = model.ffn_dim;
        let v = model.vocab_size;
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
            gate: vec![0.0; f],
            up: vec![0.0; f],
            hidden: vec![0.0; f],
            hidden_quant: vec![0; f + 12],
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

        embed_f16_lookup(model.embed_weight_f16, token, &mut self.x, h);

        use std::time::Instant;
        let profile = pos == 1;
        let mut t_qkv = 0u128;
        let mut t_attn = 0u128;
        let mut t_oproj = 0u128;
        let mut t_ffn_gu = 0u128;
        let mut t_ffn_act = 0u128;
        let mut t_down = 0u128;
        let mut t_out = 0u128;
        macro_rules! tick { () => { if profile { Instant::now() } else { Instant::now() } } }
        macro_rules! tock { ($t:expr, $s:expr) => { if profile { $t += $s.elapsed().as_nanos(); } } }

        for layer in 0..model.n_layers {
            let lw = &model.layers[layer];
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(), lw.attn_norm, self.x_norm.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }
            let mut act_scale: f32 = 0.0;
            let mut act_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.x_norm.as_ptr(), self.x_quant.as_mut_ptr(),
                    &mut act_scale, &mut act_sum, h as i32,
                );
            }
            let s = tick!();
            ternary_matmul_qkv(
                lw.wq, lw.wq_scale, &mut self.q, h,
                lw.wk, lw.wk_scale, &mut self.k, kv,
                lw.wv, lw.wv_scale, &mut self.v,
                self.x_quant.as_ptr(), act_scale, act_sum, h,
            );
            tock!(t_qkv, s);
            build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
            apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
            apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);
            for head in 0..nkv {
                let off = ((layer * nkv + head) * self.max_seq_len + pos) * hd;
                self.k_cache[off..off + hd]
                    .copy_from_slice(&self.k[head * hd..(head + 1) * hd]);
                self.v_cache[off..off + hd]
                    .copy_from_slice(&self.v[head * hd..(head + 1) * hd]);
            }
            let s = tick!();
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
            tock!(t_attn, s);
            let s = tick!();
            unsafe {
                ffi::rmsnorm_f32(
                    self.attn_out.as_ptr(), lw.attn_sub_norm, self.attn_out.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }
            let mut attn_scale: f32 = 0.0;
            let mut attn_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.attn_out.as_ptr(), self.attn_out_quant.as_mut_ptr(),
                    &mut attn_scale, &mut attn_sum, h as i32,
                );
            }
            ternary_matmul_mt(
                lw.wo, self.attn_out_quant.as_ptr(), attn_scale, attn_sum, lw.wo_scale,
                &mut self.tmp, h, h,
            );
            tock!(t_oproj, s);
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(), self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(), lw.ffn_norm, self.x_norm.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }

            let mut ffn_scale: f32 = 0.0;
            let mut ffn_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.x_norm.as_ptr(), self.x_quant.as_mut_ptr(),
                    &mut ffn_scale, &mut ffn_sum, h as i32,
                );
            }

            let s = tick!();
            ternary_matmul_parallel_pair(
                lw.w_gate, lw.w_gate_scale,
                lw.w_up, lw.w_up_scale,
                self.x_quant.as_ptr(), ffn_scale, ffn_sum,
                &mut self.gate, &mut self.up,
                f, h,
            );
            tock!(t_ffn_gu, s);

            let s = tick!();
            unsafe {
                ffi::squared_relu_mul_f32(
                    self.gate.as_ptr(), self.up.as_ptr(),
                    self.hidden.as_mut_ptr(), f as i32,
                );
            }
            unsafe {
                ffi::rmsnorm_f32(
                    self.hidden.as_ptr(), lw.ffn_sub_norm, self.hidden.as_mut_ptr(),
                    f as i32, model.rms_eps,
                );
            }
            tock!(t_ffn_act, s);

            let s = tick!();
            let mut down_scale: f32 = 0.0;
            let mut down_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.hidden.as_ptr(), self.hidden_quant.as_mut_ptr(),
                    &mut down_scale, &mut down_sum, f as i32,
                );
            }
            ternary_matmul_mt(
                lw.w_down, self.hidden_quant.as_ptr(), down_scale, down_sum, lw.w_down_scale,
                &mut self.tmp, h, f,
            );
            tock!(t_down, s);

            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(), self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);
        }

        unsafe {
            ffi::rmsnorm_f32(
                self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(),
                h as i32, model.rms_eps,
            );
        }

        let s = tick!();
        i8_output_matmul_mt(
            &model.embed_weight_i8, &model.embed_row_scales,
            &self.x_norm, &mut self.logits,
            model.vocab_size, h,
        );
        tock!(t_out, s);

        if profile {
            let ms = |ns: u128| ns as f64 / 1_000_000.0;
            let total = t_qkv + t_attn + t_oproj + t_ffn_gu + t_ffn_act + t_down + t_out;
            let pct = |ns: u128| if total > 0 { ns as f64 / total as f64 * 100.0 } else { 0.0 };
            eprintln!("\n--- profile (pos=1, 30 layers) ---");
            eprintln!("  QKV matmul:   {:6.1}ms  ({:.0}%)", ms(t_qkv), pct(t_qkv));
            eprintln!("  attention:    {:6.1}ms  ({:.0}%)", ms(t_attn), pct(t_attn));
            eprintln!("  O proj:       {:6.1}ms  ({:.0}%)", ms(t_oproj), pct(t_oproj));
            eprintln!("  FFN gate+up:  {:6.1}ms  ({:.0}%)", ms(t_ffn_gu), pct(t_ffn_gu));
            eprintln!("  FFN act+norm: {:6.1}ms  ({:.0}%)", ms(t_ffn_act), pct(t_ffn_act));
            eprintln!("  FFN down:     {:6.1}ms  ({:.0}%)", ms(t_down), pct(t_down));
            eprintln!("  output (f16): {:6.1}ms  ({:.0}%)", ms(t_out), pct(t_out));
            eprintln!("  total:        {:6.1}ms", ms(total));
        }
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
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get()).unwrap_or(1);
        let mut state = InferenceState::new(model, max_seq_len);
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
            let next = state.sample_logits(temperature);
            if next == eos_id {
                break;
            }
            output.push(next);
            if step == 0 {
                first_tok_ms = first_tok_start.elapsed().as_secs_f64() * 1000.0;
            }
            if stream {
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
}
