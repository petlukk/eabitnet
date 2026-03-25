# Batched Prefill for BitNet I2S — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the ~4.7x prefill speed gap with bitnet.cpp by batching the BitNet I2S prefill — load weights once per layer, multiply all prompt tokens.

**Architecture:** Mirror the Q4K batched prefill pattern (`gemm_q4k.rs` + `forward_llama.rs:prefill()`). Create `gemm_i2s.rs` with `BatchI8` struct and `i2s_gemm_mt()` / `i2s_fused_sqrelu_gemm_mt()`. Add `prefill()` as an `impl InferenceState` block in a new `src/prefill.rs` file (forward.rs is 439 lines — adding prefill there would exceed the 500-line limit). No new `.ea` kernels — reuse existing `i2_dot_i8_4row` and `i2_dot_i8_4row_dual`, amortizing weight loads across tokens via inner loop.

**Tech Stack:** Rust, existing Ea SIMD kernels, x86_64 only

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/gemm_i2s.rs` | **Create** | `BatchI8` struct + `i2s_gemm_mt()` + `i2s_fused_sqrelu_gemm_mt()` |
| `src/prefill.rs` | **Create** | `impl InferenceState { fn prefill() }` — batched forward pass |
| `src/forward.rs` | **Modify** | Update `generate()` to call `prefill()` instead of sequential loop |
| `src/main.rs` | **Modify** | Add `mod gemm_i2s;` and `mod prefill;` declarations |
| `src/gemm_i2s_tests.rs` | **Create** | Tests for batched GEMM correctness |

---

### Task 1: BatchI8 struct and i2s_gemm_mt

**Files:**
- Create: `src/gemm_i2s.rs`
- Create: `src/gemm_i2s_tests.rs`

- [ ] **Step 1: Write the failing test for BatchI8 quantization**

Create `src/gemm_i2s_tests.rs`:
```rust
use crate::gemm_i2s::BatchI8;

#[test]
fn batch_i8_quantize_roundtrip() {
    let dim = 256;
    let mut batch = BatchI8::new(2, dim);
    let x0: Vec<f32> = (0..dim).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let x1: Vec<f32> = (0..dim).map(|i| (i as f32) / 256.0).collect();
    batch.quantize(0, &x0);
    batch.quantize(1, &x1);
    // Check scales are nonzero
    assert!(batch.scale(0).abs() > 1e-10);
    assert!(batch.scale(1).abs() > 1e-10);
    // Check sums are computed
    assert_ne!(batch.sum(0), 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test gemm_i2s_tests::batch_i8_quantize_roundtrip -- --nocapture`
Expected: FAIL — module `gemm_i2s` not found

- [ ] **Step 3: Implement BatchI8 struct**

Create `src/gemm_i2s.rs` mirroring `src/gemm_q4k.rs:8-51` (BatchQ8K pattern):
```rust
//! GEMM-style batched I2S matmul: load weight once, multiply against N tokens.

use crate::ffi;
use crate::threadpool::ThreadPool;

/// Batched i8 activation data for N tokens (I2S quantization).
pub(crate) struct BatchI8 {
    pub n_tokens: usize,
    pub dim: usize,
    pub stride: usize,  // dim + 12 padding for kernel overshoot
    pub scales: Vec<f32>,
    pub sums: Vec<i32>,
    pub qs: Vec<i8>,
}

impl BatchI8 {
    pub fn new(n_tokens: usize, dim: usize) -> Self {
        let stride = dim + 12;
        BatchI8 {
            n_tokens, dim, stride,
            scales: vec![0.0f32; n_tokens],
            sums: vec![0i32; n_tokens],
            qs: vec![0i8; n_tokens * stride],
        }
    }

    pub fn quantize(&mut self, t: usize, src: &[f32]) {
        unsafe {
            ffi::quant_f32_i8(
                src.as_ptr(),
                self.qs.as_mut_ptr().add(t * self.stride),
                self.scales.as_mut_ptr().add(t),
                self.sums.as_mut_ptr().add(t),
                self.dim as i32,
            );
        }
    }

    pub fn qs_ptr(&self, t: usize) -> *const i8 {
        unsafe { self.qs.as_ptr().add(t * self.stride) }
    }
    pub fn scale(&self, t: usize) -> f32 { self.scales[t] }
    pub fn sum(&self, t: usize) -> i32 { self.sums[t] }
}
```

Also add `mod gemm_i2s;` to `src/main.rs` (line 10, after `mod gemm_q6k;`).

Add to the bottom of `src/gemm_i2s.rs`:
```rust
#[cfg(test)]
#[path = "gemm_i2s_tests.rs"]
mod tests;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test gemm_i2s_tests::batch_i8_quantize_roundtrip -- --nocapture`
Expected: PASS

- [ ] **Step 5: Write the failing test for i2s_gemm_mt**

Add to `src/gemm_i2s_tests.rs`:
```rust
use crate::gemm_i2s::{BatchI8, i2s_gemm_mt};
use crate::matmul::ternary_matmul_mt;
use crate::threadpool::ThreadPool;

/// Pack ternary weights {-1,0,+1} → 2-bit packed bytes.
/// (Copy the helper from matmul_tests.rs)
fn pack_ternary(weights: &[i8], n: usize) -> Vec<u8> {
    // ... same as matmul_tests.rs packing helper
}

#[test]
fn i2s_gemm_matches_sequential() {
    let pool = ThreadPool::new();
    let in_dim = 256;
    let out_dim = 64;
    let n_tokens = 4;
    let weight_scale = 1.5f32;

    // Random ternary weights
    let weights_raw: Vec<i8> = (0..out_dim * in_dim)
        .map(|i| ((i % 3) as i8 - 1))
        .collect();
    let packed = pack_ternary(&weights_raw, in_dim);

    // Random activations
    let xs: Vec<Vec<f32>> = (0..n_tokens)
        .map(|t| (0..in_dim).map(|i| ((t * in_dim + i) as f32 - 500.0) / 500.0).collect())
        .collect();

    // Quantize into batch
    let mut batch = BatchI8::new(n_tokens, in_dim);
    for t in 0..n_tokens {
        batch.quantize(t, &xs[t]);
    }

    // GEMM path
    let mut gemm_out = vec![0.0f32; n_tokens * out_dim];
    i2s_gemm_mt(packed.as_ptr(), weight_scale, &batch, &mut gemm_out, out_dim, in_dim, &pool);

    // Sequential path (reference)
    for t in 0..n_tokens {
        let mut seq_out = vec![0.0f32; out_dim];
        ternary_matmul_mt(
            packed.as_ptr(), batch.qs_ptr(t),
            batch.scale(t), batch.sum(t), weight_scale,
            &mut seq_out, out_dim, in_dim, &pool,
        );
        for r in 0..out_dim {
            assert!((gemm_out[t * out_dim + r] - seq_out[r]).abs() < 1e-4,
                "mismatch at token {t} row {r}: gemm={} seq={}", gemm_out[t * out_dim + r], seq_out[r]);
        }
    }
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test gemm_i2s_tests::i2s_gemm_matches_sequential -- --nocapture`
Expected: FAIL — `i2s_gemm_mt` not found

- [ ] **Step 7: Implement i2s_gemm_mt**

Add to `src/gemm_i2s.rs`, mirroring `gemm_q4k.rs:55-108`:
```rust
/// GEMM: weight[out_dim x in_dim] x batch[n_tokens] -> out[n_tokens * out_dim]
/// Output layout: out[t * out_dim + row]
pub(crate) fn i2s_gemm_mt(
    weight: *const u8,
    weight_scale: f32,
    batch: &BatchI8,
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    pool: &ThreadPool,
) {
    let nt = batch.n_tokens;
    let row_bytes = in_dim / 4;
    let n_threads = pool.thread_count().min(out_dim / 4).max(1);
    let w = weight as usize;
    let out_p = out.as_mut_ptr() as usize;
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let scales: Vec<f32> = (0..nt).map(|t| batch.scale(t)).collect();
    let sums: Vec<i32> = (0..nt).map(|t| batch.sum(t)).collect();

    pool.run(n_threads, |tid, n_thr| {
        let chunk = ((out_dim + n_thr - 1) / n_thr + 3) & !3;
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let weight = w as *const u8;
        let out = out_p as *mut f32;

        let mut raw4 = [0i32; 4];
        let mut r = start;
        unsafe {
            while r + 4 <= end {
                let w0 = weight.add(r * row_bytes);
                let w1 = weight.add((r + 1) * row_bytes);
                let w2 = weight.add((r + 2) * row_bytes);
                let w3 = weight.add((r + 3) * row_bytes);
                for t in 0..nt {
                    let combined = (scales[t] / 127.0) * weight_scale;
                    ffi::i2_dot_i8_4row(
                        w0, w1, w2, w3,
                        qs[t] as *const i8, raw4.as_mut_ptr(), in_dim as i32,
                    );
                    let base = out.add(t * out_dim + r);
                    for j in 0..4 {
                        *base.add(j) = (raw4[j] - sums[t]) as f32 * combined;
                    }
                }
                r += 4;
            }
            while r < end {
                for t in 0..nt {
                    let combined = (scales[t] / 127.0) * weight_scale;
                    let v = ffi::i2_dot_i8(
                        weight.add(r * row_bytes), qs[t] as *const i8, in_dim as i32,
                    );
                    *out.add(t * out_dim + r) = (v - sums[t]) as f32 * combined;
                }
                r += 1;
            }
        }
    });
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cargo test gemm_i2s_tests::i2s_gemm_matches_sequential -- --nocapture`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/gemm_i2s.rs src/gemm_i2s_tests.rs src/main.rs
git commit -m "feat: add BatchI8 and i2s_gemm_mt for batched I2S prefill"
```

---

### Task 2: Fused gate+up SquaredReLU GEMM

**Files:**
- Modify: `src/gemm_i2s.rs`
- Modify: `src/gemm_i2s_tests.rs`

- [ ] **Step 1: Write failing test for i2s_fused_sqrelu_gemm_mt**

Add to `src/gemm_i2s_tests.rs`:
```rust
use crate::gemm_i2s::i2s_fused_sqrelu_gemm_mt;

#[test]
fn i2s_fused_sqrelu_gemm_matches_sequential() {
    // Same setup as i2s_gemm_matches_sequential but with gate+up weights
    // Reference: compute gate and up separately via i2s_gemm_mt,
    // then squared_relu_mul_f32, compare to fused result.
    let pool = ThreadPool::new();
    let in_dim = 256;
    let out_dim = 64;
    let n_tokens = 3;

    // ... set up gate_weights, up_weights, activations, BatchI8 ...

    // Fused path
    let mut fused_out = vec![0.0f32; n_tokens * out_dim];
    i2s_fused_sqrelu_gemm_mt(
        gate_packed.as_ptr(), gate_scale,
        up_packed.as_ptr(), up_scale,
        &batch, &mut fused_out, out_dim, in_dim, &pool,
    );

    // Reference: gate + up separate, then squared_relu_mul
    let mut gate_out = vec![0.0f32; n_tokens * out_dim];
    let mut up_out = vec![0.0f32; n_tokens * out_dim];
    i2s_gemm_mt(gate_packed.as_ptr(), gate_scale, &batch, &mut gate_out, out_dim, in_dim, &pool);
    i2s_gemm_mt(up_packed.as_ptr(), up_scale, &batch, &mut up_out, out_dim, in_dim, &pool);
    for t in 0..n_tokens {
        for r in 0..out_dim {
            let g = gate_out[t * out_dim + r];
            let u = up_out[t * out_dim + r];
            let expected = g * g * (if g > 0.0 { 1.0 } else { 0.0 }) * u;  // squared_relu * up
            assert!((fused_out[t * out_dim + r] - expected).abs() < 1e-3,
                "mismatch at t={t} r={r}");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test gemm_i2s_tests::i2s_fused_sqrelu_gemm_matches_sequential -- --nocapture`
Expected: FAIL — function not found

- [ ] **Step 3: Implement i2s_fused_sqrelu_gemm_mt**

Add to `src/gemm_i2s.rs`, mirroring `gemm_q4k.rs:112-174` but using `i2_dot_i8_4row_dual` and `squared_relu(g) * u` instead of `silu(g) * u`:
```rust
/// Fused gate+up+SquaredReLU GEMM for I2S.
/// Output layout: out[t * out_dim + row] = squared_relu(gate[row]) * up[row]
/// Note: gate and up have SEPARATE weight_scale values.
pub(crate) fn i2s_fused_sqrelu_gemm_mt(
    w_gate: *const u8, scale_gate: f32,
    w_up: *const u8, scale_up: f32,
    batch: &BatchI8,
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    pool: &ThreadPool,
) {
    let nt = batch.n_tokens;
    let row_bytes = in_dim / 4;
    let n_threads = pool.thread_count().min(out_dim / 4).max(1);
    let wg = w_gate as usize;
    let wu = w_up as usize;
    let out_p = out.as_mut_ptr() as usize;
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let scales: Vec<f32> = (0..nt).map(|t| batch.scale(t)).collect();
    let sums: Vec<i32> = (0..nt).map(|t| batch.sum(t)).collect();

    pool.run(n_threads, |tid, n_thr| {
        let chunk = ((out_dim + n_thr - 1) / n_thr + 3) & !3;
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let wg = wg as *const u8;
        let wu = wu as *const u8;
        let out = out_p as *mut f32;

        let mut g_raw = [0i32; 4];
        let mut u_raw = [0i32; 4];
        let mut r = start;
        unsafe {
            while r + 4 <= end {
                for t in 0..nt {
                    let g_combined = (scales[t] / 127.0) * scale_gate;
                    let u_combined = (scales[t] / 127.0) * scale_up;
                    ffi::i2_dot_i8_4row_dual(
                        wg.add(r * row_bytes), wg.add((r+1) * row_bytes),
                        wg.add((r+2) * row_bytes), wg.add((r+3) * row_bytes),
                        wu.add(r * row_bytes), wu.add((r+1) * row_bytes),
                        wu.add((r+2) * row_bytes), wu.add((r+3) * row_bytes),
                        qs[t] as *const i8, g_raw.as_mut_ptr(), u_raw.as_mut_ptr(),
                        in_dim as i32,
                    );
                    let base = out.add(t * out_dim + r);
                    for j in 0..4 {
                        let g = (g_raw[j] - sums[t]) as f32 * g_combined;
                        let u = (u_raw[j] - sums[t]) as f32 * u_combined;
                        // squared_relu(g) * u = max(0,g)^2 * u
                        *base.add(j) = if g > 0.0 { g * g * u } else { 0.0 };
                    }
                }
                r += 4;
            }
            while r < end {
                for t in 0..nt {
                    let g_combined = (scales[t] / 127.0) * scale_gate;
                    let u_combined = (scales[t] / 127.0) * scale_up;
                    let gv = ffi::i2_dot_i8(wg.add(r * row_bytes), qs[t] as *const i8, in_dim as i32);
                    let uv = ffi::i2_dot_i8(wu.add(r * row_bytes), qs[t] as *const i8, in_dim as i32);
                    let g = (gv - sums[t]) as f32 * g_combined;
                    let u = (uv - sums[t]) as f32 * u_combined;
                    *out.add(t * out_dim + r) = if g > 0.0 { g * g * u } else { 0.0 };
                }
                r += 1;
            }
        }
    });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test gemm_i2s_tests::i2s_fused_sqrelu_gemm_matches_sequential -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gemm_i2s.rs src/gemm_i2s_tests.rs
git commit -m "feat: add fused gate+up+SquaredReLU GEMM for I2S batching"
```

---

### Task 3: Batched prefill in prefill.rs

**Files:**
- Create: `src/prefill.rs`
- Modify: `src/forward.rs` (update `generate()` at lines 386-389)
- Modify: `src/main.rs` (add `mod prefill;`)

- [ ] **Step 1: Create `src/prefill.rs` with `prefill()` method**

Create `src/prefill.rs` — an `impl InferenceState` block with the batched forward pass. This mirrors `forward_llama.rs:334-396` but for I2S weights.

```rust
//! Batched prefill for BitNet I2S models.

use crate::ffi;
use crate::forward::{apply_rope, build_rope_freqs, InferenceState};
use crate::gemm_i2s::{BatchI8, i2s_gemm_mt, i2s_fused_sqrelu_gemm_mt};
use crate::matmul::{embed_f16_lookup, i8_output_matmul_mt};
use crate::model::BitNetModel;

impl InferenceState {
    /// Batched prefill: load weights once per layer, multiply all prompt tokens.
    /// Sequential attention per token (attention is <1% of compute).
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
                    ffi::rmsnorm_f32(xs[t].as_ptr(), lw.attn_norm,
                        self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
                }
                batch_h.quantize(t, &self.x_norm);
            }

            // Batched QKV matmul
            i2s_gemm_mt(lw.wq, lw.wq_scale, &batch_h, &mut qs_all, h, h, &self.pool);
            i2s_gemm_mt(lw.wk, lw.wk_scale, &batch_h, &mut ks_all, kv, h, &self.pool);
            i2s_gemm_mt(lw.wv, lw.wv_scale, &batch_h, &mut vs_all, kv, h, &self.pool);

            // Per-token: RoPE, cache, attention
            for t in 0..n {
                let (q, k) = (&mut qs_all[t*h..(t+1)*h], &mut ks_all[t*kv..(t+1)*kv]);
                build_rope_freqs(&mut self.rope_freqs, hd, t, model.rope_theta);
                apply_rope(q, &self.rope_freqs, hd, nh);
                apply_rope(k, &self.rope_freqs, hd, nkv);
                for head in 0..nkv {
                    let off = ((layer * nkv + head) * self.max_seq_len + t) * hd;
                    self.k_cache[off..off+hd].copy_from_slice(&k[head*hd..(head+1)*hd]);
                    self.v_cache[off..off+hd].copy_from_slice(
                        &vs_all[t*kv+head*hd..t*kv+(head+1)*hd]);
                }
                let scale = 1.0 / (hd as f32).sqrt();
                let attn = &mut attn_all[t*h..(t+1)*h];
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
                        attn_all[t*h..].as_ptr(), lw.attn_sub_norm,
                        attn_all[t*h..].as_mut_ptr(), h as i32, model.rms_eps,
                    );
                }
                batch_h.quantize(t, &attn_all[t*h..(t+1)*h]);
            }
            i2s_gemm_mt(lw.wo, lw.wo_scale, &batch_h, &mut tmp_all, h, h, &self.pool);

            // Residual
            for t in 0..n {
                unsafe {
                    ffi::vecadd_f32(xs[t].as_ptr(), tmp_all[t*h..].as_ptr(),
                        xs[t].as_mut_ptr(), h as i32);
                }
            }

            // FFN: RMSNorm + quantize + fused gate+up+SquaredReLU
            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(xs[t].as_ptr(), lw.ffn_norm,
                        self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
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
                        hidden_all[t*f..].as_ptr(), lw.ffn_sub_norm,
                        hidden_all[t*f..].as_mut_ptr(), f as i32, model.rms_eps,
                    );
                }
                batch_f.quantize(t, &hidden_all[t*f..(t+1)*f]);
            }
            i2s_gemm_mt(lw.w_down, lw.w_down_scale, &batch_f,
                &mut tmp_all, h, f, &self.pool);

            // Residual
            for t in 0..n {
                unsafe {
                    ffi::vecadd_f32(xs[t].as_ptr(), tmp_all[t*h..].as_ptr(),
                        xs[t].as_mut_ptr(), h as i32);
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
        // Output matmul — same logic as forward() lines 309-332.
        // ARM speculative path is handled by cfg, x86 uses i8_output_matmul_mt.
        #[cfg(target_arch = "aarch64")]
        {
            if !model.embed_sketch.is_empty() {
                crate::matmul::i8_output_matmul_speculative(
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
```

Add `mod prefill;` to `src/main.rs` (after `mod gemm_i2s;`).

Note: `InferenceState` fields (`pool`, `x`, `x_norm`, `k_cache`, etc.) must be `pub(crate)` for `prefill.rs` to access them. Check `forward.rs:10-30` — if fields are private, change them to `pub(crate)`.

- [ ] **Step 2: Update generate() to call prefill()**

Modify `src/forward.rs:386-389` — replace the sequential forward loop:
```rust
// OLD:
for (i, &tok) in prompt_tokens.iter().enumerate() {
    state.forward(model, tok, i);
    output.push(tok);
}
```
with:
```rust
// NEW:
state.prefill(model, prompt_tokens);
output.extend_from_slice(prompt_tokens);
```

(Mirroring `forward_llama.rs:430-431`)

- [ ] **Step 3: Make InferenceState fields pub(crate)**

In `src/forward.rs:10-30`, change `InferenceState` fields from private to `pub(crate)` so `prefill.rs` can access them:
```rust
pub struct InferenceState {
    pub(crate) pool: ThreadPool,
    pub(crate) x: Vec<f32>,
    pub(crate) x_norm: Vec<f32>,
    // ... etc for all fields
}
```

- [ ] **Step 4: Build and verify compilation**

Run: `cargo build --release`
Expected: compiles with no errors

- [ ] **Step 5: Smoke test — run BitNet model and verify output**

Run: `./target/release/cougar --model bitnet --prompt "Hello world" --max-tokens 32`
Expected: produces coherent output, prefill tok/s should be significantly higher than before (~20 → 60+ tok/s)

- [ ] **Step 6: Commit**

```bash
git add src/prefill.rs src/forward.rs src/main.rs
git commit -m "feat: batched prefill for BitNet I2S forward pass"
```

---

### Task 4: Benchmark and verify

**Files:** None (benchmark only)

- [ ] **Step 1: Run 3x benchmark comparison (before vs after)**

Run the same benchmark as the baseline:
```bash
for i in 1 2 3; do
  echo "=== Cougar BitNet run $i ==="
  ./target/release/cougar --model bitnet \
    --prompt "Write a detailed essay about the history of computing, from the earliest mechanical calculators to modern AI systems." \
    --max-tokens 256 2>&1 | grep -E '(prefill|decode)'
done
```

Expected: prefill tok/s should be 3-5x faster, decode tok/s unchanged.

- [ ] **Step 2: Compare against bitnet.cpp**

```bash
BITNET_CLI=/home/peter/projects/BitNet/build/bin/llama-cli
BITNET_MODEL=/home/peter/projects/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
for i in 1 2 3; do
  echo "=== bitnet.cpp run $i ==="
  $BITNET_CLI -m "$BITNET_MODEL" \
    -p "Write a detailed essay about the history of computing, from the earliest mechanical calculators to modern AI systems." \
    -n 256 -t 16 --no-display-prompt 2>&1 | grep -E 'eval'
done
```

- [ ] **Step 3: Verify decode speed is unchanged**

Compare decode tok/s numbers from step 1 against baseline (17.7 tok/s avg). Should be within 5%.

- [ ] **Step 4: Commit any fixes from benchmark results**

If any issues found during benchmarking, fix and commit.
