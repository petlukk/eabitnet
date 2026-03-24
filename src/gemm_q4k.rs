//! GEMM-style batched Q4_K matmul: load weight once, multiply against N tokens.

use crate::ffi;
use crate::matmul_q4k::{q4k_4row_dot, q4k_row_dot, q4k_dual_4row_dot};
use crate::threadpool::ThreadPool;

const Q4K_BLOCK_BYTES: usize = 144;

/// Batched Q8K activation data for N tokens.
pub(crate) struct BatchQ8K {
    pub n_tokens: usize,
    pub dim: usize,
    pub n_blocks: usize,
    pub qs_stride: usize,
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub bsums: Vec<i32>,
}

impl BatchQ8K {
    pub fn new(n_tokens: usize, dim: usize) -> Self {
        let n_blocks = dim / 256;
        let qs_stride = dim + 16; // padding for narrow_f32x4_i8 overshoot
        BatchQ8K {
            n_tokens, dim, n_blocks, qs_stride,
            qs: vec![0i8; n_tokens * qs_stride],
            d: vec![0.0f32; n_tokens * n_blocks],
            bsums: vec![0i32; n_tokens * n_blocks * 16],
        }
    }

    pub fn quantize(&mut self, t: usize, src: &[f32]) {
        unsafe {
            ffi::quant_f32_q8k(
                src.as_ptr(),
                self.qs.as_mut_ptr().add(t * self.qs_stride),
                self.d.as_mut_ptr().add(t * self.n_blocks),
                self.bsums.as_mut_ptr().add(t * self.n_blocks * 16),
                self.dim as i32,
            );
        }
    }

    pub fn qs_ptr(&self, t: usize) -> *const i8 {
        unsafe { self.qs.as_ptr().add(t * self.qs_stride) }
    }
    pub fn d_ptr(&self, t: usize) -> *const f32 {
        unsafe { self.d.as_ptr().add(t * self.n_blocks) }
    }
    pub fn bsums_ptr(&self, t: usize) -> *const i32 {
        unsafe { self.bsums.as_ptr().add(t * self.n_blocks * 16) }
    }
}

/// GEMM: weight[out_dim] x batch[n_tokens] -> out[n_tokens * out_dim]
/// Output layout: out[t * out_dim + row]
pub(crate) fn q4k_gemm_mt(
    weight: *const u8,
    row_stride: usize,
    n_blocks: usize,
    batch: &BatchQ8K,
    out: &mut [f32],
    out_dim: usize,
    pool: &ThreadPool,
) {
    let nt = batch.n_tokens;
    let n_threads = pool.thread_count().min(out_dim / 4).max(1);
    let w = weight as usize;
    let out_p = out.as_mut_ptr() as usize;
    // Collect per-token pointers as usize for Send
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let ds: Vec<usize> = (0..nt).map(|t| batch.d_ptr(t) as usize).collect();
    let bs: Vec<usize> = (0..nt).map(|t| batch.bsums_ptr(t) as usize).collect();

    pool.run(n_threads, |tid, n_thr| {
        let chunk = ((out_dim + n_thr - 1) / n_thr + 3) & !3;
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let weight = w as *const u8;
        let out = out_p as *mut f32;

        let mut scores = [0.0f32; 4];
        let mut r = start;
        unsafe {
            while r + 4 <= end {
                let w0 = weight.add(r * row_stride);
                let w1 = weight.add((r + 1) * row_stride);
                let w2 = weight.add((r + 2) * row_stride);
                let w3 = weight.add((r + 3) * row_stride);
                for t in 0..nt {
                    q4k_4row_dot(w0, w1, w2, w3, n_blocks,
                        qs[t] as _, ds[t] as _, bs[t] as _, &mut scores);
                    let base = out.add(t * out_dim + r);
                    *base = scores[0]; *base.add(1) = scores[1];
                    *base.add(2) = scores[2]; *base.add(3) = scores[3];
                }
                r += 4;
            }
            while r < end {
                let wr = weight.add(r * row_stride);
                for t in 0..nt {
                    let v = q4k_row_dot(wr, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                    *out.add(t * out_dim + r) = v;
                }
                r += 1;
            }
        }
    });
}

/// Fused gate+up+SiLU GEMM: compute silu(gate) * up for all tokens.
/// Output layout: out[t * out_dim + row]
pub(crate) fn q4k_fused_silu_gemm_mt(
    w_gate: *const u8,
    w_up: *const u8,
    row_stride: usize,
    n_blocks: usize,
    batch: &BatchQ8K,
    out: &mut [f32],
    out_dim: usize,
    pool: &ThreadPool,
) {
    let nt = batch.n_tokens;
    let n_threads = pool.thread_count().min(out_dim / 4).max(1);
    let wg = w_gate as usize;
    let wu = w_up as usize;
    let out_p = out.as_mut_ptr() as usize;
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let ds: Vec<usize> = (0..nt).map(|t| batch.d_ptr(t) as usize).collect();
    let bs: Vec<usize> = (0..nt).map(|t| batch.bsums_ptr(t) as usize).collect();

    pool.run(n_threads, |tid, n_thr| {
        let chunk = ((out_dim + n_thr - 1) / n_thr + 3) & !3;
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let wg = wg as *const u8;
        let wu = wu as *const u8;
        let out = out_p as *mut f32;

        let mut g_scores = [0.0f32; 4];
        let mut u_scores = [0.0f32; 4];
        let mut r = start;
        unsafe {
            while r + 4 <= end {
                for t in 0..nt {
                    q4k_dual_4row_dot(
                        wg.add(r * row_stride), wg.add((r+1) * row_stride),
                        wg.add((r+2) * row_stride), wg.add((r+3) * row_stride),
                        wu.add(r * row_stride), wu.add((r+1) * row_stride),
                        wu.add((r+2) * row_stride), wu.add((r+3) * row_stride),
                        n_blocks, qs[t] as _, ds[t] as _, bs[t] as _,
                        &mut g_scores, &mut u_scores,
                    );
                    let base = out.add(t * out_dim + r);
                    for i in 0..4 {
                        let g = g_scores[i];
                        *base.add(i) = (g / (1.0 + (-g).exp())) * u_scores[i];
                    }
                }
                r += 4;
            }
            while r < end {
                let gw = wg.add(r * row_stride);
                let uw = wu.add(r * row_stride);
                for t in 0..nt {
                    let g = q4k_row_dot(gw, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                    let u = q4k_row_dot(uw, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                    *out.add(t * out_dim + r) = (g / (1.0 + (-g).exp())) * u;
                }
                r += 1;
            }
        }
    });
}
