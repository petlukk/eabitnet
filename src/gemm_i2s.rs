//! GEMM-style batched I2S matmul: load weight once, multiply against N tokens.

use crate::ffi;
use crate::threadpool::ThreadPool;

/// Batched i8 activation data for N tokens (I2S quantization).
pub(crate) struct BatchI8 {
    pub n_tokens: usize,
    pub dim: usize,
    pub stride: usize,
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

#[cfg(test)]
#[path = "gemm_i2s_tests.rs"]
mod tests;
