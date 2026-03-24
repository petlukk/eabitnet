//! GEMM-style batched Q6_K matmul: load weight once, multiply against N tokens.

use crate::gemm_q4k::BatchQ8K;
use crate::matmul_q6k::{q6k_4row_dot, q6k_row_dot};
use crate::threadpool::ThreadPool;

const Q6K_BLOCK_BYTES: usize = 210;

/// GEMM: Q6_K weight[out_dim] x batch[n_tokens] -> out[n_tokens * out_dim]
pub(crate) fn q6k_gemm_mt(
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
                    q6k_4row_dot(w0, w1, w2, w3, n_blocks,
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
                    let v = q6k_row_dot(wr, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                    *out.add(t * out_dim + r) = v;
                }
                r += 1;
            }
        }
    });
}
