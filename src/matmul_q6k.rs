//! Q6_K matmul dispatch: reads block layout, calls SIMD dot-product kernels.

use crate::ffi;
use crate::matmul::f16_to_f32;
use crate::threadpool::ThreadPool;

/// Bytes per Q6_K super-block (256 elements).
const Q6K_BLOCK_BYTES: usize = 210;

/// Offsets within a Q6_K block.
const Q6K_QL_OFF: usize = 0;    // ql[128]
const Q6K_QH_OFF: usize = 128;  // qh[64]
const Q6K_SC_OFF: usize = 192;  // scales[16] (int8)
const Q6K_D_OFF: usize = 208;   // d (f16, 2 bytes)

/// Compute dot product of one Q6_K weight row against Q8_K activations.
pub(crate) unsafe fn q6k_row_dot(
    weight: *const u8,
    n_blocks: usize,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i32,
) -> f32 {
    let mut result = 0.0f32;

    for blk in 0..n_blocks {
        let block_ptr = weight.add(blk * Q6K_BLOCK_BYTES);
        let d_f16 = *(block_ptr.add(Q6K_D_OFF) as *const u16);
        let blk_q8_d = *q8_d.add(blk);
        let d = f16_to_f32(d_f16) * blk_q8_d;

        result += ffi::q6k_dot_q8k(
            block_ptr.add(Q6K_QL_OFF),
            block_ptr.add(Q6K_QH_OFF),
            block_ptr.add(Q6K_SC_OFF) as *const i8,
            q8_qs.add(blk * 256),
            q8_bsums.add(blk * 16),
            1, d,
        );
    }
    result
}

/// 4-row Q6_K × Q8_K dot product with shared activations.
pub(crate) unsafe fn q6k_4row_dot(
    w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
    n_blocks: usize,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i32,
    scores: &mut [f32; 4],
) {
    scores[0] = 0.0;
    scores[1] = 0.0;
    scores[2] = 0.0;
    scores[3] = 0.0;
    let mut blk_scores = [0.0f32; 4];

    for blk in 0..n_blocks {
        let b0 = w0.add(blk * Q6K_BLOCK_BYTES);
        let b1 = w1.add(blk * Q6K_BLOCK_BYTES);
        let b2 = w2.add(blk * Q6K_BLOCK_BYTES);
        let b3 = w3.add(blk * Q6K_BLOCK_BYTES);
        let blk_q8_d = *q8_d.add(blk);

        let d0 = f16_to_f32(*(b0.add(Q6K_D_OFF) as *const u16)) * blk_q8_d;
        let d1 = f16_to_f32(*(b1.add(Q6K_D_OFF) as *const u16)) * blk_q8_d;
        let d2 = f16_to_f32(*(b2.add(Q6K_D_OFF) as *const u16)) * blk_q8_d;
        let d3 = f16_to_f32(*(b3.add(Q6K_D_OFF) as *const u16)) * blk_q8_d;

        let q8_ptr = q8_qs.add(blk * 256);
        let bsums_ptr = q8_bsums.add(blk * 16);

        ffi::q6k_dot_q8k_4row(
            b0.add(Q6K_QL_OFF), b1.add(Q6K_QL_OFF),
            b2.add(Q6K_QL_OFF), b3.add(Q6K_QL_OFF),
            b0.add(Q6K_QH_OFF), b1.add(Q6K_QH_OFF),
            b2.add(Q6K_QH_OFF), b3.add(Q6K_QH_OFF),
            b0.add(Q6K_SC_OFF) as *const i8, b1.add(Q6K_SC_OFF) as *const i8,
            b2.add(Q6K_SC_OFF) as *const i8, b3.add(Q6K_SC_OFF) as *const i8,
            q8_ptr, bsums_ptr,
            blk_scores.as_mut_ptr(), 1,
            d0, d1, d2, d3,
        );

        scores[0] += blk_scores[0];
        scores[1] += blk_scores[1];
        scores[2] += blk_scores[2];
        scores[3] += blk_scores[3];
    }
}

/// Multi-threaded Q6_K × Q8_K matrix multiplication.
pub(crate) fn q6k_matmul_mt(
    weight: *const u8,
    row_stride: usize,
    n_blocks: usize,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i32,
    out: &mut [f32],
    out_dim: usize,
    pool: &ThreadPool,
) {
    let n_threads = pool.thread_count().min(out_dim / 4).max(1);

    if n_threads <= 1 {
        let mut scores4 = [0.0f32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= out_dim {
                q6k_4row_dot(
                    weight.add(r * row_stride),
                    weight.add((r + 1) * row_stride),
                    weight.add((r + 2) * row_stride),
                    weight.add((r + 3) * row_stride),
                    n_blocks, q8_qs, q8_d, q8_bsums,
                    &mut scores4,
                );
                out[r] = scores4[0];
                out[r + 1] = scores4[1];
                out[r + 2] = scores4[2];
                out[r + 3] = scores4[3];
                r += 4;
            }
            while r < out_dim {
                out[r] = q6k_row_dot(
                    weight.add(r * row_stride),
                    n_blocks, q8_qs, q8_d, q8_bsums,
                );
                r += 1;
            }
        }
        return;
    }

    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let weight_ptr = weight as usize;
    let q8_qs_ptr = q8_qs as usize;
    let q8_d_ptr = q8_d as usize;
    let q8_bsums_ptr = q8_bsums as usize;
    let out_ptr = out.as_mut_ptr() as usize;

    pool.run(n_threads, |tid, _n_threads| {
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let count = end - start;
        let weight = weight_ptr as *const u8;
        let q8_qs = q8_qs_ptr as *const i8;
        let q8_d = q8_d_ptr as *const f32;
        let q8_bsums = q8_bsums_ptr as *const i32;
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), count)
        };

        let mut scores4 = [0.0f32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= count {
                let row = start + r;
                q6k_4row_dot(
                    weight.add(row * row_stride),
                    weight.add((row + 1) * row_stride),
                    weight.add((row + 2) * row_stride),
                    weight.add((row + 3) * row_stride),
                    n_blocks, q8_qs, q8_d, q8_bsums,
                    &mut scores4,
                );
                out_slice[r] = scores4[0];
                out_slice[r + 1] = scores4[1];
                out_slice[r + 2] = scores4[2];
                out_slice[r + 3] = scores4[3];
                r += 4;
            }
            while r < count {
                let row = start + r;
                out_slice[r] = q6k_row_dot(
                    weight.add(row * row_stride),
                    n_blocks, q8_qs, q8_d, q8_bsums,
                );
                r += 1;
            }
        }
    });
}

/// Per-thread work function for Q6_K matmul. Callable from run_split3.
pub(crate) unsafe fn q6k_matmul_work(
    weight: *const u8,
    row_stride: usize,
    n_blocks: usize,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i32,
    out: *mut f32,
    out_dim: usize,
    tid: usize,
    n_threads: usize,
) {
    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let start = tid * chunk;
    let end = (start + chunk).min(out_dim);
    if start >= end { return; }
    let count = end - start;
    let out_slice = std::slice::from_raw_parts_mut(out.add(start), count);

    let mut scores4 = [0.0f32; 4];
    let mut r = 0;
    while r + 4 <= count {
        let row = start + r;
        q6k_4row_dot(
            weight.add(row * row_stride),
            weight.add((row + 1) * row_stride),
            weight.add((row + 2) * row_stride),
            weight.add((row + 3) * row_stride),
            n_blocks, q8_qs, q8_d, q8_bsums,
            &mut scores4,
        );
        out_slice[r] = scores4[0];
        out_slice[r + 1] = scores4[1];
        out_slice[r + 2] = scores4[2];
        out_slice[r + 3] = scores4[3];
        r += 4;
    }
    while r < count {
        let row = start + r;
        out_slice[r] = q6k_row_dot(
            weight.add(row * row_stride),
            n_blocks, q8_qs, q8_d, q8_bsums,
        );
        r += 1;
    }
}

/// Dequantize a single embedding row from Q6_K block data to f32.
pub(crate) fn q6k_embed_lookup(
    embed_data: *const u8,
    token: u32,
    out: &mut [f32],
    hidden_dim: usize,
) {
    let n_blocks = hidden_dim / 256;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let row_ptr = unsafe { embed_data.add(token as usize * row_bytes) };

    for blk in 0..n_blocks {
        let block = unsafe { row_ptr.add(blk * Q6K_BLOCK_BYTES) };
        let d = f16_to_f32(unsafe { *(block.add(Q6K_D_OFF) as *const u16) });
        let ql = unsafe { block.add(Q6K_QL_OFF) };
        let qh = unsafe { block.add(Q6K_QH_OFF) };
        let scales = unsafe { block.add(Q6K_SC_OFF) };

        for half in 0..2usize {
            let ql_base = ql as usize + half * 64;
            let qh_base = qh as usize + half * 32;
            let elem_base = blk * 256 + half * 128;

            for group in 0..4usize {
                let sc0 = unsafe { *(scales.add(half * 8 + group * 2) as *const i8) } as f32;
                let sc1 = unsafe { *(scales.add(half * 8 + group * 2 + 1) as *const i8) } as f32;

                for pos in 0..32usize {
                    let ql_byte = unsafe {
                        match group {
                            0 | 1 => *(ql_base as *const u8).add(pos),
                            _ => *(ql_base as *const u8).add(32 + pos),
                        }
                    };
                    let low4 = match group {
                        0 | 2 => ql_byte & 0x0F,
                        _ => ql_byte >> 4,
                    };
                    let qh_byte = unsafe { *(qh_base as *const u8).add(pos) };
                    let high2 = match group {
                        0 => qh_byte & 0x03,
                        1 => (qh_byte >> 2) & 0x03,
                        2 => (qh_byte >> 4) & 0x03,
                        _ => (qh_byte >> 6) & 0x03,
                    };
                    let q6_unsigned = low4 | (high2 << 4);
                    let q6_signed = q6_unsigned as i8 as f32 - 32.0;
                    let scale = if pos < 16 { sc0 } else { sc1 };
                    out[elem_base + group * 32 + pos] = d * scale * q6_signed;
                }
            }
        }
    }
}
