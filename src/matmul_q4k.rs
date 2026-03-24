//! Q4_K matmul dispatch: unpacks 6-bit scales, calls SIMD dot-product kernels.

use crate::ffi;
use crate::matmul::f16_to_f32;
use crate::threadpool::ThreadPool;

/// Bytes per Q4_K super-block (256 elements).
const Q4K_BLOCK_BYTES: usize = 144;

/// Unpack Q4_K 12-byte packed scales into 8 scales + 8 mins.
///
/// Layout of the 12 packed bytes:
///   bytes 0..3:  scales[0..4] low 6 bits; bits 6-7 hold scales[4..8] bits 4-5
///   bytes 4..7:  mins[0..4] low 6 bits; bits 6-7 hold mins[4..8] bits 4-5
///   bytes 8..11: scales[4..8] low 4 bits | mins[4..8] low 4 bits
pub(crate) fn unpack_q4k_scales(packed: &[u8], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    for i in 0..4 {
        scales[i] = packed[i] & 0x3F;
        mins[i] = packed[4 + i] & 0x3F;
    }
    for i in 0..4 {
        scales[4 + i] = (packed[8 + i] & 0x0F) | ((packed[i] >> 6) << 4);
        mins[4 + i] = (packed[8 + i] >> 4) | ((packed[4 + i] >> 6) << 4);
    }
}

/// Compute dot product of one Q4_K weight row against Q8_K activations.
///
/// Iterates over super-blocks, unpacking per-block scales and d/dmin,
/// then calls the SIMD kernel with n_blocks=1 per block (since d/dmin
/// vary per block).
unsafe fn q4k_row_dot(
    weight: *const u8,
    n_blocks: usize,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i32,
) -> f32 {
    let mut result = 0.0f32;
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for blk in 0..n_blocks {
        let block_ptr = weight.add(blk * Q4K_BLOCK_BYTES);

        // Read f16 d and dmin from block header
        let d_f16 = *(block_ptr as *const u16);
        let dmin_f16 = *(block_ptr.add(2) as *const u16);
        let blk_q8_d = *q8_d.add(blk);
        let d = f16_to_f32(d_f16) * blk_q8_d;
        let dmin = f16_to_f32(dmin_f16) * blk_q8_d;

        // Unpack 6-bit scales from bytes at offset 4
        let packed = std::slice::from_raw_parts(block_ptr.add(4), 12);
        unpack_q4k_scales(packed, &mut scales, &mut mins);

        // Q4 nibbles at offset 16, Q8 data for this block
        let q4_ptr = block_ptr.add(16);
        let q8_ptr = q8_qs.add(blk * 256);
        let bsums_ptr = q8_bsums.add(blk * 16);

        result += ffi::q4k_dot_q8k(
            q4_ptr, q8_ptr, bsums_ptr,
            scales.as_ptr(), mins.as_ptr(),
            1, d, dmin,
        );
    }
    result
}

/// Compute dot product of 4 Q4_K weight rows against shared Q8_K activations.
///
/// Same per-block iteration as `q4k_row_dot`, but uses the 4-row kernel
/// to share Q8 loads across rows.
unsafe fn q4k_4row_dot(
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

    let mut sc0 = [0u8; 8];
    let mut sc1 = [0u8; 8];
    let mut sc2 = [0u8; 8];
    let mut sc3 = [0u8; 8];
    let mut mn0 = [0u8; 8];
    let mut mn1 = [0u8; 8];
    let mut mn2 = [0u8; 8];
    let mut mn3 = [0u8; 8];
    let mut blk_scores = [0.0f32; 4];

    for blk in 0..n_blocks {
        let b0 = w0.add(blk * Q4K_BLOCK_BYTES);
        let b1 = w1.add(blk * Q4K_BLOCK_BYTES);
        let b2 = w2.add(blk * Q4K_BLOCK_BYTES);
        let b3 = w3.add(blk * Q4K_BLOCK_BYTES);

        let blk_q8_d = *q8_d.add(blk);

        // Read and convert per-block d/dmin for each row
        let d0 = f16_to_f32(*(b0 as *const u16)) * blk_q8_d;
        let d1 = f16_to_f32(*(b1 as *const u16)) * blk_q8_d;
        let d2 = f16_to_f32(*(b2 as *const u16)) * blk_q8_d;
        let d3 = f16_to_f32(*(b3 as *const u16)) * blk_q8_d;
        let dm0 = f16_to_f32(*(b0.add(2) as *const u16)) * blk_q8_d;
        let dm1 = f16_to_f32(*(b1.add(2) as *const u16)) * blk_q8_d;
        let dm2 = f16_to_f32(*(b2.add(2) as *const u16)) * blk_q8_d;
        let dm3 = f16_to_f32(*(b3.add(2) as *const u16)) * blk_q8_d;

        // Unpack scales for each row
        unpack_q4k_scales(std::slice::from_raw_parts(b0.add(4), 12), &mut sc0, &mut mn0);
        unpack_q4k_scales(std::slice::from_raw_parts(b1.add(4), 12), &mut sc1, &mut mn1);
        unpack_q4k_scales(std::slice::from_raw_parts(b2.add(4), 12), &mut sc2, &mut mn2);
        unpack_q4k_scales(std::slice::from_raw_parts(b3.add(4), 12), &mut sc3, &mut mn3);

        let q8_ptr = q8_qs.add(blk * 256);
        let bsums_ptr = q8_bsums.add(blk * 16);

        ffi::q4k_dot_q8k_4row(
            b0.add(16), b1.add(16), b2.add(16), b3.add(16),
            q8_ptr, bsums_ptr,
            sc0.as_ptr(), sc1.as_ptr(), sc2.as_ptr(), sc3.as_ptr(),
            mn0.as_ptr(), mn1.as_ptr(), mn2.as_ptr(), mn3.as_ptr(),
            blk_scores.as_mut_ptr(),
            1,
            d0, d1, d2, d3,
            dm0, dm1, dm2, dm3,
        );

        scores[0] += blk_scores[0];
        scores[1] += blk_scores[1];
        scores[2] += blk_scores[2];
        scores[3] += blk_scores[3];
    }
}

/// Multi-threaded Q4_K × Q8_K matrix multiplication.
///
/// Each output row is the dot product of one Q4_K weight row against the
/// shared Q8_K-quantized activation vector. Rows are split across threads,
/// using the 4-row kernel where possible for shared Q8 loads.
///
/// # Parameters
/// - `weight`: contiguous Q4_K block data for all rows
/// - `row_stride`: bytes per weight row (n_blocks * 144)
/// - `n_blocks`: number of 256-element super-blocks per row
/// - `q8_qs`: Q8_K quantized activation values (n_blocks * 256 elements)
/// - `q8_d`: Q8_K per-block scales (n_blocks elements)
/// - `q8_bsums`: Q8_K per-block sums (n_blocks * 16 elements)
/// - `out`: output buffer (out_dim elements)
/// - `out_dim`: number of output rows
/// - `pool`: thread pool for parallel dispatch
pub(crate) fn q4k_matmul_mt(
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
                q4k_4row_dot(
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
                out[r] = q4k_row_dot(
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
        if start >= end {
            return;
        }
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
                q4k_4row_dot(
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
                out_slice[r] = q4k_row_dot(
                    weight.add(row * row_stride),
                    n_blocks, q8_qs, q8_d, q8_bsums,
                );
                r += 1;
            }
        }
    });
}

/// Dequantize a single embedding row from Q4_K block data to f32.
///
/// Each row is `n_blocks` consecutive 144-byte Q4_K super-blocks (256 elements each).
/// The row for `token` starts at `embed_data + token * n_blocks * 144`.
pub(crate) fn q4k_embed_lookup(
    embed_data: *const u8,
    token: u32,
    out: &mut [f32],
    hidden_dim: usize,
) {
    let n_blocks = hidden_dim / 256;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let row_ptr = unsafe { embed_data.add(token as usize * row_bytes) };

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for blk in 0..n_blocks {
        let block = unsafe { row_ptr.add(blk * Q4K_BLOCK_BYTES) };
        let d = f16_to_f32(unsafe { *(block as *const u16) });
        let dmin = f16_to_f32(unsafe { *(block.add(2) as *const u16) });
        let scales_raw = unsafe { std::slice::from_raw_parts(block.add(4), 12) };
        unpack_q4k_scales(scales_raw, &mut scales, &mut mins);
        let qs = unsafe { block.add(16) };

        for j in 0..4 {
            let d1 = d * scales[2 * j] as f32;
            let m1 = dmin * mins[2 * j] as f32;
            let d2 = d * scales[2 * j + 1] as f32;
            let m2 = dmin * mins[2 * j + 1] as f32;
            for k in 0..32 {
                let byte = unsafe { *qs.add(j * 32 + k) };
                out[blk * 256 + j * 64 + k] = d1 * (byte & 0xF) as f32 - m1;
                out[blk * 256 + j * 64 + 32 + k] = d2 * (byte >> 4) as f32 - m2;
            }
        }
    }
}

/// Per-thread work function for Q4_K matmul. Callable from run_split3.
pub(crate) unsafe fn q4k_matmul_work(
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
        q4k_4row_dot(
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
        out_slice[r] = q4k_row_dot(
            weight.add(row * row_stride),
            n_blocks, q8_qs, q8_d, q8_bsums,
        );
        r += 1;
    }
}

/// Fused 4-row gate+up dot product using dual kernel.
/// Computes both projections per block — Q8K stays in SIMD registers.
unsafe fn q4k_dual_4row_dot(
    gw0: *const u8, gw1: *const u8, gw2: *const u8, gw3: *const u8,
    uw0: *const u8, uw1: *const u8, uw2: *const u8, uw3: *const u8,
    n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
    g_scores: &mut [f32; 4], u_scores: &mut [f32; 4],
) {
    *g_scores = [0.0; 4]; *u_scores = [0.0; 4];
    let mut g_blk = [0.0f32; 4]; let mut u_blk = [0.0f32; 4];
    let mut gsc = [[0u8; 8]; 4]; let mut gmn = [[0u8; 8]; 4];
    let mut usc = [[0u8; 8]; 4]; let mut umn = [[0u8; 8]; 4];

    for blk in 0..n_blocks {
        let blk_q8_d = *q8_d.add(blk);
        let q8_ptr = q8_qs.add(blk * 256);
        let bsums_ptr = q8_bsums.add(blk * 16);
        let gbs = [gw0.add(blk * Q4K_BLOCK_BYTES), gw1.add(blk * Q4K_BLOCK_BYTES),
                   gw2.add(blk * Q4K_BLOCK_BYTES), gw3.add(blk * Q4K_BLOCK_BYTES)];
        let ubs = [uw0.add(blk * Q4K_BLOCK_BYTES), uw1.add(blk * Q4K_BLOCK_BYTES),
                   uw2.add(blk * Q4K_BLOCK_BYTES), uw3.add(blk * Q4K_BLOCK_BYTES)];

        let mut gd = [0f32; 4]; let mut gdm = [0f32; 4];
        let mut ud = [0f32; 4]; let mut udm = [0f32; 4];
        for i in 0..4 {
            gd[i] = f16_to_f32(*(gbs[i] as *const u16)) * blk_q8_d;
            gdm[i] = f16_to_f32(*(gbs[i].add(2) as *const u16)) * blk_q8_d;
            unpack_q4k_scales(std::slice::from_raw_parts(gbs[i].add(4), 12), &mut gsc[i], &mut gmn[i]);
            ud[i] = f16_to_f32(*(ubs[i] as *const u16)) * blk_q8_d;
            udm[i] = f16_to_f32(*(ubs[i].add(2) as *const u16)) * blk_q8_d;
            unpack_q4k_scales(std::slice::from_raw_parts(ubs[i].add(4), 12), &mut usc[i], &mut umn[i]);
        }

        ffi::q4k_dot_q8k_4row_dual(
            gbs[0].add(16), gbs[1].add(16), gbs[2].add(16), gbs[3].add(16),
            ubs[0].add(16), ubs[1].add(16), ubs[2].add(16), ubs[3].add(16),
            q8_ptr, bsums_ptr,
            gsc[0].as_ptr(), gsc[1].as_ptr(), gsc[2].as_ptr(), gsc[3].as_ptr(),
            gmn[0].as_ptr(), gmn[1].as_ptr(), gmn[2].as_ptr(), gmn[3].as_ptr(),
            usc[0].as_ptr(), usc[1].as_ptr(), usc[2].as_ptr(), usc[3].as_ptr(),
            umn[0].as_ptr(), umn[1].as_ptr(), umn[2].as_ptr(), umn[3].as_ptr(),
            g_blk.as_mut_ptr(), u_blk.as_mut_ptr(), 1,
            gd[0], gd[1], gd[2], gd[3], gdm[0], gdm[1], gdm[2], gdm[3],
            ud[0], ud[1], ud[2], ud[3], udm[0], udm[1], udm[2], udm[3],
        );

        for i in 0..4 { g_scores[i] += g_blk[i]; u_scores[i] += u_blk[i]; }
    }
}

/// Vertically fused gate+up+SiLU: dual kernel → SiLU inline → write hidden directly.
/// Eliminates gate[] and up[] memory round-trip (160KB/layer saved).
pub(crate) unsafe fn q4k_fused_gate_up_silu_work(
    w_gate: *const u8,
    w_up: *const u8,
    row_stride: usize,
    n_blocks: usize,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i32,
    hidden_out: *mut f32,
    out_dim: usize,
    tid: usize,
    n_threads: usize,
) {
    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let start = tid * chunk;
    let end = (start + chunk).min(out_dim);
    if start >= end { return; }
    let count = end - start;
    let out = std::slice::from_raw_parts_mut(hidden_out.add(start), count);

    let mut g_scores = [0.0f32; 4];
    let mut u_scores = [0.0f32; 4];
    let mut r = 0;
    while r + 4 <= count {
        let row = start + r;
        q4k_dual_4row_dot(
            w_gate.add(row * row_stride), w_gate.add((row+1) * row_stride),
            w_gate.add((row+2) * row_stride), w_gate.add((row+3) * row_stride),
            w_up.add(row * row_stride), w_up.add((row+1) * row_stride),
            w_up.add((row+2) * row_stride), w_up.add((row+3) * row_stride),
            n_blocks, q8_qs, q8_d, q8_bsums,
            &mut g_scores, &mut u_scores,
        );
        // SiLU inline: hidden[i] = silu(gate[i]) * up[i]
        for i in 0..4 {
            let g = g_scores[i];
            out[r + i] = (g / (1.0 + (-g).exp())) * u_scores[i];
        }
        r += 4;
    }
    while r < count {
        let row = start + r;
        let g = q4k_row_dot(w_gate.add(row * row_stride), n_blocks, q8_qs, q8_d, q8_bsums);
        let u = q4k_row_dot(w_up.add(row * row_stride), n_blocks, q8_qs, q8_d, q8_bsums);
        out[r] = (g / (1.0 + (-g).exp())) * u;
        r += 1;
    }
}

#[cfg(test)]
#[path = "matmul_q4k_tests.rs"]
mod tests;
