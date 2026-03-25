//! Matrix multiplication helpers for BitNet inference.

use crate::ffi;
use crate::threadpool::ThreadPool;

pub(crate) fn f16_to_f32(h: u16) -> f32 {
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

pub(crate) fn embed_f16_lookup(embed: *const u8, token: u32, out: &mut [f32], hidden_dim: usize) {
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

/// i8 output matmul: quantize x to i8, then i8×u8 dot product for each vocab row.
/// embed_i8 is u8 (i8 + 128 bias), row_scales is per-row absmax.
pub(crate) fn i8_output_matmul_mt(
    embed_i8: &[u8], row_scales: &[f32],
    x: &[f32], out: &mut [f32],
    vocab_size: usize, hidden_dim: usize,
    pool: &ThreadPool,
) {
    // Quantize x to i8 (absmax)
    let mut x_amax = 0.0f32;
    for &v in x.iter().take(hidden_dim) {
        let a = v.abs();
        if a > x_amax { x_amax = a; }
    }
    let x_inv = if x_amax > 1e-10 { 127.0 / x_amax } else { 0.0 };
    let mut x_i8 = vec![0i8; hidden_dim];
    #[cfg(not(target_arch = "aarch64"))]
    let mut x_sum: i32 = 0;
    for d in 0..hidden_dim {
        let q = (x[d] * x_inv).round().clamp(-127.0, 127.0) as i8;
        x_i8[d] = q;
        #[cfg(not(target_arch = "aarch64"))]
        { x_sum += q as i32; }
    }

    let embed_ptr = embed_i8.as_ptr() as usize;
    let scales_ptr = row_scales.as_ptr() as usize;
    let act_ptr = x_i8.as_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let x_scale = x_amax / 127.0;

    pool.run(pool.thread_count(), |tid, n_threads| {
        let chunk = ((vocab_size + n_threads - 1) / n_threads + 3) & !3;
        let start = tid * chunk;
        let end = (start + chunk).min(vocab_size);
        if start >= end { return; }
        let count = end - start;
        let embed = embed_ptr as *const u8;
        let scales = scales_ptr as *const f32;
        let act = act_ptr as *const i8;
        let out = unsafe { std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), count) };
        let h = hidden_dim;
        let mut raw4 = [0i32; 4];
        let mut r = 0;
        while r + 4 <= count {
            let row = start + r;
            unsafe {
                ffi::i8dot_4row(
                    act,
                    embed.add(row * h), embed.add((row+1) * h),
                    embed.add((row+2) * h), embed.add((row+3) * h),
                    raw4.as_mut_ptr(), h as i32,
                );
            }
            for j in 0..4 {
                let row_s = unsafe { *scales.add(row + j) };
                // ARM i8dot returns true dot (XOR-unbiased); x86 returns biased
                #[cfg(target_arch = "aarch64")]
                let corrected = raw4[j];
                #[cfg(not(target_arch = "aarch64"))]
                let corrected = raw4[j] - 128 * x_sum;
                out[r + j] = corrected as f32 * x_scale * (row_s / 127.0);
            }
            r += 4;
        }
        while r < count {
            let row = start + r;
            let raw_val = unsafe { ffi::i8dot_1row(act, embed.add(row * h), h as i32) };
            let row_s = unsafe { *scales.add(row) };
            #[cfg(target_arch = "aarch64")]
            let corrected = raw_val;
            #[cfg(not(target_arch = "aarch64"))]
            let corrected = raw_val - 128 * x_sum;
            out[r] = corrected as f32 * x_scale * (row_s / 127.0);
            r += 1;
        }
    });
}

/// Speculative i8 output matmul: pre-filter with sketch, full precision on top-K.
/// Sketch table contains every 8th byte per row for rough scoring.
#[cfg(target_arch = "aarch64")]
pub(crate) fn i8_output_matmul_speculative(
    embed_i8: &[u8], row_scales: &[f32],
    sketch: &[u8], sketch_dim: usize,
    x: &[f32], out: &mut [f32],
    vocab_size: usize, hidden_dim: usize,
    pool: &ThreadPool,
) {
    const TOP_K: usize = 512;
    const SKETCH_STRIDE: usize = 4;

    // Quantize activations
    let mut x_amax = 0.0f32;
    for &v in x.iter().take(hidden_dim) {
        let a = v.abs();
        if a > x_amax { x_amax = a; }
    }
    let x_inv = if x_amax > 1e-10 { 127.0 / x_amax } else { 0.0 };
    let mut x_i8 = vec![0i8; hidden_dim];
    for d in 0..hidden_dim {
        x_i8[d] = (x[d] * x_inv).round().clamp(-127.0, 127.0) as i8;
    }
    let x_scale = x_amax / 127.0;

    // Build subsampled activation vector (every 8th element)
    let mut act_sketch = vec![0i8; sketch_dim];
    for s in 0..sketch_dim {
        act_sketch[s] = x_i8[s * SKETCH_STRIDE];
    }

    // Phase 1: rough scores via sketch (parallel), scaled by row_scales
    let mut rough_scores = vec![0.0f32; vocab_size];
    let sketch_ptr = sketch.as_ptr() as usize;
    let act_sk_ptr = act_sketch.as_ptr() as usize;
    let rough_ptr = rough_scores.as_mut_ptr() as usize;
    let scales_ptr = row_scales.as_ptr() as usize;
    let sd = sketch_dim;
    let xsc = x_scale;

    pool.run(pool.thread_count(), |tid, n_threads| {
        let chunk = ((vocab_size + n_threads - 1) / n_threads + 3) & !3;
        let start = tid * chunk;
        let end = (start + chunk).min(vocab_size);
        if start >= end { return; }
        let sk = sketch_ptr as *const u8;
        let act = act_sk_ptr as *const i8;
        let scores = rough_ptr as *mut f32;
        let scales = scales_ptr as *const f32;
        let mut raw4 = [0i32; 4];
        let mut r = start;
        while r + 4 <= end {
            unsafe {
                ffi::i8dot_4row(
                    act,
                    sk.add(r * sd) as *const u8,
                    sk.add((r+1) * sd) as *const u8,
                    sk.add((r+2) * sd) as *const u8,
                    sk.add((r+3) * sd) as *const u8,
                    raw4.as_mut_ptr(), sd as i32,
                );
                for j in 0..4 {
                    let rs = *scales.add(r + j);
                    *scores.add(r + j) = raw4[j] as f32 * xsc * (rs / 127.0);
                }
            }
            r += 4;
        }
        while r < end {
            let v = unsafe { ffi::i8dot_1row(act, sk.add(r * sd) as *const u8, sd as i32) };
            let rs = unsafe { *scales.add(r) };
            unsafe { *scores.add(r) = v as f32 * xsc * (rs / 127.0); }
            r += 1;
        }
    });

    // Phase 2: find top-K indices
    let mut indices: Vec<u32> = (0..vocab_size as u32).collect();
    indices.select_nth_unstable_by(TOP_K, |&a, &b| {
        rough_scores[b as usize].partial_cmp(&rough_scores[a as usize]).unwrap()
    });
    let top_indices = &indices[..TOP_K];

    // Phase 3: full precision on top-K only
    // Initialize output to -inf so argmax picks from candidates
    for v in out.iter_mut() { *v = f32::NEG_INFINITY; }

    let act_ptr = x_i8.as_ptr();
    let embed_ptr = embed_i8.as_ptr();
    let h = hidden_dim;
    for &idx in top_indices {
        let row = idx as usize;
        let raw = unsafe { ffi::i8dot_1row(act_ptr, embed_ptr.add(row * h) as *const u8, h as i32) };
        let row_s = row_scales[row];
        out[row] = raw as f32 * x_scale * (row_s / 127.0);
    }
}

/// Ternary matmul with configurable thread count.
pub(crate) fn ternary_matmul_mt_n(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
    n_threads: usize, pool: &ThreadPool,
) {
    let n_threads = n_threads.min(out_dim / 4).max(1);
    let row_bytes = in_dim / 4;
    let scale = (act_scale / 127.0) * weight_scale;

    if n_threads <= 1 {
        let mut raw4 = [0i32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= out_dim {
                ffi::i2_dot_i8_4row(
                    weight.add(r * row_bytes),
                    weight.add((r + 1) * row_bytes),
                    weight.add((r + 2) * row_bytes),
                    weight.add((r + 3) * row_bytes),
                    act, raw4.as_mut_ptr(), in_dim as i32,
                );
                for j in 0..4 {
                    out[r + j] = (raw4[j] - act_sum) as f32 * scale;
                }
                r += 4;
            }
            while r < out_dim {
                let v = ffi::i2_dot_i8(weight.add(r * row_bytes), act, in_dim as i32);
                out[r] = (v - act_sum) as f32 * scale;
                r += 1;
            }
        }
        return;
    }

    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let weight_ptr = weight as usize;
    let act_ptr = act as usize;
    let out_ptr = out.as_mut_ptr() as usize;

    pool.run(n_threads, |tid, _n_threads| {
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end {
            return;
        }
        let count = end - start;
        let weight = weight_ptr as *const u8;
        let act = act_ptr as *const i8;
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), count)
        };
        let mut raw4 = [0i32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= count {
                let row = start + r;
                ffi::i2_dot_i8_4row(
                    weight.add(row * row_bytes),
                    weight.add((row + 1) * row_bytes),
                    weight.add((row + 2) * row_bytes),
                    weight.add((row + 3) * row_bytes),
                    act, raw4.as_mut_ptr(), in_dim as i32,
                );
                for j in 0..4 {
                    out_slice[r + j] = (raw4[j] - act_sum) as f32 * scale;
                }
                r += 4;
            }
            while r < count {
                let v = ffi::i2_dot_i8(
                    weight.add((start + r) * row_bytes), act, in_dim as i32,
                );
                out_slice[r] = (v - act_sum) as f32 * scale;
                r += 1;
            }
        }
    });
}

/// Ternary matmul using all available threads.
pub(crate) fn ternary_matmul_mt(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
    pool: &ThreadPool,
) {
    ternary_matmul_mt_n(weight, act, act_scale, act_sum, weight_scale, out, out_dim, in_dim, pool.thread_count(), pool);
}

/// Run Q + K + V concurrently: Q gets half threads, K and V split the other half.
pub(crate) fn ternary_matmul_qkv(
    w_q: *const u8, scale_q: f32, out_q: &mut [f32], out_dim_q: usize,
    w_k: *const u8, scale_k: f32, out_k: &mut [f32], out_dim_kv: usize,
    w_v: *const u8, scale_v: f32, out_v: &mut [f32],
    act: *const i8, act_scale: f32, act_sum: i32, in_dim: usize,
    pool: &ThreadPool,
) {
    let total = pool.thread_count();

    // Need at least 3 threads for a 3-way split; fall back to sequential.
    if total < 3 {
        ternary_matmul_mt_n(w_q, act, act_scale, act_sum, scale_q, out_q, out_dim_q, in_dim, total, pool);
        ternary_matmul_mt_n(w_k, act, act_scale, act_sum, scale_k, out_k, out_dim_kv, in_dim, total, pool);
        ternary_matmul_mt_n(w_v, act, act_scale, act_sum, scale_v, out_v, out_dim_kv, in_dim, total, pool);
        return;
    }

    let q_threads = (total / 2).max(1);
    let remaining = total - q_threads;
    let k_threads = remaining / 2;
    let v_threads = remaining - k_threads;

    let row_bytes = in_dim / 4;
    let act_ptr = act as usize;

    let make_work = |w_ptr: usize, out_ptr: usize, out_dim: usize, scale: f32| {
        move |tid: usize, n_threads: usize| {
            let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { return; }
            let count = end - start;
            let weight = w_ptr as *const u8;
            let act = act_ptr as *const i8;
            let out_slice = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), count)
            };
            let combined_scale = (act_scale / 127.0) * scale;
            let mut raw4 = [0i32; 4];
            let mut r = 0;
            unsafe {
                while r + 4 <= count {
                    let row = start + r;
                    ffi::i2_dot_i8_4row(
                        weight.add(row * row_bytes),
                        weight.add((row + 1) * row_bytes),
                        weight.add((row + 2) * row_bytes),
                        weight.add((row + 3) * row_bytes),
                        act, raw4.as_mut_ptr(), in_dim as i32,
                    );
                    for j in 0..4 {
                        out_slice[r + j] = (raw4[j] - act_sum) as f32 * combined_scale;
                    }
                    r += 4;
                }
                while r < count {
                    let v = ffi::i2_dot_i8(
                        weight.add((start + r) * row_bytes), act, in_dim as i32,
                    );
                    out_slice[r] = (v - act_sum) as f32 * combined_scale;
                    r += 1;
                }
            }
        }
    };

    pool.run_split3(
        q_threads, make_work(w_q as usize, out_q.as_mut_ptr() as usize, out_dim_q, scale_q),
        k_threads, make_work(w_k as usize, out_k.as_mut_ptr() as usize, out_dim_kv, scale_k),
        v_threads, make_work(w_v as usize, out_v.as_mut_ptr() as usize, out_dim_kv, scale_v),
    );
}

/// Fused gate+up matmul: each thread computes both projections for its rows,
/// sharing activation loads via the i2_dot_i8_4row_dual kernel.
pub(crate) fn ternary_matmul_fused_pair(
    w_a: *const u8, scale_a: f32,
    w_b: *const u8, scale_b: f32,
    act: *const i8, act_scale: f32, act_sum: i32,
    out_a: &mut [f32], out_b: &mut [f32],
    out_dim: usize, in_dim: usize,
    pool: &ThreadPool,
) {
    let total = pool.thread_count();
    let row_bytes = in_dim / 4;
    let wa_ptr = w_a as usize;
    let wb_ptr = w_b as usize;
    let act_ptr = act as usize;
    let oa_ptr = out_a.as_mut_ptr() as usize;
    let ob_ptr = out_b.as_mut_ptr() as usize;
    let scale_a_combined = (act_scale / 127.0) * scale_a;
    let scale_b_combined = (act_scale / 127.0) * scale_b;

    pool.run(total, |tid, n_threads| {
        let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let count = end - start;
        let wa = wa_ptr as *const u8;
        let wb = wb_ptr as *const u8;
        let act = act_ptr as *const i8;
        let oa = unsafe { std::slice::from_raw_parts_mut((oa_ptr as *mut f32).add(start), count) };
        let ob = unsafe { std::slice::from_raw_parts_mut((ob_ptr as *mut f32).add(start), count) };
        let mut raw_a = [0i32; 4];
        let mut raw_b = [0i32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= count {
                let row = start + r;
                ffi::i2_dot_i8_4row_dual(
                    wa.add(row * row_bytes), wa.add((row+1) * row_bytes),
                    wa.add((row+2) * row_bytes), wa.add((row+3) * row_bytes),
                    wb.add(row * row_bytes), wb.add((row+1) * row_bytes),
                    wb.add((row+2) * row_bytes), wb.add((row+3) * row_bytes),
                    act, raw_a.as_mut_ptr(), raw_b.as_mut_ptr(), in_dim as i32,
                );
                for j in 0..4 {
                    oa[r + j] = (raw_a[j] - act_sum) as f32 * scale_a_combined;
                    ob[r + j] = (raw_b[j] - act_sum) as f32 * scale_b_combined;
                }
                r += 4;
            }
            while r < count {
                let row = start + r;
                let va = ffi::i2_dot_i8(wa.add(row * row_bytes), act, in_dim as i32);
                let vb = ffi::i2_dot_i8(wb.add(row * row_bytes), act, in_dim as i32);
                oa[r] = (va - act_sum) as f32 * scale_a_combined;
                ob[r] = (vb - act_sum) as f32 * scale_b_combined;
                r += 1;
            }
        }
    });
}

#[cfg(test)]
#[path = "matmul_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "sketch_tests.rs"]
mod sketch_tests;
