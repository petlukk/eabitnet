//! Matrix multiplication helpers for BitNet inference.

use crate::ffi;

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

pub(crate) fn f32_matmul_mt(
    embed: &[f32], x: &[f32], out: &mut [f32], vocab_size: usize, hidden_dim: usize,
) {
    use std::thread;
    let n_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    // Align chunks to 4 for tiled kernel
    let chunk = ((vocab_size + n_threads - 1) / n_threads + 3) & !3;
    let embed_ptr = embed.as_ptr() as usize;
    let x_ptr = x.as_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    thread::scope(|s| {
        for t in 0..n_threads {
            let start = t * chunk;
            let end = (start + chunk).min(vocab_size);
            if start >= end { continue; }
            let count = end - start;
            s.spawn(move || {
                let rows = (embed_ptr as *const f32).wrapping_add(start * hidden_dim);
                let x = x_ptr as *const f32;
                let out = (out_ptr as *mut f32).wrapping_add(start);
                unsafe {
                    ffi::tiled_dot_4row(x, rows, out, hidden_dim as i32, count as i32);
                }
            });
        }
    });
}

#[allow(dead_code)]
pub(crate) fn f16_matmul_mt(
    embed: *const u8, x: &[f32], out: &mut [f32], vocab_size: usize, hidden_dim: usize,
) {
    use std::thread;
    let n_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let chunk = (vocab_size + n_threads - 1) / n_threads;

    let embed_ptr = embed as usize;
    let x_ptr = x.as_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;

    thread::scope(|s| {
        for t in 0..n_threads {
            let start = t * chunk;
            let end = (start + chunk).min(vocab_size);
            if start >= end {
                continue;
            }
            s.spawn(move || {
                let embed = embed_ptr as *const u8;
                let x = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, hidden_dim) };
                let out = unsafe {
                    std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
                };
                for v in 0..(end - start) {
                    let row = unsafe {
                        std::slice::from_raw_parts(
                            embed.add((start + v) * hidden_dim * 2) as *const u16,
                            hidden_dim,
                        )
                    };
                    let mut dot = 0.0f32;
                    for d in 0..hidden_dim {
                        dot += f16_to_f32(row[d]) * x[d];
                    }
                    out[v] = dot;
                }
            });
        }
    });
}

/// Ternary matmul with configurable thread count.
pub(crate) fn ternary_matmul_mt_n(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
    n_threads: usize,
) {
    let n_threads = n_threads.min(out_dim / 4).max(1);
    if n_threads <= 1 {
        let row_bytes = in_dim / 4;
        let mut raw = vec![0i32; out_dim];
        let mut r = 0;
        unsafe {
            while r + 4 <= out_dim {
                ffi::i2_dot_i8_4row(
                    weight.add(r * row_bytes),
                    weight.add((r + 1) * row_bytes),
                    weight.add((r + 2) * row_bytes),
                    weight.add((r + 3) * row_bytes),
                    act, raw[r..].as_mut_ptr(), in_dim as i32,
                );
                r += 4;
            }
            while r < out_dim {
                raw[r] = ffi::i2_dot_i8(weight.add(r * row_bytes), act, in_dim as i32);
                r += 1;
            }
        }
        let scale = (act_scale / 127.0) * weight_scale;
        for i in 0..out_dim {
            out[i] = (raw[i] - act_sum) as f32 * scale;
        }
        return;
    }

    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let row_bytes = in_dim / 4;
    let weight_ptr = weight as usize;
    let act_ptr = act as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let scale = (act_scale / 127.0) * weight_scale;

    std::thread::scope(|s| {
        for t in 0..n_threads {
            let start = t * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end {
                continue;
            }
            let count = end - start;
            s.spawn(move || {
                let weight = weight_ptr as *const u8;
                let act = act_ptr as *const i8;
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), count)
                };
                let mut raw = vec![0i32; count];
                let mut r = 0;
                unsafe {
                    while r + 4 <= count {
                        let row = start + r;
                        ffi::i2_dot_i8_4row(
                            weight.add(row * row_bytes),
                            weight.add((row + 1) * row_bytes),
                            weight.add((row + 2) * row_bytes),
                            weight.add((row + 3) * row_bytes),
                            act, raw[r..].as_mut_ptr(), in_dim as i32,
                        );
                        r += 4;
                    }
                    while r < count {
                        raw[r] = ffi::i2_dot_i8(
                            weight.add((start + r) * row_bytes), act, in_dim as i32,
                        );
                        r += 1;
                    }
                }
                for i in 0..count {
                    out_slice[i] = (raw[i] - act_sum) as f32 * scale;
                }
            });
        }
    });
}

/// Ternary matmul using all available threads.
pub(crate) fn ternary_matmul_mt(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
) {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    ternary_matmul_mt_n(weight, act, act_scale, act_sum, weight_scale, out, out_dim, in_dim, n_threads);
}

/// Run Q + K + V concurrently: Q gets half threads, K and V split the other half.
pub(crate) fn ternary_matmul_qkv(
    w_q: *const u8, scale_q: f32, out_q: &mut [f32], out_dim_q: usize,
    w_k: *const u8, scale_k: f32, out_k: &mut [f32], out_dim_kv: usize,
    w_v: *const u8, scale_v: f32, out_v: &mut [f32],
    act: *const i8, act_scale: f32, act_sum: i32, in_dim: usize,
) {
    let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let q_threads = (total / 2).max(1);
    let kv_threads = (total - q_threads).max(1);
    let k_threads = (kv_threads / 2).max(1);
    let v_threads = (kv_threads - k_threads).max(1);

    let q_ptr = out_q.as_mut_ptr() as usize;
    let k_ptr = out_k.as_mut_ptr() as usize;
    let v_ptr = out_v.as_mut_ptr() as usize;
    let act_ptr = act as usize;
    let wq = w_q as usize;
    let wk = w_k as usize;
    let wv = w_v as usize;

    std::thread::scope(|s| {
        s.spawn(|| {
            let out = unsafe { std::slice::from_raw_parts_mut(q_ptr as *mut f32, out_dim_q) };
            ternary_matmul_mt_n(wq as _, act_ptr as _, act_scale, act_sum, scale_q, out, out_dim_q, in_dim, q_threads);
        });
        s.spawn(|| {
            let out = unsafe { std::slice::from_raw_parts_mut(k_ptr as *mut f32, out_dim_kv) };
            ternary_matmul_mt_n(wk as _, act_ptr as _, act_scale, act_sum, scale_k, out, out_dim_kv, in_dim, k_threads);
        });
        s.spawn(|| {
            let out = unsafe { std::slice::from_raw_parts_mut(v_ptr as *mut f32, out_dim_kv) };
            ternary_matmul_mt_n(wv as _, act_ptr as _, act_scale, act_sum, scale_v, out, out_dim_kv, in_dim, v_threads);
        });
    });
}

/// Run two ternary matmuls concurrently, splitting threads between them.
pub(crate) fn ternary_matmul_parallel_pair(
    w_a: *const u8, scale_a: f32,
    w_b: *const u8, scale_b: f32,
    act: *const i8, act_scale: f32, act_sum: i32,
    out_a: &mut [f32], out_b: &mut [f32],
    out_dim: usize, in_dim: usize,
) {
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let half = (total / 2).max(1);

    let a_ptr = out_a.as_mut_ptr() as usize;
    let b_ptr = out_b.as_mut_ptr() as usize;
    let act_ptr = act as usize;
    let w_a_ptr = w_a as usize;
    let w_b_ptr = w_b as usize;

    std::thread::scope(|s| {
        s.spawn(|| {
            let out = unsafe { std::slice::from_raw_parts_mut(a_ptr as *mut f32, out_dim) };
            ternary_matmul_mt_n(
                w_a_ptr as *const u8, act_ptr as *const i8,
                act_scale, act_sum, scale_a, out, out_dim, in_dim, half,
            );
        });
        s.spawn(|| {
            let out = unsafe { std::slice::from_raw_parts_mut(b_ptr as *mut f32, out_dim) };
            ternary_matmul_mt_n(
                w_b_ptr as *const u8, act_ptr as *const i8,
                act_scale, act_sum, scale_b, out, out_dim, in_dim, total - half,
            );
        });
    });
}
