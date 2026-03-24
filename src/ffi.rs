//! FFI wrappers for Ea SIMD kernels (delegates to embedded dlopen table).

use crate::embed;

pub unsafe fn i2_dot_i8(weights: *const u8, activations: *const i8, n: i32) -> i32 {
    (embed::k().i2_dot_i8)(weights, activations, n)
}

pub unsafe fn i2_dot_i8_4row(
    w0: *const u8,
    w1: *const u8,
    w2: *const u8,
    w3: *const u8,
    activations: *const i8,
    scores: *mut i32,
    n: i32,
) {
    (embed::k().i2_dot_i8_4row)(w0, w1, w2, w3, activations, scores, n)
}

pub unsafe fn i2_dot_i8_4row_dual(
    gw0: *const u8, gw1: *const u8, gw2: *const u8, gw3: *const u8,
    uw0: *const u8, uw1: *const u8, uw2: *const u8, uw3: *const u8,
    activations: *const i8,
    gate_scores: *mut i32,
    up_scores: *mut i32,
    n: i32,
) {
    (embed::k().i2_dot_i8_4row_dual)(
        gw0, gw1, gw2, gw3, uw0, uw1, uw2, uw3,
        activations, gate_scores, up_scores, n,
    )
}

pub unsafe fn quant_f32_i8(
    src: *const f32,
    dst: *mut i8,
    out_scale: *mut f32,
    out_sum: *mut i32,
    n: i32,
) {
    (embed::k().quant_f32_i8)(src, dst, out_scale, out_sum, n)
}

pub unsafe fn rmsnorm_f32(x: *const f32, weight: *const f32, out: *mut f32, n: i32, eps: f32) {
    (embed::k().rmsnorm_f32)(x, weight, out, n, eps)
}

pub unsafe fn fused_attention_f32(
    q: *const f32, k_cache: *const f32, v_cache: *const f32,
    out: *mut f32, head_dim: i32, seq_len: i32, scale: f32,
) {
    (embed::k().fused_attention_f32)(q, k_cache, v_cache, out, head_dim, seq_len, scale)
}

pub unsafe fn i8dot_1row(act: *const i8, w: *const u8, n: i32) -> i32 {
    (embed::k().i8dot_1row)(act, w, n)
}

pub unsafe fn i8dot_4row(
    act: *const i8, w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
    scores: *mut i32, n: i32,
) {
    (embed::k().i8dot_4row)(act, w0, w1, w2, w3, scores, n)
}

pub unsafe fn squared_relu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: i32) {
    (embed::k().squared_relu_mul_f32)(gate, up, out, n)
}

pub unsafe fn vecadd_f32(a: *const f32, b: *const f32, out: *mut f32, n: i32) {
    (embed::k().vecadd_f32)(a, b, out, n)
}
