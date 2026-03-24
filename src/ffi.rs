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

pub unsafe fn quant_f32_q8k(
    src: *const f32,
    dst_qs: *mut i8,
    dst_d: *mut f32,
    dst_bsums: *mut i32,
    n: i32,
) {
    (embed::k().quant_f32_q8k)(src, dst_qs, dst_d, dst_bsums, n)
}

pub unsafe fn q4k_dot_q8k(
    q4: *const u8,
    q8: *const i8,
    bsums: *const i32,
    scales: *const u8,
    mins: *const u8,
    n_blocks: i32,
    d: f32,
    dmin: f32,
) -> f32 {
    (embed::k().q4k_dot_q8k)(q4, q8, bsums, scales, mins, n_blocks, d, dmin)
}

pub unsafe fn q4k_dot_q8k_4row(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    q8: *const i8,
    bsums: *const i32,
    sc0: *const u8, sc1: *const u8, sc2: *const u8, sc3: *const u8,
    mn0: *const u8, mn1: *const u8, mn2: *const u8, mn3: *const u8,
    scores: *mut f32,
    n_blocks: i32,
    d0: f32, d1: f32, d2: f32, d3: f32,
    dm0: f32, dm1: f32, dm2: f32, dm3: f32,
) {
    (embed::k().q4k_dot_q8k_4row)(
        rw0, rw1, rw2, rw3, q8, bsums,
        sc0, sc1, sc2, sc3, mn0, mn1, mn2, mn3,
        scores, n_blocks,
        d0, d1, d2, d3, dm0, dm1, dm2, dm3,
    )
}

pub unsafe fn silu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: i32) {
    (embed::k().silu_mul_f32)(gate, up, out, n)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_dot_q8k_4row_dual(
    gw0: *const u8, gw1: *const u8, gw2: *const u8, gw3: *const u8,
    uw0: *const u8, uw1: *const u8, uw2: *const u8, uw3: *const u8,
    q8: *const i8, bsums: *const i32,
    gsc0: *const u8, gsc1: *const u8, gsc2: *const u8, gsc3: *const u8,
    gmn0: *const u8, gmn1: *const u8, gmn2: *const u8, gmn3: *const u8,
    usc0: *const u8, usc1: *const u8, usc2: *const u8, usc3: *const u8,
    umn0: *const u8, umn1: *const u8, umn2: *const u8, umn3: *const u8,
    gate_scores: *mut f32, up_scores: *mut f32, n_blocks: i32,
    gd0: f32, gd1: f32, gd2: f32, gd3: f32,
    gdm0: f32, gdm1: f32, gdm2: f32, gdm3: f32,
    ud0: f32, ud1: f32, ud2: f32, ud3: f32,
    udm0: f32, udm1: f32, udm2: f32, udm3: f32,
) {
    (embed::k().q4k_dot_q8k_4row_dual)(
        gw0, gw1, gw2, gw3, uw0, uw1, uw2, uw3,
        q8, bsums,
        gsc0, gsc1, gsc2, gsc3, gmn0, gmn1, gmn2, gmn3,
        usc0, usc1, usc2, usc3, umn0, umn1, umn2, umn3,
        gate_scores, up_scores, n_blocks,
        gd0, gd1, gd2, gd3, gdm0, gdm1, gdm2, gdm3,
        ud0, ud1, ud2, ud3, udm0, udm1, udm2, udm3,
    )
}

pub unsafe fn q6k_dot_q8k(
    ql: *const u8, qh: *const u8, scales: *const i8,
    q8: *const i8, bsums: *const i32,
    n_blocks: i32, d: f32,
) -> f32 {
    (embed::k().q6k_dot_q8k)(ql, qh, scales, q8, bsums, n_blocks, d)
}

pub unsafe fn q6k_dot_q8k_4row(
    ql0: *const u8, ql1: *const u8, ql2: *const u8, ql3: *const u8,
    qh0: *const u8, qh1: *const u8, qh2: *const u8, qh3: *const u8,
    sc0: *const i8, sc1: *const i8, sc2: *const i8, sc3: *const i8,
    q8: *const i8, bsums: *const i32,
    scores: *mut f32, n_blocks: i32,
    d0: f32, d1: f32, d2: f32, d3: f32,
) {
    (embed::k().q6k_dot_q8k_4row)(
        ql0, ql1, ql2, ql3, qh0, qh1, qh2, qh3,
        sc0, sc1, sc2, sc3, q8, bsums,
        scores, n_blocks, d0, d1, d2, d3,
    )
}
