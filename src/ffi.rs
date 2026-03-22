//! FFI declarations for Ea SIMD kernels.

#[link(name = "bitnet_i2s")]
extern "C" {
    pub fn i2_dot_i8(weights: *const u8, activations: *const i8, n: i32) -> i32;
    pub fn i2_dot_i8_4row(
        w0: *const u8,
        w1: *const u8,
        w2: *const u8,
        w3: *const u8,
        activations: *const i8,
        scores: *mut i32,
        n: i32,
    );
}

#[link(name = "bitnet_quant")]
extern "C" {
    pub fn quant_f32_i8(
        src: *const f32,
        dst: *mut i8,
        out_scale: *mut f32,
        out_sum: *mut i32,
        n: i32,
    );
    pub fn pack_ternary_row(ternary: *const u8, packed: *mut u8, n: i32);
}

#[link(name = "bitnet_rmsnorm")]
extern "C" {
    pub fn rmsnorm_f32(x: *const f32, weight: *const f32, out: *mut f32, n: i32, eps: f32);
}

#[link(name = "bitnet_softmax")]
extern "C" {
    pub fn softmax_f32(x: *const f32, out: *mut f32, n: i32);
}

#[link(name = "bitnet_rope")]
extern "C" {
    pub fn rope_f32(q: *mut f32, k: *mut f32, freqs: *const f32, head_dim: i32, n_heads: i32);
}

#[link(name = "bitnet_attention")]
extern "C" {
    pub fn attn_scores_f32(
        q: *const f32,
        k_cache: *const f32,
        out: *mut f32,
        head_dim: i32,
        seq_len: i32,
        scale: f32,
    );
    pub fn attn_weighted_sum_f32(
        scores: *const f32,
        v_cache: *const f32,
        out: *mut f32,
        head_dim: i32,
        seq_len: i32,
    );
}

#[link(name = "bitnet_fused_attn")]
extern "C" {
    pub fn fused_attention_f32(
        q: *const f32, k_cache: *const f32, v_cache: *const f32,
        out: *mut f32, head_dim: i32, seq_len: i32, scale: f32,
    );
}

#[link(name = "bitnet_i8dot")]
extern "C" {
    pub fn i8dot_1row(act: *const i8, w: *const u8, n: i32) -> i32;
    pub fn i8dot_4row(
        act: *const i8, w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
        scores: *mut i32, n: i32,
    );
}

#[link(name = "bitnet_output")]
extern "C" {
    pub fn tiled_dot_4row(
        x: *const f32, rows: *const f32, out: *mut f32,
        dim: i32, n_rows: i32,
    );
}

#[link(name = "bitnet_activate")]
extern "C" {
    pub fn squared_relu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: i32);
}

#[link(name = "bitnet_vecadd")]
extern "C" {
    pub fn vecadd_f32(a: *const f32, b: *const f32, out: *mut f32, n: i32);
}
