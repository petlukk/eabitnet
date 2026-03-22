#ifndef EABITNET_H
#define EABITNET_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Ternary dot product: packed 2-bit weights × int8 activations
// Returns raw dot product (caller applies ternary offset + scale).
int32_t i2_dot_i8(const uint8_t *weights, const int8_t *activations, int32_t n);

// 4-row variant: same activations against 4 weight rows
void i2_dot_i8_4row(
    const uint8_t *w0, const uint8_t *w1,
    const uint8_t *w2, const uint8_t *w3,
    const int8_t *activations,
    int32_t *scores,
    int32_t n
);

// Per-tensor f32 → int8 activation quantization with sum output
void quant_f32_i8(const float *src, int8_t *dst, float *out_scale, int32_t *out_sum, int32_t n);

// Pack ternary values {0,1,2} into 2-bit packed bytes
void pack_ternary_row(const uint8_t *ternary, uint8_t *packed, int32_t n);

// Element-wise f32 vector addition (residual connections)
void vecadd_f32(const float *a, const float *b, float *out, int32_t n);

// RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
void rmsnorm_f32(const float *x, const float *weight, float *out, int32_t n, float eps);

// Softmax: numerically stable, fast exp polynomial
void softmax_f32(const float *x, float *out, int32_t n);

// RoPE: in-place rotation of Q and K by precomputed (cos,sin) frequency pairs
void rope_f32(float *q, float *k, const float *freqs, int32_t head_dim, int32_t n_heads);

// Attention: scaled dot products Q·K for all cached tokens
void attn_scores_f32(const float *q, const float *k_cache, float *out,
                     int32_t head_dim, int32_t seq_len, float scale);

// Attention: weighted sum out = Σ scores[t] * V[t]
void attn_weighted_sum_f32(const float *scores, const float *v_cache, float *out,
                           int32_t head_dim, int32_t seq_len);

// Squared ReLU fused: out[i] = max(0, gate[i])^2 * up[i]
void squared_relu_mul_f32(const float *gate, const float *up, float *out, int32_t n);

// Fused attention: single-pass score + online softmax + weighted V sum
void fused_attention_f32(const float *q, const float *k_cache, const float *v_cache,
                         float *out, int32_t head_dim, int32_t seq_len, float scale);

#ifdef __cplusplus
}
#endif

#endif // EABITNET_H
