#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s (%s:%d)\n", name, __FILE__, __LINE__); } \
} while(0)

#define CLOSE(a, b, tol) (fabsf((a) - (b)) < (tol))

// Reference: 3-pass attention (scores + softmax + weighted sum)
static void ref_attention(const float *q, const float *k, const float *v,
                          float *out, int hd, int seq, float scale) {
    float *scores = malloc(seq * sizeof(float));
    // Scores
    for (int t = 0; t < seq; t++) {
        float dot = 0;
        for (int d = 0; d < hd; d++) dot += q[d] * k[t * hd + d];
        scores[t] = dot * scale;
    }
    // Softmax
    float mx = scores[0];
    for (int t = 1; t < seq; t++) if (scores[t] > mx) mx = scores[t];
    float sum = 0;
    for (int t = 0; t < seq; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
    for (int t = 0; t < seq; t++) scores[t] /= sum;
    // Weighted sum
    for (int d = 0; d < hd; d++) out[d] = 0;
    for (int t = 0; t < seq; t++)
        for (int d = 0; d < hd; d++)
            out[d] += scores[t] * v[t * hd + d];
    free(scores);
}

static void test_single_token(void) {
    // seq=1: output should be exactly V[0]
    int hd = 8;
    float q[] = {1, 0, 0, 0, 0, 0, 0, 0};
    float k[] = {2, 0, 0, 0, 0, 0, 0, 0};
    float v[] = {10, 20, 30, 40, 50, 60, 70, 80};
    float out[8] = {0};
    fused_attention_f32(q, k, v, out, hd, 1, 0.5f);
    int ok = 1;
    for (int i = 0; i < hd; i++)
        if (!CLOSE(out[i], v[i], 1e-3f)) ok = 0;
    CHECK("single_token", ok);
}

static void test_two_tokens_equal(void) {
    // Two identical K vectors → equal scores → average of V[0] and V[1]
    int hd = 4;
    float q[] = {1, 0, 0, 0};
    float k[] = {1,0,0,0,  1,0,0,0};
    float v[] = {2,0,0,0,  0,2,0,0};
    float out[4] = {0}, ref[4] = {0};
    fused_attention_f32(q, k, v, out, hd, 2, 1.0f);
    ref_attention(q, k, v, ref, hd, 2, 1.0f);
    int ok = 1;
    for (int i = 0; i < hd; i++)
        if (!CLOSE(out[i], ref[i], 1e-3f)) ok = 0;
    CHECK("two_tokens_equal", ok);
}

static void test_dominant_token(void) {
    // One K has huge score → output should be close to its V
    int hd = 4;
    float q[] = {1, 0, 0, 0};
    float k[] = {100,0,0,0,  0,0,0,0,  0,0,0,0};
    float v[] = {1,2,3,4,  10,20,30,40,  100,200,300,400};
    float out[4] = {0}, ref[4] = {0};
    fused_attention_f32(q, k, v, out, hd, 3, 1.0f);
    ref_attention(q, k, v, ref, hd, 3, 1.0f);
    int ok = 1;
    for (int i = 0; i < hd; i++)
        if (!CLOSE(out[i], ref[i], 1e-2f)) ok = 0;
    CHECK("dominant_token", ok);
}

static void test_hd128_seq64(void) {
    // Realistic dimensions: head_dim=128, seq_len=64
    int hd = 128, seq = 64;
    float *q = malloc(hd * sizeof(float));
    float *k = malloc(seq * hd * sizeof(float));
    float *v = malloc(seq * hd * sizeof(float));
    float *out = malloc(hd * sizeof(float));
    float *ref = malloc(hd * sizeof(float));
    srand(42);
    for (int i = 0; i < hd; i++) q[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < seq * hd; i++) {
        k[i] = (float)rand() / RAND_MAX - 0.5f;
        v[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    float scale = 1.0f / sqrtf(128.0f);
    fused_attention_f32(q, k, v, out, hd, seq, scale);
    ref_attention(q, k, v, ref, hd, seq, scale);
    int ok = 1;
    float max_err = 0;
    for (int d = 0; d < hd; d++) {
        float err = fabsf(out[d] - ref[d]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) ok = 0;
    }
    printf("    (hd128_seq64 max_err=%.6f)\n", max_err);
    CHECK("hd128_seq64", ok);
    free(q); free(k); free(v); free(out); free(ref);
}

static void test_hd128_seq512(void) {
    // Longer context: 512 tokens
    int hd = 128, seq = 512;
    float *q = malloc(hd * sizeof(float));
    float *k = malloc(seq * hd * sizeof(float));
    float *v = malloc(seq * hd * sizeof(float));
    float *out = malloc(hd * sizeof(float));
    float *ref = malloc(hd * sizeof(float));
    srand(123);
    for (int i = 0; i < hd; i++) q[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < seq * hd; i++) {
        k[i] = (float)rand() / RAND_MAX - 0.5f;
        v[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    float scale = 1.0f / sqrtf(128.0f);
    fused_attention_f32(q, k, v, out, hd, seq, scale);
    ref_attention(q, k, v, ref, hd, seq, scale);
    int ok = 1;
    float max_err = 0;
    for (int d = 0; d < hd; d++) {
        float err = fabsf(out[d] - ref[d]);
        if (err > max_err) max_err = err;
        if (err > 1e-2f) ok = 0;
    }
    printf("    (hd128_seq512 max_err=%.6f)\n", max_err);
    CHECK("hd128_seq512", ok);
    free(q); free(k); free(v); free(out); free(ref);
}

static void test_hd128_seq2048(void) {
    // Full context: 2048 tokens
    int hd = 128, seq = 2048;
    float *q = malloc(hd * sizeof(float));
    float *k = malloc(seq * hd * sizeof(float));
    float *v = malloc(seq * hd * sizeof(float));
    float *out = malloc(hd * sizeof(float));
    float *ref = malloc(hd * sizeof(float));
    srand(999);
    for (int i = 0; i < hd; i++) q[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < seq * hd; i++) {
        k[i] = (float)rand() / RAND_MAX - 0.5f;
        v[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    float scale = 1.0f / sqrtf(128.0f);
    fused_attention_f32(q, k, v, out, hd, seq, scale);
    ref_attention(q, k, v, ref, hd, seq, scale);
    int ok = 1;
    float max_err = 0;
    for (int d = 0; d < hd; d++) {
        float err = fabsf(out[d] - ref[d]);
        if (err > max_err) max_err = err;
        if (err > 5e-2f) ok = 0;
    }
    printf("    (hd128_seq2048 max_err=%.6f)\n", max_err);
    CHECK("hd128_seq2048", ok);
    free(q); free(k); free(v); free(out); free(ref);
}

int main(void) {
    printf("test_fused_attn:\n");
    test_single_token();
    test_two_tokens_equal();
    test_dominant_token();
    test_hd128_seq64();
    test_hd128_seq512();
    test_hd128_seq2048();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
