#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

#define CLOSE(a, b, tol) (fabsf((a) - (b)) < (tol))

// Reference: scalar RoPE rotation
static void rope_ref(const float *data, const float *freqs, float *out,
                     int head_dim, int n_heads) {
    for (int h = 0; h < n_heads; h++) {
        int off = h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float r  = data[off + i];
            float im = data[off + i + 1];
            float c  = freqs[i];
            float s  = freqs[i + 1];
            out[off + i]     = r * c - im * s;
            out[off + i + 1] = r * s + im * c;
        }
    }
}

// Test basic rotation: known values
static void test_basic(void) {
    float data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float freqs[] = {cosf(0.5f), sinf(0.5f), cosf(1.0f), sinf(1.0f)};
    float out[4], ref[4];

    apply_rope_f32(data, freqs, out, 4, 1);
    rope_ref(data, freqs, ref, 4, 1);

    CHECK("basic_0", CLOSE(out[0], ref[0], 1e-5f));
    CHECK("basic_1", CLOSE(out[1], ref[1], 1e-5f));
    CHECK("basic_2", CLOSE(out[2], ref[2], 1e-5f));
    CHECK("basic_3", CLOSE(out[3], ref[3], 1e-5f));
}

// Test identity rotation (cos=1, sin=0 leaves data unchanged)
static void test_identity(void) {
    float data[] = {3.0f, 7.0f, -2.0f, 5.0f, 1.0f, -1.0f, 4.0f, 0.5f};
    float freqs[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    float out[8];

    apply_rope_f32(data, freqs, out, 8, 1);

    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], data[i], 1e-5f)) ok = 0;
    CHECK("identity_8", ok);
}

// Test 90-degree rotation (cos=0, sin=1): out[2i] = -im, out[2i+1] = r
static void test_90deg(void) {
    float data[] = {3.0f, 4.0f, 5.0f, 6.0f};
    float freqs[] = {0.0f, 1.0f, 0.0f, 1.0f};
    float out[4];

    apply_rope_f32(data, freqs, out, 4, 1);

    CHECK("90deg_0", CLOSE(out[0], -4.0f, 1e-5f));
    CHECK("90deg_1", CLOSE(out[1],  3.0f, 1e-5f));
    CHECK("90deg_2", CLOSE(out[2], -6.0f, 1e-5f));
    CHECK("90deg_3", CLOSE(out[3],  5.0f, 1e-5f));
}

// Test multiple heads get same rotation from shared freqs
static void test_multi_head(void) {
    int head_dim = 8;
    int n_heads = 4;
    int total = head_dim * n_heads;
    float *data = malloc(total * sizeof(float));
    float *freqs = malloc(head_dim * sizeof(float));
    float *out = malloc(total * sizeof(float));
    float *ref = malloc(total * sizeof(float));

    for (int i = 0; i < total; i++)
        data[i] = sinf((float)i * 0.3f);
    for (int i = 0; i < head_dim; i += 2) {
        float angle = 0.1f * (float)(i / 2);
        freqs[i] = cosf(angle);
        freqs[i + 1] = sinf(angle);
    }

    apply_rope_f32(data, freqs, out, head_dim, n_heads);
    rope_ref(data, freqs, ref, head_dim, n_heads);

    int ok = 1;
    for (int i = 0; i < total; i++)
        if (!CLOSE(out[i], ref[i], 1e-4f)) { ok = 0; break; }
    CHECK("multi_head_4x8", ok);

    free(data); free(freqs); free(out); free(ref);
}

// Test with real LLaMA dimensions: 20 heads, head_dim=128
static void test_llama_dims(void) {
    int head_dim = 128;
    int n_heads = 20;
    int total = head_dim * n_heads;
    float *data = malloc(total * sizeof(float));
    float *freqs = malloc(head_dim * sizeof(float));
    float *out = malloc(total * sizeof(float));
    float *ref = malloc(total * sizeof(float));

    for (int i = 0; i < total; i++)
        data[i] = ((float)(i % 100) - 50.0f) / 50.0f;
    for (int i = 0; i < head_dim; i += 2) {
        float angle = powf(10000.0f, -(float)(i / 2) / 64.0f);
        freqs[i] = cosf(angle);
        freqs[i + 1] = sinf(angle);
    }

    apply_rope_f32(data, freqs, out, head_dim, n_heads);
    rope_ref(data, freqs, ref, head_dim, n_heads);

    int ok = 1;
    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        float err = fabsf(out[i] - ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-4f) { ok = 0; break; }
    }
    CHECK("llama_20x128", ok);
    printf("    max_err=%.2e\n", max_err);

    free(data); free(freqs); free(out); free(ref);
}

// Test scalar cleanup path: head_dim not multiple of 4
static void test_odd_dim(void) {
    int head_dim = 6;  // 3 pairs: SIMD handles 2, scalar handles 1
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float freqs[6];
    float out[6], ref[6];

    for (int i = 0; i < 6; i += 2) {
        float angle = 0.5f * (float)(i / 2);
        freqs[i] = cosf(angle);
        freqs[i + 1] = sinf(angle);
    }

    apply_rope_f32(data, freqs, out, head_dim, 1);
    rope_ref(data, freqs, ref, head_dim, 1);

    int ok = 1;
    for (int i = 0; i < 6; i++)
        if (!CLOSE(out[i], ref[i], 1e-5f)) ok = 0;
    CHECK("odd_dim_6", ok);
}

// Test preserves vector norm (rotation is unitary)
static void test_preserves_norm(void) {
    int head_dim = 128;
    int n_heads = 1;
    float *data = malloc(head_dim * sizeof(float));
    float *freqs = malloc(head_dim * sizeof(float));
    float *out = malloc(head_dim * sizeof(float));

    for (int i = 0; i < head_dim; i++)
        data[i] = sinf((float)i);
    for (int i = 0; i < head_dim; i += 2) {
        float angle = powf(10000.0f, -(float)(i / 2) / 64.0f) * 42.0f;
        freqs[i] = cosf(angle);
        freqs[i + 1] = sinf(angle);
    }

    float norm_before = 0;
    for (int i = 0; i < head_dim; i++)
        norm_before += data[i] * data[i];

    apply_rope_f32(data, freqs, out, head_dim, n_heads);

    float norm_after = 0;
    for (int i = 0; i < head_dim; i++)
        norm_after += out[i] * out[i];

    CHECK("preserves_norm", CLOSE(norm_before, norm_after, 1e-2f));

    free(data); free(freqs); free(out);
}

int main(void) {
    printf("test_rope:\n");
    test_basic();
    test_identity();
    test_90deg();
    test_multi_head();
    test_llama_dims();
    test_odd_dim();
    test_preserves_norm();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
