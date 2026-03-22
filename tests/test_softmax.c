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

static void ref_softmax(const float *x, float *out, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { out[i] = expf(x[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

static void test_uniform(void) {
    float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    softmax_f32(x, out, 4);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        if (!CLOSE(out[i], 0.25f, 1e-3f)) ok = 0;
    CHECK("uniform_4", ok);
}

static void test_one_hot(void) {
    float x[] = {0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float out[8];
    softmax_f32(x, out, 8);
    CHECK("one_hot_peak", out[2] > 0.99f);
    float sum = 0;
    for (int i = 0; i < 8; i++) sum += out[i];
    CHECK("one_hot_sums_to_1", CLOSE(sum, 1.0f, 1e-3f));
}

static void test_negative_shift(void) {
    float x[] = {-1000.0f, -999.0f, -998.0f, -997.0f};
    float out[4], ref[4];
    ref_softmax(x, ref, 4);
    softmax_f32(x, out, 4);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        if (!CLOSE(out[i], ref[i], 1e-5f)) ok = 0;
    CHECK("negative_shift", ok);
}

static void test_known_distribution(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float out[8], ref[8];
    ref_softmax(x, ref, 8);
    softmax_f32(x, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], ref[i], 1e-5f)) ok = 0;
    CHECK("known_dist_8", ok);
}

static void test_sums_to_one(void) {
    int n = 80;
    float *x = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = (float)(i - n/2) * 0.1f;
    softmax_f32(x, out, n);
    float sum = 0;
    for (int i = 0; i < n; i++) sum += out[i];
    CHECK("sums_to_one_80", CLOSE(sum, 1.0f, 1e-5f));
    free(x); free(out);
}

static void test_large_seq(void) {
    int n = 2048;
    float *x = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = sinf((float)i * 0.01f) * 5.0f;
    ref_softmax(x, ref, n);
    softmax_f32(x, out, n);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (!CLOSE(out[i], ref[i], 1e-5f)) ok = 0;
    CHECK("large_2048", ok);
    free(x); free(out); free(ref);
}

int main(void) {
    printf("test_softmax:\n");
    test_uniform();
    test_one_hot();
    test_negative_shift();
    test_known_distribution();
    test_sums_to_one();
    test_large_seq();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
