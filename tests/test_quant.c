// test_quant.c — Validates quantization kernels against scalar reference
//
// Build & run: make test_quant

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

extern void quant_f32_i8(const float *src, int8_t *dst, float *out_scale, int32_t *out_sum, int32_t n);
extern void pack_ternary_row(const uint8_t *ternary, uint8_t *packed, int32_t n);

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

// --- quant_f32_i8 tests ---

static void ref_quant_f32_i8(const float *src, int8_t *dst, float *scale, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    *scale = amax;
    if (amax < 1e-6f) {
        memset(dst, 0, n);
        return;
    }
    float inv = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        float v = src[i] * inv;
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        dst[i] = (int8_t)(int32_t)v;
    }
}

static int test_quant(const char *name, const float *src, int n) {
    printf("  quant %s n=%d ... ", name, n);

    // Extra 12 bytes: narrow_f32x4_i8 writes 16 bytes per 4-element store
    int8_t *expected = calloc(n + 12, 1);
    int8_t *got      = calloc(n + 12, 1);
    float exp_scale, got_scale;
    int32_t got_sum;

    ref_quant_f32_i8(src, expected, &exp_scale, n);
    quant_f32_i8(src, got, &got_scale, &got_sum, n);

    int ok = 1;
    if (fabsf(exp_scale - got_scale) > 1e-6f) {
        printf("%sFAIL%s scale mismatch (expected=%f, got=%f)\n", RED, RESET, exp_scale, got_scale);
        ok = 0;
    } else {
        for (int i = 0; i < n; i++) {
            if (abs(expected[i] - got[i]) > 1) {
                printf("%sFAIL%s at [%d] expected=%d got=%d\n", RED, RESET, i, (int)expected[i], (int)got[i]);
                ok = 0;
                break;
            }
        }
    }
    if (ok) printf("%sPASS%s\n", GREEN, RESET);

    free(expected);
    free(got);
    return ok;
}

// --- pack_ternary_row tests ---

static void ref_pack(const uint8_t *ternary, uint8_t *packed, int n) {
    for (int i = 0; i < n; i += 4) {
        packed[i / 4] = (ternary[i] << 6) | (ternary[i+1] << 4) | (ternary[i+2] << 2) | ternary[i+3];
    }
}

static int test_pack(int n) {
    printf("  pack n=%d ... ", n);

    uint8_t *ternary = malloc(n);
    for (int i = 0; i < n; i++) ternary[i] = rand() % 3;

    uint8_t *expected = calloc(n / 4, 1);
    uint8_t *got      = calloc(n / 4, 1);

    ref_pack(ternary, expected, n);
    pack_ternary_row(ternary, got, n);

    int ok = (memcmp(expected, got, n / 4) == 0);
    printf("%s\n", ok ? GREEN "PASS" RESET : RED "FAIL" RESET);

    free(ternary);
    free(expected);
    free(got);
    return ok;
}

#define CHECK(name, cond) do { \
    printf("  %s ... %s\n", (name), (cond) ? GREEN "PASS" RESET : RED "FAIL" RESET); \
} while(0)

static int test_quant_activation_sum(void) {
    float src[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    int8_t dst[8 + 12];
    float scale;
    int32_t sum;
    quant_f32_i8(src, dst, &scale, &sum, 8);
    int32_t expected_sum = 0;
    for (int i = 0; i < 8; i++) expected_sum += dst[i];
    int ok = (sum == expected_sum);
    CHECK("quant_activation_sum", ok);
    return ok;
}

static int test_quant_sum_zeros(void) {
    float src[16] = {0};
    int8_t dst[16 + 12];
    float scale;
    int32_t sum;
    quant_f32_i8(src, dst, &scale, &sum, 16);
    int ok = (sum == 0);
    CHECK("quant_sum_zeros", ok);
    return ok;
}

int main(void) {
    srand(42);

    printf("=== BitNet quantization kernel tests ===\n\n");
    int pass = 0, total = 0;

    // quant_f32_i8 tests
    printf("quant_f32_i8:\n");

    // Test with random data
    int sizes[] = {8, 16, 32, 128, 1024, 4096};
    for (int s = 0; s < 6; s++) {
        int n = sizes[s];
        float *src = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            src[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        total++;
        if (test_quant("random", src, n)) pass++;
        free(src);
    }

    // Test with zeros
    {
        float zeros[32] = {0};
        total++;
        if (test_quant("zeros", zeros, 32)) pass++;
    }

    // Test with single large value
    {
        float spike[16] = {0};
        spike[7] = 100.0f;
        total++;
        if (test_quant("spike", spike, 16)) pass++;
    }

    // quant_f32_i8 activation sum tests
    total++;
    if (test_quant_activation_sum()) pass++;
    total++;
    if (test_quant_sum_zeros()) pass++;

    // pack_ternary_row tests
    printf("\npack_ternary_row:\n");
    int pack_sizes[] = {4, 16, 128, 1024, 4096};
    for (int s = 0; s < 5; s++) {
        total++;
        if (test_pack(pack_sizes[s])) pass++;
    }

    printf("\n%d/%d tests passed\n", pass, total);
    return pass == total ? 0 : 1;
}
