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

static void test_simple(void) {
    // act=[1,1,1,...,1] (16 elements), w=[2,2,...,2] as u8
    // dot = 16 * (2 * 1) = 32 (maddubs: u8*i8 pairwise)
    int8_t act[16]; for (int i = 0; i < 16; i++) act[i] = 1;
    uint8_t w[16]; for (int i = 0; i < 16; i++) w[i] = 2;
    int32_t r = i8dot_1row(act, w, 16);
    CHECK("simple_16", r == 32);
}

static void test_4row(void) {
    int n = 32;
    int8_t act[32]; for (int i = 0; i < n; i++) act[i] = 1;
    uint8_t w0[32], w1[32], w2[32], w3[32];
    for (int i = 0; i < n; i++) {
        w0[i] = 1; w1[i] = 2; w2[i] = 3; w3[i] = 4;
    }
    int32_t scores[4];
    i8dot_4row(act, w0, w1, w2, w3, scores, n);
    CHECK("4row_r0", scores[0] == 32);
    CHECK("4row_r1", scores[1] == 64);
    CHECK("4row_r2", scores[2] == 96);
    CHECK("4row_r3", scores[3] == 128);
}

static void test_bias_correction(void) {
    // Simulate quantized output projection:
    // Original f32 weights quantized to i8 with absmax scaling
    // Then bias +128 applied to make u8 for maddubs
    int n = 16;
    int8_t act[16]; for (int i = 0; i < n; i++) act[i] = 10;
    // Original weight: all zeros → quantized i8 = 0 → u8 = 128
    uint8_t w[16]; for (int i = 0; i < n; i++) w[i] = 128;
    int32_t raw = i8dot_1row(act, w, n);
    // raw = sum(128 * 10) = 16 * 1280 = 20480
    // true = raw - 128 * sum(act) = 20480 - 128 * 160 = 20480 - 20480 = 0
    int32_t act_sum = 0;
    for (int i = 0; i < n; i++) act_sum += act[i];
    int32_t corrected = raw - 128 * act_sum;
    CHECK("bias_correction_zero", corrected == 0);
}

static void test_large_dim(void) {
    int n = 2560;
    int8_t *act = malloc(n);
    uint8_t *w = malloc(n);
    for (int i = 0; i < n; i++) { act[i] = 1; w[i] = 130; } // offset weight = 2 → u8 = 130
    int32_t raw = i8dot_1row(act, w, n);
    int32_t act_sum = n; // all 1s
    int32_t corrected = raw - 128 * act_sum;
    // true dot = sum(1 * 2) = 2*2560 = 5120
    CHECK("large_2560", corrected == 5120);
    free(act); free(w);
}

int main(void) {
    printf("test_i8dot:\n");
    test_simple();
    test_4row();
    test_bias_correction();
    test_large_dim();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
