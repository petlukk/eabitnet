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

static void test_simple_add(void) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float out[8];
    vecadd_f32(a, b, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(out[i] - 9.0f) > 1e-6f) ok = 0;
    CHECK("simple_add_8", ok);
}

static void test_zeros(void) {
    float a[16] = {0};
    float b[16] = {0};
    float out[16];
    vecadd_f32(a, b, out, 16);
    int ok = 1;
    for (int i = 0; i < 16; i++)
        if (out[i] != 0.0f) ok = 0;
    CHECK("zeros_16", ok);
}

static void test_negative(void) {
    float a[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];
    vecadd_f32(a, b, out, 4);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        if (fabsf(out[i]) > 1e-6f) ok = 0;
    CHECK("negative_cancel", ok);
}

static void test_large(void) {
    int n = 2560;
    float *a = calloc(n, sizeof(float));
    float *b = calloc(n, sizeof(float));
    float *out = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) { a[i] = (float)i; b[i] = (float)(n - i); }
    vecadd_f32(a, b, out, n);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (fabsf(out[i] - (float)n) > 1e-3f) ok = 0;
    CHECK("large_2560", ok);
    free(a); free(b); free(out);
}

static void test_scalar_tail(void) {
    float a[11], b[11], out[11];
    for (int i = 0; i < 11; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    vecadd_f32(a, b, out, 11);
    int ok = 1;
    for (int i = 0; i < 11; i++)
        if (fabsf(out[i] - 3.0f) > 1e-6f) ok = 0;
    CHECK("scalar_tail_11", ok);
}

int main(void) {
    printf("test_vecadd:\n");
    test_simple_add();
    test_zeros();
    test_negative();
    test_large();
    test_scalar_tail();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
