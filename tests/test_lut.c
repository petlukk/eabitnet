// test_lut.c — Validates LUT matmul kernel against scalar reference

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

extern void prepare_lut_weights(
    const uint8_t *src, uint8_t *dst, int32_t n_rows, int32_t n_cols);
extern void lut_matmul(
    const uint8_t *weights, const uint8_t *activations,
    int32_t *scores, int32_t n_rows, int32_t n_cols);
extern void lut_matmul_tail(
    const uint8_t *weights, const int8_t *activations,
    int32_t *scores, int32_t n_rows, int32_t n_cols);

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static void ref_prepare(const uint8_t *src, uint8_t *dst,
                        int n_rows, int n_cols) {
    int stride = n_cols / 4;
    int n_groups = n_rows / 16;
    for (int col = 0; col < stride; col++) {
        for (int g = 0; g < n_groups; g++) {
            for (int r = 0; r < 16; r++) {
                dst[col * n_rows + g * 16 + r] =
                    src[(g * 16 + r) * stride + col];
            }
        }
    }
}

static void ref_ternary_matmul_rowmajor(const uint8_t *packed, const uint8_t *act,
                                        int32_t *scores, int n_rows, int n_cols) {
    int stride = n_cols / 4;
    memset(scores, 0, n_rows * sizeof(int32_t));

    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < stride; col++) {
            int act_base = col * 4;
            int8_t a0 = (int8_t)act[act_base];
            int8_t a1 = (int8_t)act[act_base + 1];
            int8_t a2 = (int8_t)act[act_base + 2];
            int8_t a3 = (int8_t)act[act_base + 3];

            uint8_t p = packed[row * stride + col];
            uint8_t w0 = (p >> 6) & 3;
            uint8_t w1 = (p >> 4) & 3;
            uint8_t w2 = (p >> 2) & 3;
            uint8_t w3 = p & 3;

            scores[row] += ((int32_t)w0 - 1) * (int32_t)a0;
            scores[row] += ((int32_t)w1 - 1) * (int32_t)a1;
            scores[row] += ((int32_t)w2 - 1) * (int32_t)a2;
            scores[row] += ((int32_t)w3 - 1) * (int32_t)a3;
        }
    }
}

static void pack_row_major(const int8_t *ternary, uint8_t *packed,
                           int n_rows, int n_cols) {
    int stride = n_cols / 4;
    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < n_cols; col += 4) {
            int base = row * n_cols + col;
            uint8_t v0 = (uint8_t)(ternary[base]     + 1);
            uint8_t v1 = (uint8_t)(ternary[base + 1]  + 1);
            uint8_t v2 = (uint8_t)(ternary[base + 2]  + 1);
            uint8_t v3 = (uint8_t)(ternary[base + 3]  + 1);
            packed[row * stride + col / 4] = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3;
        }
    }
}

static int8_t rand_ternary(void) {
    return (int8_t)((rand() % 3) - 1);
}

static int8_t rand_i8(void) {
    return (int8_t)((rand() % 256) - 128);
}

static int test_prepare(int n_rows, int n_cols) {
    printf("  prepare %dx%d ... ", n_rows, n_cols);

    int stride = n_cols / 4;
    int total_bytes = n_rows * stride;

    uint8_t *src = malloc(total_bytes);
    for (int i = 0; i < total_bytes; i++) src[i] = (uint8_t)(rand() & 0xFF);

    uint8_t *expected = calloc(total_bytes, 1);
    uint8_t *got      = calloc(total_bytes, 1);

    ref_prepare(src, expected, n_rows, n_cols);
    prepare_lut_weights(src, got, n_rows, n_cols);

    int ok = (memcmp(expected, got, total_bytes) == 0);
    printf("%s\n", ok ? GREEN "PASS" RESET : RED "FAIL" RESET);

    if (!ok) {
        for (int i = 0; i < total_bytes; i++) {
            if (expected[i] != got[i]) {
                printf("    first diff at [%d]: expected=%d got=%d\n",
                       i, expected[i], got[i]);
                break;
            }
        }
    }

    free(src); free(expected); free(got);
    return ok;
}

static int test_matmul(int n_rows, int n_cols) {
    printf("  matmul %dx%d ... ", n_rows, n_cols);

    int stride = n_cols / 4;

    int8_t *ternary = malloc(n_rows * n_cols);
    uint8_t *act = malloc(n_cols);
    for (int i = 0; i < n_rows * n_cols; i++) ternary[i] = rand_ternary();
    for (int i = 0; i < n_cols; i++) act[i] = (uint8_t)rand_i8();

    uint8_t *packed = calloc(n_rows * stride, 1);
    pack_row_major(ternary, packed, n_rows, n_cols);

    uint8_t *prepared = calloc(n_rows * stride, 1);
    prepare_lut_weights(packed, prepared, n_rows, n_cols);

    int32_t *expected = calloc(n_rows, sizeof(int32_t));
    ref_ternary_matmul_rowmajor(packed, act, expected, n_rows, n_cols);

    int32_t *got = calloc(n_rows, sizeof(int32_t));
    int main_rows = (n_rows / 16) * 16;
    int tail_rows = n_rows - main_rows;

    if (main_rows > 0) {
        lut_matmul(prepared, act, got, main_rows, n_cols);
    }
    if (tail_rows > 0) {
        lut_matmul_tail(packed + main_rows * stride, (const int8_t *)act,
                        got + main_rows, tail_rows, n_cols);
    }

    int ok = 1;
    for (int i = 0; i < n_rows; i++) {
        if (expected[i] != got[i]) {
            printf("%sFAIL%s at row %d: expected=%d got=%d\n",
                   RED, RESET, i, expected[i], got[i]);
            ok = 0;
            break;
        }
    }
    if (ok) printf("%sPASS%s\n", GREEN, RESET);

    free(ternary); free(act); free(packed);
    free(prepared); free(expected); free(got);
    return ok;
}

static double bench_matmul(int n_rows, int n_cols, int iters) {
    int stride = n_cols / 4;

    uint8_t *packed = malloc(n_rows * stride);
    for (int i = 0; i < n_rows * stride; i++) packed[i] = (uint8_t)(rand() & 0xFF);

    uint8_t *prepared = malloc(n_rows * stride);
    prepare_lut_weights(packed, prepared, n_rows, n_cols);

    uint8_t *act = malloc(n_cols);
    for (int i = 0; i < n_cols; i++) act[i] = (uint8_t)rand_i8();

    int32_t *scores = calloc(n_rows, sizeof(int32_t));

    for (int i = 0; i < 100; i++) lut_matmul(prepared, act, scores, n_rows, n_cols);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        lut_matmul(prepared, act, scores, n_rows, n_cols);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double ns = (elapsed / iters) * 1e9;

    free(packed); free(prepared); free(act); free(scores);
    return ns;
}

int main(void) {
    srand(42);

    printf("=== BitNet LUT matmul kernel tests ===\n\n");
    int pass = 0, total = 0;

    // prepare_lut_weights
    printf("prepare_lut_weights:\n");
    int prep_sizes[][2] = {{16,128}, {32,256}, {64,512}, {128,1024}};
    for (int i = 0; i < 4; i++) {
        total++;
        if (test_prepare(prep_sizes[i][0], prep_sizes[i][1])) pass++;
    }

    // lut_matmul (n_rows divisible by 16)
    printf("\nlut_matmul (aligned):\n");
    int mat_sizes[][2] = {{16,128}, {16,512}, {64,1024}, {128,4096}};
    for (int i = 0; i < 4; i++) {
        total++;
        if (test_matmul(mat_sizes[i][0], mat_sizes[i][1])) pass++;
    }

    // lut_matmul + tail (n_rows NOT divisible by 16)
    printf("\nlut_matmul + tail:\n");
    int tail_sizes[][2] = {{17,128}, {33,512}, {65,1024}};
    for (int i = 0; i < 3; i++) {
        total++;
        if (test_matmul(tail_sizes[i][0], tail_sizes[i][1])) pass++;
    }

    // Edge cases
    printf("\nEdge cases:\n");
    {
        printf("  all-zero weights 16x128 ... ");
        int n_rows = 16, n_cols = 128;
        int stride = n_cols / 4;
        uint8_t *packed = malloc(n_rows * stride);
        memset(packed, 0x55, n_rows * stride);
        uint8_t *prepared = malloc(n_rows * stride);
        prepare_lut_weights(packed, prepared, n_rows, n_cols);
        uint8_t *act = malloc(n_cols);
        for (int i = 0; i < n_cols; i++) act[i] = (uint8_t)rand_i8();
        int32_t scores[16] = {0};
        lut_matmul(prepared, act, scores, n_rows, n_cols);
        int ok = 1;
        for (int i = 0; i < n_rows; i++) {
            if (scores[i] != 0) { ok = 0; break; }
        }
        total++;
        if (ok) { pass++; printf("%sPASS%s\n", GREEN, RESET); }
        else printf("%sFAIL%s\n", RED, RESET);
        free(packed); free(prepared); free(act);
    }

    {
        printf("  act=127 weights=+1 16x128 ... ");
        int n_rows = 16, n_cols = 128;
        int stride = n_cols / 4;
        uint8_t *packed = malloc(n_rows * stride);
        memset(packed, 0xAA, n_rows * stride);
        uint8_t *prepared = malloc(n_rows * stride);
        prepare_lut_weights(packed, prepared, n_rows, n_cols);
        uint8_t act[128];
        memset(act, 127, n_cols);
        int32_t scores[16] = {0};
        lut_matmul(prepared, act, scores, n_rows, n_cols);
        int ok = 1;
        for (int i = 0; i < n_rows; i++) {
            if (scores[i] != 127 * n_cols) { ok = 0; break; }
        }
        total++;
        if (ok) { pass++; printf("%sPASS%s\n", GREEN, RESET); }
        else { printf("%sFAIL%s expected=%d got=%d\n", RED, RESET, 127*n_cols, scores[0]); }
        free(packed); free(prepared);
    }

    {
        printf("  act=-128 weights=+1 16x128 ... ");
        int n_rows = 16, n_cols = 128;
        int stride = n_cols / 4;
        uint8_t *packed = malloc(n_rows * stride);
        memset(packed, 0xAA, n_rows * stride);
        uint8_t *prepared = malloc(n_rows * stride);
        prepare_lut_weights(packed, prepared, n_rows, n_cols);
        uint8_t act_buf[128];
        memset(act_buf, 0x80, n_cols);
        int32_t scores[16] = {0};
        lut_matmul(prepared, act_buf, scores, n_rows, n_cols);
        int ok = 1;
        for (int i = 0; i < n_rows; i++) {
            if (scores[i] != -128 * n_cols) { ok = 0; break; }
        }
        total++;
        if (ok) { pass++; printf("%sPASS%s\n", GREEN, RESET); }
        else { printf("%sFAIL%s expected=%d got=%d\n", RED, RESET, -128*n_cols, scores[0]); }
        free(packed); free(prepared);
    }

    {
        printf("  alternating +1/-1 16x128 ... ");
        int n_rows = 16, n_cols = 128;
        int stride = n_cols / 4;
        int8_t *ternary = malloc(n_rows * n_cols);
        for (int i = 0; i < n_rows * n_cols; i++)
            ternary[i] = (i % 2 == 0) ? 1 : -1;
        uint8_t *packed = calloc(n_rows * stride, 1);
        pack_row_major(ternary, packed, n_rows, n_cols);
        uint8_t *prepared = malloc(n_rows * stride);
        prepare_lut_weights(packed, prepared, n_rows, n_cols);
        uint8_t act_buf[128];
        for (int i = 0; i < n_cols; i++) act_buf[i] = (uint8_t)(int8_t)1;
        int32_t expected[16] = {0};
        ref_ternary_matmul_rowmajor(packed, act_buf, expected, n_rows, n_cols);
        int32_t scores[16] = {0};
        lut_matmul(prepared, act_buf, scores, n_rows, n_cols);
        int ok = 1;
        for (int i = 0; i < n_rows; i++) {
            if (scores[i] != expected[i]) { ok = 0; break; }
        }
        total++;
        if (ok) { pass++; printf("%sPASS%s\n", GREEN, RESET); }
        else { printf("%sFAIL%s expected=%d got=%d\n", RED, RESET, expected[0], scores[0]); }
        free(ternary); free(packed); free(prepared);
    }

    printf("\n%d/%d tests passed\n", pass, total);

    if (pass == total) {
        printf("\n=== Benchmark (lut_matmul) ===\n");
        struct { int rows, cols, iters; } bench[] = {
            {64, 1024, 500000},
            {128, 4096, 100000},
            {256, 4096, 50000},
        };
        for (int i = 0; i < 3; i++) {
            double ns = bench_matmul(bench[i].rows, bench[i].cols, bench[i].iters);
            double gops = (double)bench[i].rows * bench[i].cols / ns;
            printf("  %dx%d: %.1f ns/call  (%.2f Gop/s)\n",
                   bench[i].rows, bench[i].cols, ns, gops);
        }
    }

    return pass == total ? 0 : 1;
}
