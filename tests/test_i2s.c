// test_i2s.c — Validates .ea kernel against scalar reference
//
// Build:
//   ./build_kernels.sh
//   gcc -O2 tests/test_i2s.c -Lbuild/lib -leabitnet -o build/test_i2s -lm
//
// Run:
//   LD_LIBRARY_PATH=build/lib ./build/test_i2s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// FFI declarations matching the .ea kernel exports
extern int32_t i2_dot_i8(const uint8_t *weights, const int8_t *activations, int32_t n);
extern void i2_dot_i8_4row(
    const uint8_t *w0, const uint8_t *w1,
    const uint8_t *w2, const uint8_t *w3,
    const int8_t *activations,
    int32_t *scores,
    int32_t n
);

// Scalar reference: mimics what maddubs does on 2-bit packed weights
// Pack format: byte[i] contains weights for positions i, i, i, i in groups 0-3
//   group 0 = bits 7:6, group 1 = bits 5:4, group 2 = bits 3:2, group 3 = bits 1:0
static int32_t ref_i2_dot_i8(const uint8_t *weights, const int8_t *activations, int n) {
    int32_t sum = 0;
    int n_blocks = n / 128;

    for (int blk = 0; blk < n_blocks; blk++) {
        const uint8_t *pw = weights + blk * 32;
        const int8_t  *pa = activations + blk * 128;

        for (int i = 0; i < 32; i++) {
            uint8_t packed = pw[i];
            uint8_t g0 = (packed >> 6) & 0x03;
            uint8_t g1 = (packed >> 4) & 0x03;
            uint8_t g2 = (packed >> 2) & 0x03;
            uint8_t g3 = (packed >> 0) & 0x03;

            sum += (int32_t)g0 * (int32_t)pa[i];
            sum += (int32_t)g1 * (int32_t)pa[i + 32];
            sum += (int32_t)g2 * (int32_t)pa[i + 64];
            sum += (int32_t)g3 * (int32_t)pa[i + 96];
        }
    }
    return sum;
}

// Pack ternary weights {-1,0,+1} → 2-bit {0,1,2} in BitNet layout
static void pack_ternary(const int8_t *ternary, uint8_t *packed, int n) {
    int n_blocks = n / 128;
    for (int blk = 0; blk < n_blocks; blk++) {
        for (int i = 0; i < 32; i++) {
            int base = blk * 128;
            uint8_t g0 = (uint8_t)(ternary[base + i]      + 1);  // -1→0, 0→1, +1→2
            uint8_t g1 = (uint8_t)(ternary[base + i + 32]  + 1);
            uint8_t g2 = (uint8_t)(ternary[base + i + 64]  + 1);
            uint8_t g3 = (uint8_t)(ternary[base + i + 96]  + 1);
            packed[blk * 32 + i] = (g0 << 6) | (g1 << 4) | (g2 << 2) | g3;
        }
    }
}

static int8_t rand_ternary(void) {
    return (int8_t)((rand() % 3) - 1);  // -1, 0, +1
}

static int8_t rand_i8(void) {
    return (int8_t)((rand() % 256) - 128);
}

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int test_single_row(int n) {
    printf("  single-row n=%d ... ", n);

    int8_t *ternary = malloc(n);
    int8_t *act     = malloc(n);
    uint8_t *packed = calloc(n / 4, 1);

    for (int i = 0; i < n; i++) {
        ternary[i] = rand_ternary();
        act[i] = rand_i8();
    }
    pack_ternary(ternary, packed, n);

    int32_t expected = ref_i2_dot_i8(packed, act, n);
    int32_t got      = i2_dot_i8(packed, act, n);

    int ok = (expected == got);
    printf("%s (expected=%d, got=%d)\n", ok ? GREEN "PASS" RESET : RED "FAIL" RESET, expected, got);

    free(ternary);
    free(act);
    free(packed);
    return ok;
}

static int test_4row(int n) {
    printf("  4-row n=%d ... ", n);

    int8_t *act = malloc(n);
    for (int i = 0; i < n; i++) act[i] = rand_i8();

    uint8_t *packed[4];
    int32_t expected[4];

    for (int r = 0; r < 4; r++) {
        int8_t *ternary = malloc(n);
        packed[r] = calloc(n / 4, 1);
        for (int i = 0; i < n; i++) ternary[i] = rand_ternary();
        pack_ternary(ternary, packed[r], n);
        expected[r] = ref_i2_dot_i8(packed[r], act, n);
        free(ternary);
    }

    int32_t got[4];
    i2_dot_i8_4row(packed[0], packed[1], packed[2], packed[3], act, got, n);

    int ok = 1;
    for (int r = 0; r < 4; r++) {
        if (expected[r] != got[r]) ok = 0;
    }
    printf("%s (expected=[%d,%d,%d,%d], got=[%d,%d,%d,%d])\n",
        ok ? GREEN "PASS" RESET : RED "FAIL" RESET,
        expected[0], expected[1], expected[2], expected[3],
        got[0], got[1], got[2], got[3]);

    for (int r = 0; r < 4; r++) free(packed[r]);
    free(act);
    return ok;
}

static double bench_single_row(int n, int iters) {
    int8_t *act     = malloc(n);
    uint8_t *packed = calloc(n / 4, 1);
    for (int i = 0; i < n; i++) act[i] = rand_i8();
    for (int i = 0; i < n / 4; i++) packed[i] = (uint8_t)(rand() & 0xFF);

    // Warmup
    volatile int32_t sink = 0;
    for (int i = 0; i < 100; i++) sink += i2_dot_i8(packed, act, n);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        sink += i2_dot_i8(packed, act, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double ns_per_call = (elapsed / iters) * 1e9;

    free(act);
    free(packed);
    return ns_per_call;
}

int main(void) {
    srand(42);

    printf("=== BitNet I2_S kernel tests ===\n\n");

    int pass = 0, total = 0;

    // Correctness: single row
    printf("Single-row dot product:\n");
    int sizes[] = {128, 256, 512, 1024, 4096, 16384};
    for (int i = 0; i < 6; i++) {
        total++;
        if (test_single_row(sizes[i])) pass++;
    }

    // Correctness: 4-row
    printf("\n4-row dot product:\n");
    for (int i = 0; i < 6; i++) {
        total++;
        if (test_4row(sizes[i])) pass++;
    }

    printf("\n%d/%d tests passed\n", pass, total);

    // Benchmark
    if (pass == total) {
        printf("\n=== Benchmark (single-row) ===\n");
        int bench_sizes[] = {1024, 4096, 16384};
        int bench_iters[] = {1000000, 500000, 100000};
        for (int i = 0; i < 3; i++) {
            double ns = bench_single_row(bench_sizes[i], bench_iters[i]);
            double gops = (double)bench_sizes[i] / ns;  // billions of weight-activation pairs per second
            printf("  n=%5d: %.1f ns/call  (%.2f Gop/s)\n", bench_sizes[i], ns, gops);
        }
    }

    return pass == total ? 0 : 1;
}
