// test_q4k_quant.c — Validates Q8_K quantization kernel against scalar reference
//
// Build & run:
//   EA=/root/dev/eacompute/target/release/ea
//   $EA kernels/q4k_quant.ea --lib -o build/lib/libq4k_quant.so
//   gcc -O2 -Wall tests/test_q4k_quant.c -Lbuild/lib -lq4k_quant -o build/test_q4k_quant -lm
//   LD_LIBRARY_PATH=build/lib ./build/test_q4k_quant

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

extern void quant_f32_q8k(
    const float *src,
    int8_t *dst_qs,
    float *dst_d,
    int32_t *dst_bsums,
    int32_t n
);

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

// Scalar reference implementation
static void ref_quant_f32_q8k(
    const float *src,
    int8_t *qs,
    float *d,
    int32_t *bsums,
    int n
) {
    int blocks = n / 256;
    for (int b = 0; b < blocks; b++) {
        int base = b * 256;

        // absmax
        float amax = 0;
        for (int i = 0; i < 256; i++) {
            float a = fabsf(src[base + i]);
            if (a > amax) amax = a;
        }

        if (amax < 1e-6f) {
            d[b] = 0.0f;
            memset(qs + base, 0, 256);
            for (int j = 0; j < 16; j++) bsums[b * 16 + j] = 0;
            continue;
        }

        d[b] = amax / 127.0f;
        float inv = 127.0f / amax;

        for (int i = 0; i < 256; i++) {
            float v = src[base + i] * inv;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            qs[base + i] = (int8_t)(int32_t)v;  // truncate, matching kernel
        }

        for (int g = 0; g < 16; g++) {
            int32_t sum = 0;
            for (int k = 0; k < 16; k++) {
                sum += (int32_t)qs[base + g * 16 + k];
            }
            bsums[b * 16 + g] = sum;
        }
    }
}

static int test_q8k(const char *name, const float *src, int n) {
    printf("  q8k %s n=%d ... ", name, n);

    int blocks = n / 256;
    int8_t *ref_qs  = calloc(n + 12, 1);
    int8_t *got_qs  = calloc(n + 12, 1);
    float *ref_d    = calloc(blocks, sizeof(float));
    float *got_d    = calloc(blocks, sizeof(float));
    int32_t *ref_bs = calloc(blocks * 16, sizeof(int32_t));
    int32_t *got_bs = calloc(blocks * 16, sizeof(int32_t));

    ref_quant_f32_q8k(src, ref_qs, ref_d, ref_bs, n);
    quant_f32_q8k(src, got_qs, got_d, got_bs, n);

    int ok = 1;

    // Check scales
    for (int b = 0; b < blocks && ok; b++) {
        if (fabsf(ref_d[b] - got_d[b]) > 1e-6f) {
            printf("%sFAIL%s scale[%d] expected=%f got=%f\n",
                   RED, RESET, b, ref_d[b], got_d[b]);
            ok = 0;
        }
    }

    // Check quantized values (allow +-1 for rounding)
    for (int i = 0; i < n && ok; i++) {
        if (abs(ref_qs[i] - got_qs[i]) > 1) {
            printf("%sFAIL%s qs[%d] expected=%d got=%d\n",
                   RED, RESET, i, (int)ref_qs[i], (int)got_qs[i]);
            ok = 0;
        }
    }

    // Check bsums (allow tolerance proportional to group size)
    for (int i = 0; i < blocks * 16 && ok; i++) {
        // Each bsum is sum of 16 i8 values; allow +-16 for rounding diffs
        if (abs(ref_bs[i] - got_bs[i]) > 16) {
            printf("%sFAIL%s bsums[%d] expected=%d got=%d\n",
                   RED, RESET, i, ref_bs[i], got_bs[i]);
            ok = 0;
        }
    }

    if (ok) printf("%sPASS%s\n", GREEN, RESET);

    free(ref_qs); free(got_qs);
    free(ref_d);  free(got_d);
    free(ref_bs); free(got_bs);
    return ok;
}

int main(void) {
    srand(42);

    printf("=== Q8_K quantization kernel tests ===\n\n");
    int pass = 0, total = 0;

    printf("quant_f32_q8k:\n");

    // Test 1: random f32, single block (n=256)
    {
        float src[256];
        for (int i = 0; i < 256; i++)
            src[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        total++;
        if (test_q8k("random", src, 256)) pass++;
    }

    // Test 2: random f32, two blocks (n=512)
    {
        float src[512];
        for (int i = 0; i < 512; i++)
            src[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        total++;
        if (test_q8k("random_2blk", src, 512)) pass++;
    }

    // Test 3: all zeros -> scale=0, qs=0, bsums=0
    {
        float src[256] = {0};
        total++;
        if (test_q8k("zeros", src, 256)) pass++;

        // Verify exact values for zeros
        printf("  q8k zeros_exact ... ");
        int8_t qs[256 + 12];
        float d[1];
        int32_t bs[16];
        memset(qs, 0xFF, sizeof(qs));
        quant_f32_q8k(src, qs, d, bs, 256);
        int ok = 1;
        if (d[0] != 0.0f) {
            printf("%sFAIL%s d=%f (expected 0)\n", RED, RESET, d[0]);
            ok = 0;
        }
        for (int i = 0; i < 256 && ok; i++) {
            if (qs[i] != 0) {
                printf("%sFAIL%s qs[%d]=%d (expected 0)\n",
                       RED, RESET, i, (int)qs[i]);
                ok = 0;
            }
        }
        for (int i = 0; i < 16 && ok; i++) {
            if (bs[i] != 0) {
                printf("%sFAIL%s bsums[%d]=%d (expected 0)\n",
                       RED, RESET, i, bs[i]);
                ok = 0;
            }
        }
        if (ok) printf("%sPASS%s\n", GREEN, RESET);
        total++;
        if (ok) pass++;
    }

    // Test 4: single spike -> verify scale and quantized values
    {
        float src[256] = {0};
        src[0] = 100.0f;
        total++;
        if (test_q8k("spike", src, 256)) pass++;

        // Verify spike details
        printf("  q8k spike_detail ... ");
        int8_t qs[256 + 12];
        float d[1];
        int32_t bs[16];
        quant_f32_q8k(src, qs, d, bs, 256);
        int ok = 1;
        // scale should be 100/127
        float expected_d = 100.0f / 127.0f;
        if (fabsf(d[0] - expected_d) > 1e-5f) {
            printf("%sFAIL%s d=%f (expected %f)\n",
                   RED, RESET, d[0], expected_d);
            ok = 0;
        }
        // qs[0] should be 127 (100 * 127/100 = 127)
        if (ok && qs[0] != 127) {
            printf("%sFAIL%s qs[0]=%d (expected 127)\n",
                   RED, RESET, (int)qs[0]);
            ok = 0;
        }
        // all other qs should be 0
        for (int i = 1; i < 256 && ok; i++) {
            if (qs[i] != 0) {
                printf("%sFAIL%s qs[%d]=%d (expected 0)\n",
                       RED, RESET, i, (int)qs[i]);
                ok = 0;
            }
        }
        // bsums[0] should be 127 (spike is in first group of 16)
        if (ok && bs[0] != 127) {
            printf("%sFAIL%s bsums[0]=%d (expected 127)\n",
                   RED, RESET, bs[0]);
            ok = 0;
        }
        // other bsums should be 0
        for (int i = 1; i < 16 && ok; i++) {
            if (bs[i] != 0) {
                printf("%sFAIL%s bsums[%d]=%d (expected 0)\n",
                       RED, RESET, i, bs[i]);
                ok = 0;
            }
        }
        if (ok) printf("%sPASS%s\n", GREEN, RESET);
        total++;
        if (ok) pass++;
    }

    // Test 5: larger random (n=1024, 4 blocks)
    {
        int n = 1024;
        float *src = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++)
            src[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        total++;
        if (test_q8k("random_4blk", src, n)) pass++;
        free(src);
    }

    printf("\n%d/%d tests passed\n", pass, total);
    return pass == total ? 0 : 1;
}
