// test_q6k_dot.c — Validates Q6_K × Q8_K dot product kernel against scalar reference
//
// Build & run:
//   $EA kernels/q6k_dot.ea --lib -o build/lib/libq6k_dot.so
//   gcc -O2 -Wall tests/test_q6k_dot.c -Lbuild/lib -lq6k_dot -o build/test_q6k_dot -lm
//   LD_LIBRARY_PATH=build/lib ./build/test_q6k_dot

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

extern float q6k_dot_q8k(
    const uint8_t *ql,
    const uint8_t *qh,
    const int8_t *scales,
    const int8_t *q8,
    const int32_t *bsums,
    int32_t n_blocks,
    float d
);

extern void q6k_dot_q8k_4row(
    const uint8_t *ql0, const uint8_t *ql1,
    const uint8_t *ql2, const uint8_t *ql3,
    const uint8_t *qh0, const uint8_t *qh1,
    const uint8_t *qh2, const uint8_t *qh3,
    const int8_t *sc0, const int8_t *sc1,
    const int8_t *sc2, const int8_t *sc3,
    const int8_t *q8,
    const int32_t *bsums,
    float *scores,
    int32_t n_blocks,
    float d0, float d1, float d2, float d3
);

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

// Reconstruct Q6_K element as signed value (-32 to 31)
// Layout per half (128 elements from 64 ql bytes + 32 qh bytes):
//   Group 0: ql[0..31]  low nibble,  qh[0..31] bits [0,1]
//   Group 1: ql[0..31]  high nibble, qh[0..31] bits [2,3]
//   Group 2: ql[32..63] low nibble,  qh[0..31] bits [4,5]
//   Group 3: ql[32..63] high nibble, qh[0..31] bits [6,7]
static int8_t q6k_element(const uint8_t *ql, const uint8_t *qh, int idx) {
    int half = idx / 128;
    int elem = idx % 128;
    int group = elem / 32;
    int pos = elem % 32;

    uint8_t low4, high2;
    int ql_base = half * 64;
    int qh_base = half * 32;

    switch (group) {
        case 0:
            low4 = ql[ql_base + pos] & 0x0F;
            high2 = (qh[qh_base + pos] >> 0) & 0x03;
            break;
        case 1:
            low4 = ql[ql_base + pos] >> 4;
            high2 = (qh[qh_base + pos] >> 2) & 0x03;
            break;
        case 2:
            low4 = ql[ql_base + 32 + pos] & 0x0F;
            high2 = (qh[qh_base + pos] >> 4) & 0x03;
            break;
        case 3:
            low4 = ql[ql_base + 32 + pos] >> 4;
            high2 = (qh[qh_base + pos] >> 6) & 0x03;
            break;
        default:
            return 0;
    }
    uint8_t unsigned_val = low4 | (high2 << 4);  // 0-63
    return (int8_t)(unsigned_val - 32);           // -32 to 31
}

// Scalar reference: Q6_K × Q8_K dot product
static float ref_q6k_dot_q8k(
    const uint8_t *ql, const uint8_t *qh,
    const int8_t *scales, const int8_t *q8,
    int n_blocks, float d
) {
    float result = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        int sumi = 0;
        for (int i = 0; i < 256; i++) {
            int8_t w = q6k_element(ql + blk * 128, qh + blk * 64, i);
            int scale_idx = i / 16;
            sumi += (int)scales[blk * 16 + scale_idx] * (int)w * (int)q8[blk * 256 + i];
        }
        result += d * (float)sumi;
    }
    return result;
}

// Compute bsums from q8 values (sum of 16 consecutive i8 -> i32)
static void compute_bsums(const int8_t *q8, int32_t *bsums, int n_blocks) {
    for (int blk = 0; blk < n_blocks; blk++) {
        for (int g = 0; g < 16; g++) {
            int32_t sum = 0;
            for (int k = 0; k < 16; k++) {
                sum += (int32_t)q8[blk * 256 + g * 16 + k];
            }
            bsums[blk * 16 + g] = sum;
        }
    }
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int tests_passed = 0;
static int tests_failed = 0;

static void check(const char *name, float expected, float actual, float tol) {
    float diff = fabsf(expected - actual);
    float rel = (fabsf(expected) > 1e-6f) ? diff / fabsf(expected) : diff;
    if (rel <= tol && diff <= tol * 100.0f) {
        printf(GREEN "  PASS" RESET " %s: expected=%.4f, got=%.4f (diff=%.6f)\n",
               name, expected, actual, diff);
        tests_passed++;
    } else {
        printf(RED "  FAIL" RESET " %s: expected=%.4f, got=%.4f (diff=%.6f, rel=%.6f)\n",
               name, expected, actual, diff, rel);
        tests_failed++;
    }
}

// --- Test helpers ---

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
} Q6KBlock;

// Pack a signed 6-bit value (-32 to 31) into Q6_K block at position idx
static void pack_q6k(Q6KBlock *blk, int idx, int8_t val) {
    uint8_t uval = (uint8_t)(val + 32);  // 0-63
    uint8_t low4 = uval & 0x0F;
    uint8_t high2 = (uval >> 4) & 0x03;

    int half = idx / 128;
    int elem = idx % 128;
    int group = elem / 32;
    int pos = elem % 32;

    int ql_base = half * 64;
    int qh_base = half * 32;

    switch (group) {
        case 0:
            blk->ql[ql_base + pos] = (blk->ql[ql_base + pos] & 0xF0) | low4;
            blk->qh[qh_base + pos] = (blk->qh[qh_base + pos] & ~0x03) | high2;
            break;
        case 1:
            blk->ql[ql_base + pos] = (blk->ql[ql_base + pos] & 0x0F) | (low4 << 4);
            blk->qh[qh_base + pos] = (blk->qh[qh_base + pos] & ~0x0C) | (high2 << 2);
            break;
        case 2:
            blk->ql[ql_base + 32 + pos] = (blk->ql[ql_base + 32 + pos] & 0xF0) | low4;
            blk->qh[qh_base + pos] = (blk->qh[qh_base + pos] & ~0x30) | (high2 << 4);
            break;
        case 3:
            blk->ql[ql_base + 32 + pos] = (blk->ql[ql_base + 32 + pos] & 0x0F) | (low4 << 4);
            blk->qh[qh_base + pos] = (blk->qh[qh_base + pos] & ~0xC0) | (high2 << 6);
            break;
    }
}

// --- Tests ---

void test_zeros(void) {
    printf("test_zeros:\n");
    int n = 1;
    Q6KBlock blk;
    memset(&blk, 0, sizeof(blk));
    int8_t q8[256];
    int32_t bsums[16];
    memset(q8, 0, sizeof(q8));
    memset(bsums, 0, sizeof(bsums));
    for (int i = 0; i < 16; i++) blk.scales[i] = 1;

    float ref = ref_q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, n, 1.0f);
    float kern = q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, bsums, n, 1.0f);
    check("zeros", ref, kern, 1e-5f);
}

void test_ones(void) {
    printf("test_ones:\n");
    int n = 1;
    Q6KBlock blk;
    memset(&blk, 0, sizeof(blk));
    int8_t q8[256];
    int32_t bsums[16];

    // Pack value 1 (signed) = unsigned 33 into all positions
    for (int i = 0; i < 256; i++) {
        pack_q6k(&blk, i, 1);
        q8[i] = 1;
    }
    for (int i = 0; i < 16; i++) blk.scales[i] = 1;
    compute_bsums(q8, bsums, n);

    float ref = ref_q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, n, 1.0f);
    float kern = q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, bsums, n, 1.0f);
    check("ones (w=1, x=1)", ref, kern, 1e-5f);
}

void test_negative_weights(void) {
    printf("test_negative_weights:\n");
    int n = 1;
    Q6KBlock blk;
    memset(&blk, 0, sizeof(blk));
    int8_t q8[256];
    int32_t bsums[16];

    for (int i = 0; i < 256; i++) {
        pack_q6k(&blk, i, -10);
        q8[i] = 5;
    }
    for (int i = 0; i < 16; i++) blk.scales[i] = 3;
    compute_bsums(q8, bsums, n);

    float ref = ref_q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, n, 0.5f);
    float kern = q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, bsums, n, 0.5f);
    check("negative weights (w=-10, x=5, sc=3, d=0.5)", ref, kern, 1e-5f);
}

void test_negative_scales(void) {
    printf("test_negative_scales:\n");
    int n = 1;
    Q6KBlock blk;
    memset(&blk, 0, sizeof(blk));
    int8_t q8[256];
    int32_t bsums[16];

    for (int i = 0; i < 256; i++) {
        pack_q6k(&blk, i, 15);
        q8[i] = 10;
    }
    for (int i = 0; i < 16; i++) blk.scales[i] = -5;
    compute_bsums(q8, bsums, n);

    float ref = ref_q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, n, 1.0f);
    float kern = q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, bsums, n, 1.0f);
    check("negative scales (w=15, x=10, sc=-5)", ref, kern, 1e-5f);
}

void test_max_range(void) {
    printf("test_max_range:\n");
    int n = 1;
    Q6KBlock blk;
    memset(&blk, 0, sizeof(blk));
    int8_t q8[256];
    int32_t bsums[16];

    for (int i = 0; i < 256; i++) {
        pack_q6k(&blk, i, 31);   // max positive Q6_K
        q8[i] = 127;             // max Q8_K
    }
    for (int i = 0; i < 16; i++) blk.scales[i] = 127;
    compute_bsums(q8, bsums, n);

    float ref = ref_q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, n, 1.0f);
    float kern = q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, bsums, n, 1.0f);
    check("max range (w=31, x=127, sc=127)", ref, kern, 1e-4f);
}

void test_random_single_block(void) {
    printf("test_random_single_block:\n");
    uint32_t rng = 42;
    int n = 1;
    Q6KBlock blk;
    memset(&blk, 0, sizeof(blk));
    int8_t q8[256];
    int32_t bsums[16];

    for (int i = 0; i < 256; i++) {
        int8_t w = (int8_t)((xorshift32(&rng) % 64) - 32);
        pack_q6k(&blk, i, w);
        q8[i] = (int8_t)((xorshift32(&rng) % 256) - 128);
    }
    for (int i = 0; i < 16; i++) {
        blk.scales[i] = (int8_t)((xorshift32(&rng) % 256) - 128);
    }
    compute_bsums(q8, bsums, n);

    float ref = ref_q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, n, 0.123f);
    float kern = q6k_dot_q8k(blk.ql, blk.qh, blk.scales, q8, bsums, n, 0.123f);
    check("random 1 block", ref, kern, 1e-4f);
}

void test_random_multi_block(void) {
    printf("test_random_multi_block:\n");
    uint32_t rng = 12345;
    int n = 4;
    Q6KBlock blks[4];
    memset(blks, 0, sizeof(blks));
    int8_t q8[4 * 256];
    int32_t bsums[4 * 16];

    for (int b = 0; b < n; b++) {
        for (int i = 0; i < 256; i++) {
            int8_t w = (int8_t)((xorshift32(&rng) % 64) - 32);
            pack_q6k(&blks[b], i, w);
            q8[b * 256 + i] = (int8_t)((xorshift32(&rng) % 256) - 128);
        }
        for (int i = 0; i < 16; i++) {
            blks[b].scales[i] = (int8_t)((xorshift32(&rng) % 256) - 128);
        }
    }

    // Flatten ql, qh, scales across blocks
    uint8_t all_ql[4 * 128];
    uint8_t all_qh[4 * 64];
    int8_t all_sc[4 * 16];
    for (int b = 0; b < n; b++) {
        memcpy(all_ql + b * 128, blks[b].ql, 128);
        memcpy(all_qh + b * 64, blks[b].qh, 64);
        memcpy(all_sc + b * 16, blks[b].scales, 16);
    }
    compute_bsums(q8, bsums, n);

    float ref = ref_q6k_dot_q8k(all_ql, all_qh, all_sc, q8, n, 0.05f);
    float kern = q6k_dot_q8k(all_ql, all_qh, all_sc, q8, bsums, n, 0.05f);
    check("random 4 blocks", ref, kern, 1e-4f);
}

void test_4row(void) {
    printf("test_4row:\n");
    uint32_t rng = 99;
    int n = 2;
    Q6KBlock rows[4][2];  // 4 rows × 2 blocks
    memset(rows, 0, sizeof(rows));
    int8_t q8[2 * 256];
    int32_t bsums[2 * 16];

    // Generate shared Q8 activations
    for (int i = 0; i < n * 256; i++)
        q8[i] = (int8_t)((xorshift32(&rng) % 256) - 128);
    compute_bsums(q8, bsums, n);

    // Generate 4 weight rows
    uint8_t row_ql[4][2 * 128];
    uint8_t row_qh[4][2 * 64];
    int8_t row_sc[4][2 * 16];
    float d_vals[4];

    for (int r = 0; r < 4; r++) {
        d_vals[r] = 0.01f * (float)(r + 1);
        for (int b = 0; b < n; b++) {
            for (int i = 0; i < 256; i++) {
                int8_t w = (int8_t)((xorshift32(&rng) % 64) - 32);
                pack_q6k(&rows[r][b], i, w);
            }
            for (int i = 0; i < 16; i++) {
                rows[r][b].scales[i] = (int8_t)((xorshift32(&rng) % 256) - 128);
            }
            memcpy(row_ql[r] + b * 128, rows[r][b].ql, 128);
            memcpy(row_qh[r] + b * 64, rows[r][b].qh, 64);
            memcpy(row_sc[r] + b * 16, rows[r][b].scales, 16);
        }
    }

    // Reference: 4 single-row calls
    float ref[4];
    for (int r = 0; r < 4; r++) {
        ref[r] = ref_q6k_dot_q8k(row_ql[r], row_qh[r], row_sc[r], q8, n, d_vals[r]);
    }

    // Kernel: 4-row call
    float kern[4];
    q6k_dot_q8k_4row(
        row_ql[0], row_ql[1], row_ql[2], row_ql[3],
        row_qh[0], row_qh[1], row_qh[2], row_qh[3],
        row_sc[0], row_sc[1], row_sc[2], row_sc[3],
        q8, bsums, kern, n,
        d_vals[0], d_vals[1], d_vals[2], d_vals[3]
    );

    for (int r = 0; r < 4; r++) {
        char name[32];
        snprintf(name, sizeof(name), "4row[%d]", r);
        check(name, ref[r], kern[r], 1e-4f);
    }
}

void test_4row_vs_single(void) {
    printf("test_4row_vs_single:\n");
    uint32_t rng = 777;
    int n = 1;

    Q6KBlock row_blks[4];
    memset(row_blks, 0, sizeof(row_blks));
    int8_t q8[256];
    int32_t bsums[16];

    for (int i = 0; i < 256; i++)
        q8[i] = (int8_t)((xorshift32(&rng) % 256) - 128);
    compute_bsums(q8, bsums, n);

    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 256; i++) {
            int8_t w = (int8_t)((xorshift32(&rng) % 64) - 32);
            pack_q6k(&row_blks[r], i, w);
        }
        for (int i = 0; i < 16; i++)
            row_blks[r].scales[i] = (int8_t)((xorshift32(&rng) % 256) - 128);
    }

    float d = 0.07f;

    // Single-row kernel results
    float single[4];
    for (int r = 0; r < 4; r++) {
        single[r] = q6k_dot_q8k(row_blks[r].ql, row_blks[r].qh,
                                  row_blks[r].scales, q8, bsums, n, d);
    }

    // 4-row kernel results
    float multi[4];
    q6k_dot_q8k_4row(
        row_blks[0].ql, row_blks[1].ql, row_blks[2].ql, row_blks[3].ql,
        row_blks[0].qh, row_blks[1].qh, row_blks[2].qh, row_blks[3].qh,
        row_blks[0].scales, row_blks[1].scales, row_blks[2].scales, row_blks[3].scales,
        q8, bsums, multi, n,
        d, d, d, d
    );

    for (int r = 0; r < 4; r++) {
        char name[48];
        snprintf(name, sizeof(name), "4row[%d] vs single", r);
        check(name, single[r], multi[r], 1e-6f);
    }
}

int main(void) {
    printf("=== Q6_K × Q8_K dot product tests ===\n\n");

    test_zeros();
    test_ones();
    test_negative_weights();
    test_negative_scales();
    test_max_range();
    test_random_single_block();
    test_random_multi_block();
    test_4row();
    test_4row_vs_single();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
