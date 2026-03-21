# bitnet_lut.ea — LUT-based ternary matmul kernel

## Problem

The existing `bitnet_i2s.ea` kernels use multiply-accumulate (`maddubs_i32` on x86,
`vdot_i32` on ARM) to compute ternary weight x int8 activation dot products. BitNet's
TL1/TL2 preset headers replace multiplication with table lookups — precomputing all
possible partial sums and using `pshufb`/`tbl` to look them up at inference time.

eabitnet needs a LUT kernel to match this approach. The `shuffle_bytes` intrinsic is
now available in eacompute (cross-platform: SSSE3 `pshufb` on x86, NEON `tbl` on ARM).

## Design

### Algorithm: 4-weight nibble-split LUT

Group 4 ternary weights together. Each group has 3^4 = 81 possible combinations, which
doesn't fit in a single 16-entry `shuffle_bytes` lookup. Solution: split the 81 combos
across two nibble-indexed tables (16 entries each).

**Encoding:**
- 4 ternary weights {0,1,2} encode as a single byte: `w0*27 + w1*9 + w2*3 + w3`
  (values 0..80)
- Split into high nibble (byte >> 4, range 0..5) and low nibble (byte & 0x0F, range 0..15)
- Build two u8x16 tables: `table_hi[high_nibble]` + `table_lo[low_nibble]` = partial sum
  for that weight group applied to a given activation

**Inference:**
- Activation byte used as the encoded index
- `shuffle_bytes(table_hi, activation >> 4)` + `shuffle_bytes(table_lo, activation & 0x0F)`
- Accumulate results across groups into i32x4 via widening, then `reduce_add`

### API

Three functions in a single cross-platform file (no `#[cfg]`):

```
// Offline (model load): convert ternary weights to nibble LUT tables
// Input: n ternary values {0,1,2}
// Output: n/4 * 32 bytes (two 16-byte tables per 4-weight group)
build_lut_tables(
    ternary: *restrict u8,
    out tables: *mut u8 [cap: n / 4 * 32],
    n: i32
)

// Inference: single-row dot product using precomputed LUT tables
lut_dot_i8(
    tables: *restrict u8,
    activations: *restrict i8,
    n: i32
) -> i32

// Inference: 4-row dot product (mirrors i2_dot_i8_4row API)
lut_dot_i8_4row(
    t0: *restrict u8,
    t1: *restrict u8,
    t2: *restrict u8,
    t3: *restrict u8,
    activations: *restrict i8,
    out scores: *mut i32 [cap: 4],
    n: i32
)
```

### Table layout

For n weights, `build_lut_tables` produces `n/4` groups. Each group is 32 bytes:
- Bytes 0..15: `table_hi` — indexed by `encoded_index >> 4`
- Bytes 16..31: `table_lo` — indexed by `encoded_index & 0x0F`

The two lookup results are added to reconstruct the dot product contribution of that
4-weight group.

### Cross-platform

The kernel uses only cross-platform operations:
- `shuffle_bytes(u8x16, u8x16) -> u8x16` (SSSE3/NEON)
- Bitwise: `.>>`, `.&`
- Arithmetic: `.+`, `reduce_add`
- `splat`, `load`, `store`
- Widening: `widen_u8_i32x4` or `widen_i8_f32x4` as needed for accumulation

No `#[cfg]` needed. Single `.ea` file works on both x86 and aarch64.

### eaclaw integration

Minimal changes to `llama_bridge.c`:
- At model load: call `build_lut_tables` once per weight matrix
- At inference: swap `i2_dot_i8` / `i2_dot_i8_4row` calls for `lut_dot_i8` / `lut_dot_i8_4row`
- Same pointer patterns, same output semantics
- Model profile `eaclaw --model bitnet-3b` selects LUT vs I2S path

### Files

| File | Purpose |
|------|---------|
| `kernels/bitnet_lut.ea` | All three functions |
| `tests/test_lut.c` | End-to-end tests (known vectors, random, sizes 128-16384) |

### Constraints

- All eacompute hard rules apply: <500 lines per file, end-to-end tested, no fakes
- `build_lut_tables` is scalar (runs once at load, not perf-critical)
- Dot product functions must be SIMD throughout — no scalar fallback
- Returns raw sum; caller applies ternary offset correction + scale (same as i2s)
