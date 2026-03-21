# eabitnet — Eä SIMD kernels for BitNet 1-bit LLM inference

## What

Drop-in replacement for Microsoft BitNet's hand-written C intrinsics
with portable `.ea` kernels. Part of the Eä ecosystem alongside eakv
(KV cache) and eaclaw (AI agent).

## Architecture

```
┌─────────────────────────────────────────────┐
│  eaclaw (or any app)                        │
│    eaclaw --model bitnet-3b                 │
├─────────────────────────────────────────────┤
│  llama.cpp (inference engine)               │
│    ├── eabitnet kernels                     │
│    │     ├── bitnet_i2s.ea        ✅ done   │
│    │     ├── bitnet_quant.ea      ✅ done   │
│    │     ├── bitnet_i2s_arm.ea    ✅ done   │
│    │     └── bitnet_lut.ea        ✅ done   │
│    └── eakv (KV cache)            ✅ exists │
├─────────────────────────────────────────────┤
│  eacompute (compiler)             ✅ exists │
│    ├── shuffle_bytes intrinsic    ✅ done   │
│    └── vdot_i32 intrinsic         ✅ done   │
└─────────────────────────────────────────────┘
```

## Inference pipeline

```
token in
  → embedding lookup
  → for each layer:
      → activations (f32) → quant_f32_i8 → i8       [eabitnet]
      → i8 activations × ternary weights → i2_dot_i8 [eabitnet]
      → attention: Q·K^T → softmax → ·V              [eakv]
  → output logits
```

## Kernel status

| Kernel | Functions | Tests | Status |
|--------|-----------|-------|--------|
| `bitnet_i2s.ea` | `i2_dot_i8`, `i2_dot_i8_4row` | 12/12 | Done (x86) |
| `bitnet_quant.ea` | `quant_f32_i8`, `pack_ternary_row` | 13/13 | Done (x86) |
| `bitnet_i2s_arm.ea` | `i2_dot_i8`, `i2_dot_i8_4row` | 12/12 | Done (aarch64, Pi 5) |
| `bitnet_lut.ea` | `prepare_lut_weights`, `lut_matmul`, `lut_matmul_tail` | 15/15 | Done (cross-platform) |

## Remaining work

### Kernels

- ~~**`bitnet_lut.ea`**~~ ✅ Done.
  LUT-based ternary matmul (cross-platform x86 + ARM). Processes 16 weight
  rows in parallel via `shuffle_bytes`. 15/15 tests passing, ~3.1 Gop/s.
  `prepare_lut_weights` transposes to column-interleaved layout at model load.

- ~~**`bitnet_i2s_arm.ea`**~~ ✅ Done.
  ARM NEON path for the I2_S dot product. Uses `vdot_i32` (signed×signed).
  12/12 tests passing on Pi 5, ~28 Gop/s.

### Compiler (eacompute)

- ~~**`shuffle_bytes(u8x16, u8x16) -> u8x16`**~~ ✅ Done.
  Runtime byte lookup. Maps to `vpshufb` (x86) / `tbl` (ARM).

- ~~**`vdot_i32(i8x16, i8x16) -> i32x4`**~~ ✅ Done.
  ARM NEON signed dot product. Maps to `vdotq_s32` (ARMv8.2+).

### Integration

- **Wire into eaclaw** — patch `build.rs` to compile eabitnet kernels,
  update `llama_bridge.c` to call eabitnet instead of `ggml-bitnet-mad.cpp`.
  Same pattern as eakv integration.

- **Model profile** — add `eaclaw --model bitnet-3b` config entry.
  Downloads GGUF, enables eabitnet kernel path.

## Design notes

- Weight encoding: ternary {-1, 0, +1} → 2-bit {0, 1, 2}, four per byte.
- I2_S layout (x86, QK=128): 32 packed bytes → 128 weights, 4 groups of 32.
- Dot product returns raw sum; ternary offset correction + scale applied by caller.
- `quant_f32_i8` uses `narrow_f32x4_i8` — vectors all the way, no scalar narrowing.
  Caller must allocate dst with 12 extra bytes (16-byte SIMD store writes).
- All kernels follow eacompute hard rules: <500 lines per file, end-to-end tested,
  no fake functions, no premature features.
