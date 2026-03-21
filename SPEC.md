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
│    │     ├── bitnet_i2s_arm.ea    🔲 arm    │
│    │     └── bitnet_lut.ea        🔲 lut    │
│    └── eakv (KV cache)            ✅ exists │
├─────────────────────────────────────────────┤
│  eacompute (compiler)             ✅ exists │
│    ├── shuffle_bytes intrinsic    🔲 lut    │
│    └── dot_i32 intrinsic          🔲 arm    │
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
| `bitnet_i2s_arm.ea` | same API, NEON path | — | Needs `dot_i32` intrinsic + Pi |
| `bitnet_lut.ea` | LUT-based ternary matmul | — | Needs `shuffle_bytes` intrinsic |

## Remaining work

### Kernels

- **`bitnet_lut.ea`** — LUT-based ternary matmul (x86 + ARM).
  Replaces BitNet's TL1/TL2 preset kernel headers (thousands of generated lines).
  Uses `shuffle_bytes` for 16-entry byte lookup instead of multiplication.
  Blocked on: `shuffle_bytes` intrinsic in eacompute.

- **`bitnet_i2s_arm.ea`** — ARM NEON path for the I2_S dot product.
  Same algorithm, different intrinsic (`dot_i32` instead of `maddubs_i32`).
  Blocked on: `dot_i32` intrinsic in eacompute + Raspberry Pi for testing.

### Compiler (eacompute)

- **`shuffle_bytes(u8x16, u8x16) -> u8x16`** — runtime byte lookup.
  Maps to `vpshufb` (x86) / `tbl` (ARM). Not the same as `shuffle` which is
  compile-time lane permutation. This is data-driven: indices come from weights.

- **`dot_i32(i8x16, i8x16) -> i32x4`** — ARM NEON signed dot product.
  Maps to `vdotq_s32` (ARMv8.2+). The ARM equivalent of `maddubs_i32`.

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
