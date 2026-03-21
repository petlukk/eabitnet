# eabitnet — eaclaw with BitNet via Eä kernels

## What

Standalone binary that runs eaclaw's full agent stack (tools, WhatsApp, ShellGuard,
eakv KV cache) on top of Microsoft's BitNet b1.58 2B-4T model, using Eä SIMD kernels
for ternary inference. Targets Raspberry Pi 5 (ARM aarch64, 8GB RAM).

## Two-phase approach

This integration requires two sub-projects:

### Phase 1: eaclaw refactor (prerequisite)

Make `LocalLlmProvider` model-agnostic by extracting a `ModelProfile` struct:
- Chat template (currently hardcoded Qwen2.5 `<|im_start|>/<|im_end|>`)
- Model architecture params (n_layers, n_kv_heads, head_dim)
- eakv configuration (dimensions tied to model architecture)
- Activation function (standard ReLU vs BitNet's squared ReLU)

This is a small, scoped change to eaclaw that makes it support any model,
not just Qwen2.5. eabitnet then passes a BitNet profile.

### Phase 2: eabitnet integration

eabitnet depends on the refactored eaclaw-core crate and provides:
- I2_S llama.cpp patch
- Eä kernel bridge
- BitNet model profile
- Model download and management

## Architecture

```
eabitnet (standalone Rust binary)
  ├── eaclaw (git submodule)
  │     ├── llama.cpp submodule (patched at build time)
  │     ├── eakv (KV cache compression)
  │     └── agent (tools, WhatsApp, ShellGuard, etc.)
  ├── patches/
  │     └── i2s_support.patch  (adds GGML_TYPE_I2_S to llama.cpp)
  ├── csrc/
  │     └── eabitnet_bridge.c  (vec_dot wrapper + kernel registration)
  ├── kernels/ (pre-compiled .o from eabitnet kernel repo)
  │     ├── bitnet_i2s.o       (x86, maddubs_i32)
  │     ├── bitnet_i2s_arm.o   (aarch64, vdot_i32)
  │     ├── bitnet_quant.o     (x86, quant_f32_i8 + pack_ternary_row)
  │     └── bitnet_quant_arm.o (aarch64, quant — NEEDS WRITING)
  ├── build.rs (applies patch, compiles bridge, links kernels + eaclaw)
  └── src/main.rs (creates BitNet ModelProfile, starts eaclaw agent)
```

**Build note:** eabitnet's `build.rs` owns the entire llama.cpp build (with I2_S
patch applied). eaclaw-core's `local-llm` feature must be structured so that
eabitnet can supply its own llama.cpp build instead of eaclaw doing a second one.
This avoids the double-build/double-link conflict.

## Model

Microsoft BitNet b1.58 2B-4T:
- 2 billion parameters, trained on 4 trillion tokens
- GGUF file: `ggml-model-i2_s.gguf` (~1.19 GB)
- Weight format: I2_S (2-bit ternary packed, 4 weights per byte)
- Tokenizer: standard LLaMA (handled by llama.cpp)
- Source: `huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf`
- Memory: ~0.5 GB weights + ~0.5 GB runtime = fits in Pi 5's 8 GB

## Kernels

No LUT kernel — i2s is faster for i8 activations.

| Platform | Kernel | Intrinsic | Performance |
|----------|--------|-----------|-------------|
| x86_64 | `bitnet_i2s.ea` | maddubs_i32 | ~24 Gop/s |
| aarch64 | `bitnet_i2s_arm.ea` | vdot_i32 | ~28 Gop/s (Pi 5) |
| x86_64 | `bitnet_quant.ea` | narrow_f32x4_i8 | — |
| aarch64 | `bitnet_quant_arm.ea` | **TODO** | — |

**Missing kernel:** `bitnet_quant.ea` is x86-only. Need an ARM port for Pi 5.
Same algorithm, different vector intrinsics (128-bit NEON ops only).

## I2_S patch (additions to llama.cpp)

Applied at build time via `patches/i2s_support.patch`. Pin to specific llama.cpp
commit hash for reproducibility. Adds:

1. **`GGML_TYPE_I2_S` enum value** in `ggml.h`
2. **I2_S block structure** — 32 bytes = 128 ternary weights, 4 groups of 32,
   matching `bitnet_i2s.ea` layout (QK=128). No per-block scale (pure packed bits;
   scale stored separately per tensor in the GGUF, same as Microsoft's format).
3. **GGUF type mapping** — so `ggml-model-i2_s.gguf` loads correctly
4. **Type traits entries** — both `ggml_type_traits` (basic: block_size, type_size)
   and `ggml_type_traits_cpu` (compute: vec_dot, from_float, to_float)
5. **`to_float` (dequantize)** — I2_S → f32 for non-matmul ops (layer norm, etc.)
6. **`from_float`** — f32 → I2_S packed ternary (uses `pack_ternary_row` pattern)
7. **BitNet model architecture handler** — squared ReLU activation (`x * abs(x)`),
   ternary quantization-aware layer structure

## eabitnet_bridge.c

Wraps Eä kernels to match ggml's expected `vec_dot` callback signature:

```c
// ggml vec_dot signature:
//   void (*vec_dot)(int n, float *s, size_t bs,
//                   const void *x, size_t bx,
//                   const void *y, size_t by, int nrc);

static void eabitnet_vec_dot_i2s(
    int n, float *s, size_t bs,
    const void *x, size_t bx,     // I2_S packed weights
    const void *y, size_t by,     // quantized i8 activations
    int nrc                        // number of rows (1 or 4)
) {
    if (nrc == 4) {
        int32_t scores[4];
        i2_dot_i8_4row(x, x+bx, x+2*bx, x+3*bx, y, scores, n);
        // Apply ternary offset correction + convert to float
        for (int i = 0; i < 4; i++)
            s[i*bs] = (float)scores[i] - sum_activations + scale_correction;
    } else {
        int32_t raw = i2_dot_i8(x, y, n);
        *s = (float)raw - ternary_offset_correction;
    }
}

void eabitnet_register_kernels(void) {
    ggml_type_traits_cpu[GGML_TYPE_I2_S].vec_dot     = eabitnet_vec_dot_i2s;
    ggml_type_traits_cpu[GGML_TYPE_I2_S].vec_dot_type = GGML_TYPE_Q8_0;  // or I8
    ggml_type_traits[GGML_TYPE_I2_S].block_size       = 128;
    ggml_type_traits[GGML_TYPE_I2_S].type_size        = 32;
    // from_float and to_float registered in the patch itself
}
```

**Ternary offset correction:** The i2s kernels return raw `sum(weight * activation)`
where weight is {0,1,2} not {-1,0,+1}. The correction is:
`result = raw_dot - sum(activations)` (subtracts the bias from encoding +1 offset).
The bridge computes this per call.

**vec_dot_type:** Use `GGML_TYPE_Q8_0` if `GGML_TYPE_I8` doesn't exist in the
pinned llama.cpp version. Q8_0 includes per-block scale which can be set to 1.0.
Alternatively, register I8 as a new type in the patch. Determine at implementation.

## Build flow

1. `build.rs` initializes eaclaw's llama.cpp submodule
2. Applies `patches/i2s_support.patch` (pinned to commit hash)
3. Builds patched llama.cpp via CMake (eabitnet owns this build, not eaclaw)
4. Compiles `csrc/eabitnet_bridge.c` via cc crate
5. Links pre-compiled Eä kernel `.o` files (platform-selected)
6. Depends on eaclaw-core crate (with `local-llm` feature, but llama.cpp
   build delegated to eabitnet's build.rs to avoid double-build)
7. Builds `eabitnet` binary

**Double-build avoidance:** eaclaw-core's `build.rs` must be structured so
that when used as a dependency, the llama.cpp build can be provided externally
(e.g., via environment variable `LLAMA_CPP_LIB_DIR`). This is part of the
Phase 1 eaclaw refactor.

## Runtime

```
eabitnet --model bitnet-2b
```

- First run: downloads `ggml-model-i2_s.gguf` to `~/.eabitnet/models/`
- Registers eabitnet kernels via bridge
- Loads model via patched llama.cpp (I2_S recognized)
- Creates BitNet `ModelProfile` (chat template, architecture params, eakv config)
- Starts eaclaw agent with BitNet as LLM backend
- All eaclaw features available: tools, WhatsApp, ShellGuard, eakv KV cache

## Testing

- **Unit:** eabitnet kernel tests (40/40 already passing)
- **Integration:** load GGUF, run short generation, verify coherent output
- **Cross-platform:** build on x86 (dev) and aarch64 (Pi 5)
- **Regression:** eaclaw's existing tests still pass with refactored ModelProfile

## Prerequisites before implementation

1. **eaclaw refactor** — ModelProfile struct, configurable chat template + model params
2. **ARM quant kernel** — port `bitnet_quant.ea` to aarch64
3. **eabitnet.h update** — add C declarations for quant functions
4. **Kernel .o build** — add `--obj` output mode to build_kernels.sh (currently .so only)

## Out of scope

- LUT kernel (dropped — i2s is faster for i8 activations)
- GPU backends (CPU-only, Pi 5 target)
- Custom tokenizer (BitNet uses standard LLaMA tokenizer)
