```
        /\_/\
       ( o.o )  ╔═══════════════════════════════════╗
        >╥╥<    ║  C O U G A R                      ║
       /|   |\  ║  LLM inference engine              ║
      (_|   |_) ╚═══════════════════════════════════╝
```

# Cougar

Llama.cpp is a beast, but even beasts have predators.

A ~7K-line LLM inference engine written in Rust + [Ea](https://github.com/petlukk/eacompute) SIMD kernels.
Faster than BitNet.cpp on decode. GEMM-batched prefill. ~7K LOC vs ~250K.

Zero runtime dependencies. No CUDA. No BLAS. No frameworks.
Single self-contained binary (kernels embedded). Built for speed.

📦 **~1 MB** binary (x86-64) · **1.6 MB** binary (ARM aarch64)

**Supported models:** BitNet b1.58 (I2_S) and Llama-family Q4_K_M — any GGUF file

### Performance highlights

- **BitNet decode: 18.0 tok/s** (+22% vs BitNet.cpp)
- **BitNet prefill: 47.5 tok/s** (GEMM-batched, 2.4x over sequential)
- **Llama 3B decode: 8.3 tok/s** (~1% from llama.cpp)
- **BitNet on Raspberry Pi 5: 16.1 tok/s** (4 ARM cores, stock cooler)

## Why Cougar?

- Faster than BitNet.cpp (+22% decode)
- Matches llama.cpp on decode performance
- ~7K LOC vs ~250K LOC in llama.cpp
- Zero dependencies, single binary
- Native SIMD kernels via [Ea](https://github.com/petlukk/eacompute) (AVX2 + ARM NEON)
- Runs on a Raspberry Pi 5

## Performance

### x86-64 (16 threads, AVX2, native Linux)

#### BitNet b1.58 2B-4T (256 tokens, 22-token prompt)

| | Cougar | BitNet.cpp |
|---|---|---|
| Prefill | **47.5 tok/s** | 81.2 tok/s |
| Decode | **18.0 tok/s** | 14.7 tok/s |
| | **+22% faster decode** | |

Prefill uses GEMM-style batching (load weights once, multiply all tokens). BitNet.cpp uses LUT-based matmul which is faster for prefill but slower for decode.

#### Llama 3.2 3B Instruct Q4_K_M (256 tokens, 22-token prompt)

| | Cougar | llama.cpp |
|---|---|---|
| Prefill | 19.5 tok/s | 34.5 tok/s |
| Decode | **7.8 tok/s** | 9.3 tok/s |

### ARM aarch64 (Raspberry Pi 5, 4 cores, NEON+dotprod)

#### BitNet b1.58 2B-4T

| Metric | Value |
|--------|-------|
| Decode | **16.1 tok/s** |
| Prefill | 16.3 tok/s |
| Latency | 62 ms/tok |
| Memory | ~2.0 GB |

> **Note:** Measured at 1.6 GHz (thermal throttling with stock cooler). Pi 5 max is 2.4 GHz — with active cooling, expect ~24 tok/s.

Speculative output projection (stride-4 sketch, top-512) reduces the 128K-vocab embedding scan from 328 MB to ~82 MB per token. Full ARM NEON kernel set — no x86 emulation, no fallbacks.

## Quick start

### 1. Build

```bash
# Build kernels (needs eacompute compiler)
EA=/path/to/ea make kernels

# Build binary (kernels embedded — no LD_LIBRARY_PATH needed)
cargo build --release

# Add to PATH (do this once)
ln -s $(pwd)/target/release/cougar ~/.local/bin/cougar
```

### 2. Download a model

```bash
mkdir -p ~/.cougar/models

# BitNet b1.58 2B-4T (fast, 1.7 GB)
curl -L "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf" \
  -o ~/.cougar/models/ggml-model-i2_s.gguf

# Llama 3.2 3B Instruct Q4_K_M (smarter, 1.9 GB)
curl -L "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  -o ~/.cougar/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### 3. Run

```bash
# Chat with Llama in the browser
cougar --model llama --serve

# Chat with BitNet in the terminal
cougar --model bitnet --interactive

# Single prompt
cougar --model llama --prompt "tell me a joke"

# No --model needed if a model exists in ~/.cougar/models/
cougar --prompt "Hello"
```

`--model llama` and `--model bitnet` are shorthands for the default paths. You can also pass any GGUF file path directly.

## CLI

```
cougar --model llama --serve                     # web chat UI
cougar --model bitnet --interactive              # terminal chat
cougar --model llama --prompt "tell me a joke"   # single prompt
cougar --model /path/to/model.gguf --serve       # custom model

Options:
  --model <llama|bitnet|path.gguf>   Model shorthand or file path
  --prompt <text>         Generate from a single prompt
  --interactive           Interactive REPL (stdin/stdout)
  --serve                 Web chat UI with SSE streaming
  --max-tokens N          Maximum tokens to generate (default: 128)
  --temperature T         Sampling temperature, 0 = greedy (default: 0)
  --repetition-penalty F  Penalize repeated tokens (default: 1.1)
  --max-seq-len N         Maximum sequence length (default: 2048)
  --port N                Server port (default: 8080)
```

## Architecture

```
cougar/
  kernels/     21 Ea SIMD kernels (11 x86 AVX2 + 10 ARM NEON)
  src/         23 Rust modules (82 tests)
  tests/       3 C kernel test harnesses (29 tests)
  build.rs     kernel embedding + ABI hash
```

### Inference pipeline

Two forward paths dispatch based on GGUF weight type:

**BitNet (I2_S):** RMSNorm -> i8 quantize -> ternary matmul (i2 x i8 via maddubs) -> squared ReLU -> i8 output projection. Fused gate+up dual kernel shares activation loads. GEMM-style batched prefill loads weights once per layer.

**Llama Q4_K_M (mixed Q4_K + Q6_K):** RMSNorm -> Q8_K quantize -> Q4_K/Q6_K matmul (nibble x i8 via maddubs, 6-bit scale unpacking) -> SiLU (fused inline) -> GEMM-style batched prefill with L1 weight reuse.

Both paths use a persistent condvar-based thread pool with QKV `run_split3` concurrent dispatch.

### Key optimizations

- **Persistent thread pool** -- condvar-based, zero per-dispatch allocation
- **QKV run_split3** -- Q/K/V projections concurrent in single dispatch
- **Fused gate+up+SiLU** -- vertical fusion eliminates intermediate buffers
- **Dual 4-row kernels** -- gate+up share activation loads at register level
- **GEMM-style prefill** -- weight rows loaded once, reused across all prompt tokens
- **Q6_K mixed dispatch** -- per-tensor Q4_K/Q6_K detection for Q4_K_M models
- **Tied embedding fallback** -- handles models without separate output.weight
- **Speculative output projection** (ARM) -- stride-4 sketch pre-filters 128K vocab to top-512 candidates, 8x bandwidth reduction

<details>
<summary>Kernel table (21 kernels)</summary>

| Kernel | Lines | What |
|--------|------:|------|
| `q4k_dot.ea` | 342 | Q4_K dot product: 1-row, 4-row, 4-row dual |
| `q4k_dot_arm.ea` | 329 | Q4_K dot product (ARM NEON) |
| `q6k_dot.ea` | 256 | Q6_K dot product: 1-row, 4-row |
| `q6k_dot_arm.ea` | 243 | Q6_K dot product (ARM NEON) |
| `bitnet_i2s.ea` | 242 | Ternary matmul: 1-row, 4-row, 4-row dual (x86) |
| `bitnet_i2s_arm.ea` | 215 | Ternary matmul: 1-row, 4-row, 4-row dual (ARM NEON) |
| `bitnet_fused_attn.ea` | 120 | Single-pass online softmax attention |
| `bitnet_fused_attn_arm.ea` | 103 | Single-pass online softmax attention (ARM NEON) |
| `bitnet_i8dot.ea` | 106 | i8 x u8 dot for quantized output |
| `bitnet_i8dot_arm.ea` | 102 | i8 x i8 dot for quantized output (ARM NEON) |
| `bitnet_quant.ea` | 105 | f32 -> i8 quantization + activation sum |
| `q4k_quant.ea` | 88 | f32 -> Q8_K quantization + bsums |
| `bitnet_quant_arm.ea` | 78 | f32 -> i8 quantization (ARM NEON) |
| `q4k_quant_arm.ea` | 84 | f32 -> Q8_K quantization (ARM NEON) |
| `bitnet_rmsnorm.ea` | 54 | RMS normalization |
| `bitnet_rmsnorm_arm.ea` | 52 | RMS normalization (ARM NEON) |
| `bitnet_activate.ea` | 32 | Squared ReLU x up (fused) |
| `bitnet_activate_arm.ea` | 31 | Squared ReLU x up (ARM NEON) |
| `rope.ea` | 70 | Rotary position embedding (shuffle+fma) |
| `bitnet_vecadd.ea` | 17 | Residual vector add |
| `bitnet_vecadd_arm.ea` | 17 | Residual vector add (ARM NEON) |

</details>

## Tests

111 tests total, zero warnings:

| Suite | Tests |
|---|---|
| Rust (`cargo test`) | 82 |
| C kernel (Q6K dot) | 15 |
| C kernel (Q4K dot) | 7 |
| C kernel (Q8K quant) | 7 |

## Building

Requires:
- [eacompute](https://github.com/petlukk/eacompute) compiler (`ea` binary)
- Rust 1.63+
- x86-64 with AVX2 + FMA (or ARM with NEON for BitNet)

```bash
# Build eacompute first
cd ~/projects/eacompute && cargo build --release --features=llvm

# Build cougar
cd ~/projects/cougar
EA=/path/to/ea make kernels    # compile .ea -> .so
cargo build --release           # kernels embedded in binary
cargo test                      # 79 Rust tests
```

### Cross-compile for ARM (Raspberry Pi 5)

```bash
# Install cross toolchain (Ubuntu/Debian)
sudo apt install gcc-aarch64-linux-gnu
rustup target add aarch64-unknown-linux-gnu

# Build ARM kernels
EA=/path/to/ea
for f in kernels/*_arm.ea; do
  stem=$(basename "$f" _arm.ea)
  CC=aarch64-linux-gnu-gcc $EA "$f" --lib \
    --target-triple=aarch64-unknown-linux-gnu --dotprod \
    -o build/lib-arm/lib${stem}.so
done
# Generic kernels (activate, rmsnorm, vecadd, quant, fused_attn)
for f in kernels/bitnet_{activate,rmsnorm,vecadd,quant,fused_attn}_arm.ea \
         kernels/q4k_quant_arm.ea; do
  stem=$(basename "$f" _arm.ea)
  CC=aarch64-linux-gnu-gcc $EA "$f" --lib \
    --target-triple=aarch64-unknown-linux-gnu \
    -o build/lib-arm/lib${stem}.so
done

# Build binary
cargo build --release --target aarch64-unknown-linux-gnu

# Copy to Pi and run
scp target/aarch64-unknown-linux-gnu/release/cougar pi@raspberrypi:/tmp/
ssh pi@raspberrypi '/tmp/cougar --prompt "Hello world" --max-tokens 20'
```

Ea kernels are compiled to `.so`, embedded via `include_bytes!` at build time, extracted to `~/.cougar/lib/` on first run, and loaded via `dlopen`. No `LD_LIBRARY_PATH` needed.
