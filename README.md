# eabitnet

Standalone BitNet b1.58 2B-4T inference via [Ea](https://github.com/petlukk/eacompute) SIMD kernels. No llama.cpp. No dependencies.

**10 tok/s** on 16 threads. **~3200 lines** total. **118 tests**.

## Quick start

```bash
# Build kernels (needs eacompute compiler)
make kernels

# Build runner
RUSTFLAGS="-L build/lib" cargo build --release

# Run
LD_LIBRARY_PATH=build/lib ./target/release/eabitnet \
  --model path/to/ggml-model-i2_s.gguf \
  --prompt "The capital of France is"
```

## What it does

Loads a BitNet b1.58 2B-4T GGUF file and generates text. The entire inference pipeline runs on Ea SIMD kernels:

- **Ternary matmul** (`i2_dot_i8`) -- 2-bit packed weights x i8 activations via `maddubs_i32`
- **Fused attention** (`fused_attention_f32`) -- single-pass online softmax, no scores buffer
- **RMSNorm, softmax, RoPE, squared ReLU** -- all Ea kernels
- **i8 output projection** (`i8dot_4row`) -- quantized embedding for 4x bandwidth reduction
- **Multi-threaded** -- `std::thread::scope` partitions work across all cores

## Performance

BitNet b1.58 2B-4T (30 layers, 2560 hidden, 20 heads, 128K vocab) on x86-64:

| Metric | Value |
|--------|-------|
| Decode throughput | 10.0 tok/s |
| Decode latency | ~100 ms/tok |
| Prefill throughput | 10.0 tok/s |
| Threads | 16 |
| Binary size | 644 KB |
| Model RSS | ~2.0 GB |

### Optimization history

```
 1.5 tok/s  single-threaded baseline
 5.0 tok/s  +multi-threaded matmul
 5.6 tok/s  +F32 output pre-conversion
 6.5 tok/s  +parallel K/V, gate/up pairs
 7.4 tok/s  +concurrent Q+K+V
10.0 tok/s  +i8 quantized output projection
```

### Per-stage profile (100ms/tok)

```
FFN gate+up:   28ms (27%)   2x ternary matmul, parallel pair
QKV:           22ms (21%)   concurrent Q+K+V
FFN down:      21ms (20%)   1x ternary matmul
O proj:        19ms (18%)   1x ternary matmul
output (i8):   13ms (13%)   i8 quantized, was 49ms with f32
attention:      0.2ms        fused online softmax
```

## Kernels

13 Ea kernels, 1248 lines total:

| Kernel | Lines | What |
|--------|------:|------|
| `bitnet_i2s.ea` | 145 | Ternary matmul (x86 AVX2) |
| `bitnet_i2s_arm.ea` | 145 | Ternary matmul (ARM NEON) |
| `bitnet_fused_attn.ea` | 120 | Single-pass online softmax attention |
| `bitnet_output.ea` | 121 | Tiled 4-row f32 dot (FMA) |
| `bitnet_quant.ea` | 105 | f32->i8 quantization + activation sum |
| `bitnet_softmax.ea` | 85 | Numerically stable softmax |
| `bitnet_attention.ea` | 86 | 3-pass attention (legacy) |
| `bitnet_i8dot.ea` | 62 | i8xu8 dot for quantized output |
| `bitnet_rmsnorm.ea` | 54 | RMS normalization |
| `bitnet_rope.ea` | 41 | Rotary position encoding |
| `bitnet_activate.ea` | 32 | Squared ReLU x up (fused) |
| `bitnet_vecadd.ea` | 17 | Residual vector add |
| `bitnet_lut.ea` | 235 | LUT matmul (unused) |

## Architecture

```
eabitnet/
  kernels/    13 Ea kernels (.ea -> .so via eacompute)
  src/        7 Rust modules (gguf, tokenizer, model, forward, matmul, ffi, main)
  tests/      13 C test harnesses (118 tests)
```

The model runs BitNet b1.58 2B-4T with:
- 30 layers, 2560 hidden, 20 attention heads (GQA 20:5), 128 head dim
- Squared ReLU activation, RMSNorm with sub-norms
- RoPE (theta=500000), tied F16 embeddings
- I2_S ternary weights (2-bit packed, per-tensor absmean scale)

## Model

Download from HuggingFace:

```bash
mkdir -p ~/.eabitnet/models
curl -L "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf" \
  -o ~/.eabitnet/models/ggml-model-i2_s.gguf
```

## Model quality

BitNet b1.58 2B-4T is a **2 billion parameter** model with **1-bit weights**. It's small and fast, not smart. Set your expectations accordingly:

| Works well | Don't expect |
|---|---|
| Factual recall ("capital of France") | Math ("what is 6+6") |
| Code completion (`def fibonacci(n):`) | Complex reasoning |
| Text continuation | Long coherent essays |
| Pattern matching | Following instructions |

Greedy decoding (temperature=0) tends to repeat after 20-30 tokens. This is normal for small models without top-p/top-k sampling.

The point of eabitnet is proving the **inference engine**, not the model. If you want better answers, wait for larger BitNet models and drop them in — the runner handles any BitNet GGUF.

## CLI

```
eabitnet --model <path.gguf> --prompt <text> [options]

Options:
  --max-tokens N      Maximum tokens to generate (default: 128)
  --temperature T     Sampling temperature, 0 = greedy (default: 0)
  --max-seq-len N     Maximum sequence length (default: 2048)
```

## Building

Requires:
- [eacompute](https://github.com/petlukk/eacompute) compiler (`ea` binary)
- Rust 1.63+ (for `std::thread::scope`)
- GCC (for kernel test harnesses)
- x86-64 with AVX2 + FMA

```bash
# Build eacompute first
cd ~/projects/eacompute && cargo build --release --features=llvm

# Build eabitnet
cd ~/projects/eabitnet
make kernels                              # compile .ea -> .so
RUSTFLAGS="-L build/lib" cargo build --release
make test                                 # 102 kernel tests
RUSTFLAGS="-L build/lib" cargo test       # 16 Rust tests
```

### Linking note

The Ea kernels compile to shared libraries (`.so` files) in `build/lib/`. Both the Rust compiler and the runtime need to find them:

```bash
# At compile time — tell rustc where to find the .so files
RUSTFLAGS="-L build/lib" cargo build --release

# At runtime — tell the dynamic linker where to find them
LD_LIBRARY_PATH=build/lib ./target/release/eabitnet --model ...
```

To avoid typing `LD_LIBRARY_PATH` every time, add the full path to your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
export LD_LIBRARY_PATH="$HOME/projects/eabitnet/build/lib:$LD_LIBRARY_PATH"
```

Or install the libraries system-wide:

```bash
sudo cp build/lib/*.so /usr/local/lib/
sudo ldconfig
```
