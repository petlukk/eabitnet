```
        /\_/\
       ( o.o )  ╔═══════════════════════════════════╗
        >╥╥<    ║  C O U G A R                      ║
       /|   |\  ║  BitNet b1.58 inference engine     ║
      (_|   |_) ╚═══════════════════════════════════╝
```

# Cougar

Llama.cpp is a beast, but even beasts have predators. 🐾

A ~4,300-line BitNet b1.58 engine written in Rust + [Eä](https://github.com/petlukk/eacompute) SIMD kernels. No llama.cpp. No dependencies. Single binary with embedded kernels, interactive REPL, and web chat UI.

🚀 **18.8 tok/s** (AVX2 / 16 threads) · 📦 **869 KB** binary (kernels embedded) · 🚫 Zero dependencies

It doesn't just run models. It hunts them.

## Quick start

```bash
# Build kernels (needs eacompute compiler)
make kernels

# Build (kernels embedded in binary — no LD_LIBRARY_PATH needed)
cargo build --release

# Single prompt
./target/release/cougar --model path/to/ggml-model-i2_s.gguf \
  --prompt "The capital of France is"

# Interactive chat
./target/release/cougar --model path/to/ggml-model-i2_s.gguf --interactive

# Web chat UI
./target/release/cougar --model path/to/ggml-model-i2_s.gguf --serve
# Open http://localhost:8080
```

## What it does

Loads a BitNet b1.58 2B-4T GGUF file and generates text. The entire inference pipeline runs on Ea SIMD kernels:

- **Ternary matmul** (`i2_dot_i8`) -- 2-bit packed weights x i8 activations via `maddubs_i32`
- **Fused attention** (`fused_attention_f32`) -- single-pass online softmax, no scores buffer
- **RMSNorm, softmax, RoPE, squared ReLU** -- all Ea kernels
- **i8 output projection** (`i8dot_4row`) -- quantized embedding for 4x bandwidth reduction
- **Persistent thread pool** -- condvar-based pool, zero per-dispatch overhead

## Performance

BitNet b1.58 2B-4T (30 layers, 2560 hidden, 20 heads, 128K vocab) on x86-64:

| Metric | Value |
|--------|-------|
| Decode throughput | **18.8 tok/s** |
| Decode latency | **53.3 ms/tok** |
| Prefill throughput | 17.6 tok/s |
| Threads | 16 |
| Binary size | 869 KB (kernels embedded) |
| Model RSS | ~2.0 GB |

### Optimization history

```
 1.5 tok/s  single-threaded baseline
 5.0 tok/s  +multi-threaded matmul
 5.6 tok/s  +F32 output pre-conversion
 6.5 tok/s  +parallel K/V, gate/up pairs
 7.4 tok/s  +concurrent Q+K+V
10.0 tok/s  +i8 quantized output projection
18.8 tok/s  +persistent thread pool (beats bitnet.cpp 15.05)
```

### Per-stage profile (53.3ms/tok)

```
FFN gate+up:  18.4ms (35%)   2x ternary matmul, parallel pair
output (i8):  12.1ms (23%)   i8 quantized embedding projection
FFN down:      9.1ms (17%)   1x ternary matmul
QKV:           7.0ms (13%)   concurrent Q+K+V via run_split3
O proj:        5.1ms (10%)   1x ternary matmul
attention:     0.2ms          fused online softmax
```

## Kernels

8 Ea kernels, 680 lines total:

| Kernel | Lines | What |
|--------|------:|------|
| `bitnet_i2s.ea` | 145 | Ternary matmul (x86 AVX2) |
| `bitnet_i2s_arm.ea` | 145 | Ternary matmul (ARM NEON) |
| `bitnet_fused_attn.ea` | 120 | Single-pass online softmax attention |
| `bitnet_quant.ea` | 105 | f32->i8 quantization + activation sum |
| `bitnet_i8dot.ea` | 62 | i8xu8 dot for quantized output |
| `bitnet_rmsnorm.ea` | 54 | RMS normalization |
| `bitnet_activate.ea` | 32 | Squared ReLU x up (fused) |
| `bitnet_vecadd.ea` | 17 | Residual vector add |

## Architecture

```
cougar/
  kernels/    8 Ea kernels (.ea -> .so, embedded in binary)
  src/        11 Rust modules + test files (70 tests)
  tests/      13 C test harnesses (102 tests)
  build.rs    kernel embedding + ABI hash
```

The model runs BitNet b1.58 2B-4T with:
- 30 layers, 2560 hidden, 20 attention heads (GQA 20:5), 128 head dim
- Squared ReLU activation, RMSNorm with sub-norms
- RoPE (theta=500000), tied F16 embeddings
- I2_S ternary weights (2-bit packed, per-tensor absmean scale)

## Model

Download from HuggingFace:

```bash
mkdir -p ~/.cougar/models
curl -L "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf" \
  -o ~/.cougar/models/ggml-model-i2_s.gguf
```

## Model quality

BitNet b1.58 2B-4T is a **2 billion parameter** model with **1-bit weights**. It's small and fast, not smart. Set your expectations accordingly:

| Works well | Don't expect |
|---|---|
| Factual recall ("capital of France") | Math ("what is 6+6") |
| Code completion (`def fibonacci(n):`) | Complex reasoning |
| Text continuation | Long coherent essays |
| Pattern matching | Following instructions |

Repetition penalty (default 1.1) prevents loops. Temperature sampling with proper softmax is supported for more varied output.

The point of cougar is proving the **inference engine**, not the model. If you want better answers, wait for larger BitNet models and drop them in — the runner handles any BitNet GGUF.

## CLI

```
cougar --model <path.gguf> --prompt <text> [options]
cougar --model <path.gguf> --interactive [options]
cougar --model <path.gguf> --serve [--port 8080] [options]

Options:
  --max-tokens N          Maximum tokens to generate (default: 128)
  --temperature T         Sampling temperature, 0 = greedy (default: 0)
  --repetition-penalty F  Penalize repeated tokens (default: 1.1)
  --max-seq-len N         Maximum sequence length (default: 2048)
  --interactive           Interactive REPL mode
  --serve                 Web chat UI (SSE streaming)
  --port N                Server port (default: 8080)
```

## Building

Requires:
- [eacompute](https://github.com/petlukk/eacompute) compiler (`ea` binary)
- Rust 1.63+
- GCC (for kernel test harnesses)
- x86-64 with AVX2 + FMA

```bash
# Build eacompute first
cd ~/projects/eacompute && cargo build --release --features=llvm

# Build cougar
cd ~/projects/cougar
make kernels                    # compile .ea -> .so
cargo build --release           # kernels embedded in binary
cargo test                      # 70 Rust tests
```

### Kernel embedding

Ea kernels are compiled to `.so` files by `make kernels`, then embedded in the binary via `include_bytes!` at build time. On first run, they're extracted to `~/.cougar/lib/v{VERSION}-{HASH}/` and loaded via `dlopen`. No `LD_LIBRARY_PATH` needed.
