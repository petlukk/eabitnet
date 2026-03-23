# Future Optimizations

## Baseline comparison (2026-03-22)

BitNet b1.58 2B-4T, 16 threads, x86-64 (AVX2 + FMA), greedy decoding:

| Metric | bitnet.cpp (official) | Cougar (before pool) | Cougar (after pool) |
|--------|----------------------|---------------------|---------------------|
| Decode throughput | 15.05 tok/s | 10.0 tok/s | **17.4 tok/s** |
| Decode latency | 66.5 ms/tok | 100 ms/tok | **57.6 ms/tok** |
| Prefill throughput | 16.48 tok/s | 10.0 tok/s | **17.1 tok/s** |
| Load time | 1454 ms | — | — |
| Kernel type | I2_S + llamafile | Ea SIMD | Ea SIMD |

Cougar now **15% faster** than bitnet.cpp on decode throughput.

---

## ~~1. Thread Pool vs `std::thread::scope`~~ DONE (2026-03-23)

Replaced all `std::thread::scope` with a persistent condvar-based thread pool. Eliminated ~2000+ thread create/teardown cycles per token. Also flattened nested spawning in QKV and gate+up matmuls.

**Result:** 10.0 → 17.4 tok/s (+74%). QKV matmul: 20.5ms → 6.7ms. Total: 98ms → 57.6ms/tok.

## 2. LUT (Look-Up Table) Kernels

Microsoft uses thousands of lines of generated code for LUT-based ternary matmul.

**Why it's faster:** Instead of computing the result of a ternary multiplication, they look it up in a table that sits in L1 cache.

**Our opportunity:** We already have `bitnet_lut.ea` (235 lines). If we optimize it and wire it into the main inference loop, we can dramatically cut computation time in the ternary matmuls.

## 3. Micro-batching & Prefetching

Microsoft's I2_S implementation is extremely aggressive with software prefetching.

**Strategy:** In our Ea loops, start reading the next block of weights before finishing the current one. On a modern CPU this can hide almost the entire memory access latency.
