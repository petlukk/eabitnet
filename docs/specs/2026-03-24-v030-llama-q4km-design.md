# Cougar v0.3.0 — Llama 3.2 3B Q4_K_M Support

## Goal

Run Llama 3.2 3B Instruct (Q4_K_M quantization) through Cougar with native Eä Q4 kernels. Benchmark against llama.cpp on the same hardware.

## Non-Goals

- No other quant types (Q8_0, Q5_K, F16) — just Q4_K_M
- No other architectures beyond Llama family (Mistral/Qwen share the arch, so they'd work too)
- No batched prefill
- No KV cache quantization
- No re-packing weights at load time (start with direct GGUF layout)

## Q4_K_M Block Structure

Source of truth: `ggml-common.h` line 295.

```c
// Super-block: 256 elements, 144 bytes, ~4.5 bits/weight
typedef struct {
    ggml_half d;             // 2 bytes — super-block scale
    ggml_half dmin;          // 2 bytes — super-block min scale
    uint8_t scales[12];      // 12 bytes — 8 sub-block scales + 8 sub-block mins, 6-bit packed
    uint8_t qs[128];         // 128 bytes — 256 × 4-bit weights (2 per byte)
} block_q4_K;
```

**Sub-block structure:** 8 sub-blocks of 32 elements each. Each sub-block has its own 6-bit scale and 6-bit min, packed into the 12-byte `scales` array.

**Weight formula:** `x[i] = d * scale_j * nibble[i] - dmin * min_j` where `j` is the sub-block index.

**6-bit scale packing (12 bytes → 8 scales + 8 mins):**
```
bytes 0-3:   scales[0..3] low 6 bits, scales[4..7] bits 4-5 in bits 6-7
bytes 4-7:   mins[0..3] low 6 bits, mins[4..7] bits 4-5 in bits 6-7
bytes 8-11:  scales[4..7] low 4 bits | mins[4..7] low 4 bits
```

Unpacking uses the `kmask1/kmask2/kmask3` pattern from llama.cpp:
```c
kmask1 = 0x3f3f3f3f  // low 6 bits of each byte
kmask2 = 0x0f0f0f0f  // low 4 bits of each byte
kmask3 = 0x03030303  // low 2 bits of each byte
```

## Q8_K Activation Quantization

llama.cpp quantizes f32 activations to Q8_K before the dot product. The inner loop is Q4×Q8 via `maddubs`, not Q4×f32.

```c
// Intermediate quantization block: 256 elements, 292 bytes
typedef struct {
    float   d;              // 4 bytes — delta (absmax scale)
    int8_t  qs[256];        // 256 bytes — quantized values
    int16_t bsums[16];      // 32 bytes — sum of quants in groups of 16
} block_q8_K;
```

**Why Q8_K:** `maddubs_epi16(u4, i8)` gives full SIMD throughput. Q4×f32 would require dequantizing to f32 first, losing the integer pipeline advantage.

**bsums optimization:** Pre-computed sums per 16-element group enable fast min correction without iterating over all elements. The formula is: `min_contribution = -dmin * sum(mins_j * bsums_j)`.

## New Eä Kernels

### `kernels/q4k_dot.ea` (~200 LOC)

Native Q4_K × Q8_K dot product. Two exported functions:

**`q4k_dot_q8k(weights: *u8, activations: *i8, act_bsums: *i16, n_blocks: i32, d_scale: f32, dmin_scale: f32, scales_unpacked: *u8) -> f32`**

Inner loop per super-block (4 iterations of 64 elements):
1. Load 32 bytes of packed Q4 nibbles
2. Extract low nibbles: `q4bits & 0x0F`
3. Extract high nibbles: `q4bits >> 4`  (these correspond to the next 32-element sub-block)
4. Load 32 + 32 bytes of Q8_K activations
5. `maddubs_i16(q4_lo, q8_lo)` → multiply by sub-block scale → accumulate
6. `maddubs_i16(q4_hi, q8_hi)` → multiply by sub-block scale → accumulate
7. Min correction via bsums (accumulated separately)

**`q4k_dot_q8k_4row(...) `**

4-row variant with shared activation loads, same ILP pattern as `i2_dot_i8_4row`.

### `kernels/bitnet_silu.ea` (~30 LOC)

Fused SwiGLU activation: `out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]`

Uses Eä's `exp` intrinsic on f32x8 (or f32x4 on SSE). The division is: multiply gate by sigmoid, then multiply by up.

### `kernels/q4k_quant.ea` (~80 LOC)

Quantize f32 activations to Q8_K format:
1. Compute absmax per 256-element block → scale `d = max/127`
2. Quantize: `qs[i] = round(x[i] / d)` clamped to [-127, 127]
3. Compute `bsums[j]` = sum of `qs` in each 16-element group

## Forward Pass: `src/forward_llama.rs` (~300 LOC)

Separate forward function for Llama architecture. No sub-norms, SiLU activation, Q4 matmul.

```
Per layer:
  RMSNorm(x, attn_norm_weight)
  Quantize x → Q8_K (scale + bsums)
  Q4_K matmul: Q → wq × x_q8k
  Q4_K matmul: K → wk × x_q8k
  Q4_K matmul: V → wv × x_q8k
  RoPE(Q, K, pos, theta)
  Cache K, V
  Fused attention(Q, K_cache, V_cache) → attn_out
  Quantize attn_out → Q8_K
  Q4_K matmul: O → wo × attn_out_q8k
  Residual: x = x + O

  RMSNorm(x, ffn_norm_weight)
  Quantize x → Q8_K
  Q4_K matmul: gate → w_gate × x_q8k
  Q4_K matmul: up   → w_up × x_q8k
  SwiGLU: hidden = silu(gate) * up
  Quantize hidden → Q8_K
  Q4_K matmul: down → w_down × hidden_q8k
  Residual: x = x + down

Output:
  RMSNorm(x, output_norm_weight)
  Quantize x → Q8_K
  Q4_K matmul: logits → output_weight × x_q8k
  Sample
```

**Key difference from BitNet:** Activations quantized to Q8_K (with bsums) before each matmul, not the simpler absmax i8 quantization. The Q8_K bsums enable the fast min correction in the Q4 dot product.

**Threading:** Same pattern as BitNet — `pool.run()` for all matmuls, `run_split3()` for QKV if ≥3 threads, sequential fallback otherwise.

**f32 throughout:** All activation buffers (x, q, k, v, attn_out, gate, up, hidden) stay f32 between operations. Q8_K quantization happens immediately before each matmul and results are written back as f32.

## Model Loading: `src/model.rs` changes

### ModelConfig

Read from GGUF metadata:

```rust
pub enum QuantType { I2S, Q4K }
pub enum Activation { SquaredReLU, SiLU }

pub struct ModelConfig {
    pub quant_type: QuantType,
    pub activation: Activation,
    pub has_sub_norms: bool,
    // existing fields: n_layers, hidden_dim, n_heads, n_kv_heads, etc.
}
```

Determined by:
- `quant_type`: Read from first weight tensor's type field in GGUF
- `activation`: `"silu"` for Llama, `"squared_relu"` for BitNet (read from `{arch}.activation_function`, default to `"silu"`)
- `has_sub_norms`: Check if `blk.0.attn_sub_norm.weight` tensor exists

### Q4_K_M Weight Loading

Q4_K_M weights are stored in GGUF as raw `block_q4_K` arrays. The loader reads them as-is (byte pointer + tensor offset). No repacking.

The 6-bit scale unpacking happens in the Eä kernel (or in a Rust helper called once per super-block before the kernel call, to keep the kernel focused on the dot product).

**Decision:** Unpack scales in Rust before calling the kernel. The kernel receives pre-unpacked `u8` scales and mins arrays. This keeps the kernel clean and the unpacking is O(n_blocks) not O(n_elements) — negligible cost.

### Embedding

Llama 3.2 3B Q4_K_M has the output weight (`output.weight`) as a separate tensor, not tied to token embeddings. Token embeddings (`token_embd.weight`) are typically Q4_K_M too.

For embedding lookup: dequantize the single row on the fly (256 elements at a time, 30 super-blocks for hidden_dim=3072). This is once per token, not in the hot loop.

## Dispatch: `src/main.rs`

```rust
match model.config.quant_type {
    QuantType::I2S => {
        // existing BitNet path
        forward::generate(model, tokens, ..., on_token)
    }
    QuantType::Q4K => {
        forward_llama::generate(model, tokens, ..., on_token)
    }
}
```

Comment in dispatch: `// Tested architectures: "llama" (Llama 3.2, Mistral, Qwen). Other Llama-family models may work but are untested.`

## File Plan

| File | Action | Est. LOC | Under 500? |
|---|---|---|---|
| `kernels/q4k_dot.ea` | Create | ~200 | Yes |
| `kernels/bitnet_silu.ea` | Create | ~30 | Yes |
| `kernels/q4k_quant.ea` | Create | ~80 | Yes |
| `src/forward_llama.rs` | Create | ~300 | Yes |
| `src/model.rs` | Modify | +60 (Q4K loading, ModelConfig) | ~310 total |
| `src/matmul_q4k.rs` | Create | ~200 (Q4K matmul dispatch) | Yes |
| `src/ffi.rs` | Modify | +20 (new kernel wrappers) | ~90 total |
| `src/embed.rs` | Modify | +10 (new .so files) | ~145 total |
| `build.rs` | Modify | +3 (new kernel names) | ~57 total |
| `src/main.rs` | Modify | +15 (dispatch logic) | ~165 total |
| `src/forward.rs` | Untouched | — | — |

All under 500. No existing files broken.

## Testing

### Kernel Tests
- `tests/test_q4k_dot.c`: Reference scalar Q4_K×Q8_K dot product, verify against Eä kernel
- `tests/test_silu.c`: Reference SiLU, verify against kernel
- `tests/test_q4k_quant.c`: Reference Q8_K quantization, verify against kernel

### Rust Tests
- Q4_K_M weight loading from a minimal GGUF fixture
- Q4K matmul dispatch correctness (small known case)
- ModelConfig detection from GGUF metadata
- `forward_llama::generate()` callback contract (same as BitNet test)

### End-to-End
- Download Llama 3.2 3B Instruct Q4_K_M
- Run through Cougar, verify coherent output
- Benchmark: `cougar --model llama-3.2-3b-q4km.gguf --prompt "..." --max-tokens 64`

## Benchmark Plan

1. Download: `huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf`
2. llama.cpp baseline: `llama-bench -m model.gguf -t 2 -n 64 -p 0`
3. Cougar: `cougar --model model.gguf --prompt "The capital of France is" --max-tokens 64`
4. Compare: tok/s decode, ms/tok, memory usage

**Benchmark fairness notes:**
- Same machine, same model file, same thread count
- llama.cpp uses Q4_K_M with its AVX2 kernel — we use Q4_K_M with our Eä kernel
- Both do Q4×Q8 dot products (same algorithmic approach)
- Report numbers honestly; if llama.cpp is faster, say so

## Risks

1. **6-bit scale unpacking complexity** — The packed scales format is non-trivial. Pre-unpacking in Rust mitigates this but adds a per-block overhead. If it matters, move unpacking into the kernel later.
2. **exp() in SiLU kernel** — Need to verify Eä's `exp` intrinsic exists and is accurate enough for inference. If not, use a polynomial approximation.
3. **Memory** — Llama 3.2 3B Q4_K_M is ~2 GB. With 7.8 GB RAM and KV cache, it should fit but may be tight.
4. **Tested architecture string** — Llama 3.2 identifies as `"llama"` in GGUF. Other models may use different strings. We only test `"llama"`.
