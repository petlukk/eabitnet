# Cougar v0.3.0 — Llama 3.2 3B Q4_K_M Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Llama 3.2 3B Q4_K_M through Cougar with native Eä Q4 kernels and benchmark against llama.cpp.

**Architecture:** Separate forward pass for Llama (`forward_llama.rs`) alongside existing BitNet path (`forward.rs`). New Eä kernels for Q4_K×Q8_K dot product and SiLU activation. Model loader detects quant type from GGUF tensors and dispatches accordingly. Q4_K 6-bit scales unpacked in Rust; kernel receives clean scale/min arrays.

**Tech Stack:** Rust (std only), Eä SIMD kernels (x86 AVX2), GGUF format.

**Spec:** `docs/specs/2026-03-24-v030-llama-q4km-design.md`

**Hard rules:** No file > 500 LOC. Every feature tested. No stubs. No premature features. Delete don't comment.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/gguf.rs` | Fix | Correct Q4_K byte size calculation (160→144) |
| `src/model.rs` | Modify | Add `ModelConfig`, `QuantType`, Q4_K_M weight loading, architecture dispatch |
| `kernels/q4k_dot.ea` | Create | Q4_K × Q8_K dot product (single-row + 4-row) |
| `kernels/q4k_quant.ea` | Create | f32 → Q8_K quantization (scale + bsums) |
| `kernels/bitnet_silu.ea` | Create | Fused SwiGLU: silu(gate) * up |
| `tests/test_q4k_dot.c` | Create | C reference + validation for Q4K kernel |
| `tests/test_q4k_quant.c` | Create | C reference + validation for Q8K quant kernel |
| `tests/test_silu.c` | Create | C reference + validation for SiLU kernel |
| `src/ffi.rs` | Modify | Add FFI wrappers for new kernels |
| `src/embed.rs` | Modify | Embed new .so files |
| `build.rs` | Modify | Add new kernel names to embed list |
| `src/matmul_q4k.rs` | Create | Q4_K matmul dispatch (thread pool, scale unpacking) |
| `src/forward_llama.rs` | Create | Llama forward pass (SiLU, no sub-norms, Q4K matmul) |
| `src/main.rs` | Modify | Detect quant type, dispatch to BitNet or Llama forward |

---

### Task 1: Fix Q4_K byte size in GGUF parser

The GGUF parser calculates Q4_K tensor size as 160 bytes/block but the actual `block_q4_K` is 144 bytes. This would cause incorrect tensor data offsets for Q4_K_M models.

**Files:**
- Modify: `src/gguf.rs:178` and `src/gguf.rs:205-217`

- [ ] **Step 1: Verify the bug**

Read `src/gguf.rs` line 178. Current: `12 => Ok((5, 256))` which computes `5*256/8 = 160 bytes/block`. Actual Q4_K block is 144 bytes (`2+2+12+128`).

The `tensor_byte_size` function uses a generic `n_blocks * bits * block / 8` formula that doesn't work for K-quant types which have complex block layouts. Need a special case like I2_S already has.

- [ ] **Step 2: Fix tensor_byte_size for Q4_K**

In `src/gguf.rs`, add a special case in `tensor_byte_size()` for Q4_K (dtype 12):

```rust
if dtype == 12 {
    // Q4_K: block_q4_K = 2(d) + 2(dmin) + 12(scales) + 128(qs) = 144 bytes per 256 elements
    let n_blocks = (n_elements as usize + 255) / 256;
    return Ok(n_blocks * 144);
}
```

- [ ] **Step 3: Run existing tests**

Run: `cargo test gguf`
Expected: All GGUF tests pass (the fix doesn't affect I2_S which has its own special case).

- [ ] **Step 4: Commit**

```bash
git add src/gguf.rs
git commit -m "fix: correct Q4_K byte size in GGUF parser (160→144 bytes/block)"
```

---

### Task 2: ModelConfig and architecture detection

Add config types and detection logic so `main.rs` can dispatch to the right forward pass.

**Files:**
- Modify: `src/model.rs`
- Modify: `src/main.rs`

- [ ] **Step 1: Add config types to model.rs**

At the top of `src/model.rs`, after the imports:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantType { I2S, Q4K }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation { SquaredReLU, SiLU }
```

Add fields to `BitNetModel` (rename later, or keep as-is since it's the same struct shape):

```rust
pub quant_type: QuantType,
pub activation: Activation,
pub has_sub_norms: bool,
```

- [ ] **Step 2: Detect config in from_gguf()**

In `BitNetModel::from_gguf()`, after reading architecture metadata:

```rust
// Detect quant type from first weight tensor
let first_weight_name = format!("blk.0.attn_q.weight");
let first_weight_idx = *gguf.tensor_map.get(&first_weight_name)
    .or_else(|| gguf.tensor_map.get("blk.0.attn_qkv.weight"))
    .ok_or("cannot find first weight tensor to detect quant type")?;
let quant_type = match gguf.tensors[first_weight_idx].dtype {
    36 => QuantType::I2S,
    12 => QuantType::Q4K,
    dt => return Err(format!("unsupported weight quant type: {dt}")),
};

// Detect activation
let activation = match gguf.get_str(&key("activation_function")) {
    Some("squared_relu") => Activation::SquaredReLU,
    _ => Activation::SiLU, // default for Llama family
};

// Detect sub-norms
let has_sub_norms = gguf.tensor_map.contains_key("blk.0.attn_sub_norm.weight");
```

- [ ] **Step 3: Branch weight loading based on quant_type**

In `from_gguf()`, the weight loading section needs to branch:
- `QuantType::I2S` → existing `load_i2s_tensor` path
- `QuantType::Q4K` → store raw byte pointer (no repacking needed, Q4_K blocks read directly)

For Q4K, each layer weight is just a raw pointer into the GGUF mmap. No repacking, no scale extraction (scales are inside the block).

Also handle embedding: Q4K models may have Q4K or F16 embeddings. Check tensor dtype and branch.

- [ ] **Step 4: Update main.rs dispatch**

```rust
match model.quant_type {
    QuantType::I2S => {
        // existing BitNet path (forward.rs)
        let (output, _, _) = InferenceState::generate(
            &model, &tokens, max_tokens, temperature, repetition_penalty,
            tokenizer.eos_id, max_seq_len, |_| {},
        );
        // ...
    }
    QuantType::Q4K => {
        // Llama path (forward_llama.rs) — implemented in Task 7
        eprintln!("error: Q4_K_M support not yet implemented");
        std::process::exit(1);
    }
}
```

Same dispatch for `--interactive` and `--serve` modes.

- [ ] **Step 5: Run all tests**

Run: `cargo test`
Expected: 70 passed. Existing BitNet path unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/model.rs src/main.rs
git commit -m "feat: add ModelConfig with quant type and activation detection from GGUF"
```

---

### Task 3: Q8_K quantization kernel

Quantize f32 activations to Q8_K format (scale + i8 values + bsums per 16-element group). Required before Q4K matmul.

**Files:**
- Create: `kernels/q4k_quant.ea`
- Create: `tests/test_q4k_quant.c`

- [ ] **Step 1: Write the C reference test**

Create `tests/test_q4k_quant.c` with a scalar reference implementation:

```c
// Reference: quantize f32[256] → Q8_K block
// d = max(abs(x)) / 127
// qs[i] = round(x[i] / d)
// bsums[j] = sum(qs[j*16 .. (j+1)*16])
void ref_quant_q8k(const float *x, int8_t *qs, float *d_out, int16_t *bsums, int n) {
    // n must be multiple of 256
    for (int blk = 0; blk < n/256; blk++) {
        const float *xb = x + blk * 256;
        int8_t *qb = qs + blk * 256;
        float amax = 0;
        for (int i = 0; i < 256; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        float scale = amax > 1e-10f ? amax / 127.0f : 0.0f;
        float inv = scale > 0 ? 1.0f / scale : 0.0f;
        d_out[blk] = scale;
        for (int i = 0; i < 256; i++) {
            int v = (int)roundf(xb[i] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qb[i] = (int8_t)v;
        }
        for (int j = 0; j < 16; j++) {
            int16_t sum = 0;
            for (int k = 0; k < 16; k++) sum += qb[j*16 + k];
            bsums[blk * 16 + j] = sum;
        }
    }
}
```

Test: quantize a known f32 array, compare kernel output to reference.

- [ ] **Step 2: Write the Eä kernel**

Create `kernels/q4k_quant.ea`:

```
export func quant_f32_q8k(
    src: *restrict f32,
    dst_qs: *mut i8,
    dst_d: *mut f32,
    dst_bsums: *mut i16,
    n: i32  // must be multiple of 256
)
```

Per 256-element block:
1. SIMD absmax scan (same pattern as existing `quant_f32_i8`)
2. Scale and quantize to i8
3. Compute bsums: sum 16 consecutive i8 values → i16 (using `maddubs` or horizontal add)

- [ ] **Step 3: Compile and test**

```bash
EA=/root/dev/eacompute/target/release/ea $EA kernels/q4k_quant.ea --lib -o build/lib/libq4k_quant.so
gcc -O2 tests/test_q4k_quant.c -Lbuild/lib -lq4k_quant -o build/test_q4k_quant -lm
LD_LIBRARY_PATH=build/lib ./build/test_q4k_quant
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add kernels/q4k_quant.ea tests/test_q4k_quant.c
git commit -m "feat: Q8_K quantization kernel (f32 → i8 + bsums)"
```

---

### Task 4: Q4_K × Q8_K dot product kernel

The core compute kernel. Operates on packed Q4_K nibbles × Q8_K i8 activations.

**Files:**
- Create: `kernels/q4k_dot.ea`
- Create: `tests/test_q4k_dot.c`

- [ ] **Step 1: Write the C reference test**

Create `tests/test_q4k_dot.c` with scalar reference:

```c
// Reference Q4_K × Q8_K dot product for one super-block (256 elements)
// weights: 128 bytes packed nibbles (qs)
// activations: 256 × i8 (Q8_K qs)
// scales[8]: pre-unpacked sub-block scales
// mins[8]: pre-unpacked sub-block mins
// Returns: d * sum(scale_j * dot(q4_sub_j, q8_sub_j)) - dmin * sum(min_j * bsums_j)
float ref_q4k_dot_q8k(
    const uint8_t *q4,        // packed nibbles, 128 bytes
    const int8_t *q8,         // Q8_K quantized activations, 256 bytes
    const int16_t *bsums,     // Q8_K bsums, 16 values
    const uint8_t *scales,    // 8 pre-unpacked sub-block scales
    const uint8_t *mins,      // 8 pre-unpacked sub-block mins
    float d,                  // weight super-block scale * activation scale
    float dmin                // weight super-block min * activation scale
) {
    int sumi = 0;
    int summs = 0;
    for (int j = 0; j < 8; j++) {
        // bsums covers 16-element groups; each sub-block of 32 elements spans 2 bsums
        summs += mins[j] * (bsums[2*j] + bsums[2*j+1]);
    }
    for (int j = 0; j < 4; j++) {  // 4 iterations of 64 elements
        int s0 = 0, s1 = 0;
        for (int k = 0; k < 32; k++) {
            s0 += (q4[j*32 + k] & 0xF) * q8[j*64 + k];       // low nibble
            s1 += (q4[j*32 + k] >> 4)  * q8[j*64 + 32 + k];  // high nibble
        }
        sumi += s0 * scales[2*j] + s1 * scales[2*j+1];
    }
    return d * sumi - dmin * summs;
}
```

Tests: generate deterministic weights and activations, verify kernel matches reference.

- [ ] **Step 2: Write the Eä kernel — single row**

Create `kernels/q4k_dot.ea`:

```
export func q4k_dot_q8k(
    q4: *restrict u8,           // packed nibbles (128 bytes per super-block)
    q8: *restrict i8,           // Q8_K activations (256 bytes per super-block)
    bsums: *restrict i16,       // Q8_K bsums (16 per super-block)
    scales: *restrict u8,       // pre-unpacked scales (8 per super-block)
    mins: *restrict u8,         // pre-unpacked mins (8 per super-block)
    n_blocks: i32,              // number of super-blocks
    d: f32,                     // combined weight.d * activation.d
    dmin: f32                   // combined weight.dmin * activation.d
) -> f32
```

Inner loop per super-block, 4 iterations of 64 elements:
1. Load 32 bytes Q4: `let q4v: u8x16 = load(q4, offset)`
2. Low nibbles: `q4v .& splat(0x0F_u8)` — these are u8x16 values 0-15
3. High nibbles: `q4v .>> splat(4_u8)`
4. Load Q8: `let q8lo: i8x16 = load(q8, offset)`, `let q8hi: i8x16 = load(q8, offset+32)`
5. `maddubs_i16(q4_lo, q8lo)` → scale by `scales[2*j]` → accumulate
6. `maddubs_i16(q4_hi, q8hi)` → scale by `scales[2*j+1]` → accumulate
7. Min correction: `summs += mins[j] * (bsums[2*j] + bsums[2*j+1])`
8. Final: `d * sumi_f32 - dmin * summs_f32`

Note: `maddubs_i16` takes `(u8x16, i8x16)` — Q4 nibbles are unsigned (0-15), Q8 values are signed. This matches the operand order.

- [ ] **Step 3: Write the 4-row variant**

```
export func q4k_dot_q8k_4row(
    rw0: *restrict u8, rw1: *restrict u8, rw2: *restrict u8, rw3: *restrict u8,
    q8: *restrict i8,
    bsums: *restrict i16,
    sc0: *restrict u8, sc1: *restrict u8, sc2: *restrict u8, sc3: *restrict u8,
    mn0: *restrict u8, mn1: *restrict u8, mn2: *restrict u8, mn3: *restrict u8,
    out: *mut f32,  // 4 results
    n_blocks: i32,
    d0: f32, d1: f32, d2: f32, d3: f32,
    dm0: f32, dm1: f32, dm2: f32, dm3: f32
)
```

Shared Q8 activation loads across 4 weight rows. Group-first interleaving for ILP (same pattern as `i2_dot_i8_4row`).

Note: this has many parameters. If Eä doesn't support this many function args, split into a struct or reduce to 2-row variant.

- [ ] **Step 4: Compile and test**

```bash
EA=/root/dev/eacompute/target/release/ea $EA kernels/q4k_dot.ea --lib -o build/lib/libq4k_dot.so
gcc -O2 tests/test_q4k_dot.c -Lbuild/lib -lq4k_dot -o build/test_q4k_dot -lm
LD_LIBRARY_PATH=build/lib ./build/test_q4k_dot
```

- [ ] **Step 5: Commit**

```bash
git add kernels/q4k_dot.ea tests/test_q4k_dot.c
git commit -m "feat: Q4_K × Q8_K dot product kernel (single-row + 4-row)"
```

---

### Task 5: SiLU (SwiGLU) activation kernel

**Files:**
- Create: `kernels/bitnet_silu.ea`
- Create: `tests/test_silu.c`

- [ ] **Step 1: Write C reference test**

```c
// Reference: out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//          = gate[i] * sigmoid(gate[i]) * up[i]
void ref_silu_mul(const float *gate, const float *up, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + expf(-gate[i]));
        out[i] = gate[i] * s * up[i];
    }
}
```

- [ ] **Step 2: Write the Eä kernel**

Create `kernels/bitnet_silu.ea`:

```
export func silu_mul_f32(
    gate: *restrict f32,
    up: *restrict f32,
    out: *mut f32,
    n: i32
)
```

Per 4 (or 8 with AVX) elements:
1. Load gate, load up
2. `neg_gate = splat(0.0) .- gate` (negate)
3. `exp_neg = exp(neg_gate)` (Eä has `exp` — used in fused_attn.ea)
4. `sigmoid = splat(1.0) ./ (splat(1.0) .+ exp_neg)`
5. `out = gate .* sigmoid .* up`

- [ ] **Step 3: Compile and test**

```bash
EA=/root/dev/eacompute/target/release/ea $EA kernels/bitnet_silu.ea --lib -o build/lib/libbitnet_silu.so
gcc -O2 tests/test_silu.c -Lbuild/lib -lbitnet_silu -o build/test_silu -lm
LD_LIBRARY_PATH=build/lib ./build/test_silu
```

- [ ] **Step 4: Commit**

```bash
git add kernels/bitnet_silu.ea tests/test_silu.c
git commit -m "feat: SiLU (SwiGLU) activation kernel"
```

---

### Task 6: FFI + embedding for new kernels

Wire the new kernels into the binary.

**Files:**
- Modify: `build.rs` (add kernel names)
- Modify: `src/embed.rs` (add to KernelTable)
- Modify: `src/ffi.rs` (add wrapper functions)
- Modify: `build_kernels.sh` or `Makefile` (build new .ea files)

- [ ] **Step 1: Update build.rs kernel list**

Add `"q4k_dot"`, `"q4k_quant"`, `"bitnet_silu"` to the kernels array.

- [ ] **Step 2: Update embed.rs KernelTable**

Add function pointer fields:
```rust
pub q4k_dot_q8k: unsafe extern "C" fn(...) -> f32,
pub q4k_dot_q8k_4row: unsafe extern "C" fn(...),
pub quant_f32_q8k: unsafe extern "C" fn(...),
pub silu_mul_f32: unsafe extern "C" fn(...),
```

Add dlopen + dlsym calls in `load()`.

- [ ] **Step 3: Update ffi.rs wrappers**

Add `pub unsafe fn` wrappers for each new kernel function.

- [ ] **Step 4: Build all kernels and test**

```bash
EA=/root/dev/eacompute/target/release/ea make kernels
cargo build --release
cargo test
```

Expected: All existing 70 tests pass. New kernels embedded in binary.

- [ ] **Step 5: Commit**

```bash
git add build.rs src/embed.rs src/ffi.rs Makefile
git commit -m "feat: embed Q4K and SiLU kernels in binary"
```

---

### Task 7: Q4_K matmul dispatch

Rust-side matmul function that unpacks Q4_K 6-bit scales and dispatches to the kernel.

**Files:**
- Create: `src/matmul_q4k.rs`
- Test: add tests to existing `src/matmul_tests.rs` or new test section

- [ ] **Step 1: Write the 6-bit scale unpacking function**

```rust
/// Unpack Q4_K 12-byte packed scales into 8 scales + 8 mins (u8 each).
/// Uses the kmask1/kmask2/kmask3 pattern from ggml.
pub fn unpack_q4k_scales(packed: &[u8; 12], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    // bytes 0-3: scales[0..4] low 6 bits, bits 6-7 are scales[4..8] bits 4-5
    // bytes 4-7: mins[0..4] low 6 bits, bits 6-7 are mins[4..8] bits 4-5
    // bytes 8-11: scales[4..8] low 4 bits | mins[4..8] low 4 bits
    for i in 0..4 {
        scales[i] = packed[i] & 0x3F;
        mins[i] = packed[4 + i] & 0x3F;
    }
    for i in 0..4 {
        let hi_byte = packed[8 + i];
        scales[4 + i] = (hi_byte & 0x0F) | ((packed[i] >> 6) << 4);
        mins[4 + i] = (hi_byte >> 4) | ((packed[4 + i] >> 6) << 4);
    }
}
```

- [ ] **Step 2: Write test for scale unpacking**

Test with known values from llama.cpp's dequantize_row_q4_K reference.

- [ ] **Step 3: Write Q4K matmul function**

```rust
pub fn q4k_matmul_mt(
    weight: *const u8,          // raw Q4_K blocks
    act_q8: *const i8,          // Q8_K quantized activations
    act_d: f32,                 // Q8_K activation scale
    act_bsums: *const i16,      // Q8_K bsums
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,              // must be multiple of 256
    pool: &ThreadPool,
)
```

Per row:
1. Compute block pointers: `weight + row * block_stride`
2. For each super-block, unpack scales (12 bytes → 8 scales + 8 mins)
3. Read `d` (f16→f32) and `dmin` (f16→f32) from block header
4. Call `ffi::q4k_dot_q8k()` (or `q4k_dot_q8k_4row` for 4 rows at once)
5. Write f32 result to output

Threading: same `pool.run(n_threads, ...)` pattern as `ternary_matmul_mt_n`.

- [ ] **Step 4: Test with known values**

Create a small test with hand-crafted Q4_K blocks and known activations, verify output matches scalar reference.

- [ ] **Step 5: Commit**

```bash
git add src/matmul_q4k.rs src/matmul_tests.rs
git commit -m "feat: Q4_K matmul dispatch with 6-bit scale unpacking"
```

---

### Task 8: Q4_K_M weight loading in model.rs

Load Q4_K_M weights from GGUF into the model struct.

**Files:**
- Modify: `src/model.rs`

- [ ] **Step 1: Add Q4K layer weight fields**

Q4_K_M layers need different storage than I2S:
- Weight pointer (raw bytes, `*const u8` into GGUF mmap)
- Block count per tensor (for iteration)
- No per-tensor scale (scales are inside blocks)

Add a `Q4KLayerWeights` struct or extend existing `LayerWeights` with optional fields.

Cleanest approach: parallel struct since the fields are completely different:

```rust
pub struct Q4KLayerWeights {
    pub attn_norm: *const f32,
    pub wq: *const u8, pub wq_blocks: usize,
    pub wk: *const u8, pub wk_blocks: usize,
    pub wv: *const u8, pub wv_blocks: usize,
    pub wo: *const u8, pub wo_blocks: usize,
    pub ffn_norm: *const f32,
    pub w_gate: *const u8, pub w_gate_blocks: usize,
    pub w_up: *const u8, pub w_up_blocks: usize,
    pub w_down: *const u8, pub w_down_blocks: usize,
}
```

- [ ] **Step 2: Load Q4K weights in from_gguf**

For Q4K, weights are raw pointers into the GGUF mmap:

```rust
fn load_q4k_tensor(gguf: &GgufFile, name: &str) -> Result<(*const u8, usize), String> {
    let data = gguf.tensor_data(name).ok_or_else(|| format!("missing tensor: {name}"))?;
    let n_blocks = data.len() / 144; // block_q4_K = 144 bytes
    Ok((data.as_ptr(), n_blocks))
}
```

No repacking needed. The GGUF mmap data IS the block array.

- [ ] **Step 3: Handle embeddings**

Llama 3.2 3B may have:
- `token_embd.weight` — could be Q4_K or F16 (check tensor dtype)
- `output.weight` — separate tensor (not tied), typically Q6_K or Q4_K

For Q4_K embedding lookup: dequantize single row at token index. Write a `q4k_embed_lookup()` function.

For output projection: use `q4k_matmul_mt` against the full vocab (same as i8_output_matmul_mt but with Q4K kernel).

- [ ] **Step 4: Test model loading**

Download Llama 3.2 3B Q4_K_M (or use a small test GGUF), verify:
- ModelConfig detects Q4K quant type
- All 32 layers load without error
- Tensor shapes match expectations (3072 hidden, 8192 ffn, 32 heads)

- [ ] **Step 5: Commit**

```bash
git add src/model.rs
git commit -m "feat: Q4_K_M weight loading from GGUF"
```

---

### Task 9: Llama forward pass

The main inference loop for Llama architecture.

**Files:**
- Create: `src/forward_llama.rs`
- Modify: `src/main.rs` (wire up)

- [ ] **Step 1: Create forward_llama.rs skeleton**

```rust
//! Transformer forward pass for Llama architecture (Q4_K_M weights).

pub struct LlamaState {
    pool: ThreadPool,
    x: Vec<f32>,
    x_norm: Vec<f32>,
    // Q8_K buffers (reused across layers)
    x_q8: Vec<i8>,         // 256-aligned
    x_q8_d: Vec<f32>,      // one scale per super-block
    x_q8_bsums: Vec<i16>,  // 16 per super-block
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    tmp: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    rope_freqs: Vec<f32>,
    max_seq_len: usize,
}
```

- [ ] **Step 2: Implement forward() per layer**

Per layer:
1. `rmsnorm_f32(x, norm_weight, x_norm)` (reuse existing kernel)
2. `quant_f32_q8k(x_norm, x_q8, x_q8_d, x_q8_bsums)` (new kernel)
3. Q4K matmul for Q, K, V (using `q4k_matmul_mt`)
4. RoPE (reuse existing `build_rope_freqs` + `apply_rope`)
5. KV cache store
6. Fused attention (reuse existing `fused_attention_f32`)
7. Q8K quantize attn_out
8. Q4K matmul for O projection
9. Residual add (reuse `vecadd_f32`)
10. RMSNorm → Q8K quantize → Q4K matmul gate+up
11. SiLU activation (`silu_mul_f32`, new kernel) — **NOT squared_relu**
12. Q8K quantize hidden → Q4K matmul down
13. Residual add

Key differences from BitNet forward:
- **No sub-norms** (no attn_sub_norm, no ffn_sub_norm)
- **SiLU** instead of squared ReLU
- **Q8K quantization** instead of simple i8 quantization
- **Q4K matmul** instead of ternary matmul
- **f32 activations** between operations (Q8K is only for matmul input)

- [ ] **Step 3: Implement generate()**

Same structure as BitNet generate: prefill loop + decode loop with `on_token` callback. Copy the pattern from `forward.rs:generate()` but use `LlamaState` and `forward_llama::forward()`.

Include performance profiling with the `prof!` macro pattern.

- [ ] **Step 4: Wire into main.rs**

Replace the placeholder error with actual dispatch:

```rust
QuantType::Q4K => {
    forward_llama::generate(&model, &tokens, max_tokens, temperature,
        repetition_penalty, tokenizer.eos_id, max_seq_len, on_token_callback);
}
```

Wire for all three modes (prompt, interactive, serve).

- [ ] **Step 5: Run all tests**

```bash
cargo test
```

Expected: All tests pass (existing + any new forward_llama tests).

- [ ] **Step 6: Commit**

```bash
git add src/forward_llama.rs src/main.rs
git commit -m "feat: Llama forward pass with Q4_K_M matmul and SiLU"
```

---

### Task 10: End-to-end test and benchmark

Download model, run inference, benchmark against llama.cpp.

**Files:**
- None (manual testing)

- [ ] **Step 1: Download model**

```bash
mkdir -p ~/.cougar/models
# Llama 3.2 3B Instruct Q4_K_M (~2 GB)
curl -L "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  -o ~/.cougar/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

- [ ] **Step 2: Test Cougar**

```bash
cargo build --release
./target/release/cougar --model ~/.cougar/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "The capital of France is" --max-tokens 32
```

Expected: Coherent output, performance stats printed.

- [ ] **Step 3: Test interactive mode**

```bash
echo "What is 2+2?" | ./target/release/cougar \
  --model ~/.cougar/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf --interactive --max-tokens 32
```

- [ ] **Step 4: Benchmark llama.cpp**

```bash
# Use eaclaw's pre-built llama-bench
LLAMA_BENCH=/root/dev/eaclaw/target/debug/build/eaclaw-core-*/out/bin/llama-bench
$LLAMA_BENCH -m ~/.cougar/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -t 2 -n 64 -p 0
```

Record: tok/s, ms/tok.

- [ ] **Step 5: Benchmark Cougar**

```bash
./target/release/cougar --model ~/.cougar/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "The capital of France is" --max-tokens 64
```

Record: tok/s, ms/tok from the perf output.

- [ ] **Step 6: Run hard rules audit**

Check all files under 500 lines. All new features tested. No dead code.

- [ ] **Step 7: Update README**

Add Q4_K_M to supported formats. Add Llama 3.2 3B to tested models. Update LOC count.

- [ ] **Step 8: Commit, tag, push**

```bash
git add README.md
git commit -m "docs: add Q4_K_M support to README, update LOC"
# Bump version
# Edit Cargo.toml: version = "0.3.0"
cargo build --release
git add Cargo.toml
git commit -m "release: bump version to 0.3.0"
git tag -a v0.3.0 -m "Cougar v0.3.0 — Llama 3.2 3B Q4_K_M support"
git push && git push origin v0.3.0
```
