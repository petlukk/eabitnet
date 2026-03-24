# Cougar UX Upgrade — Design Spec

## Problem

Three UX pain points:
1. `LD_LIBRARY_PATH=build/lib` required on every run
2. Single prompt per invocation — no interactive conversation
3. No way for non-technical users to experience the model

## Solution

Three features added to the single Cougar binary. No external dependencies.

## 1. Embedded Kernels (no LD_LIBRARY_PATH)

Kernel .so files embedded via `include_bytes!` at compile time. Extracted on first run to `~/.cougar/lib/v{APP_VERSION}-{KERNEL_HASH}/`. Binary uses `dlopen` to load from there.

**Kernel ABI hash**: SHA-256 of concatenated .so bytes, truncated to 6 hex chars. Embedded as a const at build time. Prevents silent mismatches when kernels are rebuilt without bumping app version.

**Extraction logic**: On startup, check if `~/.cougar/lib/v{VERSION}-{HASH}/` exists. If not, create it and write all .so files. Then `dlopen` each kernel from that path.

**Build change**: New `build.rs` that:
- Reads `build/lib/*.so` files
- Computes the kernel hash
- Generates `include_bytes!` declarations
- Writes the hash as a const

**Impact**: `./cougar --model model.gguf --prompt "..."` just works. No env vars.

## 2. Interactive REPL (`--interactive`)

```
$ cougar --model model.gguf --interactive
cougar> 30 layers, 2560d, 20 heads, 128256 vocab

You: the capital of France is
 Paris, and the largest city in France is Marseille.

You: /quit
```

- Reads stdin line by line
- Tokens stream to stdout as generated (via `on_token` callback)
- Settings via existing flags: `--temperature`, `--repetition-penalty`, `--max-tokens`
- `/quit` or Ctrl-D to exit
- KV cache maintained across turns (conversation context)

## 3. Web Chat UI (`--serve`)

```
$ cougar --model model.gguf --serve
cougar> serving at http://localhost:8080
```

### HTTP Server

Minimal Rust HTTP server. Raw TCP + HTTP/1.1 parsing. Hard-scoped to exactly 3 routes:

| Route | Method | Response |
|---|---|---|
| `/` | GET | Single HTML page (embedded, all inline) |
| `/api/model` | GET | JSON: `{"layers":30, "hidden":2560, ...}` |
| `/api/generate` | POST | SSE stream of tokens |

Nothing else. No headers parsing complexity, no chunked encoding, no keep-alive. Read full POST body, respond, close. Everything else returns 404.

### SSE Token Stream

`POST /api/generate` with body `{"prompt": "...", "temperature": 0.7, ...}`

Response is `text/event-stream`:
```
data: {"token": " Paris", "tps": 9.3}
data: {"token": ",", "tps": 9.4}
data: {"token": " and", "tps": 9.2}
data: [DONE]
```

Structured events include token text and current throughput. Extensible for future fields.

### Chat Interface

Single HTML file with inline CSS/JS. Dark theme.

**Components:**
- Header: Cougar ASCII logo + model info (layers, dims, vocab)
- Chat area: user/assistant message bubbles, auto-scroll
- Token timeline: live-updating progress bar showing current tok/s: `[████████████████] 9.3 tok/s`
- Settings panel: temperature, repetition penalty, max tokens (sliders/inputs)
- Input: text area + send button (Enter to send)

**No:**
- No conversation history persistence
- No multiple concurrent users (single-threaded inference)
- No HTTPS
- No model selection (one model per server instance)

## 4. Streaming Generate (callback refactor)

Refactor `InferenceState::generate()` to accept a per-token callback:

```rust
pub fn generate(
    model: &BitNetModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    repetition_penalty: f32,
    eos_id: u32,
    max_seq_len: usize,
    on_token: impl FnMut(u32),
) -> (Vec<u32>, f64, f64)
```

**Zero-cost contract:**
- `FnMut` is inlined by LLVM — no virtual dispatch
- Callback must NOT allocate, lock, or format inside the hot loop
- REPL callback: `write!()` to pre-locked stdout handle
- Server callback: `write!()` directly to TCP stream
- Single-prompt mode: `|_| {}` — compiler eliminates entirely

Existing callers pass a no-op closure. No performance regression.

## New Files

| File | LOC (est) | Purpose |
|---|---|---|
| `build.rs` | ~60 | Embed kernels, compute ABI hash |
| `src/embed.rs` | ~80 | Extract kernels to ~/.cougar/lib/, dlopen |
| `src/repl.rs` | ~80 | Interactive REPL loop |
| `src/server.rs` | ~400 | HTTP server, SSE streaming, embedded HTML |

All under 500 lines. No external crate dependencies.

## Changes to Existing Files

| File | Change |
|---|---|
| `main.rs` | Add `--interactive`, `--serve`, `--port` flags. Route to repl/server. |
| `forward.rs` | Add `on_token: impl FnMut(u32)` param to `generate()`. Call inside decode loop. |
| `ffi.rs` | Switch from `#[link]` to runtime `dlopen` via `embed.rs` function pointers. |

## Testing

- `generate()` callback: unit test that collects tokens via `Vec::push` callback, verifies same output as before
- REPL: manual test (interactive)
- Server: manual test (browser) + test that `/api/model` returns valid JSON
- Kernel embedding: test that extraction creates files and hash matches
