# Thread Pool Design — Persistent Worker Threads for Cougar

**Date:** 2026-03-23
**Goal:** Replace per-matmul `std::thread::scope` with a persistent thread pool to eliminate ~240 thread create/teardown cycles per token. Target: 10 tok/s → 12-13 tok/s.

## Problem

Every matmul call in `matmul.rs` creates fresh threads via `std::thread::scope` and tears them down on completion. With 30 layers and ~8 thread spawns per layer, that's 240 OS thread lifecycle events per decoded token. At ~20-50μs each, this wastes 5-12ms per token.

## Design

### ThreadPool struct (`src/threadpool.rs`)

A pool of N worker threads (N = `std::thread::available_parallelism()`) created once at startup. Workers sleep on a shared condvar between dispatches. Zero external dependencies.

### API

```rust
pub struct ThreadPool { /* workers, shared state */ }

impl ThreadPool {
    /// Spawn N workers (N = available_parallelism).
    pub fn new() -> Self;

    /// Run `f` on the first `n` workers. Blocks until all complete.
    /// f(thread_id, n_threads) — thread_id is 0..n.
    pub fn run(&self, n: usize, f: impl Fn(usize, usize) + Send + Sync);

    /// Run f1 on threads 0..n1, f2 on threads n1..n1+n2 concurrently.
    /// Blocks until both complete.
    pub fn run_split2(&self, n1: usize, f1: ..., n2: usize, f2: ...);

    /// Run f1/f2/f3 on disjoint thread ranges concurrently.
    /// For QKV: Q on 0..n1, K on n1..n1+n2, V on n1+n2..n1+n2+n3.
    pub fn run_split3(&self, n1: usize, f1: ..., n2: usize, f2: ..., n3: usize, f3: ...);

    /// Number of threads in the pool.
    pub fn thread_count(&self) -> usize;
}
```

### Internals

- **Shared state:** `Arc<Mutex<WorkState>>` + `Condvar` for waking workers.
- **Work descriptor:** Function pointer(s) + thread range assignments, stored in shared state. Safe to use raw pointers because dispatcher blocks until all workers finish.
- **Worker loop:** lock → wait on condvar → wake → read work → unlock → execute → atomically decrement done counter → if last, notify dispatcher condvar → loop.
- **Completion:** `AtomicUsize` done counter. Dispatcher waits on a second condvar, notified by the last finishing worker.
- **Shutdown:** `Drop` impl sets shutdown flag, notifies all workers, joins all threads.

### Asymmetric dispatch

`run_split2` and `run_split3` support the existing concurrent matmul patterns:
- QKV: Q on 8 threads, K on 4, V on 4 (via `run_split3`)
- gate+up: 8 threads each (via `run_split2`)
- Single matmuls (output proj, down proj, embedding): all N threads (via `run`)

Each worker checks its ID against the ranges to determine which function to execute.

### Integration

- `InferenceState` gains a `pool: ThreadPool` field, initialized once in `new()`.
- `forward()` passes `&self.pool` to matmul functions.
- All 6 matmul functions in `matmul.rs` take `&ThreadPool` instead of spawning threads.
- All `std::thread::scope` calls replaced with `pool.run()` / `pool.run_split2()` / `pool.run_split3()`.

### What doesn't change

- FFI kernel calls — identical pointer math.
- Eä kernel files — untouched.
- Forward loop structure — same operation order.
- Thread count — same N, just persistent.

### Matmul function mapping

| Function | Current | New |
|----------|---------|-----|
| `ternary_matmul_mt_n` | `thread::scope`, N threads | `pool.run(N, ...)` |
| `i8_output_matmul_mt` | `thread::scope`, N threads | `pool.run(N, ...)` |
| `f32_matmul_mt` | `thread::scope`, N threads | `pool.run(N, ...)` |
| `f16_matmul_mt` | `thread::scope`, N threads | `pool.run(N, ...)` |
| `ternary_matmul_qkv` | `thread::scope`, 8+4+4 | `pool.run_split3(8, ..., 4, ..., 4, ...)` |
| `ternary_matmul_parallel_pair` | `thread::scope`, 8+8 | `pool.run_split2(8, ..., 8, ...)` |

### Expected impact

- Eliminate ~240 thread create/teardown cycles per token
- Save 5-10ms per token
- 10 tok/s → ~12-13 tok/s
