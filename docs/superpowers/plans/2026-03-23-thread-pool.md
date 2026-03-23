# Thread Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-matmul `std::thread::scope` with a persistent thread pool to eliminate ~2000+ thread create/teardown cycles per token and push throughput from 10 to 12-13 tok/s.

**Architecture:** New `src/threadpool.rs` with condvar-based worker pool. Workers sleep between dispatches. Supports uniform and asymmetric (split2/split3) dispatch. Owned by `InferenceState`, passed to matmul functions.

**Tech Stack:** Rust stdlib only (`std::thread`, `std::sync::{Mutex, Condvar, Arc}`, `std::sync::atomic::AtomicUsize`)

**Spec:** `docs/superpowers/specs/2026-03-23-thread-pool-design.md`

---

### Task 1: ThreadPool — basic `run()` dispatch

**Files:**
- Create: `src/threadpool.rs`
- Modify: `src/main.rs:1` (add `mod threadpool;`)

- [ ] **Step 1: Write the failing test**

Add a test at the bottom of `src/threadpool.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_run_all_threads_execute() {
        let pool = ThreadPool::new();
        let count = AtomicUsize::new(0);
        pool.run(pool.thread_count(), |_tid, _n| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), pool.thread_count());
    }

    #[test]
    fn test_run_subset_of_threads() {
        let pool = ThreadPool::new();
        let count = AtomicUsize::new(0);
        pool.run(4, |_tid, _n| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_run_writes_to_slices() {
        let pool = ThreadPool::new();
        let n = pool.thread_count();
        let mut data = vec![0u32; n];
        let ptr = data.as_mut_ptr() as usize;
        pool.run(n, |tid, _n| {
            unsafe { *(ptr as *mut u32).add(tid) = tid as u32 + 1; }
        });
        for i in 0..n {
            assert_eq!(data[i], i as u32 + 1);
        }
    }

    #[test]
    fn test_run_multiple_dispatches() {
        let pool = ThreadPool::new();
        let count = AtomicUsize::new(0);
        for _ in 0..100 {
            pool.run(pool.thread_count(), |_tid, _n| {
                count.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(count.load(Ordering::Relaxed), pool.thread_count() * 100);
    }
}
```

- [ ] **Step 2: Write minimal ThreadPool with `run()`**

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

struct WorkState {
    /// Raw pointer to the closure. Transmuted from &dyn Fn(usize, usize).
    /// Safety: dispatcher blocks until all workers complete, so the closure's
    /// stack frame is alive for the entire execution.
    func: *const dyn Fn(usize, usize),
    /// How many threads should execute this batch.
    n_active: usize,
    /// Incremented each dispatch so workers can detect new work.
    generation: u64,
    /// Set to true to shut down all workers.
    shutdown: bool,
}

unsafe impl Send for WorkState {}
unsafe impl Sync for WorkState {}

pub struct ThreadPool {
    shared: Arc<(Mutex<WorkState>, Condvar)>,
    done: Arc<AtomicUsize>,
    done_signal: Arc<(Mutex<bool>, Condvar)>,
    workers: Vec<JoinHandle<()>>,
    n_threads: usize,
}

impl ThreadPool {
    pub fn new() -> Self {
        let n_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let shared = Arc::new((
            Mutex::new(WorkState {
                func: std::ptr::null(),
                n_active: 0,
                generation: 0,
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let done = Arc::new(AtomicUsize::new(0));
        let done_signal = Arc::new((Mutex::new(false), Condvar::new()));

        let mut workers = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let shared = Arc::clone(&shared);
            let done = Arc::clone(&done);
            let done_signal = Arc::clone(&done_signal);
            let handle = thread::spawn(move || {
                let mut last_gen: u64 = 0;
                loop {
                    let (func_ptr, n_active);
                    {
                        let (lock, cvar) = &*shared;
                        let mut state = lock.lock().unwrap();
                        while state.generation == last_gen && !state.shutdown {
                            state = cvar.wait(state).unwrap();
                        }
                        if state.shutdown {
                            return;
                        }
                        last_gen = state.generation;
                        func_ptr = state.func;
                        n_active = state.n_active;
                    }
                    if tid < n_active {
                        let f = unsafe { &*func_ptr };
                        f(tid, n_active);
                    }
                    if done.fetch_sub(1, Ordering::AcqRel) == 1 {
                        let (lock, cvar) = &*done_signal;
                        let mut finished = lock.lock().unwrap();
                        *finished = true;
                        cvar.notify_one();
                    }
                }
            });
            workers.push(handle);
        }

        ThreadPool { shared, done, done_signal, workers, n_threads }
    }

    pub fn thread_count(&self) -> usize {
        self.n_threads
    }

    pub fn run(&self, n: usize, f: impl Fn(usize, usize) + Send + Sync) {
        debug_assert!(n <= self.n_threads, "n ({n}) > pool size ({})", self.n_threads);
        if n == 0 { return; }

        // Reset done counter — all n_threads must check in (even if idle)
        self.done.store(self.n_threads, Ordering::Release);
        {
            let mut finished = self.done_signal.0.lock().unwrap();
            *finished = false;
        }

        // Store closure as raw pointer and dispatch
        let func_ref: &dyn Fn(usize, usize) = &f;
        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock().unwrap();
            state.func = func_ref as *const dyn Fn(usize, usize);
            state.n_active = n;
            state.generation += 1;
            cvar.notify_all();
        }

        // Wait for all threads to finish
        {
            let (lock, cvar) = &*self.done_signal;
            let mut finished = lock.lock().unwrap();
            while !*finished {
                finished = cvar.wait(finished).unwrap();
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock().unwrap();
            state.shutdown = true;
            cvar.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}
```

- [ ] **Step 3: Add `mod threadpool;` to main.rs**

In `src/main.rs`, add after the other mod declarations:

```rust
mod threadpool;
```

- [ ] **Step 4: Run tests**

Run: `RUSTFLAGS="-L $(pwd)/build/lib" cargo test -- threadpool -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/threadpool.rs src/main.rs
git commit -m "feat: add ThreadPool with condvar-based run() dispatch"
```

---

### Task 2: Add `run_split2` and `run_split3`

**Files:**
- Modify: `src/threadpool.rs`

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/threadpool.rs`:

```rust
#[test]
fn test_run_split2() {
    let pool = ThreadPool::new();
    let n = pool.thread_count();
    let half = n / 2;
    let rest = n - half;
    let count_a = AtomicUsize::new(0);
    let count_b = AtomicUsize::new(0);
    pool.run_split2(
        half, |_tid, _n| { count_a.fetch_add(1, Ordering::Relaxed); },
        rest, |_tid, _n| { count_b.fetch_add(1, Ordering::Relaxed); },
    );
    assert_eq!(count_a.load(Ordering::Relaxed), half);
    assert_eq!(count_b.load(Ordering::Relaxed), rest);
}

#[test]
fn test_run_split3() {
    let pool = ThreadPool::new();
    let n = pool.thread_count();
    let n1 = n / 2;
    let n2 = n / 4;
    let n3 = n - n1 - n2;
    let c1 = AtomicUsize::new(0);
    let c2 = AtomicUsize::new(0);
    let c3 = AtomicUsize::new(0);
    pool.run_split3(
        n1, |_tid, _n| { c1.fetch_add(1, Ordering::Relaxed); },
        n2, |_tid, _n| { c2.fetch_add(1, Ordering::Relaxed); },
        n3, |_tid, _n| { c3.fetch_add(1, Ordering::Relaxed); },
    );
    assert_eq!(c1.load(Ordering::Relaxed), n1);
    assert_eq!(c2.load(Ordering::Relaxed), n2);
    assert_eq!(c3.load(Ordering::Relaxed), n3);
}

#[test]
fn test_split2_thread_ids_are_local() {
    let pool = ThreadPool::new();
    let n = pool.thread_count();
    let half = n / 2;
    let rest = n - half;
    let mut ids_a = vec![0usize; half];
    let mut ids_b = vec![0usize; rest];
    let ptr_a = ids_a.as_mut_ptr() as usize;
    let ptr_b = ids_b.as_mut_ptr() as usize;
    pool.run_split2(
        half, |tid, _n| { unsafe { *(ptr_a as *mut usize).add(tid) = tid; } },
        rest, |tid, _n| { unsafe { *(ptr_b as *mut usize).add(tid) = tid; } },
    );
    for i in 0..half { assert_eq!(ids_a[i], i); }
    for i in 0..rest { assert_eq!(ids_b[i], i); }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `RUSTFLAGS="-L $(pwd)/build/lib" cargo test -- threadpool -v`
Expected: FAIL — `run_split2` and `run_split3` not found.

- [ ] **Step 3: Implement `run_split2` and `run_split3`**

Refactor `WorkState` to hold up to 3 function slots with boundary-based dispatch. Each worker checks its global tid against the boundaries to pick which function to run, remapping to a 0-based local index within its group.

Refactored `WorkState`:

```rust
struct WorkState {
    /// Up to 3 function pointers for split dispatch.
    /// Safety: dispatcher blocks until all workers complete, so the closures'
    /// stack frames are alive for the entire execution.
    funcs: [*const dyn Fn(usize, usize); 3],
    /// Cumulative thread boundaries: group 0 = [0, bounds[0]),
    /// group 1 = [bounds[0], bounds[1]), group 2 = [bounds[1], bounds[2]).
    bounds: [usize; 3],
    n_groups: usize,
    generation: u64,
    shutdown: bool,
}
```

Refactored worker dispatch (replaces the `if tid < n_active` block):

```rust
let n_active_total = bounds[n_groups - 1];
if tid < n_active_total {
    if tid < bounds[0] {
        let f = unsafe { &*funcs[0] };
        f(tid, bounds[0]);
    } else if n_groups >= 2 && tid < bounds[1] {
        let f = unsafe { &*funcs[1] };
        f(tid - bounds[0], bounds[1] - bounds[0]);
    } else if n_groups >= 3 && tid < bounds[2] {
        let f = unsafe { &*funcs[2] };
        f(tid - bounds[1], bounds[2] - bounds[1]);
    }
}
```

Extract shared dispatch logic into private `dispatch` method:

```rust
fn dispatch(&self, funcs: [*const dyn Fn(usize, usize); 3], bounds: [usize; 3], n_groups: usize) {
    self.done.store(self.n_threads, Ordering::Release);
    {
        let mut finished = self.done_signal.0.lock().unwrap();
        *finished = false;
    }
    {
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().unwrap();
        state.funcs = funcs;
        state.bounds = bounds;
        state.n_groups = n_groups;
        state.generation += 1;
        cvar.notify_all();
    }
    {
        let (lock, cvar) = &*self.done_signal;
        let mut finished = lock.lock().unwrap();
        while !*finished {
            finished = cvar.wait(finished).unwrap();
        }
    }
}
```

Refactor `run()` to use `dispatch`:

```rust
pub fn run(&self, n: usize, f: impl Fn(usize, usize) + Send + Sync) {
    debug_assert!(n <= self.n_threads);
    if n == 0 { return; }
    let func_ref: &dyn Fn(usize, usize) = &f;
    self.dispatch(
        [func_ref as *const _, std::ptr::null(), std::ptr::null()],
        [n, 0, 0], 1,
    );
}
```

`run_split2`:

```rust
pub fn run_split2(
    &self,
    n1: usize, f1: impl Fn(usize, usize) + Send + Sync,
    n2: usize, f2: impl Fn(usize, usize) + Send + Sync,
) {
    debug_assert!(n1 + n2 <= self.n_threads, "split2 {} + {} > pool {}", n1, n2, self.n_threads);
    if n1 + n2 == 0 { return; }
    let r1: &dyn Fn(usize, usize) = &f1;
    let r2: &dyn Fn(usize, usize) = &f2;
    self.dispatch(
        [r1 as *const _, r2 as *const _, std::ptr::null()],
        [n1, n1 + n2, 0], 2,
    );
}
```

`run_split3`:

```rust
pub fn run_split3(
    &self,
    n1: usize, f1: impl Fn(usize, usize) + Send + Sync,
    n2: usize, f2: impl Fn(usize, usize) + Send + Sync,
    n3: usize, f3: impl Fn(usize, usize) + Send + Sync,
) {
    debug_assert!(n1 + n2 + n3 <= self.n_threads, "split3 {} + {} + {} > pool {}", n1, n2, n3, self.n_threads);
    if n1 + n2 + n3 == 0 { return; }
    let r1: &dyn Fn(usize, usize) = &f1;
    let r2: &dyn Fn(usize, usize) = &f2;
    let r3: &dyn Fn(usize, usize) = &f3;
    self.dispatch(
        [r1 as *const _, r2 as *const _, r3 as *const _],
        [n1, n1 + n2, n1 + n2 + n3], 3,
    );
}
```

**Note:** `run()` and split variants are not safe for concurrent calls from multiple threads. This matches the usage pattern (`forward()` takes `&mut self`). Add a comment documenting this.

- [ ] **Step 4: Run tests**

Run: `RUSTFLAGS="-L $(pwd)/build/lib" cargo test -- threadpool -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/threadpool.rs
git commit -m "feat: add run_split2/run_split3 for asymmetric thread dispatch"
```

---

### Task 3: Wire ThreadPool into InferenceState and `ternary_matmul_mt_n`

**Files:**
- Modify: `src/forward.rs:7-8,62-87,89` (add pool field, pass to matmul)
- Modify: `src/matmul.rs:191-284` (ternary_matmul_mt_n and ternary_matmul_mt)

- [ ] **Step 1: Update `ternary_matmul_mt_n` to accept `&ThreadPool`**

Replace `std::thread::scope` block (lines 232-272) with `pool.run()`. The closure captures the same data via usize pointers as before, but uses `tid` and `n_threads` from the pool dispatch to compute its chunk. Preserve the `n_threads <= 1` early-return path (lines 198-223) — dispatching through the pool for a single thread adds unnecessary synchronization overhead.

New signature:

```rust
pub(crate) fn ternary_matmul_mt_n(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
    n_threads: usize, pool: &ThreadPool,
)
```

Replace the `std::thread::scope` block with:

```rust
pool.run(n_threads, |tid, n_threads| {
    // Same per-thread body as current spawn closure,
    // using tid to compute start/end chunk
    let start = tid * chunk;
    let end = (start + chunk).min(out_dim);
    // ... rest identical
});
```

- [ ] **Step 2: Update `ternary_matmul_mt` to accept and forward `&ThreadPool`**

```rust
pub(crate) fn ternary_matmul_mt(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
    pool: &ThreadPool,
) {
    ternary_matmul_mt_n(weight, act, act_scale, act_sum, weight_scale,
        out, out_dim, in_dim, pool.thread_count(), pool);
}
```

- [ ] **Step 3: Add `pool` field to `InferenceState`**

In `src/forward.rs`, add to the struct (before buffer fields):

```rust
use crate::threadpool::ThreadPool;

pub struct InferenceState {
    pool: ThreadPool,
    // ... existing fields
}
```

Initialize in `new()`:

```rust
pool: ThreadPool::new(),
```

- [ ] **Step 4: Update `forward()` calls to `ternary_matmul_mt`**

Pass `&self.pool` to the two `ternary_matmul_mt` calls at lines 179 and 241.

- [ ] **Step 5: Build and run**

Run: `RUSTFLAGS="-L $(pwd)/build/lib" cargo build --release && LD_LIBRARY_PATH=build/lib ./target/release/cougar --model ~/.eabitnet/models/ggml-model-i2_s.gguf --prompt "Hello" --max-tokens 16`
Expected: Compiles, generates text. O proj and FFN down now use pool.

- [ ] **Step 6: Commit**

```bash
git add src/threadpool.rs src/matmul.rs src/forward.rs
git commit -m "feat: wire ThreadPool into ternary_matmul_mt — O proj and down proj use pool"
```

---

### Task 4: Convert `ternary_matmul_qkv` to `run_split3`

**Files:**
- Modify: `src/matmul.rs:288-322`
- Modify: `src/forward.rs:130-135` (pass pool)

- [ ] **Step 1: Rewrite `ternary_matmul_qkv` to use `pool.run_split3`**

New signature adds `pool: &ThreadPool`. Remove `std::thread::scope`. Compute q_threads/k_threads/v_threads from `pool.thread_count()`. Inline the per-thread work from `ternary_matmul_mt_n` into each split closure (chunk calculation, i2_dot_i8 loop, scaling). No more delegation to `ternary_matmul_mt_n`.

```rust
pub(crate) fn ternary_matmul_qkv(
    w_q: *const u8, scale_q: f32, out_q: &mut [f32], out_dim_q: usize,
    w_k: *const u8, scale_k: f32, out_k: &mut [f32], out_dim_kv: usize,
    w_v: *const u8, scale_v: f32, out_v: &mut [f32],
    act: *const i8, act_scale: f32, act_sum: i32, in_dim: usize,
    pool: &ThreadPool,
) {
    let total = pool.thread_count();
    // Safe split: sum always <= total (no .max(1) inflation)
    let q_threads = (total / 2).max(1);
    let remaining = total - q_threads;
    let k_threads = remaining / 2;
    let v_threads = remaining - k_threads;
    // On tiny machines (total=1): q=1, k=0, v=0 — k/v run inline on dispatcher
    // On total=2: q=1, k=1, v=0 — v runs inline (fine, dimensions are small)

    let row_bytes = in_dim / 4;
    // Precompute scales
    let q_scale = (act_scale / 127.0) * scale_q;
    let k_scale = (act_scale / 127.0) * scale_k;
    let v_scale = (act_scale / 127.0) * scale_v;

    // Convert to usize for Send
    let act_ptr = act as usize;
    let wq = w_q as usize;
    let wk = w_k as usize;
    let wv = w_v as usize;
    let q_ptr = out_q.as_mut_ptr() as usize;
    let k_ptr = out_k.as_mut_ptr() as usize;
    let v_ptr = out_v.as_mut_ptr() as usize;

    // Helper closure builder for ternary matmul work
    let make_work = |w_ptr: usize, out_ptr: usize, out_dim: usize, scale: f32| {
        move |tid: usize, n_threads: usize| {
            let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { return; }
            let count = end - start;
            let weight = w_ptr as *const u8;
            let act = act_ptr as *const i8;
            let out_slice = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), count)
            };
            let mut raw = vec![0i32; count];
            let mut r = 0;
            unsafe {
                while r + 4 <= count {
                    let row = start + r;
                    ffi::i2_dot_i8_4row(
                        weight.add(row * row_bytes),
                        weight.add((row + 1) * row_bytes),
                        weight.add((row + 2) * row_bytes),
                        weight.add((row + 3) * row_bytes),
                        act, raw[r..].as_mut_ptr(), in_dim as i32,
                    );
                    r += 4;
                }
                while r < count {
                    raw[r] = ffi::i2_dot_i8(
                        weight.add((start + r) * row_bytes), act, in_dim as i32,
                    );
                    r += 1;
                }
            }
            for i in 0..count {
                out_slice[i] = (raw[i] - act_sum) as f32 * scale;
            }
        }
    };

    pool.run_split3(
        q_threads, make_work(wq, q_ptr, out_dim_q, q_scale),
        k_threads, make_work(wk, k_ptr, out_dim_kv, k_scale),
        v_threads, make_work(wv, v_ptr, out_dim_kv, v_scale),
    );
}
```

- [ ] **Step 2: Update call site in forward.rs**

Pass `&self.pool` to `ternary_matmul_qkv` at line 130.

- [ ] **Step 3: Build and run**

Run: `RUSTFLAGS="-L $(pwd)/build/lib" cargo build --release && LD_LIBRARY_PATH=build/lib ./target/release/cougar --model ~/.eabitnet/models/ggml-model-i2_s.gguf --prompt "Hello" --max-tokens 16`
Expected: Compiles, generates same text. QKV now single-level dispatch.

- [ ] **Step 4: Commit**

```bash
git add src/matmul.rs src/forward.rs
git commit -m "feat: flatten ternary_matmul_qkv to pool.run_split3 — no nested spawning"
```

---

### Task 5: Convert `ternary_matmul_parallel_pair` to `run_split2`

**Files:**
- Modify: `src/matmul.rs:324-359`
- Modify: `src/forward.rs:208-214` (pass pool)

- [ ] **Step 1: Rewrite `ternary_matmul_parallel_pair` to use `pool.run_split2`**

Same pattern as Task 4. New signature adds `pool: &ThreadPool`. Inline the per-thread work. Use the same `make_work` helper pattern or inline directly.

- [ ] **Step 2: Update call site in forward.rs**

Pass `&self.pool` to `ternary_matmul_parallel_pair` at line 208.

- [ ] **Step 3: Build and run**

Run: `RUSTFLAGS="-L $(pwd)/build/lib" cargo build --release && LD_LIBRARY_PATH=build/lib ./target/release/cougar --model ~/.eabitnet/models/ggml-model-i2_s.gguf --prompt "Hello" --max-tokens 16`
Expected: Compiles, generates same text. gate+up now single-level dispatch.

- [ ] **Step 4: Commit**

```bash
git add src/matmul.rs src/forward.rs
git commit -m "feat: flatten ternary_matmul_parallel_pair to pool.run_split2"
```

---

### Task 6: Convert `i8_output_matmul_mt`

**Files:**
- Modify: `src/matmul.rs:44-116`
- Modify: `src/forward.rs:264-268` (pass pool)

- [ ] **Step 1: Update `i8_output_matmul_mt` to accept `&ThreadPool`**

New signature:

```rust
pub(crate) fn i8_output_matmul_mt(
    embed_i8: &[u8], row_scales: &[f32],
    x: &[f32], out: &mut [f32],
    vocab_size: usize, hidden_dim: usize,
    pool: &ThreadPool,
)
```

Keep the sequential quantization (lines 53-65) on the caller thread. Replace `thread::scope` (line 74) with `pool.run(pool.thread_count(), ...)`. Remove the internal `use std::thread` and `available_parallelism()` call. Same closure body, using `tid`/`n_threads` from the pool dispatch to compute chunks.

- [ ] **Step 2: Update call site in forward.rs**

Pass `&self.pool`.

- [ ] **Step 3: Build and run**

Run: `RUSTFLAGS="-L $(pwd)/build/lib" cargo build --release && LD_LIBRARY_PATH=build/lib ./target/release/cougar --model ~/.eabitnet/models/ggml-model-i2_s.gguf --prompt "Hello" --max-tokens 16`
Expected: Compiles, generates same text.

- [ ] **Step 4: Commit**

```bash
git add src/matmul.rs src/forward.rs
git commit -m "feat: convert i8_output_matmul_mt to use thread pool"
```

---

### Task 7: Convert remaining matmul functions and cleanup

**Files:**
- Modify: `src/matmul.rs` (f32_matmul_mt, f16_matmul_mt)

- [ ] **Step 1: Update `f32_matmul_mt` and `f16_matmul_mt`**

Both get `pool: &ThreadPool` parameter. Replace `thread::scope` with `pool.run()`. These are `#[allow(dead_code)]` but should be converted for consistency.

- [ ] **Step 2: Remove all function-local `use std::thread;` from matmul.rs**

Each of `i8_output_matmul_mt`, `f32_matmul_mt`, `f16_matmul_mt` has `use std::thread;` inside the function body. Remove them all — no thread imports needed anymore.

- [ ] **Step 3: Build and run tests**

Run: `RUSTFLAGS="-L $(pwd)/build/lib" cargo test -v && RUSTFLAGS="-L $(pwd)/build/lib" cargo build --release`
Expected: All tests pass, release builds clean.

- [ ] **Step 4: Commit**

```bash
git add src/matmul.rs
git commit -m "feat: convert remaining matmul functions to thread pool, remove thread::scope"
```

---

### Task 8: Benchmark and verify

**Files:** None (read-only)

- [ ] **Step 1: Run full benchmark**

Run: `LD_LIBRARY_PATH=build/lib ./target/release/cougar --model ~/.eabitnet/models/ggml-model-i2_s.gguf --prompt "Hello" --max-tokens 128`

Record: tok/s, ms/tok, profile breakdown.

- [ ] **Step 2: Compare against baseline**

Baseline: 10.2 tok/s, 98.1 ms/tok.
Target: 12-13 tok/s.
Check profile for which sections improved most (QKV and gate+up should show biggest gains due to nested spawn elimination).

- [ ] **Step 3: Commit benchmark note**

Update `docs/future_optimizations.md` with the new numbers and mark thread pool as done.

```bash
git add docs/future_optimizations.md
git commit -m "docs: thread pool results — X tok/s (was 10.2)"
```
