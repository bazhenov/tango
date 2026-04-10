# Commpage: Process-Isolated Benchmarking via Shared Memory

## Motivation

Today tango loads both the baseline (B) and candidate (C) benchmarks into **the same process** using `dlopen`/`LoadLibrary`. This is clever but has drawbacks:

1. **Platform hacks** -- Linux PIE patching (`linux.rs`) and Windows IAT patching (`windows.rs`) exist solely to make executables loadable as shared libraries. These are fragile and there is no guarantee they will continue to work in the future. Especially on Linux where this approach was explicitly banned by dyld.
2. **Insufficient isolation** -- Operating systems provide ways to collect additional per-process information (e.g. memory allocations), but these are often defined at the process level, not the thread level. Running both benchmarks in the same process prevents effective use of these tools.

Running each benchmark in its **own process** and reading measurements through shared memory (the "commpage") solves both problems.

## Architecture Overview

```
                        ┌──────────────────────────┐
                        │     USER invokes:        │
                        │  ./candidate compare     │
                        │       ./baseline         │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │     RUNNER  (R)          │
                        │  Reads commpage          │
                        │  Computes statistics     │
                        │  Prints results          │
                        └──┬────────────────────┬──┘
                   stdin/  │                    │  stdin/
                   stdout  │                    │  stdout
                  (JSON-   │                    │  (JSON-
                   RPC)    │                    │   RPC)
                           ▼                    ▼
                 ┌─────────────────┐  ┌─────────────────┐
                 │  CANDIDATE (C)  │  │  BASELINE (B)   │
                 │  Writes lane C  │  │  Writes lane B  │
                 │  Reads lane B   │  │  Reads lane C   │
                 └────────┬────────┘  └────────┬────────┘
                          │                    │
                          ▼                    ▼
                 ┌─────────────────────────────────────┐
                 │           COMMPAGE (shm)            │
                 │  ┌───────────┐  ┌───────────┐       │
                 │  │  Lane C   │  │  Lane B   │       │
                 │  │  (C data) │  │  (B data) │       │
                 │  └───────────┘  └───────────┘       │
                 └─────────────────────────────────────┘
                          ▲          ▲
                          │  reads   │
                          └────┬─────┘
                          RUNNER (R)
```

### Roles

| Role              | Process                              | Description                                                                                                                                                                                           |
| ----------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Runner (R)**    | The initial process the user invokes | Spawns C and B, sends commands via JSON-RPC over stdio, reads measurements from commpage, computes statistics, prints results. Passive during measurement -- does not participate in synchronization. |
| **Candidate (C)** | Spawned by R (copy of itself)        | Writes samples to lane C, reads B's write_cursor (lane B) to stay in lock-step.                                                                                                                       |
| **Baseline (B)**  | Spawned by R (the second executable) | Writes samples to lane B, reads C's write_cursor (lane C) to stay in lock-step.                                                                                                                       |

## Commpage Design

A **commpage** is a single shared memory region accessible to all three processes (R, C, B). It contains two **lanes** -- one for the candidate, one for the baseline. Each child writes to its own lane and reads the peer's write_cursor to synchronize. R only reads, and does not participate in the synchronization.

### Memory Layout

Each lane contains N = 128 sample slots (N must be a power of two). The two lanes are laid out contiguously, with each lane's write_cursor immediately preceding its sample array.

```
Offset  Size    Field                Description
──────  ────    ─────                ───────────
0x000   8       magic                0x54414E474F_434D50 ("TANGOCMP")
0x008   4       version              Commpage protocol version (1)
0x00C   4       flags                AtomicU32: flags (see below)

Lane C (Candidate):
0x010   8       write_cursor_c       AtomicU64: C's monotonic sample counter
0x018   N*8     samples_c[N]         C's ring buffer (128 * 8 = 1024 bytes)

Lane B (Baseline):
0x418   8       write_cursor_b       AtomicU64: B's monotonic sample counter
0x420   N*8     samples_b[N]         B's ring buffer (128 * 8 = 1024 bytes)
```

Total size: 16 + 2 \* (8 + 1024) = **2080 bytes**. Fits in a single page.

**`flags` (AtomicU32 at 0x00C):**

| Bit  | Name     | Description                                      |
| ---- | -------- | ------------------------------------------------ |
| 0    | `STOP`   | Set by R to signal early termination (`-t` mode) |
| 1-31 | reserved | For future use                                   |

R sets the `STOP` bit when the time budget is exhausted. C and B check it before each sample. In `--samples` mode it is never set.

**`write_cursor` encoding:**

Each `write_cursor` is a monotonically increasing counter -- the total number of samples written by that child. The slot index is `write_cursor & (N - 1)` (low 7 bits).

```
  ┌────┬───────────────────────────┬────────────┐
  │done│  sample count (bits 7..62)│ slot (0..6)│
  │ 63 │                           │            │
  └────┴───────────────────────────┴────────────┘
```

**`DONE` flag (bit 63):** C/B sets this bit when it exits the measurement loop -- whether normally (all samples taken) or abruptly (error, panic). R reads `write_cursor` atomically and know both the sample count and whether the child is finished. The peer also checks this bit in the barrier spin-wait to avoid spinning forever if the other side exits early.

There is **no read_cursor in shared memory**. R tracks its read position per lane locally.

Each **Sample** slot (8 bytes):

A single `u64` holding the total elapsed nanoseconds for the sample. The iteration count is fixed for the entire benchmark run (set via `run_benchmark(iterations=...)`), so it does not need to be stored per sample.

### Peer Synchronization via Write Cursors

C and B synchronize directly with each other through the commpage -- **R does not participate**. After writing sample N, each child spin-waits (busy loop, no OS yield) until the peer's write_cursor reaches N. This acts as a per-sample barrier: neither child starts sample N+1 until both have completed sample N.

```
  C (Candidate, lane C)              B (Baseline, lane B)
  ─────────────────────              ────────────────────
  take sample 0                      take sample 0
  write to lane C                    write to lane B
  wc_c.fetch_add(1, Release) → 1     wc_b.fetch_add(1, Release) → 1
  spin until wc_b >= 1        ◄──►   spin until wc_c >= 1
  ── barrier ──                      ── barrier ──
  take sample 1                      take sample 1
  write to lane C                    write to lane B
  wc_c.fetch_add(1, Release) → 2     wc_b.fetch_add(1, Release) → 2
  spin until wc_b >= 2        ◄──►   spin until wc_c >= 2
  ── barrier ──                      ── barrier ──
  ...                                ...
  after num_samples:                 after num_samples:
  return RPC response                return RPC response
```

The first sample may start slightly offset (one child receives the RPC before the other), but the barrier after sample 0 aligns them. From sample 1 onward, both children start each measurement nearly simultaneously, under the same system conditions.

R knows the number of samples upfront and passes it via `run_benchmark(num_samples=...)`. Both children run exactly that many samples and return. In case of `-t` (bench duration) R pass `num_samples=0` and control children via `STOP` bit in a `flag` field of the commpage.

C/B may only access the commpage while `run_benchmark()` is in-flight. R may access the commpage at any time (e.g. to drain samples for progress reporting).

### Capacity Sizing

N = **128 samples per lane** (128 \* 8 = 1 KiB per lane). At ~10ms per sample, 128 slots represent ~1.3 seconds of data. R only needs to drain faster than one buffer's worth per second, which is trivially achievable.

### Lock-Free Protocol

**Child (C or B) -- measurement loop (inside `run_benchmark()`):**

```
let my_wc = &commpage.write_cursor[my_lane]
let peer_wc = &commpage.write_cursor[peer_lane]
let my_samples = &commpage.samples[my_lane]

let sample_no = 0;
loop {
    if num_samples > 0 && sample_no >= num_samples { break; }  // --samples mode
    if commpage.stop() { break; }                               // -t mode (early termination)
    elapsed_ns = take measurement (call benchmark function)
    slot = sample_no & (N - 1)
    my_samples[slot] = elapsed_ns
    sample_no += 1
    my_wc.store(sample_no, Release)   // publish sample
    // barrier: wait for peer to also complete this sample (or exit)
    loop {
        let peer = peer_wc.load(Acquire)
        if peer & DONE_BIT != 0 { break; }     // peer exited early
        if peer >= sample_no { break; }          // peer caught up
        spin;
    }
}
my_wc.fetch_or(DONE_BIT, Release)   // signal done to peer and R
return RPC response
```

**Runner (R) -- passive reader:**

```
// --samples mode: num_samples = S (known upfront)
// -t mode:        num_samples = 0 (run until stop_flag)
send run_benchmark(num_samples=S) RPC to C (do not wait for response)
send run_benchmark(num_samples=S) RPC to B (do not wait for response)

if time_budget_mode {
    // monitor elapsed time, drain commpage periodically
    while elapsed < time_budget {
        drain samples for progress reporting
    }
    commpage.stop_flag.store(1, Release)
}

read RPC response from C
read RPC response from B
drain remaining samples from both lanes
```

## Child Spawning and Initialization

R spawns each child as a hidden `__worker` subcommand with two required flags:

```
./executable __worker --shmem /tango-XXXX --role candidate
./executable __worker --shmem /tango-XXXX --role baseline
```

| Flag      | Description                                                                       |
| --------- | --------------------------------------------------------------------------------- |
| `--shmem` | OS-level shared memory name (e.g. `/tango-XXXX`)                                  |
| `--role`  | `candidate` (write lane C, peer lane B) or `baseline` (write lane B, peer lane C) |

The child opens the commpage on startup and enters the JSON-RPC dispatch loop. R discovers available benchmarks by sending a `list_benchmarks` request.

## JSON-RPC Protocol (over stdio)

Communication between R and C/B uses newline-delimited JSON-RPC 2.0 over the child's stdin/stdout. This carries **control messages only** -- measurement data flows through the commpage.

### Methods (R -> Child)

```jsonc
// List available benchmarks
{"jsonrpc":"2.0","method":"list_benchmarks","params":null,"id":1}
// Response: {"jsonrpc":"2.0","result":{"benchmarks":["bench_a","bench_b"]},"id":1}

// Estimate iterations for a given time budget
{"jsonrpc":"2.0","method":"estimate_iterations","params":{"index":0,"time_ms":10,"seed":42},"id":2}
// Response: {"jsonrpc":"2.0","result":{"iterations":50000},"id":2}

// Start the measurement loop. The child takes samples, writing each to its
// commpage lane and synchronizing with the peer via write_cursors.
// num_samples > 0: run exactly that many samples (--samples mode)
// num_samples = 0: run until stop_flag is set by R (-t mode)
{"jsonrpc":"2.0","method":"run_benchmark","params":{"index":0,"seed":42,"iterations":1000,"num_samples":300},"id":3}
// Response (sent after loop exits):
// {"jsonrpc":"2.0","result":{"samples_written":300},"id":3}

// Shutdown gracefully
{"jsonrpc":"2.0","method":"shutdown","id":4}
```

### Why JSON-RPC + commpage (not pure JSON-RPC or pure shmem)?

- **JSON-RPC for control**: Flexible, debuggable, easy to evolve. Control messages are infrequent so serialization cost is irrelevant.
- **Commpage for data**: Zero-copy, zero-syscall measurement reporting. The child writes samples without waiting for an acknowledgement. This is critical because any IPC synchronization on the measurement path would add latency noise.

## Implementation Plan

### Phase 1: Commpage Module

**New file: `tango-bench/src/commpage.rs`**

1. Define the commpage layout as `#[repr(C)]` structs: `Commpage`, and `Lane`.
2. Implement `Commpage::create(name) -> Result<Commpage>` -- creates shared memory region via the `shared_memory` crate, initializes header and both lanes.
3. Implement `Commpage::open(name) -> Result<Commpage>` -- opens an existing region by name, validates magic/version.
4. implement `Lane::push_sample(elapsed_ns: u64)` writes sample and increments write_cursor, `Lane::wait_for_peer(write_cursor_value: u64)` spin-waits until peer's write_cursor catches up.
5. (for runner): `Lane::drain_samples(from: u64) -> Vec<Sample>` reads new samples from a lane.
6. Unit tests: create + open in same process, write N samples from two threads acting as C/B, verify peer synchronization and drain.

**Dependencies:** `shared_memory` crate (already in workspace for `tests/shmem.rs`).

### Phase 2: Child Worker Mode

**New file: `tango-bench/src/worker.rs`**

The child process needs a way to enter "worker mode" where it reads JSON-RPC commands from stdin and writes measurement samples to the commpage.

1. Add a hidden CLI subcommand: `__worker --shmem <name> --role <role>`.
2. When `__worker` is invoked:
   - Open the commpage.
   - Initialize list of benchmarks.
   - Enter command loop reading JSON-RPC from stdin.
3. The `run_benchmark` command:
   - Prepares state from the benchmark index and seed passed in params.
   - Loops `num_samples` times: take measurement, write sample to own lane, `wait_for_peer()`.
   - Returns RPC response after all samples are written.
4. The `estimate_iterations` command prepares state from index/seed and estimates iteration count for a given time budget.
5. The child's stderr is forwarded to the runner's stderr for debugging.

### Phase 3: Runner-Side Child Handle

**New file: `tango-bench/src/child.rs`**

A `ChildHandle` struct that the runner uses to manage one child process.

Methods:

- `spawn(executable: &Path, commpage: &Commpage, role: Role) -> Result<ChildHandle>` -- spawns child with `__worker --shmem <name> --role <role>` CLI arguments.
- `list_benchmarks() -> Result<Vec<String>>` -- queries the child for available benchmark names via JSON-RPC.
- `estimate_iterations(index: usize, seed: u64, time_ms: u32) -> usize`
- `start_benchmark(index: usize, seed: u64, iterations: usize, num_samples: usize)` -- sends `run_benchmark` RPC **without** waiting for the response. Child enters its measurement loop.
- `finish_benchmark()` -- reads the RPC response (blocks until child finishes all samples).
- `drain_samples()` -- reads new samples from this child's commpage lane.
- `shutdown()` -- sends shutdown, waits for child to exit.

Note: R creates a single `Commpage` and passes it to both `ChildHandle`s. Each handle knows its role.

### Phase 4: Integrate with Existing CLI

**Modify: `tango-bench/src/cli.rs`**

1. Replace `Spi::for_self()` and `Spi::for_library()` in `run_paired_test` with `ChildHandle::spawn()`.
   - R creates one `Commpage`.
   - The candidate is spawned from `env::current_exe()` with `role: candidate` (lane C).
   - The baseline is spawned from the provided path with `role: baseline` (lane B).
2. The measurement loop simplifies -- R is passive:
   - Call `child_c.start_benchmark(iterations, num_samples)` and `child_b.start_benchmark(iterations, num_samples)`.
   - C and B synchronize with each other via commpage write_cursors. R does not drive the loop.
   - R calls `child_c.finish_benchmark()` and `child_b.finish_benchmark()` (blocks until both RPCs return).
   - R drains all samples from both lanes.
   - The `--parallel` flag is no longer needed (process isolation replaces thread isolation).
3. **Solo mode stays in-process** -- it is designed for use with system profilers (perf, Instruments, etc.) where an extra fork barrier would complicate profiling and debugging.
4. Remove PIE patching (`linux.rs`) and IAT patching (`windows.rs`) once the new path is validated -- these are no longer needed since we don't load executables as libraries.

### Phase 5: Wire Up `tango_main!` Macro

**Modify: `tango-bench/src/lib.rs`** (or `dylib.rs`)

The `tango_main!()` macro generates a `main()` that calls `cli::run()`. The `__worker` subcommand is handled by clap inside `cli::run()`, so no special detection is needed in the macro.

### Phase 6: Testing

1. **Unit tests** for commpage: create, write, read, overflow, flags (with forking).
2. **Integration tests** for worker: spawn a child in worker mode, send JSON-RPC commands, verify commpage contains expected samples.
3. **End-to-end tests**: extend `cli_tests.rs` with the existing test scenarios (`compare`, `solo`, panic handling, regression detection) running against the new process-isolated mode.
4. **Cross-platform**: test on Linux and macOS (Windows support can follow since POSIX shmem is available on both Unix platforms; Windows would use `CreateFileMapping`).

### Phase 7: Cleanup

1. Remove `linux.rs` (PIE patching) and `windows.rs` (IAT patching).
2. Remove `libloading` dependency.
3. Remove the in-process `Spi` / `VTable` / FFI machinery and the `tango_*` C FFI exports in `dylib.rs`. No backward compatibility is needed.
4. Simplify error handling now that child crashes are isolated (runner gets an exit code + stderr, not a segfault).

## File Change Summary

| File                            | Action     | Description                                                         |
| ------------------------------- | ---------- | ------------------------------------------------------------------- |
| `tango-bench/src/commpage.rs`   | **New**    | Shared memory commpage implementation (using `shared_memory` crate) |
| `tango-bench/src/worker.rs`     | **New**    | Child worker mode (JSON-RPC + commpage writer)                      |
| `tango-bench/src/child.rs`      | **New**    | Runner-side child process handle                                    |
| `tango-bench/src/cli.rs`        | **Modify** | Use ChildHandle instead of Spi for compare; solo stays in-process   |
| `tango-bench/src/lib.rs`        | **Modify** | Update `tango_main!` to detect worker mode                          |
| `tango-bench/src/dylib.rs`      | **Remove** | FFI vtable and `tango_*` exports no longer needed                   |
| `tango-bench/src/linux.rs`      | **Remove** | PIE patching no longer needed                                       |
| `tango-bench/src/windows.rs`    | **Remove** | IAT patching no longer needed                                       |
| `tango-bench/Cargo.toml`        | **Modify** | Add `shared_memory`, `serde`/`serde_json`; remove `libloading`      |
| `tango-test/tests/cli_tests.rs` | **Modify** | Add tests for new process-isolated mode                             |

## Design Decisions

1. **Fixed iteration count per benchmark run.** Dynamic iteration adjustment and `SamplerKind` (flat/linear/random) are removed. The system uses a fixed iteration count estimated by `estimate_iterations()` before the benchmark starts (equivalent to the current `SamplerFlat` behavior).

2. **State preparation is the benchmark's responsibility.** `prepare_state` is removed from the RPC protocol. Benchmarks that only read data can reuse the same state across iterations. Benchmarks that mutate data (e.g. sorting) must pre-create enough copies of the input to cover all iterations within a sample.

3. **Two modes for controlling benchmark duration:**
   - `--samples N`: R passes `num_samples = N`. Children run exactly N samples and return.
   - `-t DURATION`: R passes `num_samples = 0`. Children run indefinitely until R sets `stop_flag = 1` in the commpage when the time budget is exhausted.

4. **First-sample alignment is acceptable.** The barrier after sample 0 aligns subsequent samples. If the first sample turns out to be noisy, it can be discarded during analysis.

## Open Questions

1. **Spin-wait power/thermal impact?** C and B busy-spin waiting for the peer after each sample. On a system with few cores, two processes spinning could cause contention. Should there be a configurable fallback to `pause`/`_mm_pause` hints in the spin loops?
