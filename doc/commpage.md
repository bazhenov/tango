# Commpage: Process-Isolated Benchmarking via Shared Memory

## Motivation

Tango previously loaded both the baseline (B) and candidate (C) benchmarks into **the same process** using `dlopen`/`LoadLibrary`. This had drawbacks:

1. **Platform hacks** -- Linux PIE patching (`linux.rs`) and Windows IAT patching (`windows.rs`) existed solely to make executables loadable as shared libraries. These were fragile and not guaranteed to keep working. On Linux this approach was explicitly prohibited by dyld.
2. **Insufficient isolation** -- OS-level per-process metrics (memory allocations, page faults, etc.) could not distinguish between the two benchmarks sharing a process.

Running each benchmark in its **own process** and reading measurements through shared memory (the "commpage") solves both problems.

## Architecture

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
| **Candidate (C)** | Spawned by R (copy of itself)        | Writes samples to lane C, reads B's write_cursor to stay in lock-step.                                                                                                                                |
| **Baseline (B)**  | Spawned by R (the second executable) | Writes samples to lane B, reads C's write_cursor to stay in lock-step.                                                                                                                                |

## Commpage

A **commpage** is a single shared memory region accessible to all three processes (R, C, B). It contains two **lanes** -- one for the candidate, one for the baseline. Each child writes to its own lane and reads the peer's write_cursor to synchronize. R only reads.

### Memory Layout

Each lane contains N = 128 sample slots (N must be a power of two). The two lanes are laid out contiguously.

```
Offset  Size    Field                Description
──────  ────    ─────                ───────────
0x000   8       magic                0x54414E47_4F434D50 ("TANGOCMP")
0x008   4       version              Commpage protocol version (1)
0x00C   4       flags                AtomicU32: bit 0 = STOP flag

Lane C (Candidate):
0x010   8       write_cursor_c       AtomicU64: C's monotonic sample counter
0x018   N*8     samples_c[N]         C's ring buffer (128 * 8 = 1024 bytes)

Lane B (Baseline):
0x418   8       write_cursor_b       AtomicU64: B's monotonic sample counter
0x420   N*8     samples_b[N]         B's ring buffer (128 * 8 = 1024 bytes)
```

Total size: 16 + 2 \* (8 + 1024) = **2080 bytes**. Fits in a single page.

### Write Cursor Encoding

Each `write_cursor` is a monotonically increasing counter -- the total number of samples written by that child. The ring buffer slot index is `write_cursor & (N - 1)` (low 7 bits). Bit 63 is the `DONE` flag, set when a child exits the measurement loop.

```
  ┌────┬───────────────────────────┬────────────┐
  │done│  sample count (bits 7..62)│ slot (0..6)│
  │ 63 │                           │            │
  └────┴───────────────────────────┴────────────┘
```

### Samples

Each sample slot is a `u64` holding total elapsed nanoseconds. The iteration count is fixed for the entire benchmark run (set via `run_benchmark(iterations=...)`), so it does not need to be stored per sample.

### Peer Synchronization

C and B synchronize directly with each other through the commpage -- **R does not participate**. After writing sample N, each child spin-waits until the peer's write_cursor reaches N. This acts as a per-sample barrier: neither child starts sample N+1 until both have completed sample N.

The first sample may start slightly offset (one child receives the RPC before the other), but the barrier after sample 0 aligns them. From sample 1 onward, both children start each measurement nearly simultaneously, under the same system conditions.

The spin-wait also checks the peer's DONE bit to avoid spinning forever if the peer exits early.

### Duration Modes

- **`--samples N`**: R passes `num_samples = N`. Children run exactly N samples and return.
- **`-t DURATION`**: R passes `num_samples = 0`. Children run until R sets the `STOP` flag in the commpage when the time budget is exhausted.

### Ring Buffer Overwrite Protection

If R cannot drain samples fast enough (e.g. under heavy load or with very fast benchmarks), the ring buffer may wrap around and overwrite unread slots. `drain_samples()` detects this: it fills the gap with a sentinel value `MISSED_SAMPLE` (`u64::MAX`) and returns `Err(skipped)` with the number of overwritten samples. R then:

1. Emits a warning showing the number of skipped observations per child.
2. Filters out any sample pair where either side is `MISSED_SAMPLE` before computing statistics.

This guarantees that results are never silently corrupted by stale ring-buffer data.

### Capacity

N = **128 samples per lane** (1 KiB per lane). At ~10ms per sample, 128 slots represent ~1.3 seconds of data. R drains samples in the background while children are running.

## Child Spawning

R spawns each child as a hidden `__worker` clap subcommand with two required flags:

```
./candidate __worker --shmem /tango-XXXX --role candidate
./baseline __worker --shmem /tango-XXXX --role baseline
```

The child opens the commpage on startup and enters the JSON-RPC dispatch loop. R discovers available benchmarks by sending a `list_benchmarks` request.

Solo mode stays in-process -- it does not use the commpage or child processes. This keeps profiling with tools like `perf` and Instruments straightforward.

## JSON-RPC Protocol (over stdio)

Communication between R and C/B uses newline-delimited JSON-RPC 2.0 over the child's stdin/stdout. This carries **control messages only** -- measurement data flows through the commpage.

```jsonc
// List available benchmarks
{"jsonrpc":"2.0","method":"list_benchmarks","params":null,"id":1}
// Response: {"jsonrpc":"2.0","result":{"benchmarks":["bench_a","bench_b"]},"id":1}

// Estimate iterations for a given time budget
{"jsonrpc":"2.0","method":"estimate_iterations","params":{"index":0,"time_ms":10,"seed":42},"id":2}
// Response: {"jsonrpc":"2.0","result":{"iterations":50000},"id":2}

// Run measurement loop; child writes samples to commpage, synchronizes with peer
{"jsonrpc":"2.0","method":"run_benchmark","params":{"index":0,"seed":42,"iterations":1000,"num_samples":300},"id":3}
// Response (after loop exits): {"jsonrpc":"2.0","result":{"samples_written":300},"id":3}

// Shutdown gracefully
{"jsonrpc":"2.0","method":"shutdown","id":4}
```

### Why JSON-RPC + commpage?

- **JSON-RPC for control**: Flexible, debuggable, easy to evolve. Control messages are infrequent so serialization cost is irrelevant.
- **Commpage for data**: Zero-copy, zero-syscall measurement reporting. The child writes samples without waiting for an acknowledgement. Any IPC synchronization on the measurement path would add latency noise.

## Key Design Decisions

1. **Fixed iteration count per benchmark run.** The iteration count is estimated once by `estimate_iterations()` before the benchmark starts, then held constant for all samples in that run.

2. **Each RPC call prepares its own state.** Both `estimate_iterations` and `run_benchmark` accept a benchmark `index` and `seed`, and create a fresh sampler internally. There is no persistent "selected benchmark" state on the worker.

3. **Solo mode stays in-process.** It is designed for use with system profilers where an extra process boundary would complicate profiling and debugging.

## Source Files

| File                          | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| `tango-bench/src/commpage.rs` | Shared memory layout, lane operations, synchronization  |
| `tango-bench/src/worker.rs`   | Child worker mode (JSON-RPC dispatch + commpage writer) |
| `tango-bench/src/child.rs`    | Runner-side child process handle                        |
| `tango-bench/src/protocol.rs` | Shared method names and param/result types              |
| `tango-bench/src/cli.rs`      | CLI entry point, compare and solo test orchestration    |
