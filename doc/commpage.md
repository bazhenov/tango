# Commpage: Lock-Step Synchronization via Shared Memory

## Motivation

Tango runs the baseline (B) and candidate (C) benchmarks in **separate processes** for full isolation (distinct address spaces, independent OS-level metrics). The two children must start each sample at nearly the same instant so they experience identical system conditions. A shared memory region -- the "commpage" -- provides a low-latency, zero-syscall synchronization barrier between B and C.

All **measurement data** (sample timings) flows back to the runner through JSON-RPC responses. The commpage carries **no measurement data** -- it exists solely so that two children can execute in lock-step.

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
                        │  Receives samples via    │
                        │    JSON-RPC responses    │
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
                 │  Writes cursor  │  │  Writes cursor  │
                 │  Reads B cursor │  │  Reads C cursor │
                 └────────┬────────┘  └────────┬────────┘
                          │                    │
                          ▼                    ▼
                 ┌─────────────────────────────────────┐
                 │           COMMPAGE (shm)            │
                 │   cursor_c (AtomicU64)              │
                 │   cursor_b (AtomicU64)              │
                 └─────────────────────────────────────┘
```

### Roles

| Role              | Process                              | Description                                                                                                                                                                                     |
| ----------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Runner (R)**    | The initial process the user invokes | Spawns C and B, sends commands via JSON-RPC over stdio, receives samples in JSON-RPC responses, computes statistics, prints results. Passive during measurement -- does not touch the commpage. |
| **Candidate (C)** | Spawned by R (copy of itself)        | Advances `cursor_c`, reads `cursor_b` to stay in lock-step. Returns samples to R via JSON-RPC.                                                                                                  |
| **Baseline (B)**  | Spawned by R (the second executable) | Advances `cursor_b`, reads `cursor_c` to stay in lock-step. Returns samples to R via JSON-RPC.                                                                                                  |

## Commpage Layout

The commpage is a single shared memory region mapped into all three processes. It contains only the header and two cursors -- no sample data.

```
Offset  Size    Field                Description
──────  ────    ─────                ───────────
0x000   8       magic                0x54414E47_4F434D50 ("TANGOCMP")
0x008   4       version              Commpage protocol version (2)
0x00C   4       flags                AtomicU32: bit 0 = STOP flag
0x010   8       cursor_c             AtomicU64: C's monotonic sample counter
0x018   8       cursor_b             AtomicU64: B's monotonic sample counter
```

Total size: **32 bytes**. Fits in a single cache line on most architectures.

### Cursor Encoding

Each cursor is a monotonically increasing counter -- the total number of samples completed by that child. Bit 63 is the `DONE` flag, set when a child exits the measurement loop.

```
  ┌────┬──────────────────────────────────────────┐
  │done│         sample count (bits 0..62)        │
  │ 63 │                                          │
  └────┴──────────────────────────────────────────┘
```

### Peer Synchronization

C and B synchronize directly with each other through the commpage -- **R does not participate**. After completing sample N, each child advances its own cursor and then spin-waits until the peer's cursor reaches N. This acts as a per-sample barrier: neither child starts sample N+1 until both have completed sample N.

The first sample may start slightly offset (one child receives the RPC before the other), but the barrier after sample 0 aligns them. From sample 1 onward, both children start each measurement nearly simultaneously, under the same system conditions.

The spin-wait also checks the peer's DONE bit to avoid spinning forever if the peer exits early.

### Duration Modes

- **`--samples N`**: R passes `num_samples = N`. Children run exactly N samples and return.
- **`-t DURATION`**: R passes `num_samples = 0`. Children run until R sets the `STOP` flag in the commpage when the time budget is exhausted.

## Child Spawning

R spawns each child as a hidden `__worker` clap subcommand with two required flags:

```
./candidate __worker --shmem /tango-XXXX --role candidate
./baseline __worker --shmem /tango-XXXX --role baseline
```

The child opens the commpage on startup and enters the JSON-RPC dispatch loop. R discovers available benchmarks by sending a `list_benchmarks` request.

Solo mode stays in-process -- it does not use the commpage or child processes. This keeps profiling with tools like `perf` and Instruments straightforward.

## JSON-RPC Protocol (over stdio)

Communication between R and C/B uses newline-delimited JSON-RPC 2.0 over the child's stdin/stdout. This carries both **control messages** and **measurement results**.

```jsonc
// List available benchmarks
{"jsonrpc":"2.0","method":"list_benchmarks","params":null,"id":1}
// Response: {"jsonrpc":"2.0","result":{"benchmarks":["bench_a","bench_b"]},"id":1}

// Estimate iterations for a given time budget
{"jsonrpc":"2.0","method":"estimate_iterations","params":{"index":0,"time_ms":10,"seed":42},"id":2}
// Response: {"jsonrpc":"2.0","result":{"iterations":50000},"id":2}

// Run measurement loop; child synchronizes with peer via commpage,
// then returns all samples in the response
{"jsonrpc":"2.0","method":"run_benchmark","params":{"index":0,"seed":42,"iterations":1000,"num_samples":300},"id":3}
// Response: {"jsonrpc":"2.0","result":{"samples":[48201,48150,48180,...]},"id":3}

// Shutdown gracefully
{"jsonrpc":"2.0","method":"shutdown","id":4}
```

### Why JSON-RPC + commpage?

- **JSON-RPC for control and data**: Flexible, debuggable, easy to evolve. The samples array is returned once after the measurement loop completes, so serialization cost is a one-time expense per benchmark run.
- **Commpage for synchronization**: The per-sample barrier between B and C must be zero-syscall and sub-microsecond. Atomic spin-waits on shared memory are the lowest-latency mechanism available. Using JSON-RPC (or any pipe-based IPC) for this would add unacceptable latency noise to every sample.

## Key Design Decisions

1. **Fixed iteration count per benchmark run.** The iteration count is estimated once by `estimate_iterations()` before the benchmark starts, then held constant for all samples in that run.

2. **Each RPC call prepares its own state.** Both `estimate_iterations` and `run_benchmark` accept a benchmark `index` and `seed`, and create a fresh sampler internally. There is no persistent "selected benchmark" state on the worker.

3. **Solo mode stays in-process.** It is designed for use with system profilers where an extra process boundary would complicate profiling and debugging.

4. **No measurement data in the commpage.** Samples are returned in the JSON-RPC response after the measurement loop finishes. This eliminates the ring buffer, drain logic, missed-sample handling, and any need for R to read the commpage during measurement.

## Source Files

| File                          | Description                                              |
| ----------------------------- | -------------------------------------------------------- |
| `tango-bench/src/commpage.rs` | Shared memory layout, cursor operations, synchronization |
| `tango-bench/src/worker.rs`   | Child worker mode (JSON-RPC dispatch + commpage sync)    |
| `tango-bench/src/child.rs`    | Runner-side child process handle                         |
| `tango-bench/src/protocol.rs` | Shared method names and param/result types               |
| `tango-bench/src/cli.rs`      | CLI entry point, compare and solo test orchestration     |
