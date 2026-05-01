# Auxiliary Metrics Implementation Plan

## Context

Tango.rs currently records only a single u64 (elapsed nanoseconds) per sample in its shared-memory commpage. This document proposes allowing children to expose additional measurements (CPU
system time, CPU user time, etc.) alongside the primary time metric. The Runner discovers available aux metrics, requests specific ones by code, and children write [Time, aux_1, ..., aux_n] per sample into
the same lane.

## Motivation

Auxialary metrics allows to gather more information per each sample that is usually require for one of several reasons

- gathering information that allows to veto main sample (things like system/user cpu time proportion)
- additional information like memory allocation etc.

---

Approach: Stride-Based Packing in Existing Ring Buffer

Each logical sample occupies stride = 1 + num_aux_metrics consecutive u64 slots. The write_cursor counts stride – amount of slots taken by the sample. With stride=2, effective capacity drops from 128 to 64 samples -- acceptable
given the 50ms drain interval. No commpage layout or version changes needed.

---

Implementation Steps

Phase 1: Protocol & Data Types (protocol.rs, lib.rs)

tango-bench/src/protocol.rs:

- Add AuxMetricInfo { display_name: String, code: String }
- Add METHOD_LIST_AUX_METRICS = "list_aux_metrics" + request/result types
- Add aux_metrics: Vec<String> to RunBenchmarkParams with #[serde(default)]

tango-bench/src/lib.rs:

- Define AuxMetric trait: code() -> &str, display_name() -> &str
- Add pub mod aux_metrics with built-in impls: CpuSystem, CpuUser (reuse platform-specific code from existing CpuTime metric in platform.rs)
- Add builtin_aux_metrics() -> Vec<Box<dyn AuxMetric>>
- Add default measure_with_aux(&mut self, iterations, &[&dyn AuxMetric]) -> SmallVec/array on ErasedSampler -- reads counters before/after self.measure()

Phase 2: Commpage Stride Support (commpage.rs)

- Add push_sample_group(&self, sample_no: u64, values: &[u64]) -- writes stride u64s at (sample_no \* stride + i) & (NUM_SLOTS - 1) for each value, then advances write_cursor
- Keep existing push_sample as push_sample_group(n, &[v]) wrapper
- Add unit tests for stride 2, 3, 4 including wrap-around and skip detection

Phase 3: Worker (worker.rs)

- Add list_aux_metrics RPC handler -- returns builtin_aux_metrics() info
- Modify run_benchmark: resolve params.aux_metrics codes to AuxMetric instances, compute stride
- Fast path: when aux_metrics is empty, keep original tight loop (measure + push_sample)
- Aux path: use pre-allocated buffer, call measure_with_aux, write via push_sample_group
- Return stride in RunBenchmarkResult

Phase 4: Child Handle (child.rs)

- Add list_aux_metrics(index) -> Vec<AuxMetricInfo>
- Modify start_benchmark to accept aux_metrics: Vec<String>
- Modify drain_samples to accept stride: usize parameter

Phase 5: CLI & Reporting (cli.rs)

- Add --aux-metrics <codes> flag to PairedOpts (comma-separated, e.g. "cpu_system,cpu_user")
- After list_benchmarks, call list_aux_metrics on both children; validate requested codes exist in both
- Pass aux metric codes to start_benchmark; compute stride = 1 + len
- Drain with stride; reshape flat Vec<u64> into per-metric channels:
  - Time: samples[i * stride]
  - Aux metric k: samples[i * stride + 1 + k]
- Primary calculate_run_result operates on time pairs only (unchanged)
- For each aux metric, compute a Summary for baseline and candidate; display in reporters
- DefaultReporter: one extra line per aux metric showing baseline/candidate means
- VerboseReporter: full table per aux metric
- CSV dump: extend columns baseline_time,candidate_time,iterations,aux_1_b,aux_1_c,...

Phase 6: Tests

- commpage.rs: stride-aware push/drain, wrap-around, skip detection
- protocol.rs: serde backward compat (missing aux_metrics field deserializes to empty vec)
- Integration (tango-test/tests/cli_tests.rs): end-to-end test with --aux-metrics cpu_user

---

Backward Compatibility

- Old child + new runner: aux_metrics absent in params -> #[serde(default)] -> empty vec -> stride=1
- New child + old runner: no aux_metrics sent -> empty vec -> stride=1; stride missing in response -> default 1
- No commpage version bump needed

Performance Considerations

- When no aux metrics requested, the hot loop is unchanged (fast path)
- Aux counter reads (read_counter) happen outside the primary timing window, adding no noise to time
- Pre-allocate a stack buffer for measure_with_aux to avoid per-sample heap allocation
- Effective ring buffer capacity: 128/stride logical samples (64 for stride=2, 42 for stride=3)

Files to Modify

┌─────────────────────────────┬────────────────────────────────────────────────────────────────┐
│ File │ Changes │
├─────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ tango-bench/src/protocol.rs │ New types, extended params/result │
├─────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ tango-bench/src/lib.rs │ AuxMetric trait, built-ins, measure_with_aux │
├─────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ tango-bench/src/commpage.rs │ push_sample_group, drain_samples_strided │
├─────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ tango-bench/src/worker.rs │ list_aux_metrics handler, strided measurement loop │
├─────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ tango-bench/src/child.rs │ list_aux_metrics, updated start_benchmark/drain_samples │
├─────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ tango-bench/src/cli.rs │ --aux-metrics flag, stride-aware drain, aux metric reporting │
├─────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ tango-bench/src/platform.rs │ Per-thread kernel/user time helpers (for built-in aux metrics) │
└─────────────────────────────┴────────────────────────────────────────────────────────────────┘

Verification

1. cargo test -p tango-bench -- unit tests pass (commpage stride, serde compat)
2. cargo test -p tango-test -- integration tests pass
3. Manual: build sleep_10 example, run ./sleep_10 compare --aux-metrics cpu_user and verify aux metric columns appear in output
4. Manual: run old binary against new binary to verify backward compat (no crash, stride=1 behavior)
