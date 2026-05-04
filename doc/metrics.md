# Metrics

## Overview

Tango-bench collects two kinds of measurements during a benchmark run:

1. **Main metric (wall-clock time)** -- the primary measurement, collected per sample.
2. **Auxiliary metrics (cpu_sys, cpu_usr)** -- supplementary resource-consumption measurements, collected once per entire benchmark run.

## Main Metric: Wall-Clock Time

The main metric is implemented via the `Metric` trait (`WallClock`). It wraps each iteration batch in a timing call.

Wall-clock time is measured **per sample** -- each iteration batch produces one `u64` value pushed into a `Vec<u64>`. These samples are then used for full statistical analysis (medians, confidence intervals, etc.).

## Auxiliary Metrics: CPU Time

Auxiliary metrics are defined by the `AuxMetricEntry` struct:

```rust
pub struct AuxMetricEntry {
    pub id: &'static str,       // e.g. "cpu_sys", "cpu_usr"
    pub start: fn() -> u64,     // snapshot before the run
    pub finish: fn(u64) -> u64, // delta after the run
}
```

Two auxiliary metrics are currently available:

| ID        | What it measures                               |
| --------- | ---------------------------------------------- |
| `cpu_sys` | Kernel/system CPU time (via OS resource usage) |
| `cpu_usr` | User-space CPU time (via OS resource usage)    |

## Key Differences in Collection

| Aspect               | Main metric (time)                        | Auxiliary metrics (cpu_sys/cpu_usr)                    |
| -------------------- | ----------------------------------------- | ------------------------------------------------------ |
| **Granularity**      | Per sample (per iteration batch)          | Single aggregate for the entire run                    |
| **Collection point** | Inside the sampling loop                  | Outside -- `start()` before the loop, `finish()` after |
| **Source**           | Wall-clock (fixed)                        | can be extended                                        |
| **Used for**         | Statistical analysis of benchmark results | Diagnostics (e.g. detecting high system-time bias)     |

Auxiliary metrics are reported in verbose mode and are used to warn the user when system CPU time exceeds sane proportion of wall time, which can indicate that OS overhead is distorting benchmark results.
