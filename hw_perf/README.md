# hw_perf

macOS hardware CPU counter profiler. Analog of a `perf stat` from Linux. Based on a [prototype](https://gist.github.com/ibireme/173517c208c7dc333ba962c1f0d67d12) created by [ibireme](https://github.com/ibireme).

# Building from source

```console
$ git clone https://github.com/bazhenov/hw_perf
$ cd hw_perf
$ make
```

# Usage

Tracing requires root privileges.

1. profiling target process till the end:
   ```console
   # hw_perf -- target_process
   ```
1. profiling `WindowServer` for a 5 seconds:
   ```console
   # hw_perf -p `pgrep WindowServer` -- sleep 5
   loaded db: icelake (Intel Ice Lake)
   number of fixed counters: 4
   number of configurable counters: 8
   CPU tick frequency: 1000000000

     Perfomance counters stats for 'sleep 5'

     thread: 876

           62351 cycles         # CPU_CLK_UNHALTED.THREAD
           17467 instructions   # INST_RETIRED.ANY
            3505 branches       # BR_INST_RETIRED.ALL_BRANCHES
             265 branch-misses  # BR_MISP_RETIRED.ALL_BRANCHES
        0.001268 seconds elapsed

     thread: 2286

          223492 cycles         # CPU_CLK_UNHALTED.THREAD
          109630 instructions   # INST_RETIRED.ANY
           21596 branches       # BR_INST_RETIRED.ALL_BRANCHES
            1065 branch-misses  # BR_MISP_RETIRED.ALL_BRANCHES
        0.000881 seconds elapsed
   ```
