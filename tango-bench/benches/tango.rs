use std::{cell::UnsafeCell, hint::black_box, thread};

use rand::{distributions::Standard, rngs::SmallRng, Rng, SeedableRng};
use tango_bench::{
    benchmark_fn,
    commpage::{Commpage, Role},
    iqr_variance_thresholds, tango_benchmarks, IntoBenchmarks, Summary,
};

/// Wrapper to share `Commpage` across threads. This is intentionally unsafe —
/// the benchmark tests the shared-memory synchronization built into Commpage.
struct SharedCommpage(UnsafeCell<Commpage>);
unsafe impl Send for SharedCommpage {}
unsafe impl Sync for SharedCommpage {}

fn benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("summary", move |b| {
            let rnd = SmallRng::seed_from_u64(b.seed);
            let input: Vec<i64> = rnd.sample_iter(Standard).take(1000).collect();
            b.iter(move || Summary::from(&input))
        }),
        benchmark_fn("iqr", move |b| {
            let rnd = SmallRng::seed_from_u64(b.seed);
            let input: Vec<f64> = rnd.sample_iter(Standard).take(1000).collect();
            b.iter(move || iqr_variance_thresholds(input.clone()))
        }),
        benchmark_fn("measure_empty_function", move |p| {
            let mut bench = benchmark_fn("_", |b| b.iter(|| 42));
            let mut state = bench.prepare_state(p.seed);
            p.iter(move || state.measure(1))
        }),
        benchmark_fn("advance_cursor", move |p| {
            const N: u64 = 1000000;
            p.iter(move || {
                let comm = SharedCommpage(UnsafeCell::new(
                    Commpage::create().expect("Unable to create commpage"),
                ));
                let comm = &comm;

                thread::scope(|s| {
                    s.spawn(|| {
                        advance_commpage(
                            unsafe { &mut *comm.0.get() },
                            Role::Candidate,
                            black_box(N),
                        );
                    });
                    advance_commpage(unsafe { &mut *comm.0.get() }, Role::Baseline, black_box(N));
                });
            })
        }),
    ]
}

fn advance_commpage(commpage: &mut Commpage, role: Role, pos: u64) {
    for i in 0..pos {
        commpage.set_cursor_value(role, i);
    }
}

tango_benchmarks!(benchmarks());
