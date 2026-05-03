//! Child worker mode: reads JSON-RPC commands from stdin, writes samples to commpage.

use crate::{
    commpage::{Commpage, Role},
    protocol::{self, *},
    Benchmark,
};
use anyhow::{anyhow, Result};
use jsonrpc_types::v2::*;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    io::{self, BufRead, BufReader, Write},
    process::ExitCode,
};

/// Entry point for a child process in worker mode.
///
/// Opens the commpage from the given shmem name, then enters the JSON-RPC
/// dispatch loop reading commands from stdin and writing responses to stdout.
pub(crate) fn run_worker(
    shmem_name: &str,
    role: Role,
    benchmarks: Vec<Benchmark>,
) -> Result<ExitCode> {
    let commpage = Commpage::open(shmem_name)?;

    let mut state = WorkerState {
        benchmarks,
        commpage,
        role,
    };

    let stdin = BufReader::new(io::stdin());
    let mut out = io::stdout().lock();

    // increasing priority of a test thread, to minimize effect of CPU scheduler
    #[cfg(target_os = "macos")]
    unsafe {
        use libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE;
        libc::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }

    let mut lines = stdin.lines();
    while let Some(Ok(line)) = lines.next() {
        let response = match serde_json::from_str::<MethodCall>(&line) {
            Ok(r) => match r.method.as_str() {
                protocol::METHOD_LIST_BENCHMARKS => rpc_handle(&r, |()| state.list_benchmarks()),
                protocol::METHOD_ESTIMATE_ITERATIONS => {
                    rpc_handle(&r, |p| state.estimate_iterations(p))
                }
                protocol::METHOD_RUN_BENCHMARK => rpc_handle(&r, |p| state.run_benchmark(p)),
                protocol::METHOD_SHUTDOWN => break,
                _ => jsonrpc_error(&r.id, Error::method_not_found()),
            },
            Err(e) => Output::<Value>::failure(
                Error {
                    code: ErrorCode::ParseError,
                    message: format!("Parse error: {e}"),
                    data: None,
                },
                None,
            ),
        };
        let _ = writeln!(out, "{}", serde_json::to_string(&response).unwrap());
        let _ = out.flush();
    }
    Ok(ExitCode::SUCCESS)
}

struct WorkerState {
    commpage: Commpage,
    role: Role,
    benchmarks: Vec<Benchmark>,
}

impl WorkerState {
    fn list_benchmarks(&self) -> Result<ListBenchmarksResult> {
        let benchmarks = self
            .benchmarks
            .iter()
            .map(|b| b.name().to_string())
            .collect();
        Ok(ListBenchmarksResult { benchmarks })
    }

    fn prepare_sampler(
        &mut self,
        index: usize,
        seed: u64,
    ) -> Result<Box<dyn crate::ErasedSampler>> {
        let len = self.benchmarks.len();
        let bench = self
            .benchmarks
            .get_mut(index)
            .ok_or_else(|| anyhow!("Index {index} out of range (have {len} benchmarks)"))?;
        Ok(bench.prepare_state(seed))
    }

    fn estimate_iterations(
        &mut self,
        params: EstimateIterationsParams,
    ) -> Result<EstimateIterationsResult> {
        let mut sampler = self.prepare_sampler(params.index, params.seed)?;
        let iterations = sampler.estimate_iterations(params.time_ms);
        Ok(EstimateIterationsResult { iterations })
    }

    fn run_benchmark(&mut self, params: RunBenchmarkParams) -> Result<RunBenchmarkResult> {
        let mut sampler = self.prepare_sampler(params.index, params.seed)?;
        let mut samples = if params.num_samples > 0 {
            Vec::with_capacity(params.num_samples)
        } else {
            Vec::new()
        };

        let mut sample_no = 0u64;

        #[cfg(feature = "stack-randomize")]
        let mut stack_randomizer = stack_randomizer::StackRandomizer::new(params.seed);

        // We use value 0 to self synchronize two processes at the start of benchmark execution.
        // So cursor value is a sample index being collected right now. Hence the number
        // of collected samples is cursor_value - 1;

        if cfg!(not(feature = "no-sync")) {
            self.commpage.advance_cursor(self.role, sample_no + 1);
            self.commpage
                .wait_for_cursor_value(self.role.peer(), sample_no + 1);
        }

        // Terminate conditions differs if the explicit number of samples given.
        // Yes – collect given amount
        //  No – run until STOP bit is not set
        while (params.num_samples > 0 && sample_no < params.num_samples as u64)
            || (params.num_samples == 0 && !self.commpage.is_stopped())
        {
            #[cfg(feature = "stack-randomize")]
            let elapsed_ns = stack_randomizer.measure(&mut *sampler, params.iterations);
            #[cfg(not(feature = "stack-randomize"))]
            let elapsed_ns = sampler.measure(params.iterations);

            samples.push(elapsed_ns);

            // Advance cursor and wait for peer
            sample_no += 1;
            if cfg!(not(feature = "no-sync")) {
                self.commpage.advance_cursor(self.role, sample_no + 1);
                if !self
                    .commpage
                    .wait_for_cursor_value(self.role.peer(), sample_no + 1)
                {
                    // Peer exited early
                    break;
                }
            }
        }

        self.commpage.mark_done(self.role);

        Ok(RunBenchmarkResult { samples })
    }
}

fn rpc_handle<P: DeserializeOwned, R: Serialize>(
    req: &MethodCall,
    f: impl FnOnce(P) -> Result<R>,
) -> Output {
    let params_value = params_to_value(&req.params);
    match serde_json::from_value::<P>(params_value) {
        Ok(p) => match f(p) {
            Ok(r) => jsonrpc_success(&req.id, r),
            Err(e) => server_error(&req.id, &e.to_string()),
        },
        Err(e) => jsonrpc_error(&req.id, Error::invalid_params(e)),
    }
}

fn jsonrpc_success<T: Serialize>(id: &Id, result: T) -> Output {
    let result = serde_json::to_value(result).unwrap();
    Output::success(result, id.clone())
}

fn jsonrpc_error(id: &Id, error: Error) -> Output {
    Output::failure(error, Some(id.clone()))
}

fn server_error(id: &Id, message: &str) -> Output {
    jsonrpc_error(
        id,
        Error {
            code: ErrorCode::ServerError(-32000),
            message: message.to_string(),
            data: None,
        },
    )
}

fn params_to_value(params: &Option<Params>) -> Value {
    params
        .as_ref()
        .map(|p| serde_json::to_value(p).unwrap_or_default())
        .unwrap_or_default()
}

#[cfg(feature = "stack-randomize")]
mod stack_randomizer {
    use crate::ErasedSampler;
    use alloca::with_alloca;
    use rand::{distributions, rngs::SmallRng, Rng, SeedableRng};

    pub struct StackRandomizer {
        rng: SmallRng,
        distr: distributions::Uniform<usize>,
    }

    impl StackRandomizer {
        pub fn new(seed: u64) -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seed),
                distr: rand::distributions::Uniform::new(0, 64),
            }
        }

        pub fn measure(&mut self, sampler: &mut dyn ErasedSampler, iterations: usize) -> u64 {
            with_alloca(self.rng.sample(self.distr), |_| sampler.measure(iterations))
        }
    }
}
