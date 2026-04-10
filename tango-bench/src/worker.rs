//! Child worker mode: reads JSON-RPC commands from stdin, writes samples to commpage.

use crate::{
    commpage::{Commpage, CommpageError},
    protocol::{self, *},
    Benchmark, ErasedSampler,
};
use anyhow::{anyhow, bail, Result};
use jsonrpc_types::v2::*;
use serde::{de::DeserializeOwned, Serialize};
use std::io::{self, BufRead, BufReader, Write};

/// Entry point for a child process in worker mode.
///
/// Reads newline-delimited JSON-RPC 2.0 from stdin, writes responses to stdout.
/// Measurement data flows through the commpage, not through JSON-RPC.
pub fn run_worker() {
    let stdin = BufReader::new(io::stdin());
    let mut out = io::stdout().lock();
    let mut state = WorkerState::default();

    // increasing priority of a test thread, to minimize effect of CPU scheduler
    #[cfg(target_os = "macos")]
    unsafe {
        use libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE;
        libc::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }

    let mut lines = stdin.lines();
    while let Some(Ok(line)) = lines.next() {
        if line.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<MethodCall>(&line) {
            Ok(r) if r.method == protocol::METHOD_SHUTDOWN => break,
            Ok(r) => dispatch(&r, &mut state),
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
}

#[derive(Default)]
struct WorkerState {
    commpage: Option<Commpage>,
    role: Option<crate::commpage::Role>,
    benchmarks: Vec<Benchmark>,
    sampler: Option<Box<dyn ErasedSampler>>,
}

impl WorkerState {
    fn init(&mut self, params: InitParams) -> Result<InitResult> {
        match Commpage::open(&params.shmem_name) {
            Ok(cp) => {
                self.commpage = Some(cp);
                self.role = Some(params.role);
            }
            Err(CommpageError::Shmem(e)) => {
                bail!("Failed to open shared memory: {e}");
            }
            Err(e) => {
                bail!("Commpage error: {e}");
            }
        }

        let benchmarks =
            crate::take_benchmarks().ok_or_else(|| anyhow!("No benchmarks registered"))?;

        let names: Vec<String> = benchmarks.iter().map(|b| b.name().to_string()).collect();
        self.benchmarks = benchmarks;

        Ok(InitResult { benchmarks: names })
    }

    fn select(&mut self, params: SelectParams) -> Result<()> {
        if params.index >= self.benchmarks.len() {
            bail!(
                "Index {} out of range (have {} benchmarks)",
                params.index,
                self.benchmarks.len()
            );
        }

        let sampler = self.benchmarks[params.index].prepare_state(0);
        self.sampler = Some(sampler);
        Ok(())
    }

    fn estimate_iterations(
        &mut self,
        params: EstimateIterationsParams,
    ) -> Result<EstimateIterationsResult> {
        let sampler = self
            .sampler
            .as_mut()
            .ok_or_else(|| anyhow!("No benchmark selected (call select first)"))?;

        let iterations = sampler.estimate_iterations(params.time_ms);
        Ok(EstimateIterationsResult { iterations })
    }

    fn run_benchmark(&mut self, params: RunBenchmarkParams) -> Result<RunBenchmarkResult> {
        let cp = self
            .commpage
            .as_ref()
            .ok_or_else(|| anyhow!("Commpage not initialized"))?;
        let r = self.role.ok_or_else(|| anyhow!("Role not set"))?;
        let sampler = self
            .sampler
            .as_mut()
            .ok_or_else(|| anyhow!("No benchmark selected (call select first)"))?;

        let my_lane = cp.get_lane(r);
        let peer_lane = cp.peer_lane(r);

        let mut samples_written = 0;
        loop {
            // Check termination conditions
            if params.num_samples > 0 && samples_written >= params.num_samples as u64 {
                break;
            }
            if params.num_samples == 0 && cp.is_stopped() {
                break;
            }

            // Take measurement
            let elapsed_ns = sampler.measure(params.iterations);

            // Write sample to commpage
            my_lane.push_sample(samples_written, elapsed_ns);
            samples_written += 1;

            // Barrier: wait for peer
            if !peer_lane.wait_for_cursor(samples_written) {
                // Peer exited early
                break;
            }
        }

        my_lane.mark_done();

        Ok(RunBenchmarkResult { samples_written })
    }
}

fn dispatch(req: &MethodCall, state: &mut WorkerState) -> Output {
    match req.method.as_str() {
        protocol::METHOD_INIT => rpc_handle(req, |p| state.init(p)),
        protocol::METHOD_SELECT => rpc_handle(req, |p| state.select(p)),
        protocol::METHOD_ESTIMATE_ITERATIONS => rpc_handle(req, |p| state.estimate_iterations(p)),
        protocol::METHOD_RUN_BENCHMARK => rpc_handle(req, |p| state.run_benchmark(p)),
        _ => jsonrpc_error(&req.id, Error::method_not_found()),
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
