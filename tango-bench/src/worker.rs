//! Child worker mode: reads JSON-RPC commands from stdin, writes samples to commpage.

use crate::commpage::{Commpage, CommpageError};
use crate::protocol::{self, *};
use crate::{Benchmark, ErasedSampler};
use jsonrpc_types::v2::*;
use serde::Serialize;
use std::io::{self, BufRead, BufReader, Write};

/// Entry point for a child process in worker mode.
///
/// Reads newline-delimited JSON-RPC 2.0 from stdin, writes responses to stdout.
/// Measurement data flows through the commpage, not through JSON-RPC.
pub fn run_worker() {
    let stdin = BufReader::new(io::stdin());
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut state = WorkerState::default();

    #[cfg(target_os = "macos")]
    unsafe {
        use libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE;
        libc::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }

    for line in stdin.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.is_empty() {
            continue;
        }

        let req: MethodCall = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = Output::<Value>::failure(
                    Error {
                        code: ErrorCode::ParseError,
                        message: format!("Parse error: {e}"),
                        data: None,
                    },
                    None,
                );
                let _ = writeln!(out, "{}", serde_json::to_string(&resp).unwrap());
                let _ = out.flush();
                continue;
            }
        };

        let is_shutdown = req.method == protocol::method::SHUTDOWN;
        let resp = dispatch(&req, &mut state);
        let _ = writeln!(out, "{}", serde_json::to_string(&resp).unwrap());
        let _ = out.flush();

        if is_shutdown {
            break;
        }
    }
}

#[derive(Default)]
struct WorkerState {
    commpage: Option<Commpage>,
    role: Option<crate::commpage::Role>,
    benchmarks: Vec<Benchmark>,
    sampler: Option<Box<dyn ErasedSampler>>,
}

fn dispatch(req: &MethodCall, state: &mut WorkerState) -> Output {
    match req.method.as_str() {
        protocol::method::INIT => handle_init(req, state),
        protocol::method::SELECT => handle_select(req, state),
        protocol::method::ESTIMATE_ITERATIONS => handle_estimate_iterations(req, state),
        protocol::method::RUN_BENCHMARK => handle_run_benchmark(req, state),
        protocol::method::SHUTDOWN => jsonrpc_success(&req.id, Value::Null),
        _ => jsonrpc_error(&req.id, Error::method_not_found()),
    }
}

fn handle_init(req: &MethodCall, state: &mut WorkerState) -> Output {
    let params: InitParams = match serde_json::from_value(params_to_value(&req.params)) {
        Ok(p) => p,
        Err(e) => return jsonrpc_error(&req.id, Error::invalid_params(e)),
    };

    match Commpage::open(&params.shmem_name) {
        Ok(cp) => {
            state.commpage = Some(cp);
            state.role = Some(params.role);
        }
        Err(CommpageError::Shmem(e)) => {
            return server_error(&req.id, &format!("Failed to open shared memory: {e}"));
        }
        Err(e) => {
            return server_error(&req.id, &format!("Commpage error: {e}"));
        }
    }

    let benchmarks = match crate::take_benchmarks() {
        Some(b) => b,
        None => {
            return server_error(&req.id, "No benchmarks registered");
        }
    };

    let names: Vec<String> = benchmarks.iter().map(|b| b.name().to_string()).collect();
    state.benchmarks = benchmarks;

    jsonrpc_success(&req.id, InitResult { benchmarks: names })
}

fn handle_select(req: &MethodCall, state: &mut WorkerState) -> Output {
    let params: SelectParams = match serde_json::from_value(params_to_value(&req.params)) {
        Ok(p) => p,
        Err(e) => return jsonrpc_error(&req.id, Error::invalid_params(e)),
    };

    if params.index >= state.benchmarks.len() {
        return jsonrpc_error(
            &req.id,
            Error::invalid_params(format!(
                "Index {} out of range (have {} benchmarks)",
                params.index,
                state.benchmarks.len()
            )),
        );
    }

    // Create a sampler (prepares state) for the selected benchmark
    let sampler = state.benchmarks[params.index].prepare_state(0);
    state.sampler = Some(sampler);
    jsonrpc_success(&req.id, Value::Null)
}

fn handle_estimate_iterations(req: &MethodCall, state: &mut WorkerState) -> Output {
    let params: EstimateIterationsParams =
        match serde_json::from_value(params_to_value(&req.params)) {
            Ok(p) => p,
            Err(e) => return jsonrpc_error(&req.id, Error::invalid_params(e)),
        };

    let sampler = match &mut state.sampler {
        Some(s) => s,
        None => return server_error(&req.id, "No benchmark selected (call select first)"),
    };

    let iterations = sampler.estimate_iterations(params.time_ms);
    jsonrpc_success(&req.id, EstimateIterationsResult { iterations })
}

fn handle_run_benchmark(req: &MethodCall, state: &mut WorkerState) -> Output {
    let params: RunBenchmarkParams = match serde_json::from_value(params_to_value(&req.params)) {
        Ok(p) => p,
        Err(e) => return jsonrpc_error(&req.id, Error::invalid_params(e)),
    };

    let cp = match &state.commpage {
        Some(c) => c,
        None => return server_error(&req.id, "Commpage not initialized"),
    };
    let r = match state.role {
        Some(r) => r,
        None => return server_error(&req.id, "Role not set"),
    };

    let sampler = match &mut state.sampler {
        Some(s) => s,
        None => return server_error(&req.id, "No benchmark selected (call select first)"),
    };

    let my_lane = cp.my_lane(r);
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

    jsonrpc_success(&req.id, RunBenchmarkResult { samples_written })
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
