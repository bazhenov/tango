//! Child worker mode: reads JSON-RPC commands from stdin, writes samples to commpage.

use crate::commpage::{Commpage, CommpageError, Role};
use crate::{Benchmark, ErasedSampler};
use jsonrpc_types::v2::*;
use serde::Deserialize;
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

        let is_shutdown = req.method == "shutdown";
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
    role: Option<Role>,
    benchmarks: Vec<Benchmark>,
    sampler: Option<Box<dyn ErasedSampler>>,
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

fn dispatch(req: &MethodCall, state: &mut WorkerState) -> Output {
    match req.method.as_str() {
        "init" => handle_init(req, state),
        "select" => handle_select(req, state),
        "estimate_iterations" => handle_estimate_iterations(req, state),
        "run_benchmark" => handle_run_benchmark(req, state),
        "shutdown" => {
            let id = req.id.clone();
            let value = Value::Null;
            Output::success(value, id)
        }
        _ => jsonrpc_error(&req.id, Error::method_not_found()),
    }
}

fn handle_init(req: &MethodCall, state: &mut WorkerState) -> Output {
    #[derive(Deserialize)]
    struct InitParams {
        shmem_name: String,
        role: Role,
    }

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

    let result = serde_json::json!({ "benchmarks": names });
    {
        let id = req.id.clone();
        Output::success(result, id)
    }
}

fn handle_select(req: &MethodCall, state: &mut WorkerState) -> Output {
    #[derive(Deserialize)]
    struct SelectParams {
        index: usize,
    }

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

    {
        let id = req.id.clone();
        let value = Value::Null;
        Output::success(value, id)
    }
}

fn handle_estimate_iterations(req: &MethodCall, state: &mut WorkerState) -> Output {
    #[derive(Deserialize)]
    struct EstimateParams {
        time_ms: u32,
    }

    let params: EstimateParams = match serde_json::from_value(params_to_value(&req.params)) {
        Ok(p) => p,
        Err(e) => return jsonrpc_error(&req.id, Error::invalid_params(e)),
    };

    let sampler = match &mut state.sampler {
        Some(s) => s,
        None => return server_error(&req.id, "No benchmark selected (call select first)"),
    };

    let iters = sampler.estimate_iterations(params.time_ms);
    {
        let id = req.id.clone();
        let value = serde_json::json!({ "iterations": iters });
        Output::success(value, id)
    }
}

fn handle_run_benchmark(req: &MethodCall, state: &mut WorkerState) -> Output {
    #[derive(Deserialize)]
    struct RunParams {
        iterations: usize,
        num_samples: usize,
    }

    let params: RunParams = match serde_json::from_value(params_to_value(&req.params)) {
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

    let mut sample_no: u64 = 0;
    loop {
        // Check termination conditions
        if params.num_samples > 0 && sample_no >= params.num_samples as u64 {
            break;
        }
        if params.num_samples == 0 && cp.is_stopped() {
            break;
        }

        // Take measurement
        let elapsed_ns = sampler.measure(params.iterations);

        // Write sample to commpage
        my_lane.push_sample(sample_no, elapsed_ns);
        sample_no += 1;

        // Barrier: wait for peer
        if !peer_lane.wait_for_cursor(sample_no) {
            // Peer exited early
            break;
        }
    }

    my_lane.mark_done();

    {
        let id = req.id.clone();
        let value = serde_json::json!({ "samples_written": sample_no });
        Output::success(value, id)
    }
}
