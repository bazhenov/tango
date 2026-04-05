//! Child worker mode: reads JSON-RPC commands from stdin, writes samples to commpage.

use crate::commpage::{Commpage, CommpageError, Role};
use crate::{Benchmark, ErasedSampler};
use libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE;
use serde::{Deserialize, Serialize};
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

    unsafe { libc::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };

    for line in stdin.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(None, -32700, &format!("Parse error: {e}"));
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

fn dispatch(req: &JsonRpcRequest, state: &mut WorkerState) -> JsonRpcResponse {
    match req.method.as_str() {
        "init" => handle_init(req, state),
        "select" => handle_select(req, state),
        "estimate_iterations" => handle_estimate_iterations(req, state),
        "run_benchmark" => handle_run_benchmark(req, state),
        "shutdown" => JsonRpcResponse::result(req.id.clone(), serde_json::Value::Null),
        _ => JsonRpcResponse::error(req.id.clone(), -32601, "Method not found"),
    }
}

fn handle_init(req: &JsonRpcRequest, state: &mut WorkerState) -> JsonRpcResponse {
    #[derive(Deserialize)]
    struct Params {
        shmem_name: String,
        role: Role,
    }

    let params: Params = match serde_json::from_value(req.params.clone().unwrap_or_default()) {
        Ok(p) => p,
        Err(e) => {
            return JsonRpcResponse::error(req.id.clone(), -32602, &format!("Invalid params: {e}"))
        }
    };

    match Commpage::open(&params.shmem_name) {
        Ok(cp) => {
            state.commpage = Some(cp);
            state.role = Some(params.role);
        }
        Err(CommpageError::Shmem(e)) => {
            return JsonRpcResponse::error(
                req.id.clone(),
                -32000,
                &format!("Failed to open shared memory: {e}"),
            );
        }
        Err(e) => {
            return JsonRpcResponse::error(req.id.clone(), -32000, &format!("Commpage error: {e}"));
        }
    }

    let benchmarks = match crate::take_benchmarks() {
        Some(b) => b,
        None => {
            return JsonRpcResponse::error(req.id.clone(), -32000, "No benchmarks registered");
        }
    };

    let names: Vec<String> = benchmarks.iter().map(|b| b.name().to_string()).collect();
    state.benchmarks = benchmarks;

    let result = serde_json::json!({ "benchmarks": names });
    JsonRpcResponse::result(req.id.clone(), result)
}

fn handle_select(req: &JsonRpcRequest, state: &mut WorkerState) -> JsonRpcResponse {
    #[derive(Deserialize)]
    struct Params {
        index: usize,
    }

    let params: Params = match serde_json::from_value(req.params.clone().unwrap_or_default()) {
        Ok(p) => p,
        Err(e) => {
            return JsonRpcResponse::error(req.id.clone(), -32602, &format!("Invalid params: {e}"))
        }
    };

    if params.index >= state.benchmarks.len() {
        return JsonRpcResponse::error(
            req.id.clone(),
            -32602,
            &format!(
                "Index {} out of range (have {} benchmarks)",
                params.index,
                state.benchmarks.len()
            ),
        );
    }

    // Create a sampler (prepares state) for the selected benchmark
    let sampler = state.benchmarks[params.index].prepare_state(0);
    state.sampler = Some(sampler);

    JsonRpcResponse::result(req.id.clone(), serde_json::Value::Null)
}

fn handle_estimate_iterations(req: &JsonRpcRequest, state: &mut WorkerState) -> JsonRpcResponse {
    #[derive(Deserialize)]
    struct Params {
        time_ms: u32,
    }

    let params: Params = match serde_json::from_value(req.params.clone().unwrap_or_default()) {
        Ok(p) => p,
        Err(e) => {
            return JsonRpcResponse::error(req.id.clone(), -32602, &format!("Invalid params: {e}"))
        }
    };

    let sampler = match &mut state.sampler {
        Some(s) => s,
        None => {
            return JsonRpcResponse::error(
                req.id.clone(),
                -32000,
                "No benchmark selected (call select first)",
            )
        }
    };

    let iters = sampler.estimate_iterations(params.time_ms);
    JsonRpcResponse::result(req.id.clone(), serde_json::json!({ "iterations": iters }))
}

fn handle_run_benchmark(req: &JsonRpcRequest, state: &mut WorkerState) -> JsonRpcResponse {
    #[derive(Deserialize)]
    struct Params {
        iterations: usize,
        num_samples: usize,
    }

    let params: Params = match serde_json::from_value(req.params.clone().unwrap_or_default()) {
        Ok(p) => p,
        Err(e) => {
            return JsonRpcResponse::error(req.id.clone(), -32602, &format!("Invalid params: {e}"))
        }
    };

    let cp = match &state.commpage {
        Some(c) => c,
        None => return JsonRpcResponse::error(req.id.clone(), -32000, "Commpage not initialized"),
    };
    let r = match state.role {
        Some(r) => r,
        None => return JsonRpcResponse::error(req.id.clone(), -32000, "Role not set"),
    };

    let sampler = match &mut state.sampler {
        Some(s) => s,
        None => {
            return JsonRpcResponse::error(
                req.id.clone(),
                -32000,
                "No benchmark selected (call select first)",
            )
        }
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

    JsonRpcResponse::result(
        req.id.clone(),
        serde_json::json!({ "samples_written": sample_no }),
    )
}

// --- JSON-RPC 2.0 types ---

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    method: String,
    params: Option<serde_json::Value>,
    id: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    fn result(id: Option<serde_json::Value>, value: serde_json::Value) -> Self {
        JsonRpcResponse {
            jsonrpc: "2.0",
            result: Some(value),
            error: None,
            id,
        }
    }

    fn error(id: Option<serde_json::Value>, code: i32, message: &str) -> Self {
        JsonRpcResponse {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
            }),
            id,
        }
    }
}
