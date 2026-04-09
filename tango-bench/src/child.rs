//! Runner-side child process handle.
//!
//! The runner spawns child processes in worker mode and communicates with them
//! via JSON-RPC over stdin/stdout. Measurement data flows through the commpage.

use crate::{
    commpage::{Commpage, Role},
    protocol::{self, *},
};
use anyhow::{bail, Context, Result};
use jsonrpc_types::v2::*;
use serde::Serialize;
use serde_json::Value;
use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

pub struct ChildHandle {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    role: Role,
    /// Next id for JSON RPC request
    req_next_id: u64,
    /// Runner's local read position for this child's lane
    read_pos: u64,
}

impl ChildHandle {
    /// Spawn a child in worker mode and send the `init` command.
    ///
    /// Returns the list of benchmark names the child reported.
    pub fn spawn(
        executable: &Path,
        commpage: &Commpage,
        role: Role,
    ) -> Result<(Self, Vec<String>)> {
        let mut child = Command::new(executable)
            .arg(WORKER_COMMAND)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to spawn: {}", executable.display()))?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        let mut handle = ChildHandle {
            process: child,
            stdin,
            stdout,
            role,
            req_next_id: 1,
            read_pos: 0,
        };

        // Send init
        let params = InitParams {
            shmem_name: commpage.os_id().to_string(),
            role,
        };
        let result = handle.call(protocol::method::INIT, params)?;
        let init: InitResult =
            serde_json::from_value(result).context("Failed to parse init response")?;

        Ok((handle, init.benchmarks))
    }

    /// Select a benchmark by index.
    pub fn select(&mut self, idx: usize) -> Result<()> {
        self.call(protocol::method::SELECT, SelectParams { index: idx })?;
        Ok(())
    }

    /// Estimate iterations for a given time budget (in ms).
    pub fn estimate_iterations(&mut self, time_ms: u32) -> Result<usize> {
        let result = self.call(
            protocol::method::ESTIMATE_ITERATIONS,
            EstimateIterationsParams { time_ms },
        )?;
        let est: EstimateIterationsResult = serde_json::from_value(result)?;
        Ok(est.iterations)
    }

    /// Start the measurement loop (sends `run_benchmark` RPC without waiting for response).
    pub fn start_benchmark(&mut self, iterations: usize, num_samples: usize) -> Result<u64> {
        let id = self.req_next_id;
        self.req_next_id += 1;
        let params = RunBenchmarkParams {
            iterations,
            num_samples,
        };
        let req = MethodCall {
            jsonrpc: Version::V2_0,
            method: protocol::method::RUN_BENCHMARK.to_string(),
            params: value_to_params(serde_json::to_value(params)?),
            id: Id::Num(id),
        };
        let line = serde_json::to_string(&req)?;
        writeln!(self.stdin, "{}", line)?;
        self.stdin.flush()?;
        Ok(id)
    }

    /// Wait for the `run_benchmark` response (blocks until child finishes all samples).
    pub fn finish_benchmark(&mut self) -> Result<u64> {
        let resp = self.read_response()?;
        let r: RunBenchmarkResult = serde_json::from_value(resp)?;
        Ok(r.samples_written)
    }

    /// Reset local read position (call before each new benchmark run).
    pub fn reset_read_pos(&mut self) {
        self.read_pos = 0;
    }

    /// Drain new samples from this child's commpage lane.
    pub fn drain_samples(&mut self, commpage: &Commpage) -> Vec<u64> {
        let lane = commpage.my_lane(self.role);
        let (samples, new_pos) = lane.drain_samples(self.read_pos);
        self.read_pos = new_pos;
        samples
    }

    /// Send shutdown and wait for the child to exit.
    pub fn shutdown(mut self) -> Result<()> {
        // Best-effort: send shutdown, ignore write errors (child may have already exited)
        let _ = self.call(protocol::method::SHUTDOWN, Value::Null);
        self.process.wait()?;
        Ok(())
    }

    /// The role of this child.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Send a JSON-RPC request and wait for the response.
    fn call<T: Serialize>(&mut self, method: &str, params: T) -> Result<Value> {
        let params = serde_json::to_value(params)?;
        let id = self.req_next_id;
        self.req_next_id += 1;

        let req = MethodCall {
            jsonrpc: Version::V2_0,
            method: method.to_string(),
            params: value_to_params(params),
            id: Id::Num(id),
        };

        let line = serde_json::to_string(&req)?;
        writeln!(self.stdin, "{}", line)
            .with_context(|| format!("Failed to send '{method}' to child"))?;
        self.stdin.flush()?;

        self.read_response()
    }

    fn read_response(&mut self) -> Result<Value> {
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .context("Failed to read response from child")?;

        if line.is_empty() {
            bail!("Child process closed stdout unexpectedly");
        }

        let resp: Output =
            serde_json::from_str(&line).context("Failed to parse JSON-RPC response")?;

        match resp {
            Output::Success(s) => Ok(s.result),
            Output::Failure(f) => {
                bail!(
                    "Child RPC error {}: {}",
                    f.error.code.code(),
                    f.error.message
                )
            }
        }
    }
}

impl Drop for ChildHandle {
    fn drop(&mut self) {
        // Kill child if still running
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

fn value_to_params(v: Value) -> Option<Params> {
    match v {
        Value::Object(map) => Some(Params::Map(map)),
        Value::Array(arr) => Some(Params::Array(arr)),
        _ => None,
    }
}
