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
use serde::{de::DeserializeOwned, Serialize};
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
}

impl ChildHandle {
    /// Spawn a child in worker mode.
    ///
    /// Passes shmem name and role via CLI arguments.
    pub fn spawn(executable: &Path, commpage: &Commpage, role: Role) -> Result<Self> {
        let role_str = match role {
            Role::Candidate => "candidate",
            Role::Baseline => "baseline",
        };

        let mut child = Command::new(executable)
            .args(["__worker", "--shmem", commpage.os_id(), "--role", role_str])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to spawn: {}", executable.display()))?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        Ok(ChildHandle {
            process: child,
            stdin,
            stdout,
            role,
            req_next_id: 1,
        })
    }

    /// Query the child for its list of benchmark names.
    pub fn list_benchmarks(&mut self) -> Result<Vec<String>> {
        let result = self.call(protocol::METHOD_LIST_BENCHMARKS, Value::Null)?;
        let list: ListBenchmarksResult = serde_json::from_value(result)?;
        Ok(list.benchmarks)
    }

    /// Estimate iterations for a given time budget (in ms).
    pub fn estimate_iterations(&mut self, index: usize, seed: u64, time_ms: u32) -> Result<usize> {
        let result = self.call(
            protocol::METHOD_ESTIMATE_ITERATIONS,
            EstimateIterationsParams {
                index,
                time_ms,
                seed,
            },
        )?;
        let est: EstimateIterationsResult = serde_json::from_value(result)?;
        Ok(est.iterations)
    }

    /// Start the measurement loop (sends `run_benchmark` RPC without waiting for response).
    pub fn start_benchmark(
        &mut self,
        index: usize,
        seed: u64,
        iterations: usize,
        num_samples: usize,
    ) -> Result<()> {
        self.send_request(
            protocol::METHOD_RUN_BENCHMARK,
            RunBenchmarkParams {
                index,
                seed,
                iterations,
                num_samples,
            },
        )?;
        Ok(())
    }

    /// Wait for the `run_benchmark` response (blocks until child finishes all samples).
    pub fn finish_benchmark(&mut self) -> Result<RunBenchmarkResult> {
        self.read_response::<RunBenchmarkResult>()
    }

    /// Drain new samples from this child's commpage lane.
    /// Returns the number of samples that were skipped (overwritten before reading).
    pub fn drain_samples(&mut self, commpage: &Commpage, samples: &mut Vec<u64>) -> usize {
        commpage
            .get_lane(self.role)
            .drain_samples(samples)
            .err()
            .unwrap_or_default()
    }

    /// Send shutdown and wait for the child to exit.
    pub fn shutdown(mut self) -> Result<()> {
        // Best-effort: send shutdown, ignore write errors (child may have already exited)
        let _ = self.call(protocol::METHOD_SHUTDOWN, Value::Null);
        self.process.wait()?;
        Ok(())
    }

    /// The role of this child.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Send a JSON-RPC request and wait for the response.
    fn call<T: Serialize>(&mut self, method: &str, params: T) -> Result<Value> {
        self.send_request(method, params)?;
        self.read_response()
    }

    fn send_request<T: Serialize>(&mut self, method: &str, params: T) -> Result<()> {
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
        Ok(())
    }

    fn read_response<T: DeserializeOwned>(&mut self) -> Result<T> {
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .context("Failed to read response from child")?;

        if line.is_empty() {
            bail!("Child process closed stdout unexpectedly");
        }

        let resp = serde_json::from_str::<Output<T>>(&line)
            .context("Failed to parse JSON-RPC response")?;

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
