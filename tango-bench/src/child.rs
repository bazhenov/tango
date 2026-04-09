//! Runner-side child process handle.
//!
//! The runner spawns child processes in worker mode and communicates with them
//! via JSON-RPC over stdin/stdout. Measurement data flows through the commpage.

use crate::commpage::{Commpage, Role};
use anyhow::{bail, Context};
use jsonrpc_types::v2::*;
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub struct ChildHandle {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    role: Role,
    next_id: u64,
    /// Runner's local read position for this child's lane
    local_read_pos: u64,
}

fn value_to_params(v: serde_json::Value) -> Option<Params> {
    match v {
        serde_json::Value::Object(map) => Some(Params::Map(map)),
        serde_json::Value::Array(arr) => Some(Params::Array(arr)),
        _ => None,
    }
}

impl ChildHandle {
    /// Spawn a child in worker mode and send the `init` command.
    ///
    /// Returns the list of benchmark names the child reported.
    pub fn spawn(
        executable: &Path,
        commpage: &Commpage,
        role: Role,
    ) -> anyhow::Result<(Self, Vec<String>)> {
        let mut child = Command::new(executable)
            .arg("__worker")
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
            next_id: 1,
            local_read_pos: 0,
        };

        // Send init
        let result = handle.call(
            "init",
            serde_json::json!({
                "shmem_name": commpage.os_id(),
                "role": role,
            }),
        )?;

        #[derive(Deserialize)]
        struct InitResult {
            benchmarks: Vec<String>,
        }

        let init: InitResult =
            serde_json::from_value(result).context("Failed to parse init response")?;

        Ok((handle, init.benchmarks))
    }

    /// Select a benchmark by index.
    pub fn select(&mut self, idx: usize) -> anyhow::Result<()> {
        self.call("select", serde_json::json!({ "index": idx }))?;
        Ok(())
    }

    /// Estimate iterations for a given time budget (in ms).
    pub fn estimate_iterations(&mut self, time_ms: u32) -> anyhow::Result<usize> {
        let result = self.call(
            "estimate_iterations",
            serde_json::json!({ "time_ms": time_ms }),
        )?;

        #[derive(Deserialize)]
        struct EstResult {
            iterations: usize,
        }
        let est: EstResult = serde_json::from_value(result)?;
        Ok(est.iterations)
    }

    /// Start the measurement loop (sends `run_benchmark` RPC without waiting for response).
    pub fn start_benchmark(
        &mut self,
        iterations: usize,
        num_samples: usize,
    ) -> anyhow::Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let req = MethodCall {
            jsonrpc: Version::V2_0,
            method: "run_benchmark".to_string(),
            params: value_to_params(serde_json::json!({
                "iterations": iterations,
                "num_samples": num_samples,
            })),
            id: Id::Num(id),
        };
        let line = serde_json::to_string(&req)?;
        writeln!(self.stdin, "{}", line)?;
        self.stdin.flush()?;
        Ok(id)
    }

    /// Wait for the `run_benchmark` response (blocks until child finishes all samples).
    pub fn finish_benchmark(&mut self) -> anyhow::Result<u64> {
        let resp = self.read_response()?;

        #[derive(Deserialize)]
        struct RunResult {
            samples_written: u64,
        }
        let r: RunResult = serde_json::from_value(resp)?;
        Ok(r.samples_written)
    }

    /// Reset local read position (call before each new benchmark run).
    pub fn reset_read_pos(&mut self) {
        self.local_read_pos = 0;
    }

    /// Drain new samples from this child's commpage lane.
    pub fn drain_samples(&mut self, commpage: &Commpage) -> Vec<u64> {
        let lane = commpage.my_lane(self.role);
        let (samples, new_pos) = lane.drain_samples(self.local_read_pos);
        self.local_read_pos = new_pos;
        samples
    }

    /// Send shutdown and wait for the child to exit.
    pub fn shutdown(mut self) -> anyhow::Result<()> {
        // Best-effort: send shutdown, ignore write errors (child may have already exited)
        let _ = self.call("shutdown", serde_json::Value::Null);
        self.process.wait()?;
        Ok(())
    }

    /// The role of this child.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Send a JSON-RPC request and wait for the response.
    fn call(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let id = self.next_id;
        self.next_id += 1;

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

    fn read_response(&mut self) -> anyhow::Result<serde_json::Value> {
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
