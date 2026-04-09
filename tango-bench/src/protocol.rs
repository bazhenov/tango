//! Shared JSON-RPC protocol definitions: method names, parameter types, and result types
//! used by both the worker (server) and child handle (client).

use crate::commpage::Role;
use serde::{Deserialize, Serialize};

pub const WORKER_COMMAND: &str = "__worker";

pub mod method {
    pub const INIT: &str = "init";
    pub const SELECT: &str = "select";
    pub const ESTIMATE_ITERATIONS: &str = "estimate_iterations";
    pub const RUN_BENCHMARK: &str = "run_benchmark";
    pub const SHUTDOWN: &str = "shutdown";
}

#[derive(Serialize, Deserialize)]
pub struct InitParams {
    #[serde(rename = "shmem_name")]
    pub shmem_name: String,
    #[serde(rename = "role")]
    pub role: Role,
}

#[derive(Serialize, Deserialize)]
pub struct InitResult {
    #[serde(rename = "benchmarks")]
    pub benchmarks: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SelectParams {
    #[serde(rename = "index")]
    pub index: usize,
}

#[derive(Serialize, Deserialize)]
pub struct EstimateIterationsParams {
    #[serde(rename = "time_ms")]
    pub time_ms: u32,
}

#[derive(Serialize, Deserialize)]
pub struct EstimateIterationsResult {
    #[serde(rename = "iterations")]
    pub iterations: usize,
}

#[derive(Serialize, Deserialize)]
pub struct RunBenchmarkParams {
    #[serde(rename = "iterations")]
    pub iterations: usize,
    #[serde(rename = "num_samples")]
    pub num_samples: usize,
}

#[derive(Serialize, Deserialize)]
pub struct RunBenchmarkResult {
    #[serde(rename = "samples_written")]
    pub samples_written: u64,
}
