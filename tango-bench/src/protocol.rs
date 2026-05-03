//! Shared JSON-RPC protocol definitions: method names, parameter types, and result types
//! used by both the worker (server) and child handle (client).

use serde::{Deserialize, Serialize};

pub const METHOD_LIST_BENCHMARKS: &str = "list_benchmarks";
pub const METHOD_ESTIMATE_ITERATIONS: &str = "estimate_iterations";
pub const METHOD_RUN_BENCHMARK: &str = "run_benchmark";
pub const METHOD_SHUTDOWN: &str = "shutdown";

#[derive(Serialize, Deserialize)]
pub struct ListBenchmarksResult {
    #[serde(rename = "benchmarks")]
    pub benchmarks: Vec<String>,
    #[serde(rename = "aux_metrics", default)]
    pub aux_metrics: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct EstimateIterationsParams {
    #[serde(rename = "index")]
    pub index: usize,
    #[serde(rename = "time_ms")]
    pub time_ms: u32,
    #[serde(rename = "seed")]
    pub seed: u64,
}

#[derive(Serialize, Deserialize)]
pub struct EstimateIterationsResult {
    #[serde(rename = "iterations")]
    pub iterations: usize,
}

#[derive(Serialize, Deserialize)]
pub struct RunBenchmarkParams {
    #[serde(rename = "index")]
    pub index: usize,
    #[serde(rename = "seed")]
    pub seed: u64,
    #[serde(rename = "iterations")]
    pub iterations: usize,
    #[serde(rename = "num_samples")]
    pub num_samples: usize,
    /// Auxiliary metric ids to measure alongside time.
    #[serde(rename = "aux_metrics")]
    pub aux_metrics: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct RunBenchmarkResult {
    #[serde(rename = "samples")]
    pub samples: Vec<u64>,
    /// Auxiliary metrics samples, in the same order as `aux_metrics` in the request.
    #[serde(rename = "aux_metrics")]
    pub aux_metrics: Vec<u64>,
}
