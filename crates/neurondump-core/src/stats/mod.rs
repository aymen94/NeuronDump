//! Streaming weight statistics. Each tensor is dequantized in bounded row chunks
//! (at most ~1M floats per thread) and reduced in two passes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use neurondump_gguf::{can_dequantize, GgmlType, GgufFile, TensorInfo};

use crate::arch::{parse_tensor_name, per_layer, Hyperparams};
use crate::util::{check_cancel, Progress, SCHEMA_VERSION, Instant};

mod tensor;
mod model;
pub use tensor::*;
pub use model::*;

pub const HIST_BINS: usize = 64;
const CHUNK_ELEMS: u64 = 1 << 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub min: f32,
    pub max: f32,
    pub bins: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStats {
    pub name: String,
    pub layer: Option<u32>,
    pub role: String,
    pub ggml_type: String,
    pub dims: Vec<u64>,
    pub n: u64,
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub std: f64,
    pub l2: f64,
    /// Fraction of exact zeros.
    pub sparsity: f64,
    /// Values with |x - mean| > 6 std.
    pub outliers: u64,
    pub outlier_fraction: f64,
    pub nan_or_inf: u64,
    pub histogram: Histogram,
    /// L2 norm of each row (`dims[0]`-length vectors): output units of a projection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_norms: Option<Vec<f32>>,
    /// L2 norm of each column: input units (e.g. FFN neurons of `ffn_down`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub col_norms: Option<Vec<f32>>,
    /// L2 norm of each slice along the last dim of 3D tensors (MoE experts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slice_norms: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadNorms {
    pub layer: u32,
    pub head_dim: u64,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronNorms {
    pub layer: u32,
    /// sqrt(|up_i|² + |gate_i|² + |down_:,i|²) per FFN neuron.
    pub norms: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertNorms {
    pub layer: u32,
    pub norms: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsReport {
    pub schema_version: u32,
    pub model: String,
    pub digest: String,
    pub tensors: Vec<TensorStats>,
    pub heads: Vec<HeadNorms>,
    pub ffn_neurons: Vec<NeuronNorms>,
    pub experts: Vec<ExpertNorms>,
    pub skipped: Vec<(String, String)>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatsOptions {
    /// Only these tensors (exact names); all when empty.
    #[serde(default)]
    pub tensors: Vec<String>,
    /// Only tensors of these layers; all when empty.
    #[serde(default)]
    pub layers: Vec<u32>,
}

fn wants_rows(role: &str) -> bool {
    matches!(role, "attn_q" | "attn_k" | "attn_v" | "attn_qkv" | "ffn_up" | "ffn_gate" | "ffn_gate_up" | "token_embd" | "output")
}

fn wants_cols(role: &str) -> bool {
    matches!(role, "attn_output" | "ffn_down")
}
