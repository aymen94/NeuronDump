//! Runtime tracing types and the pluggable backend. The candle-based implementation
//! lives in `neurondump-trace` and is registered by the binary when built with the
//! `trace` feature, so core stays free of inference dependencies.

use std::sync::OnceLock;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::model::ModelHandle;
use crate::util::Progress;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceRequest {
    #[serde(default)]
    pub prompt: Option<String>,
    /// Corpus mode: each entry is traced and statistics are aggregated.
    #[serde(default)]
    pub corpus: Vec<String>,
    /// Corpus file (read by core before tracing).
    #[serde(default)]
    pub corpus_file: Option<std::path::PathBuf>,
    /// Number of logit-lens tokens / top neurons per layer.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Truncate inputs to this many tokens.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// `auto`, `cpu`, `cuda` or `cuda:N` (None = auto).
    #[serde(default)]
    pub device: Option<String>,
}

/// How a run went: where it ran and how long it took.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunStats {
    /// `cpu` or `cuda:N`.
    pub device: String,
    pub device_name: Option<String>,
    /// Weight loading time (0 when the model was already loaded).
    pub load_ms: u64,
    /// Weights reused from an earlier job.
    pub cached_weights: bool,
    /// Tokens processed per second over all forward passes of the run.
    pub tokens_per_s: f64,
    pub forward_passes: usize,
    /// Time per layer of one measured forward pass (synchronized), in ms.
    #[serde(default)]
    pub layer_ms: Vec<f32>,
    /// Highest VRAM use seen while the run was going.
    #[serde(default)]
    pub peak_gpu: Option<crate::gpu::GpuSample>,
    pub total_ms: u64,
}

/// A small binary measurement for drawing (never weights): little-endian `f32` or `u16`.
#[derive(Debug, Clone, Default)]
pub struct TensorBlob {
    pub name: String,
    /// `f32` or `u16` (u16 = value × 65535, for maps in [0, 1]).
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
    /// Optional names along the first axis (e.g. node ids of a sweep).
    pub labels: Vec<String>,
}

impl TensorBlob {
    pub fn f32(name: impl Into<String>, shape: Vec<usize>, v: &[f32]) -> Self {
        Self { name: name.into(), dtype: "f32", shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect(), labels: vec![] }
    }

    /// Values in [0, 1] stored as u16.
    pub fn unit_u16(name: impl Into<String>, shape: Vec<usize>, v: &[f32]) -> Self {
        Self {
            name: name.into(),
            dtype: "u16",
            shape,
            data: v.iter().flat_map(|x| ((x.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes()).collect(),
            labels: vec![],
        }
    }
}

/// A model held in memory by the backend for reuse across jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModel {
    pub model: String,
    pub digest: String,
    pub device: String,
    pub weights_bytes: u64,
    pub idle_s: u64,
    pub uses: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensToken {
    pub id: u32,
    pub text: String,
    pub prob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTrace {
    pub index: u32,
    /// Mean L2 norm of the residual stream after this layer (over positions).
    pub hidden_norm: f32,
    /// Cosine similarity between the layer's input and output hidden states
    /// (≈1 means the layer barely changes the residual stream).
    pub cos_prev: Option<f32>,
    /// Mean attention entropy per head (nats), last position or averaged.
    pub head_entropy: Vec<f32>,
    /// Top activated FFN neurons (index, mean |activation|).
    pub top_neurons: Vec<(u32, f32)>,
    /// MoE router: experts chosen for the last position.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub router: Vec<u32>,
    /// Logit lens at the last position.
    pub logit_lens: Vec<LensToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenContribution {
    pub id: u32,
    pub text: String,
    /// Cosine similarity of the token's final hidden state with the pooled embedding.
    pub contribution: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub backend: String,
    pub supported: bool,
    pub tokens: Vec<(u32, String)>,
    pub layers: Vec<LayerTrace>,
    /// Final next-token prediction (decoders).
    pub prediction: Vec<LensToken>,
    /// Embedding models: per-token contribution to the pooled embedding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding_contributions: Vec<TokenContribution>,
    /// Corpus mode: number of texts aggregated (layer values are means).
    pub samples: usize,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
    #[serde(default)]
    pub stats: Option<RunStats>,
    /// Binary measurements for drawing; stored next to the job result, not in the JSON.
    #[serde(skip)]
    pub blobs: Vec<TensorBlob>,
}

/// In-memory cuts applied during a forward pass (nothing is written to disk).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Ablation {
    /// Layers replaced by the identity (the residual stream passes through unchanged).
    #[serde(default)]
    pub skip_layers: Vec<u32>,
    /// (layer, block) whose contribution is set to zero: `attention`, `ffn`, `ssm`, `moe`.
    #[serde(default)]
    pub zero_blocks: Vec<(u32, String)>,
    /// (layer, head): the head's output is set to zero before the output projection.
    #[serde(default)]
    pub zero_heads: Vec<(u32, u32)>,
    /// (layer, first, last) FFN neurons (inclusive) set to zero.
    #[serde(default)]
    pub zero_neurons: Vec<(u32, u32, u32)>,
    /// (layer, expert).
    #[serde(default)]
    pub zero_experts: Vec<(u32, u32)>,
    /// Whole tensors set to zero (by GGUF name).
    #[serde(default)]
    pub zero_tensors: Vec<String>,
    /// Token id ranges (inclusive) that can no longer be predicted.
    #[serde(default)]
    pub mask_tokens: Vec<(u32, u32)>,
}

impl Ablation {
    pub fn is_empty(&self) -> bool {
        self == &Ablation::default()
    }

    pub fn masks_token(&self, id: u32) -> bool {
        self.mask_tokens.iter().any(|(a, b)| id >= *a && id <= *b)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimRequest {
    pub ablation: Ablation,
    /// Node ids the ablation was built from (reported back).
    #[serde(default)]
    pub cut: Vec<String>,
    #[serde(default)]
    pub prompts: Vec<String>,
    /// Texts for perplexity / mean KL.
    #[serde(default)]
    pub corpus: Vec<String>,
    /// Greedy tokens generated per prompt.
    #[serde(default)]
    pub gen_tokens: usize,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub max_tokens: usize,
    #[serde(default)]
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensDiff {
    pub layer: u32,
    pub baseline: Option<LensToken>,
    pub ablated: Option<LensToken>,
    /// Probability of the baseline's final top token at this layer.
    pub baseline_target_prob: f32,
    pub ablated_target_prob: f32,
    pub skipped: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimSide {
    pub top_k: Vec<LensToken>,
    pub generated: String,
    pub generated_tokens: usize,
    pub tokens_per_s: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_norm: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimPrompt {
    pub prompt: String,
    pub n_tokens: usize,
    pub baseline: SimSide,
    pub ablated: SimSide,
    /// KL(baseline || ablated) of the next-token distribution, in nats.
    pub kl: Option<f64>,
    pub top1_changed: bool,
    /// Embedding models: cosine similarity of the pooled embeddings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_cosine: Option<f32>,
    pub lens: Vec<LensDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusEval {
    pub samples: usize,
    pub tokens: usize,
    pub baseline_ppl: Option<f64>,
    pub ablated_ppl: Option<f64>,
    /// Mean KL(baseline || ablated) over every predicted position.
    pub mean_kl: Option<f64>,
    /// Fraction of positions where the top-1 prediction is unchanged.
    pub top1_agreement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub backend: String,
    pub cut: Vec<String>,
    pub ablation: Ablation,
    pub prompts: Vec<SimPrompt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus: Option<CorpusEval>,
    /// Bytes of weights still in use (what a slimmed file would hold), baseline vs ablated.
    pub weights_bytes_baseline: u64,
    pub weights_bytes_ablated: u64,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
    #[serde(default)]
    pub stats: Option<RunStats>,
    #[serde(skip)]
    pub blobs: Vec<TensorBlob>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepTarget {
    pub id: String,
    pub label: String,
    pub ablation: Ablation,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SweepRequest {
    /// `layer`, `head` or `expert`.
    pub level: String,
    #[serde(default)]
    pub layer: Option<u32>,
    #[serde(default)]
    pub corpus: Vec<String>,
    #[serde(default)]
    pub corpus_file: Option<std::path::PathBuf>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Filled by core from the graph.
    #[serde(default)]
    pub targets: Vec<SweepTarget>,
    #[serde(default)]
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepEntry {
    pub id: String,
    pub label: String,
    pub mean_kl: f64,
    pub ppl: Option<f64>,
    /// ppl(ablated) / ppl(baseline) - 1.
    pub ppl_increase: Option<f64>,
    pub top1_agreement: f64,
    /// 1 = most damaging.
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepReport {
    pub schema_version: u32,
    pub model: String,
    pub digest: String,
    pub architecture: String,
    pub level: String,
    pub layer: Option<u32>,
    pub samples: usize,
    pub tokens: usize,
    pub baseline_ppl: Option<f64>,
    /// In target order.
    pub entries: Vec<SweepEntry>,
    /// Node ids, most damaging first.
    pub ranking: Vec<String>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
    #[serde(default)]
    pub stats: Option<RunStats>,
    #[serde(skip)]
    pub blobs: Vec<TensorBlob>,
}

pub trait TraceBackend: Send + Sync {
    fn name(&self) -> &str;
    fn supported_architectures(&self) -> Vec<String>;
    fn trace(&self, model: &ModelHandle, req: &TraceRequest, progress: &dyn Progress) -> Result<TraceReport>;
    fn simulate(&self, _model: &ModelHandle, _req: &SimRequest, _progress: &dyn Progress) -> Result<SimReport> {
        bail!("this trace backend cannot simulate ablations")
    }
    fn sweep(&self, _model: &ModelHandle, _req: &SweepRequest, _progress: &dyn Progress) -> Result<SweepReport> {
        bail!("this trace backend cannot run sweeps")
    }
    /// Built with CUDA support.
    fn cuda_compiled(&self) -> bool {
        false
    }
    /// A CUDA device answers.
    fn cuda_available(&self) -> bool {
        false
    }
    /// Models currently held in memory.
    fn loaded(&self) -> Vec<LoadedModel> {
        vec![]
    }
    /// Frees held models (all, or one digest); returns how many were freed.
    fn unload(&self, _digest: Option<&str>) -> usize {
        0
    }
}

/// Checks that runtime simulation is available for this model.
pub fn require_backend(m: &ModelHandle) -> Result<&'static dyn TraceBackend> {
    let Some(b) = backend() else {
        bail!("simulation needs the trace backend; rebuild with `cargo build --features trace` (static impact still works with --static)");
    };
    let arch = m.architecture().to_string();
    if !b.supported_architectures().iter().any(|a| *a == arch) {
        bail!(
            "simulation unsupported for architecture {arch:?} (supported: {}); static impact still works with --static",
            b.supported_architectures().join(", ")
        );
    }
    Ok(b)
}

/// Splits a corpus file into texts (blank-line separated), like `trace --corpus`.
pub fn read_corpus(path: &std::path::Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading corpus {}: {e}", path.display()))?;
    Ok(text.split("\n\n").map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect())
}

static BACKEND: OnceLock<Box<dyn TraceBackend>> = OnceLock::new();

pub fn register_backend(b: Box<dyn TraceBackend>) {
    let _ = BACKEND.set(b);
}

pub fn backend() -> Option<&'static dyn TraceBackend> {
    BACKEND.get().map(|b| b.as_ref())
}

pub fn supported_architectures() -> Vec<String> {
    backend().map(|b| b.supported_architectures()).unwrap_or_default()
}

pub fn trace(m: &ModelHandle, req: &TraceRequest, p: &dyn Progress) -> Result<TraceReport> {
    let mut req = req.clone();
    if let Some(f) = &req.corpus_file {
        let text = std::fs::read_to_string(f)?;
        req.corpus.extend(text.split("\n\n").map(str::trim).filter(|s| !s.is_empty()).map(String::from));
    }
    if req.prompt.is_none() && req.corpus.is_empty() {
        bail!("trace needs a prompt or a corpus");
    }
    let Some(b) = backend() else {
        bail!("trace support is not compiled in; rebuild with `cargo build --features trace`");
    };
    let arch = m.architecture().to_string();
    if !b.supported_architectures().iter().any(|a| *a == arch) {
        bail!(
            "trace unsupported for architecture {arch:?} (supported: {}); static analysis is still available",
            b.supported_architectures().join(", ")
        );
    }
    b.trace(m, &req, p)
}
