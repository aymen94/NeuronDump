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
}

/// Replaces or scales the input embedding row of one token (in memory only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbedEdit {
    pub token: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
    /// Use this token's row instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replace_with: Option<u32>,
}

/// Adds `strength * (mean h_layer(a) - mean h_layer(b))` to the residual stream after `layer`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Steer {
    pub layer: u32,
    pub a: String,
    pub b: String,
    pub strength: f32,
}

/// Replaces a layer's output (or one head's output) with the one from another prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Patch {
    pub layer: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<u32>,
    pub source: String,
    /// Every position aligned from the end; default only the last position.
    #[serde(default)]
    pub all_positions: bool,
}

/// In-memory interventions applied during a forward pass (nothing is written to disk).
/// Ablations are the zeroing / skipping subset; edits add scales, embedding edits, logit
/// biases, steering and patching.
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
    /// (layer, factor): the layer's whole contribution to the residual stream is scaled.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale_layers: Vec<(u32, f32)>,
    /// (layer, block, factor), block = attention, ssm, ffn, moe.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale_blocks: Vec<(u32, String, f32)>,
    /// (layer, head, factor).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale_heads: Vec<(u32, u32, f32)>,
    /// (layer, first, last, factor), inclusive neuron range.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale_neurons: Vec<(u32, u32, u32, f32)>,
    /// (layer, expert, factor).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale_experts: Vec<(u32, u32, f32)>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embed_edits: Vec<EmbedEdit>,
    /// (token, bias) added to the logits.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logit_bias: Vec<(u32, f32)>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steer: Vec<Steer>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patch: Vec<Patch>,
}

/// The general name: an ablation is an intervention that only zeroes and skips.
pub type Intervention = Ablation;

impl Ablation {
    pub fn is_empty(&self) -> bool {
        self == &Ablation::default()
    }

    /// Merges another intervention into this one (lists are concatenated).
    pub fn merge(&mut self, o: Ablation) {
        self.skip_layers.extend(o.skip_layers);
        self.zero_blocks.extend(o.zero_blocks);
        self.zero_heads.extend(o.zero_heads);
        self.zero_neurons.extend(o.zero_neurons);
        self.zero_experts.extend(o.zero_experts);
        self.zero_tensors.extend(o.zero_tensors);
        self.mask_tokens.extend(o.mask_tokens);
        self.scale_layers.extend(o.scale_layers);
        self.scale_blocks.extend(o.scale_blocks);
        self.scale_heads.extend(o.scale_heads);
        self.scale_neurons.extend(o.scale_neurons);
        self.scale_experts.extend(o.scale_experts);
        self.embed_edits.extend(o.embed_edits);
        self.logit_bias.extend(o.logit_bias);
        self.steer.extend(o.steer);
        self.patch.extend(o.patch);
        self.skip_layers.sort_unstable();
        self.skip_layers.dedup();
        self.zero_tensors.sort();
        self.zero_tensors.dedup();
    }

    pub fn masks_token(&self, id: u32) -> bool {
        self.mask_tokens.iter().any(|(a, b)| id >= *a && id <= *b)
    }
}

/// Token sampling. The default (temperature 0) is greedy and deterministic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    #[serde(default)]
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// (generation step, token id): the token chosen at that step regardless of the logits.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forced: Vec<(usize, u32)>,
}

impl Sampling {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// Direct logit attribution of one component (linearized through the final norm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAttribution {
    pub layer: u32,
    /// Push of the attention (or mixer) output towards `chosen` minus `runner_up`, in logits.
    pub attn: f32,
    pub ffn: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadAttribution {
    pub layer: u32,
    pub head: u32,
    pub value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronAttribution {
    pub layer: u32,
    pub neuron: u32,
    pub activation: f32,
    pub value: f32,
}

/// Why a token was chosen at one position: final candidates and how much each layer,
/// head and FFN pushed the chosen token against the runner-up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub position: usize,
    pub candidates: Vec<LensToken>,
    pub chosen: LensToken,
    pub runner_up: LensToken,
    /// logit(chosen) - logit(runner_up) as computed by the model.
    pub logit_diff: f32,
    /// Sum of all attributions, in exact f32. `logit_diff` comes from the quantized forward pass,
    /// so on quantized models the two can differ by a few tenths of a logit.
    pub attributed: f32,
    pub embedding: f32,
    pub layers: Vec<LayerAttribution>,
    /// Largest |value| first.
    pub top_heads: Vec<HeadAttribution>,
    pub top_neurons: Vec<NeuronAttribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JourneyToken {
    pub id: u32,
    pub text: String,
    pub special: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JourneyLayer {
    pub index: u32,
    /// L2 norm of the residual stream at the position after this layer.
    pub residual_norm: f32,
    /// Logit lens at the position.
    pub lens: Vec<LensToken>,
    /// Probability and rank of the final prediction at this layer.
    pub target_prob: f32,
    pub target_rank: u32,
    /// [head][key position 0..=position] attention weights of the query at the position.
    pub attention: Vec<Vec<f32>>,
    /// (key position, weight averaged over heads), largest first.
    pub attended: Vec<(usize, f32)>,
    /// (bucket node id, mean |activation|), largest first.
    pub ffn_buckets: Vec<(String, f32)>,
    pub top_neurons: Vec<(u32, f32)>,
    pub skipped: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JourneyRequest {
    /// Text to tokenize (already chat-templated by core).
    #[serde(default)]
    pub text: String,
    /// Token ids appended after the text (e.g. generated tokens).
    #[serde(default)]
    pub append: Vec<u32>,
    /// Position to inspect; negative counts from the end (-1 = last).
    #[serde(default = "minus_one")]
    pub token: i64,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub max_tokens: usize,
    #[serde(default)]
    pub ablation: Ablation,
    /// Graph neuron bucket size per layer (from the graph), for bucket ids.
    #[serde(default)]
    pub buckets: Vec<(u32, u64)>,
}

fn minus_one() -> i64 {
    -1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JourneyReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub backend: String,
    pub edited: bool,
    pub text: String,
    pub tokens: Vec<JourneyToken>,
    pub position: usize,
    /// Token at position + 1 when it exists (what the model should predict).
    pub next: Option<JourneyToken>,
    pub layers: Vec<JourneyLayer>,
    /// Final prediction at the position.
    pub prediction: Vec<LensToken>,
    /// First layer where the final prediction is top-1 / in the top 5 of the logit lens.
    pub first_top1: Option<u32>,
    pub first_top5: Option<u32>,
    pub decision: Option<Decision>,
    /// Graph node ids that contribute most (for highlighting in the graph).
    pub highlight: Vec<String>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateRequest {
    #[serde(default)]
    pub text: String,
    pub steps: usize,
    /// Decision breakdown for every step.
    #[serde(default)]
    pub explain: bool,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub max_tokens: usize,
    #[serde(default)]
    pub ablation: Ablation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenStep {
    pub index: usize,
    /// Position in the sequence the prediction was made at.
    pub position: usize,
    pub token: LensToken,
    /// Entropy of the next-token distribution (nats).
    pub entropy: f32,
    pub alternatives: Vec<LensToken>,
    pub forced: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<Decision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub backend: String,
    pub edited: bool,
    pub text: String,
    pub prompt_tokens: Vec<JourneyToken>,
    pub steps: Vec<GenStep>,
    pub generated: String,
    pub generated_ids: Vec<u32>,
    pub sampling: Sampling,
    pub tokens_per_s: f64,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimRequest {
    pub ablation: Ablation,
    /// Node ids the ablation was built from (reported back).
    #[serde(default)]
    pub cut: Vec<String>,
    #[serde(default)]
    pub prompts: Vec<String>,
    /// Edited side's texts (same length as `prompts`) when prompt edits change them.
    #[serde(default)]
    pub edited_prompts: Vec<String>,
    /// Sampling of the edited side (baseline is always greedy).
    #[serde(default)]
    pub sampling: Sampling,
    /// Decision breakdown of the next token for both sides.
    #[serde(default)]
    pub decisions: bool,
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
    /// The edited side's text when a prompt edit changed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edited_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_baseline: Option<Decision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_edited: Option<Decision>,
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
    fn journey(&self, _model: &ModelHandle, _req: &JourneyRequest, _progress: &dyn Progress) -> Result<JourneyReport> {
        bail!("this trace backend cannot follow tokens")
    }
    fn generate(&self, _model: &ModelHandle, _req: &GenerateRequest, _progress: &dyn Progress) -> Result<GenerateReport> {
        bail!("this trace backend cannot generate")
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

/// Follows one token through the model (feature `trace`, supported architectures only).
pub fn journey(m: &ModelHandle, req: &JourneyRequest, p: &dyn Progress) -> Result<JourneyReport> {
    require_backend(m)?.journey(m, req, p)
}

/// Generates tokens with a per-step timeline (feature `trace`, supported architectures only).
pub fn generate(m: &ModelHandle, req: &GenerateRequest, p: &dyn Progress) -> Result<GenerateReport> {
    require_backend(m)?.generate(m, req, p)
}
