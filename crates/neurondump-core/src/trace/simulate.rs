//! Simulation (baseline vs edited) and sweep request / report types.

use super::*;

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
