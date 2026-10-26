//! Token journey and generation types: attributions, decision breakdown, per-layer journey, steps.

use super::*;

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

pub(super) fn minus_one() -> i64 {
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
