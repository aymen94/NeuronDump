//! Interventions for the forward pass: ablation masks, embedding edits, steering, patching, sampling.

use super::*;

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
