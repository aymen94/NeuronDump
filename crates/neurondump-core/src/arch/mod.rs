//! Architecture-agnostic structure inference from metadata and tensor names.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::Serialize;

use neurondump_gguf::{GgmlType, GgufFile, TensorInfo, Value};

use crate::util::value_json;

mod roles;
mod analyze;
pub use roles::*;
pub use analyze::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mixer {
    FullAttention,
    SlidingAttention,
    LinearAttention,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnKind {
    Dense,
    Moe,
    MoeWithShared,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Decoder,
    Encoder,
    Moe,
    Hybrid,
    VisionProjector,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerInfo {
    pub index: u32,
    pub mixer: Mixer,
    pub ffn: FfnKind,
    pub params: u64,
    pub bytes: u64,
    pub n_tensors: usize,
    pub roles: Vec<String>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub ffn_length: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypeUsage {
    pub ggml_type: GgmlType,
    pub tensors: usize,
    pub params: u64,
    pub bytes: u64,
    pub bits_per_weight: Option<f64>,
}

/// Hyperparameters pulled from the standard `<arch>.*` keys (absent ones are None).
#[derive(Debug, Clone, Default, Serialize)]
pub struct Hyperparams {
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    pub gqa_group: Option<u64>,
    pub rope_dimension_count: Option<u64>,
    pub rope_freq_base: Option<f64>,
    pub rope_scaling_type: Option<String>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub expert_feed_forward_length: Option<u64>,
    pub sliding_window: Option<u64>,
    pub ssm_conv_kernel: Option<u64>,
    pub ssm_state_size: Option<u64>,
    pub ssm_inner_size: Option<u64>,
    pub full_attention_interval: Option<u64>,
    pub vocab_size: Option<u64>,
    pub layer_norm_rms_epsilon: Option<f64>,
    pub layer_norm_epsilon: Option<f64>,
    pub pooling_type: Option<u64>,
    pub causal: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureReport {
    pub name: String,
    pub kind: ModelKind,
    /// All `<arch>.*` keys (prefix stripped), as JSON.
    pub keys: IndexMap<String, serde_json::Value>,
    pub hyperparams: Hyperparams,
    pub total_params: u64,
    pub params_by_component: BTreeMap<String, u64>,
    pub bytes_by_component: BTreeMap<String, u64>,
    pub types: Vec<TypeUsage>,
    pub layers: Vec<LayerInfo>,
    pub layer_kinds: BTreeMap<String, Vec<u32>>,
    pub unknown_roles: Vec<String>,
    pub tied_embeddings: bool,
    pub warnings: Vec<String>,
}

/// Value for layer `i` of a key that may be a scalar or a per-layer array.
pub fn per_layer(v: Option<&Value>, i: usize) -> Option<u64> {
    match v? {
        Value::Array(a) => a.get_i64(i).and_then(|x| u64::try_from(x).ok()),
        v => v.as_u64(),
    }
}

/// Max over a scalar or per-layer array key.
fn scalar_or_max(v: Option<&Value>) -> Option<u64> {
    match v? {
        Value::Array(a) => (0..a.len() as usize).filter_map(|i| a.get_i64(i)).max().and_then(|x| u64::try_from(x).ok()),
        v => v.as_u64(),
    }
}

impl Hyperparams {
    pub fn from_gguf(g: &GgufFile) -> Self {
        let k = |s: &str| g.arch_key(s);
        let u = |s: &str| k(s).and_then(Value::as_u64);
        let f = |s: &str| k(s).and_then(Value::as_f64);
        let head_count = scalar_or_max(k("attention.head_count"));
        let head_count_kv = scalar_or_max(k("attention.head_count_kv")).or(head_count);
        let vocab_size = u("vocab_size")
            .or_else(|| g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).map(|a| a.len()))
            .or_else(|| g.tensor("token_embd.weight").and_then(|t| t.dims.get(1).copied()));
        Self {
            context_length: u("context_length"),
            embedding_length: u("embedding_length"),
            block_count: u("block_count"),
            feed_forward_length: scalar_or_max(k("feed_forward_length")),
            head_count,
            head_count_kv,
            key_length: u("attention.key_length"),
            value_length: u("attention.value_length"),
            gqa_group: match (head_count, head_count_kv) {
                (Some(h), Some(kv)) if kv > 0 => Some(h / kv),
                _ => None,
            },
            rope_dimension_count: u("rope.dimension_count"),
            rope_freq_base: f("rope.freq_base"),
            rope_scaling_type: k("rope.scaling.type").and_then(Value::as_str).map(str::to_string),
            expert_count: u("expert_count"),
            expert_used_count: u("expert_used_count"),
            expert_feed_forward_length: u("expert_feed_forward_length"),
            sliding_window: u("attention.sliding_window"),
            ssm_conv_kernel: u("ssm.conv_kernel"),
            ssm_state_size: u("ssm.state_size"),
            ssm_inner_size: u("ssm.inner_size"),
            full_attention_interval: u("full_attention_interval"),
            vocab_size,
            layer_norm_rms_epsilon: f("attention.layer_norm_rms_epsilon"),
            layer_norm_epsilon: f("attention.layer_norm_epsilon"),
            pooling_type: u("pooling_type"),
            causal: k("attention.causal").and_then(Value::as_bool),
        }
    }
}
