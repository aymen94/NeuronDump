//! Architecture-agnostic structure inference from metadata and tensor names.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use serde::Serialize;

use neurondump_gguf::{GgmlType, GgufFile, TensorInfo, Value};

use crate::util::value_json;

/// Coarse component a tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Embedding,
    Attention,
    LinearAttention,
    Ffn,
    Moe,
    Norm,
    Output,
    Vision,
    /// Multi-token-prediction / next-n speculative heads (`mtp.*`, `nextn.*`).
    Mtp,
    Other,
}

impl Component {
    pub fn as_str(self) -> &'static str {
        match self {
            Component::Embedding => "embedding",
            Component::Attention => "attention",
            Component::LinearAttention => "linear_attention",
            Component::Ffn => "ffn",
            Component::Moe => "moe",
            Component::Norm => "norm",
            Component::Output => "output",
            Component::Vision => "vision",
            Component::Mtp => "mtp",
            Component::Other => "other",
        }
    }
}

/// Parsed tensor name: `blk.{layer}.{role}.{param}` or a top-level `{role}.{param}`.
#[derive(Debug, Clone, Serialize)]
pub struct TensorRole {
    pub layer: Option<u32>,
    /// e.g. `attn_q`, `ffn_down_exps`; `unknown:<suffix>` for unrecognized names.
    pub role: String,
    /// `weight`, `bias`, `scale`, ... (last name component), empty if absent.
    pub param: String,
    pub component: Component,
    pub known: bool,
}

const KNOWN_LAYER_ROLES: &[&str] = &[
    "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output", "attn_out", "attn_q_norm", "attn_k_norm",
    "attn_q_a", "attn_q_b", "attn_q_a_norm", "attn_kv_a_mqa", "attn_kv_a_norm", "attn_kv_b", "attn_k_b",
    "attn_v_b", "attn_sinks", "attn_gate", "attn_norm", "attn_norm_2", "attn_post_norm", "attn_rel_b",
    "attn_output_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "layer_output_norm", "ffn_gate",
    "ffn_up", "ffn_down", "ffn_gate_up", "ffn_act", "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps",
    "ffn_down_exps", "ffn_gate_up_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    "ffn_gate_inp_shexp", "exp_probs_b", "ffn_exp_probs_b", "ffn_post_norm", "ffn_sub_norm", "ffn_norm_exps",
    "ssm_in", "ssm_conv1d", "ssm_x", "ssm_dt", "ssm_a", "ssm_d", "ssm_out", "ssm_norm", "ssm_beta_alpha",
    "ssm_ba", "ssm_beta", "ssm_alpha", "layer_out_norm", "attn_sub_norm", "time_mix_lerp_x", "channel_mix_key",
    "rope_freqs", "per_layer_inp_gate", "per_layer_proj", "per_layer_post_norm", "altup_router", "laurel_l",
    "laurel_r", "laurel_post_norm",
];

const KNOWN_TOP_ROLES: &[&str] = &[
    "token_embd", "token_embd_norm", "token_types", "pos_embd", "output", "output_norm", "rope_freqs",
    "cls", "cls.output", "rope_factors_long", "rope_factors_short", "per_layer_token_embd", "per_layer_model_proj",
    "per_layer_proj_norm", "altup_proj", "altup_unembd_proj",
];

pub fn component_of(role: &str, top_level: bool) -> Component {
    let r = role.strip_prefix("unknown:").unwrap_or(role);
    if r.starts_with("mtp.") || r.starts_with("nextn.") {
        return Component::Mtp;
    }
    if top_level {
        return match r {
            "token_embd" | "pos_embd" | "token_types" | "per_layer_token_embd" => Component::Embedding,
            "output" | "cls" | "cls.output" => Component::Output,
            _ if r.contains("norm") => Component::Norm,
            _ if r.starts_with("v.") || r.starts_with("mm.") || r.starts_with("a.") => Component::Vision,
            _ => Component::Other,
        };
    }
    if r.contains("norm") {
        Component::Norm
    } else if r.contains("_exps") || r.contains("shexp") || r.contains("gate_inp") || r.contains("exp_probs") {
        Component::Moe
    } else if r.starts_with("ssm_") || r.starts_with("time_mix") || r.starts_with("channel_mix") {
        Component::LinearAttention
    } else if r.starts_with("attn_") {
        Component::Attention
    } else if r.starts_with("ffn_") {
        Component::Ffn
    } else {
        Component::Other
    }
}

pub fn parse_tensor_name(name: &str) -> TensorRole {
    let (layer, rest) = match name.strip_prefix("blk.") {
        Some(r) => match r.split_once('.') {
            Some((n, rest)) => match n.parse::<u32>() {
                Ok(n) => (Some(n), rest),
                Err(_) => (None, name),
            },
            None => (None, name),
        },
        None => (None, name),
    };
    let (role, param) = match rest.rsplit_once('.') {
        Some((r, p)) if matches!(p, "weight" | "bias" | "scale" | "scales" | "a" | "b" | "inv_freq") => (r, p),
        _ => (rest, ""),
    };
    let known = if role.starts_with("mtp.") || role.starts_with("nextn.") {
        true
    } else if layer.is_some() {
        KNOWN_LAYER_ROLES.contains(&role)
    } else {
        KNOWN_TOP_ROLES.contains(&role) || role.starts_with("v.") || role.starts_with("mm.")
    };
    let role_s = if known { role.to_string() } else { format!("unknown:{role}") };
    TensorRole { layer, component: component_of(role, layer.is_none()), role: role_s, param: param.to_string(), known }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

fn layer_kind(roles: &[&str]) -> (Mixer, FfnKind) {
    let has = |p: &dyn Fn(&str) -> bool| roles.iter().any(|r| p(r));
    let attn = has(&|r| matches!(r, "attn_q" | "attn_k" | "attn_v" | "attn_qkv" | "attn_q_a" | "attn_q_b" | "attn_kv_a_mqa"));
    let linear = has(&|r| r.starts_with("ssm_") || r.starts_with("time_mix"));
    let mixer = if linear {
        Mixer::LinearAttention
    } else if attn {
        Mixer::FullAttention
    } else {
        Mixer::None
    };
    let moe = has(&|r| r.contains("_exps"));
    let shared = has(&|r| r.contains("shexp"));
    let dense = has(&|r| matches!(r, "ffn_up" | "ffn_down" | "ffn_gate" | "ffn_gate_up"));
    let ffn = match (moe, shared, dense) {
        (true, true, _) => FfnKind::MoeWithShared,
        (true, false, _) => FfnKind::Moe,
        (false, _, true) => FfnKind::Dense,
        _ => FfnKind::None,
    };
    (mixer, ffn)
}

/// `sliding_window_pattern` is either a per-layer bool/int array or a period N where
/// every N-th layer is global and the rest use the sliding window.
fn is_sliding(pattern: Option<&Value>, i: usize) -> bool {
    match pattern {
        Some(Value::Array(neurondump_gguf::Array::Bool(v))) => v.get(i).copied().unwrap_or(false),
        Some(Value::Array(a)) => a.get_i64(i).is_some_and(|v| v != 0),
        Some(v) => v.as_u64().is_some_and(|n| n > 1 && (i as u64 + 1) % n != 0),
        None => false,
    }
}

pub fn tensor_roles(g: &GgufFile) -> Vec<TensorRole> {
    g.tensors.iter().map(|t| parse_tensor_name(&t.name)).collect()
}

pub fn analyze_architecture(g: &GgufFile) -> ArchitectureReport {
    let name = g.architecture().unwrap_or("unknown").to_string();
    let mut warnings = Vec::new();
    if g.architecture().is_none() {
        warnings.push("general.architecture is missing".into());
    }
    let prefix = format!("{name}.");
    let keys: IndexMap<String, serde_json::Value> = g
        .metadata
        .iter()
        .filter_map(|(k, v)| k.strip_prefix(&prefix).map(|s| (s.to_string(), value_json(v, Some(64)))))
        .collect();
    let hp = Hyperparams::from_gguf(g);

    let roles = tensor_roles(g);
    let mut params_by_component: BTreeMap<String, u64> = BTreeMap::new();
    let mut bytes_by_component: BTreeMap<String, u64> = BTreeMap::new();
    let mut types: BTreeMap<GgmlType, TypeUsage> = BTreeMap::new();
    let mut unknown_roles: Vec<String> = Vec::new();
    let mut per_layer_t: BTreeMap<u32, Vec<(&TensorInfo, &TensorRole)>> = BTreeMap::new();

    for (t, r) in g.tensors.iter().zip(&roles) {
        *params_by_component.entry(r.component.as_str().into()).or_default() += t.n_elements;
        *bytes_by_component.entry(r.component.as_str().into()).or_default() += t.n_bytes;
        let tu = types.entry(t.ggml_type).or_insert(TypeUsage {
            ggml_type: t.ggml_type,
            tensors: 0,
            params: 0,
            bytes: 0,
            bits_per_weight: t.ggml_type.bits_per_weight(),
        });
        tu.tensors += 1;
        tu.params += t.n_elements;
        tu.bytes += t.n_bytes;
        if !r.known && !unknown_roles.contains(&r.role) {
            unknown_roles.push(r.role.clone());
        }
        if let Some(l) = r.layer {
            per_layer_t.entry(l).or_default().push((t, r));
        }
    }

    let hc_key = g.arch_key("attention.head_count");
    let hkv_key = g.arch_key("attention.head_count_kv");
    let ff_key = g.arch_key("feed_forward_length");
    let swa_pattern = g.arch_key("attention.sliding_window_pattern");
    let mut layers = Vec::new();
    let mut layer_kinds: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for (idx, ts) in &per_layer_t {
        let role_names: Vec<&str> = ts.iter().map(|(_, r)| r.role.strip_prefix("unknown:").unwrap_or(&r.role)).collect();
        let (mut mixer, ffn) = layer_kind(&role_names);
        // Per-layer head_count_kv == 0 marks a non-attention (recurrent) layer in hybrids.
        let hkv = per_layer(hkv_key, *idx as usize);
        if mixer == Mixer::FullAttention && is_sliding(swa_pattern, *idx as usize) {
            mixer = Mixer::SlidingAttention;
        }
        let mut uniq: Vec<String> = Vec::new();
        for r in &role_names {
            if !uniq.iter().any(|u| u == r) {
                uniq.push(r.to_string());
            }
        }
        let info = LayerInfo {
            index: *idx,
            mixer,
            ffn,
            params: ts.iter().map(|(t, _)| t.n_elements).sum(),
            bytes: ts.iter().map(|(t, _)| t.n_bytes).sum(),
            n_tensors: ts.len(),
            roles: uniq,
            head_count: per_layer(hc_key, *idx as usize),
            head_count_kv: hkv,
            ffn_length: per_layer(ff_key, *idx as usize),
        };
        let kind_label = format!(
            "{}+{}",
            serde_json::to_value(mixer).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default(),
            serde_json::to_value(ffn).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
        );
        layer_kinds.entry(kind_label).or_default().push(*idx);
        layers.push(info);
    }

    if let Some(bc) = hp.block_count {
        if bc as usize != layers.len() && !layers.is_empty() {
            warnings.push(format!("block_count is {bc} but tensors exist for {} layers", layers.len()));
        }
    }

    let is_projector = name == "clip" || g.metadata.keys().any(|k| k.starts_with("clip."));
    let mixers: std::collections::HashSet<_> =
        layers.iter().map(|l| if l.mixer == Mixer::SlidingAttention { Mixer::FullAttention } else { l.mixer }).collect();
    // Pooling alone does not make an encoder: qwen3-embedding is a causal decoder with
    // last-token pooling.
    let encoder = hp.causal == Some(false) || name.contains("bert") || name == "t5encoder";
    if hp.pooling_type.is_some_and(|p| p > 0) && !encoder {
        warnings.push(format!("{name} is a causal decoder used as an embedding model (pooling_type {})", hp.pooling_type.unwrap_or(0)));
    }
    let kind = if is_projector {
        ModelKind::VisionProjector
    } else if mixers.len() > 1 {
        ModelKind::Hybrid
    } else if layers.iter().any(|l| matches!(l.ffn, FfnKind::Moe | FfnKind::MoeWithShared)) || hp.expert_count.unwrap_or(0) > 0 {
        ModelKind::Moe
    } else if encoder {
        ModelKind::Encoder
    } else if layers.is_empty() {
        ModelKind::Unknown
    } else {
        ModelKind::Decoder
    };

    let tied_embeddings = g.tensor("token_embd.weight").is_some() && g.tensor("output.weight").is_none() && !encoder;

    ArchitectureReport {
        name,
        kind,
        keys,
        hyperparams: hp,
        total_params: g.total_params(),
        params_by_component,
        bytes_by_component,
        types: types.into_values().collect(),
        layers,
        layer_kinds,
        unknown_roles,
        tied_embeddings,
        warnings,
    }
}
