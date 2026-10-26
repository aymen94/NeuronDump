//! Tensor names → layer, role and component (attention, FFN, MoE, norm, embedding...).

use super::*;

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

pub(super) const KNOWN_LAYER_ROLES: &[&str] = &[
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

pub(super) const KNOWN_TOP_ROLES: &[&str] = &[
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

pub fn tensor_roles(g: &GgufFile) -> Vec<TensorRole> {
    g.tensors.iter().map(|t| parse_tensor_name(&t.name)).collect()
}
