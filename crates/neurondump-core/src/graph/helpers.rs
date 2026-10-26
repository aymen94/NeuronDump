//! Id, label and block / stage classification helpers shared by building and viewing.

use super::*;

pub(super) fn tensor_id(name: &str) -> String {
    format!("t.{name}")
}

pub(super) fn block_label(b: &str) -> &'static str {
    match b {
        "attention" => "attention",
        "ssm" => "linear attention (SSM)",
        "ffn" => "feed-forward",
        "moe" => "mixture of experts",
        "mtp" => "next-token prediction head",
        _ => "other",
    }
}

pub(super) fn block_short(b: &str) -> &'static str {
    match b {
        "attention" => "attn",
        "ssm" => "ssm",
        "ffn" => "ffn",
        "moe" => "moe",
        "mtp" => "mtp",
        _ => "other",
    }
}

pub(super) fn block_from_short(s: &str) -> Option<&'static str> {
    Some(match s {
        "attn" | "attention" => "attention",
        "ssm" => "ssm",
        "ffn" => "ffn",
        "moe" => "moe",
        "mtp" => "mtp",
        "other" => "other",
        _ => return None,
    })
}

/// Which block of a layer a tensor role belongs to.
pub(super) fn block_of(role: &str, comp: Component, has_ssm: bool, has_ffn: bool, has_moe: bool) -> &'static str {
    let r = role.strip_prefix("unknown:").unwrap_or(role);
    if r.starts_with("nextn.") || r.starts_with("mtp.") || comp == Component::Mtp {
        return "mtp";
    }
    if comp == Component::Moe {
        return "moe";
    }
    if comp == Component::LinearAttention || r.starts_with("ssm_") {
        return "ssm";
    }
    let mixer = if has_ssm { "ssm" } else { "attention" };
    let ffn = if has_ffn || !has_moe { "ffn" } else { "moe" };
    if r.starts_with("attn_") || r == "post_attention_norm" {
        // BERT-style post-norms belong to the residual stream after the attention add.
        return mixer;
    }
    if r.starts_with("ffn_") || matches!(r, "post_ffw_norm" | "layer_output_norm" | "layer_out_norm") {
        return ffn;
    }
    "other"
}

/// Position of a tensor inside its block's dataflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Stage {
    Pre,
    In,
    Out,
    Post,
    /// Norm applied to the residual stream after the add (BERT post-norm).
    PostAdd,
}

pub(super) fn stage_of(role: &str) -> Stage {
    let r = role.strip_prefix("unknown:").unwrap_or(role);
    match r {
        "attn_norm" | "ffn_norm" | "attn_norm_2" => Stage::Pre,
        "attn_output_norm" | "layer_output_norm" | "layer_out_norm" => Stage::PostAdd,
        "post_attention_norm" | "post_ffw_norm" | "attn_post_norm" | "ffn_post_norm" => Stage::Post,
        "attn_output" | "attn_out" | "ffn_down" | "ffn_down_exps" | "ffn_down_shexp" | "ssm_out" => Stage::Out,
        _ => Stage::In,
    }
}

pub(super) fn ancestors_of(nodes: &IndexMap<String, Node>, id: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = nodes.get(id).and_then(|n| n.parent.clone());
    while let Some(p) = cur {
        cur = nodes.get(&p).and_then(|n| n.parent.clone());
        out.push(p);
    }
    out
}

pub(super) fn new_node(id: &str, kind: NodeKind, label: &str, parent: Option<&str>) -> Node {
    Node {
        id: id.to_string(),
        kind,
        label: label.to_string(),
        parent: parent.map(String::from),
        layer: None,
        block: None,
        layer_kind: None,
        params: 0,
        bytes: 0,
        quant: vec![],
        stats: None,
        importance: Importance { score: 0.5, source: "structural: not measured (run `sweep`)".into() },
        activation: None,
        contains: vec![],
        removable: Removable::Risky,
        removable_reason: String::new(),
        lazy: false,
        info: Value::Null,
    }
}

pub(super) fn enum_str<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default()
}

pub(super) fn layer_kind_label(l: &LayerInfo) -> String {
    format!("{}+{}", enum_str(&l.mixer), enum_str(&l.ffn))
}

/// Parses `layer.N...` / `blk.N` / `LN` style prefixes.
pub(super) fn parse_layer_prefix(s: &str) -> Option<(u32, &str)> {
    let rest = s.strip_prefix("layer.").or_else(|| s.strip_prefix("blk."))?;
    let (n, tail) = match rest.split_once('.') {
        Some((n, t)) => (n, t),
        None => (rest, ""),
    };
    Some((n.parse().ok()?, tail))
}

// --- build -----------------------------------------------------------------------------
