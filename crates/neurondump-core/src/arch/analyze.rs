//! `analyze_architecture`: layers, their kinds (full / sliding / linear attention, dense / MoE), sizes.

use super::*;

pub(super) fn layer_kind(roles: &[&str]) -> (Mixer, FfnKind) {
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
pub(super) fn is_sliding(pattern: Option<&Value>, i: usize) -> bool {
    match pattern {
        Some(Value::Array(neurondump_gguf::Array::Bool(v))) => v.get(i).copied().unwrap_or(false),
        Some(Value::Array(a)) => a.get_i64(i).is_some_and(|v| v != 0),
        Some(v) => v.as_u64().is_some_and(|n| n > 1 && (i as u64 + 1) % n != 0),
        None => false,
    }
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
