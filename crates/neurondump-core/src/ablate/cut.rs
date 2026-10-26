//! Resolving cut node ids to targets, and the matching in-memory `Ablation` for simulation.

use super::*;

pub(super) fn parse_vocab_range(id: &str, n_vocab: u32) -> Option<(u32, u32)> {
    let r = id.strip_prefix("vocab:")?;
    let (a, b) = match r.split_once('-') {
        Some((a, b)) => (a.trim().parse().ok()?, b.trim().parse().ok()?),
        None => {
            let a: u32 = r.trim().parse().ok()?;
            (a, a)
        }
    };
    (a <= b && b < n_vocab).then_some((a, b))
}

/// Tensors whose output is the block's contribution to the residual stream.
pub(crate) fn block_out_tensors(graph: &ModelGraph, block_id: &str) -> Vec<String> {
    let ts: Vec<&Node> = graph.tensors_under(block_id);
    let outs: Vec<String> = ts
        .iter()
        .filter(|t| {
            let role = t.info["role"].as_str().unwrap_or("");
            matches!(role, "attn_output" | "attn_out" | "ffn_down" | "ffn_down_exps" | "ffn_down_shexp" | "ssm_out")
        })
        .filter_map(|t| t.info["name"].as_str().map(String::from))
        .collect();
    if outs.is_empty() {
        ts.iter().filter_map(|t| t.info["name"].as_str().map(String::from)).collect()
    } else {
        outs
    }
}

/// Resolves ids, drops descendants of other cut ids, and classifies each target.
pub fn resolve_cut(graph: &ModelGraph, vocab: Option<&Vocab>, ids: &[String]) -> (Vec<CutTarget>, Vec<String>, Vec<String>) {
    let n_vocab = vocab.map_or(0, |v| v.tokens.len() as u32);
    let mut unknown = Vec::new();
    let mut resolved: Vec<String> = Vec::new();
    for raw in ids.iter().flat_map(|s| s.split(',')).map(str::trim).filter(|s| !s.is_empty()) {
        match graph.resolve_id(raw) {
            Some(id) if !id.starts_with("vocab:") || parse_vocab_range(&id, n_vocab).is_some() => {
                if !resolved.contains(&id) {
                    resolved.push(id);
                }
            }
            _ => unknown.push(raw.to_string()),
        }
    }
    let set: HashSet<String> = resolved.iter().cloned().collect();
    let mut redundant = Vec::new();
    let mut out = Vec::new();
    for id in &resolved {
        if graph.ancestors(id).iter().any(|a| set.contains(a)) {
            redundant.push(id.clone());
            continue;
        }
        if let Some((a, b)) = parse_vocab_range(id, n_vocab) {
            let n = (b - a + 1) as u64;
            let per_token: u64 = graph
                .nodes
                .values()
                .filter(|n| matches!(n.info["name"].as_str(), Some("token_embd.weight") | Some("output.weight")))
                .map(|n| n.bytes / n_vocab.max(1) as u64)
                .sum();
            out.push(CutTarget {
                id: id.clone(),
                kind: NodeKind::Token,
                label: format!("tokens {a}-{b}"),
                mode: CutMode::Remove,
                params: n,
                bytes: n * per_token,
                removable: Removable::Risky,
                reason: "removed from the vocabulary (the model can no longer read or write them)".into(),
            });
            continue;
        }
        let Some(n) = graph.get(id) else { continue };
        let mode = match n.kind {
            NodeKind::Op | NodeKind::Model => CutMode::None,
            NodeKind::Layer | NodeKind::Token => CutMode::Remove,
            NodeKind::Section => match n.id.as_str() {
                "projector" | "template" | "mtp" | "vision" | "extra" => CutMode::Remove,
                "body" => CutMode::Remove,
                _ => CutMode::Zero,
            },
            _ => CutMode::Zero,
        };
        out.push(CutTarget {
            id: n.id.clone(),
            kind: n.kind,
            label: n.label.clone(),
            mode,
            params: n.params,
            bytes: n.bytes,
            removable: if mode == CutMode::None { Removable::BreaksModel } else { n.removable },
            reason: if mode == CutMode::None { "residual stream operation: cannot be cut".into() } else { n.removable_reason.clone() },
        });
    }
    (out, unknown, redundant)
}

/// The in-memory ablation equivalent to a set of cut targets.
pub fn ablation_for(graph: &ModelGraph, targets: &[CutTarget], n_vocab: u32) -> Ablation {
    let mut a = Ablation::default();
    for t in targets {
        if let Some(r) = parse_vocab_range(&t.id, n_vocab) {
            a.mask_tokens.push(r);
            continue;
        }
        let Some(n) = graph.get(&t.id) else { continue };
        match n.kind {
            NodeKind::Layer => {
                if let Some(l) = n.layer {
                    a.skip_layers.push(l);
                }
            }
            NodeKind::Section if n.id == "body" => {
                a.skip_layers.extend(graph.nodes.values().filter(|x| x.kind == NodeKind::Layer).filter_map(|x| x.layer));
            }
            NodeKind::Section | NodeKind::Tensor => {
                if !matches!(n.id.as_str(), "projector" | "template" | "tokenizer") && n.info["virtual"].as_bool() != Some(true) {
                    a.zero_tensors.extend(graph.tensors_under(&n.id).iter().filter_map(|x| x.info["name"].as_str().map(String::from)));
                }
                if n.id == "output.head" {
                    // Tied head: zeroing it means zeroing the shared embedding matrix.
                    a.zero_tensors.push("token_embd.weight".into());
                }
            }
            NodeKind::Block => {
                if let (Some(l), Some(b)) = (n.layer, n.block.clone()) {
                    if matches!(b.as_str(), "attention" | "ffn" | "ssm" | "moe") {
                        a.zero_blocks.push((l, b));
                    } else {
                        a.zero_tensors.extend(graph.tensors_under(&n.id).iter().filter_map(|x| x.info["name"].as_str().map(String::from)));
                    }
                }
            }
            NodeKind::KvGroup => {
                if let (Some(l), Some(hs)) = (n.layer, n.info["heads"].as_array()) {
                    a.zero_heads.extend(hs.iter().filter_map(|h| h.as_u64()).map(|h| (l, h as u32)));
                }
            }
            NodeKind::Head => {
                if let (Some(l), Some(h)) = (n.layer, n.info["head"].as_u64()) {
                    a.zero_heads.push((l, h as u32));
                }
            }
            NodeKind::NeuronBucket => {
                if let (Some(l), Some(s), Some(e)) = (n.layer, n.info["start"].as_u64(), n.info["end"].as_u64()) {
                    a.zero_neurons.push((l, s as u32, e as u32));
                }
            }
            NodeKind::Expert => {
                if let (Some(l), Some(e)) = (n.layer, n.info["expert"].as_u64()) {
                    a.zero_experts.push((l, e as u32));
                }
            }
            NodeKind::Token => {
                if let Some(id) = n.info["id"].as_u64() {
                    a.mask_tokens.push((id as u32, id as u32));
                }
            }
            _ => {}
        }
    }
    a.skip_layers.sort_unstable();
    a.skip_layers.dedup();
    a.zero_tensors.sort();
    a.zero_tensors.dedup();
    a
}
