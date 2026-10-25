//! "What happens if I cut this?": static impact of removing or zeroing graph nodes, the
//! matching in-memory [`Ablation`] for simulation, sweeps, and conversion to a [`PrunePlan`].
//!
//! Every cut is either a **remove** (layers, tokens, the projector, the chat template,
//! optional tensors: the file gets smaller) or a **zero** (blocks, heads, neuron buckets,
//! experts, single tensors: weights are set to 0 in place, so every shape stays the same
//! and llama.cpp still loads the file, but it does not get smaller).

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use neurondump_gguf::{GgmlType, GgufFile};

use crate::analyze::{row_bytes, vocab_tensors, Candidate, CandidateKind, PrunePlan, Risk};
use crate::graph::{EdgeKind, ModelGraph, Node, NodeKind, Removable};
use crate::model::ModelHandle;
use crate::tokenizer::Vocab;
use crate::trace::{Ablation, SimReport, SimRequest, SweepReport, SweepRequest, SweepTarget};
use crate::util::{human_bytes, Progress, SCHEMA_VERSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CutMode {
    /// Deleted from the file (smaller file).
    Remove,
    /// Weights set to zero in place (same shapes, same size).
    Zero,
    /// Cannot be cut (residual stream ops).
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutTarget {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub mode: CutMode,
    pub params: u64,
    pub bytes: u64,
    pub removable: Removable,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticImpact {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub cut: Vec<CutTarget>,
    /// Ids that matched nothing.
    pub unknown: Vec<String>,
    /// Ids dropped because an ancestor is also cut.
    pub redundant: Vec<String>,
    pub total_params: u64,
    pub params_removed: u64,
    pub params_zeroed: u64,
    pub percent_of_model: f64,
    pub original_size: u64,
    pub bytes_removed: u64,
    /// Size change from requantizing tensors whose columns are zeroed (Q8_0), usually > 0.
    pub requant_bytes_delta: i64,
    pub new_size: u64,
    /// Bytes saved outside the main GGUF (vision projector file).
    pub other_files_removed: u64,
    pub verdict: Removable,
    pub reasons: Vec<String>,
    /// Nodes left without a path from the input to the output.
    pub disconnected: Vec<String>,
    pub dependencies: Vec<String>,
    pub llama_cpp_loads: bool,
    pub legality: Vec<String>,
    /// (old, new) indices of layers that move.
    pub layer_renumbering: Vec<(u32, u32)>,
    /// What `slim` would do.
    pub file_ops: Vec<String>,
    /// Refused: verdict is breaks_model and `force` was not given.
    pub refused: bool,
    pub ablation: Ablation,
    pub simulation_supported: bool,
    pub simulation_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblateResult {
    pub impact: StaticImpact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simulation: Option<SimReport>,
}

fn parse_vocab_range(id: &str, n_vocab: u32) -> Option<(u32, u32)> {
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

/// One zeroing operation on a tensor (element ranges are half-open).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZeroOp {
    pub tensor: String,
    #[serde(default)]
    pub all: bool,
    /// Ranges of rows (along dims[1]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<(u64, u64)>,
    /// Ranges of elements inside every row (along dims[0]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cols: Vec<(u64, u64)>,
    /// Ranges along dims[2] (MoE experts).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slices: Vec<(u64, u64)>,
}

/// File-level zero operations equivalent to the zeroing part of an ablation.
pub fn zero_ops(g: &GgufFile, graph: &ModelGraph, a: &Ablation) -> Vec<ZeroOp> {
    let mut ops: BTreeMap<String, ZeroOp> = BTreeMap::new();
    fn get<'a>(ops: &'a mut BTreeMap<String, ZeroOp>, g: &GgufFile, name: &str) -> Option<&'a mut ZeroOp> {
        g.tensor(name)?;
        Some(ops.entry(name.to_string()).or_insert_with(|| ZeroOp { tensor: name.to_string(), ..Default::default() }))
    }
    macro_rules! op {
        ($name:expr) => {
            get(&mut ops, g, &$name)
        };
    }
    for t in &a.zero_tensors {
        if let Some(o) = op!(t) {
            o.all = true;
        }
    }
    for (l, b) in &a.zero_blocks {
        let bid = format!("layer.{l}.{}", match b.as_str() {
            "attention" => "attn",
            other => other,
        });
        for name in block_out_tensors(graph, &bid) {
            if let Some(o) = op!(name) {
                o.all = true;
            }
            let bias = name.trim_end_matches(".weight").to_string() + ".bias";
            if let Some(o) = op!(bias) {
                o.all = true;
            }
        }
    }
    for (l, h) in &a.zero_heads {
        let hd = graph.dims.get(l).map_or(0, |d| d.head_dim);
        if hd == 0 {
            continue;
        }
        let range = (*h as u64 * hd, (*h as u64 + 1) * hd);
        if let Some(o) = op!(format!("blk.{l}.attn_output.weight")) {
            o.cols.push(range);
        }
    }
    for (l, s, e) in &a.zero_neurons {
        let n_ff = graph.dims.get(l).map_or(0, |d| d.n_ff);
        let range = (*s as u64, *e as u64 + 1);
        let gate = format!("blk.{l}.ffn_gate.weight");
        let up = format!("blk.{l}.ffn_up.weight");
        let fused = g.tensor(&gate).is_none() && g.tensor(&up).and_then(|t| t.dims.get(1).copied()) == Some(2 * n_ff);
        if let Some(o) = op!(up) {
            o.rows.push(range);
            if fused {
                o.rows.push((range.0 + n_ff, range.1 + n_ff));
            }
        }
        if let Some(o) = op!(gate) {
            o.rows.push(range);
        }
        for b in [format!("blk.{l}.ffn_up.bias"), format!("blk.{l}.ffn_gate.bias")] {
            if let Some(o) = op!(b) {
                o.cols.push(range);
            }
        }
    }
    for (l, e) in &a.zero_experts {
        for r in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"] {
            if let Some(o) = op!(format!("blk.{l}.{r}.weight")) {
                o.slices.push((*e as u64, *e as u64 + 1));
            }
        }
    }
    ops.into_values().collect()
}

/// Bytes added by zeroing columns of quantized tensors that need requantizing to Q8_0.
fn requant_delta(g: &GgufFile, ops: &[ZeroOp]) -> (i64, Vec<String>) {
    let mut delta = 0i64;
    let mut names = Vec::new();
    for o in ops.iter().filter(|o| !o.all && !o.cols.is_empty()) {
        let Some(t) = g.tensor(&o.tensor) else { continue };
        if !needs_requant(t.ggml_type, &o.cols) {
            continue;
        }
        if let Some(q) = GgmlType::Q8_0.bytes_for(t.n_elements) {
            delta += q as i64 - t.n_bytes as i64;
            names.push(o.tensor.clone());
        }
    }
    (delta, names)
}

/// Column zeroing works in place for float types and block-aligned ranges.
pub fn needs_requant(ty: GgmlType, cols: &[(u64, u64)]) -> bool {
    let Some(layout) = ty.layout() else { return true };
    if !layout.quantized {
        return false;
    }
    cols.iter().any(|(a, b)| a % layout.block_size != 0 || b % layout.block_size != 0)
}

fn reachable(starts: &[String], adj: &BTreeMap<String, Vec<String>>, blocked: &HashSet<String>) -> HashSet<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut q: VecDeque<String> = starts.iter().filter(|s| !blocked.contains(*s)).cloned().collect();
    while let Some(x) = q.pop_front() {
        if !seen.insert(x.clone()) {
            continue;
        }
        for y in adj.get(&x).into_iter().flatten() {
            if !blocked.contains(y) && !seen.contains(y) {
                q.push_back(y.clone());
            }
        }
    }
    seen
}

pub fn static_impact(m: &ModelHandle, graph: &ModelGraph, vocab: Option<&Vocab>, ids: &[String], force: bool) -> StaticImpact {
    let g = &m.gguf;
    let n_vocab = vocab.map_or(0, |v| v.tokens.len() as u32);
    let (targets, unknown, redundant) = resolve_cut(graph, vocab, ids);
    let ablation = ablation_for(graph, &targets, n_vocab);
    let mut reasons = Vec::new();
    let mut dependencies = Vec::new();
    let mut legality = Vec::new();
    let mut file_ops = Vec::new();
    let mut verdict = targets.iter().map(|t| t.removable).max().unwrap_or(Removable::Safe);
    let mut llama_ok = true;
    for t in &targets {
        reasons.push(format!("{}: {} ({})", t.id, t.removable.as_str(), t.reason));
    }
    if targets.is_empty() {
        reasons.push("nothing selected".into());
    }

    // --- sizes -----------------------------------------------------------------------
    let mut params_removed = 0u64;
    let mut params_zeroed = 0u64;
    let mut bytes_removed = 0u64;
    let mut other_files = 0u64;
    for t in &targets {
        match t.mode {
            CutMode::Remove if t.id == "projector" => other_files += t.bytes,
            CutMode::Remove if t.id == "template" => bytes_removed += t.bytes,
            CutMode::Remove => {
                params_removed += t.params;
                bytes_removed += t.bytes;
            }
            CutMode::Zero => params_zeroed += t.params,
            CutMode::None => {}
        }
    }
    let ops = zero_ops(g, graph, &ablation);
    let (delta, requant) = requant_delta(g, &ops);
    if !requant.is_empty() {
        file_ops.push(format!(
            "requantize {} tensor(s) to Q8_0 to zero unaligned columns ({}{}): {}",
            requant.len(),
            if delta >= 0 { "+" } else { "-" },
            human_bytes(delta.unsigned_abs()),
            requant.join(", ")
        ));
    }

    // --- layers ----------------------------------------------------------------------
    let n_layers = graph.n_layers;
    let removed: BTreeSet<u32> = ablation.skip_layers.iter().copied().collect();
    let mut renumber = Vec::new();
    if !removed.is_empty() {
        if removed.len() as u32 >= n_layers {
            verdict = Removable::BreaksModel;
            llama_ok = false;
            reasons.push("every layer is removed".into());
        }
        let mut next = 0;
        for l in 0..n_layers {
            if removed.contains(&l) {
                continue;
            }
            if l != next {
                renumber.push((l, next));
            }
            next += 1;
        }
        file_ops.push(format!("remove layer(s) {:?}; block_count {} -> {}", removed, n_layers, n_layers as usize - removed.len()));
        legality.push(format!("layers renumbered: {} later layer(s) move down", renumber.len()));
        if let Some(p) = graph.pattern_period {
            // Removing whole aligned periods keeps the hybrid pattern intact.
            let nextn = g.arch_key("nextn_predict_layers").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let main: BTreeSet<u32> = removed.iter().copied().filter(|l| *l < n_layers.saturating_sub(nextn)).collect();
            let p = p as u32;
            let whole = main.iter().all(|l| (l / p * p..l / p * p + p).all(|x| main.contains(&x)));
            if whole && !main.is_empty() {
                verdict = verdict.min(Removable::Risky).max(targets.iter().filter(|t| t.kind != NodeKind::Layer).map(|t| t.removable).max().unwrap_or(Removable::Safe));
                reasons.push(format!("removed layers form whole periods of {p} (the hybrid pattern stays aligned)"));
            } else if !main.is_empty() {
                verdict = Removable::BreaksModel;
                llama_ok = false;
                reasons.push(format!("hybrid pattern (period {p}) broken: remove whole aligned groups of {p} layers instead"));
            }
        }
    }

    // --- dependencies ------------------------------------------------------------------
    let cut_ids: HashSet<&str> = targets.iter().map(|t| t.id.as_str()).collect();
    let cut_under = |x: &str| cut_ids.contains(x) || graph.ancestors(x).iter().any(|a| cut_ids.contains(a.as_str()));
    if graph.tied_embeddings && (cut_under("t.token_embd.weight") || cut_under("output.head")) {
        dependencies.push("tied embeddings: token_embd is also the output head, so cutting one cuts both".into());
    }
    if let Some(v) = vocab {
        let tpl_specials: BTreeSet<u32> = graph
            .nodes
            .values()
            .filter(|n| n.kind == NodeKind::Token && n.info["roles"].as_array().is_some_and(|r| r.iter().any(|x| x == "template")))
            .filter_map(|n| n.info["id"].as_u64().map(|x| x as u32))
            .collect();
        let structural: BTreeSet<u32> = v.summary.specials.iter().filter(|s| s.role == "bos" || s.role == "eos").map(|s| s.id).collect();
        let hit_tpl: Vec<u32> = tpl_specials.iter().copied().filter(|id| ablation.masks_token(*id)).collect();
        let hit_struct: Vec<u32> = structural.iter().copied().filter(|id| ablation.masks_token(*id)).collect();
        if !hit_tpl.is_empty() {
            dependencies.push(format!("the chat template uses {} of the cut tokens ({:?}): chat formatting breaks", hit_tpl.len(), hit_tpl));
            verdict = verdict.max(Removable::Risky);
        }
        if !hit_struct.is_empty() {
            dependencies.push(format!("bos/eos token(s) {hit_struct:?} are cut: slim always keeps them, simulation masks them"));
            verdict = Removable::BreaksModel;
        }
        if !ablation.mask_tokens.is_empty() {
            let n: u32 = ablation.mask_tokens.iter().map(|(a, b)| b - a + 1).sum();
            file_ops.push(format!("remove {n} token(s) from the vocabulary, token_embd/output rows and merges"));
        }
    }
    if cut_ids.contains("tokenizer") {
        dependencies.push("without the tokenizer no text can be encoded".into());
    }

    // --- graph connectivity -------------------------------------------------------------
    let mut fwd: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut bwd: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for e in graph.edges.iter().filter(|e| e.kind == EdgeKind::Dataflow) {
        fwd.entry(e.from.clone()).or_default().push(e.to.clone());
        bwd.entry(e.to.clone()).or_default().push(e.from.clone());
    }
    // Skipped layers become identities: connect the input of each run of consecutive
    // removed layers to its output.
    let mut runs: Vec<Vec<u32>> = Vec::new();
    for &l in &removed {
        match runs.last_mut() {
            Some(r) if r.last() == Some(&(l.wrapping_sub(1))) => r.push(l),
            _ => runs.push(vec![l]),
        }
    }
    for run in &runs {
        let lids: Vec<String> = run.iter().map(|l| format!("layer.{l}")).collect();
        let inside = |x: &str| lids.iter().any(|lid| x == lid || graph.ancestors(x).contains(lid));
        let ins: Vec<String> = graph.edges.iter().filter(|e| e.kind == EdgeKind::Dataflow && !inside(&e.from) && inside(&e.to)).map(|e| e.from.clone()).collect();
        let outs: Vec<String> = graph.edges.iter().filter(|e| e.kind == EdgeKind::Dataflow && inside(&e.from) && !inside(&e.to)).map(|e| e.to.clone()).collect();
        for a in &ins {
            for b in &outs {
                fwd.entry(a.clone()).or_default().push(b.clone());
                bwd.entry(b.clone()).or_default().push(a.clone());
            }
        }
    }
    let blocked: HashSet<String> = graph
        .nodes
        .keys()
        .filter(|id| {
            let ablated_tensor = graph.nodes.get(*id).and_then(|n| n.info["name"].as_str()).is_some_and(|nm| ablation.zero_tensors.iter().any(|z| z == nm));
            cut_under(id) || ablated_tensor
        })
        .cloned()
        .collect();
    let source = if graph.nodes.contains_key("t.token_embd.weight") { "t.token_embd.weight" } else { "input" };
    let sink = if graph.nodes.contains_key("output.logits") { "output.logits" } else { "output.embedding" };
    let from_src = reachable(&[source.to_string()], &fwd, &blocked);
    let to_sink = reachable(&[sink.to_string()], &bwd, &blocked);
    // Nodes already off the main path without any cut (e.g. speculative next-n layers).
    let none = HashSet::new();
    let base_src = reachable(&[source.to_string()], &fwd, &none);
    let base_sink = reachable(&[sink.to_string()], &bwd, &none);
    let flow_nodes: BTreeSet<&String> = fwd.keys().chain(bwd.keys()).collect();
    let mut disconnected: BTreeSet<String> = BTreeSet::new();
    for n in flow_nodes {
        if blocked.contains(n) || (from_src.contains(n) && to_sink.contains(n)) || !(base_src.contains(n) && base_sink.contains(n)) {
            continue;
        }
        // Report at block / layer granularity.
        let anc = graph.ancestors(n);
        let shown = anc.iter().find(|a| graph.nodes.get(*a).is_some_and(|x| x.kind == NodeKind::Block)).cloned().unwrap_or_else(|| n.clone());
        if !shown.ends_with("mtp") && graph.nodes.get(&shown).is_none_or(|x| x.block.as_deref() != Some("mtp")) {
            disconnected.insert(shown);
        }
    }
    if !from_src.contains(sink) {
        verdict = Removable::BreaksModel;
        reasons.push("no path from the input embeddings to the output remains".into());
    }
    if !disconnected.is_empty() {
        reasons.push(format!("{} node(s) are cut off from the input or the output", disconnected.len()));
    }

    // --- file operations / legality -----------------------------------------------------
    for t in &targets {
        match (t.mode, t.id.as_str()) {
            (CutMode::Remove, "projector") => file_ops.push("drop the vision projector from the Modelfile".into()),
            (CutMode::Remove, "template") => file_ops.push("strip tokenizer.chat_template and the Modelfile TEMPLATE".into()),
            (CutMode::Remove, "mtp" | "vision" | "extra") => {
                file_ops.push(format!("drop the {} tensors of {}", graph.tensors_under(&t.id).len(), t.id));
                legality.push(format!("{}: not verified that llama.cpp loads the file without these tensors", t.id));
            }
            (CutMode::Zero, _) if t.removable == Removable::BreaksModel => {
                legality.push(format!("{}: zeroed tensors keep their shapes, so the file loads, but the model's output is garbage", t.id));
            }
            _ => {}
        }
    }
    if !ops.is_empty() {
        file_ops.push(format!("zero weights in place in {} tensor(s) (shapes unchanged)", ops.len()));
        legality.push("zeroed weights keep every tensor shape: llama.cpp loads the file unchanged".into());
    }
    if targets.iter().any(|t| t.mode == CutMode::None) {
        llama_ok = false;
    }

    // --- simulation support -----------------------------------------------------------------
    let mut sim_notes = Vec::new();
    let backend_ok = crate::trace::require_backend(m);
    let mut sim_supported = backend_ok.is_ok();
    if let Err(e) = &backend_ok {
        sim_notes.push(format!("{e:#}"));
    }
    if !ablation.zero_experts.is_empty() || ablation.zero_blocks.iter().any(|(_, b)| b == "moe" || b == "ssm") {
        sim_supported = false;
        sim_notes.push("the runtime forward pass has no MoE / SSM support: experts and SSM blocks can only be cut statically".into());
    }
    for t in &targets {
        if matches!(t.id.as_str(), "projector" | "template") {
            sim_notes.push(format!("{} has no effect on a raw-text forward pass", t.id));
        }
        if t.id == "tokenizer" {
            sim_supported = false;
            sim_notes.push("the tokenizer cannot be simulated away".into());
        }
    }
    if !ablation.mask_tokens.is_empty() {
        sim_notes.push("simulation masks the tokens in the output distribution; slim removes them from the vocabulary".into());
    }

    let refused = verdict == Removable::BreaksModel && !force;
    let total = graph.total_params.max(1);
    let new_size = (graph.file_size as i64 - bytes_removed as i64 + delta).max(0) as u64;
    StaticImpact {
        schema_version: SCHEMA_VERSION,
        model: graph.model.clone(),
        architecture: graph.architecture.clone(),
        cut: targets,
        unknown,
        redundant,
        total_params: graph.total_params,
        params_removed,
        params_zeroed,
        percent_of_model: 100.0 * (params_removed + params_zeroed) as f64 / total as f64,
        original_size: graph.file_size,
        bytes_removed,
        requant_bytes_delta: delta,
        new_size,
        other_files_removed: other_files,
        verdict,
        reasons,
        disconnected: {
            let mut d: Vec<String> = disconnected.into_iter().collect();
            d.sort_by_key(|x| (x.strip_prefix("layer.").and_then(|r| r.split('.').next()).and_then(|n| n.parse::<u32>().ok()).unwrap_or(u32::MAX), x.clone()));
            d
        },
        dependencies,
        llama_cpp_loads: llama_ok,
        legality,
        layer_renumbering: renumber,
        file_ops,
        refused,
        ablation,
        simulation_supported: sim_supported,
        simulation_notes: sim_notes,
    }
}

/// Runs the in-memory ablation through the trace backend.
pub fn simulate(m: &ModelHandle, impact: &StaticImpact, mut req: SimRequest, p: &dyn Progress) -> Result<SimReport> {
    if impact.refused {
        bail!("refused: this cut is `breaks_model` ({}); pass --force to simulate it anyway", impact.reasons.join("; "));
    }
    if !impact.simulation_supported {
        bail!("simulation is not available for this cut: {}", impact.simulation_notes.join("; "));
    }
    let b = crate::trace::require_backend(m)?;
    req.ablation = impact.ablation.clone();
    req.cut = impact.cut.iter().map(|c| c.id.clone()).collect();
    if req.prompts.is_empty() && req.corpus.is_empty() {
        req.prompts.push("The name of the star that Earth orbits is".into());
    }
    b.simulate(m, &req, p)
}

/// Builds sweep targets from the graph and runs them.
pub fn sweep(m: &ModelHandle, graph: &ModelGraph, mut req: SweepRequest, p: &dyn Progress) -> Result<SweepReport> {
    if let Some(f) = &req.corpus_file {
        req.corpus.extend(crate::trace::read_corpus(f)?);
    }
    if req.corpus.is_empty() {
        bail!("sweep needs a corpus (--corpus file, texts separated by blank lines)");
    }
    let b = crate::trace::require_backend(m)?;
    req.targets = match req.level.as_str() {
        "layer" | "layers" => graph
            .nodes
            .values()
            .filter(|n| n.kind == NodeKind::Layer)
            .filter_map(|n| {
                let l = n.layer?;
                Some(SweepTarget { id: n.id.clone(), label: n.label.clone(), ablation: Ablation { skip_layers: vec![l], ..Default::default() } })
            })
            .collect(),
        "head" | "heads" => {
            let l = req.layer.ok_or_else(|| anyhow::anyhow!("--level head needs --layer N"))?;
            let d = graph.dims.get(&l).ok_or_else(|| anyhow::anyhow!("no layer {l}"))?;
            if d.n_head == 0 {
                bail!("layer {l} has no attention heads");
            }
            (0..d.n_head as u32)
                .map(|h| SweepTarget {
                    id: format!("layer.{l}.attn.head.{h}"),
                    label: format!("layer {l} head {h}"),
                    ablation: Ablation { zero_heads: vec![(l, h)], ..Default::default() },
                })
                .collect()
        }
        "expert" | "experts" => bail!("expert sweeps need a MoE forward pass, which the trace backend does not have yet"),
        other => bail!("unknown sweep level {other:?} (layer, head, expert)"),
    };
    b.sweep(m, &req, p)
}

// --- plans ---------------------------------------------------------------------------------

fn plan_base(m: &ModelHandle, notes: Vec<String>) -> PrunePlan {
    PrunePlan {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        digest: m.entry.digest.clone(),
        architecture: m.architecture().to_string(),
        original_size: m.gguf.file_size,
        candidates: vec![],
        notes,
    }
}

/// A prune plan that applies exactly this cut (all candidates selected).
pub fn plan_from_impact(m: &ModelHandle, graph: &ModelGraph, impact: &StaticImpact, force: bool) -> Result<PrunePlan> {
    if impact.verdict == Removable::BreaksModel && !force {
        bail!("refused: the cut is `breaks_model`; pass --force to write the plan anyway");
    }
    if let Some(t) = impact.cut.iter().find(|t| t.mode == CutMode::None) {
        bail!("{} cannot be cut", t.id);
    }
    let g = &m.gguf;
    let a = &impact.ablation;
    let mut plan = plan_base(m, vec![format!("from ablation of {}", impact.cut.iter().map(|c| c.id.as_str()).collect::<Vec<_>>().join(", "))]);
    let hybrid_ok = graph.pattern_period.is_none() || impact.verdict != Removable::BreaksModel || force;
    for l in &a.skip_layers {
        let bytes = graph.nodes.get(&format!("layer.{l}")).map_or(0, |n| n.bytes);
        plan.candidates.push(Candidate {
            id: format!("layer_{l}"),
            kind: CandidateKind::LayerRedundancy,
            title: format!("Remove layer {l} (ablation)"),
            detail: "Selected by ablation. Later layers are renumbered.".into(),
            risk: Risk::High,
            bytes_saved: bytes,
            applicable: hybrid_ok,
            reason: (!hybrid_ok).then(|| "hybrid layer pattern would break".into()),
            selected: true,
            data: json!({ "layer": l, "score": null, "allow_pattern": graph.pattern_period.is_some() }),
        });
    }
    let ops = zero_ops(g, graph, a);
    if !ops.is_empty() {
        let (delta, _) = requant_delta(g, &ops);
        plan.candidates.push(Candidate {
            id: "zero_weights".into(),
            kind: CandidateKind::ZeroWeights,
            title: format!("Zero weights in {} tensor(s) (blocks / heads / neurons / experts)", ops.len()),
            detail: format!(
                "Weights set to 0 in place; shapes are unchanged so llama.cpp loads the file. Saves no space{}.",
                if delta > 0 { format!(" (unaligned column zeroing requantizes to Q8_0: +{})", human_bytes(delta as u64)) } else { String::new() }
            ),
            risk: Risk::High,
            bytes_saved: 0,
            applicable: true,
            reason: None,
            selected: true,
            data: json!({ "ops": ops }),
        });
    }
    if !a.mask_tokens.is_empty() {
        let ids: Vec<u32> = a.mask_tokens.iter().flat_map(|(s, e)| *s..=*e).collect();
        let n_vocab = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
        let per: u64 = vocab_tensors(g, n_vocab).iter().map(|t| row_bytes(t)).sum();
        plan.candidates.push(Candidate {
            id: "cut_tokens".into(),
            kind: CandidateKind::ReservedTokens,
            title: format!("Remove {} token(s) (ablation)", ids.len()),
            detail: "Removed from the vocabulary; bos/eos and other metadata-referenced tokens are always kept.".into(),
            risk: Risk::High,
            bytes_saved: ids.len() as u64 * per,
            applicable: true,
            reason: None,
            selected: true,
            data: json!({ "remove_ids": ids }),
        });
    }
    for t in &impact.cut {
        match t.id.as_str() {
            "projector" => {
                if let Some(pp) = &m.entry.projector {
                    plan.candidates.push(Candidate {
                        id: "unused_projector".into(),
                        kind: CandidateKind::UnusedProjector,
                        title: "Omit the vision projector".into(),
                        detail: "The Modelfile will not reference the projector GGUF.".into(),
                        risk: Risk::Low,
                        bytes_saved: t.bytes,
                        applicable: true,
                        reason: None,
                        selected: true,
                        data: json!({ "projector": pp }),
                    });
                }
            }
            "template" => plan.candidates.push(Candidate {
                id: "strip_template".into(),
                kind: CandidateKind::StripTemplate,
                title: "Strip the chat template".into(),
                detail: "Removes tokenizer.chat_template and the Modelfile TEMPLATE; prompts are passed as raw text.".into(),
                risk: Risk::Medium,
                bytes_saved: t.bytes,
                applicable: true,
                reason: None,
                selected: true,
                data: json!({}),
            }),
            "mtp" | "vision" | "extra" => {
                let names: Vec<String> = graph.tensors_under(&t.id).iter().filter_map(|x| x.info["name"].as_str().map(String::from)).collect();
                plan.candidates.push(Candidate {
                    id: format!("drop_{}", t.id),
                    kind: CandidateKind::DropTensors,
                    title: format!("Drop the {} tensors of {}", names.len(), t.label),
                    detail: "Not verified to load in llama.cpp: check the slimmed file.".into(),
                    risk: Risk::Medium,
                    bytes_saved: t.bytes,
                    applicable: true,
                    reason: None,
                    selected: true,
                    data: json!({ "tensors": names }),
                });
            }
            _ => {}
        }
    }
    if plan.candidates.is_empty() {
        bail!("this cut has nothing to write to a file");
    }
    Ok(plan)
}

/// The `top` least damaging targets of a sweep as a prune plan.
pub fn plan_from_sweep(m: &ModelHandle, graph: &ModelGraph, vocab: Option<&Vocab>, sw: &SweepReport, top: usize, force: bool) -> Result<(PrunePlan, StaticImpact)> {
    if sw.digest != m.entry.digest {
        bail!("sweep was made for {} ({}), not {}", sw.model, sw.digest, m.entry.name);
    }
    let mut entries: Vec<_> = sw.entries.iter().collect();
    entries.sort_by(|a, b| a.mean_kl.partial_cmp(&b.mean_kl).unwrap_or(std::cmp::Ordering::Equal));
    let ids: Vec<String> = entries.iter().take(top.max(1)).map(|e| e.id.clone()).collect();
    let impact = static_impact(m, graph, vocab, &ids, force);
    let mut plan = plan_from_impact(m, graph, &impact, force)?;
    plan.notes.push(format!(
        "{} least damaging of {} {}(s) from a sweep over {} sample(s): {}",
        ids.len(),
        sw.entries.len(),
        sw.level,
        sw.samples,
        entries.iter().take(top.max(1)).map(|e| format!("{} (KL {:.4})", e.id, e.mean_kl)).collect::<Vec<_>>().join(", ")
    ));
    Ok((plan, impact))
}

/// Sweep results as graph importance.
pub fn sweep_importance(sw: &SweepReport) -> crate::graph::SweepImportance {
    crate::graph::SweepImportance { damage: sw.entries.iter().map(|e| (e.id.clone(), e.mean_kl)).collect() }
}

pub fn impact_json(i: &StaticImpact) -> Value {
    serde_json::to_value(i).unwrap_or(Value::Null)
}
