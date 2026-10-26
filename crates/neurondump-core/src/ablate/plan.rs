//! Turning a cut or a sweep ranking into a prune plan; sweep importance; impact JSON.

use super::*;

pub(super) fn plan_base(m: &ModelHandle, notes: Vec<String>) -> PrunePlan {
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
