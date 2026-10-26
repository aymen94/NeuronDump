//! Simulated ablation and sweeps through the trace backend.

use super::*;

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
