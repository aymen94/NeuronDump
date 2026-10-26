//! Applying stats, traces and sweeps to nodes (importance, activation) and child summaries.

use super::*;

impl ModelGraph {
    pub(super) fn apply_measurements(&mut self, inputs: &GraphInputs, notes: &mut Vec<String>) {
        // Structural importance first; measurements override it.
        let last_main = self.n_layers.saturating_sub(1);
        for n in self.nodes.values_mut() {
            if n.removable == Removable::BreaksModel {
                n.importance = Importance { score: 1.0, source: "structural: required".into() };
            } else if n.kind == NodeKind::Layer && (n.layer == Some(0) || n.layer == Some(last_main)) {
                n.importance = Importance { score: 0.9, source: "structural: edge layer (not measured)".into() };
            } else if n.removable == Removable::Safe {
                n.importance = Importance { score: 0.0, source: "structural: optional".into() };
            }
        }
        // Weight stats.
        if let Some(s) = inputs.stats {
            let by_name: HashMap<&str, &crate::stats::TensorStats> = s.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
            for n in self.nodes.values_mut().filter(|n| n.kind == NodeKind::Tensor) {
                if let Some(t) = n.info["name"].as_str().and_then(|nm| by_name.get(nm)) {
                    n.stats = Some(StatsSummary {
                        mean: Some(t.mean),
                        std: Some(t.std),
                        l2: t.l2,
                        sparsity: Some(t.sparsity),
                        outlier_fraction: Some(t.outlier_fraction),
                    });
                }
            }
            // Blocks / layers: combined L2 norm.
            let ids: Vec<String> = self.nodes.values().filter(|n| matches!(n.kind, NodeKind::Block | NodeKind::Layer)).map(|n| n.id.clone()).collect();
            for id in ids {
                let prefix_children: Vec<f64> = self
                    .nodes
                    .values()
                    .filter(|c| c.kind == NodeKind::Tensor && ancestors_of(&self.nodes, &c.id).contains(&id))
                    .filter_map(|c| c.stats.as_ref().map(|s| s.l2 * s.l2))
                    .collect();
                if !prefix_children.is_empty() {
                    if let Some(n) = self.nodes.get_mut(&id) {
                        n.stats = Some(StatsSummary { mean: None, std: None, l2: prefix_children.iter().sum::<f64>().sqrt(), sparsity: None, outlier_fraction: None });
                    }
                }
            }
            for h in &s.heads {
                let score: Vec<f32> = if !h.o.is_empty() && h.o.len() == h.q.len() { h.q.iter().zip(&h.o).map(|(a, b)| a * b).collect() } else { h.q.clone() };
                let max = score.iter().cloned().fold(0f32, f32::max);
                for (i, v) in score.iter().enumerate() {
                    let id = format!("layer.{}.attn.head.{i}", h.layer);
                    self.fine_stats.insert(id.clone(), *v as f64);
                    if max > 0.0 {
                        self.fine_importance.insert(id, Importance { score: v / max, source: "stats: |Q|·|O| norm relative to the layer's largest head".into() });
                    }
                }
            }
            for nn in &s.ffn_neurons {
                let Some(d) = self.dims.get(&nn.layer) else { continue };
                let b = d.bucket.max(1) as usize;
                let means: Vec<f32> = nn.norms.chunks(b).map(|c| c.iter().sum::<f32>() / c.len().max(1) as f32).collect();
                let max = means.iter().cloned().fold(0f32, f32::max);
                for (i, v) in means.iter().enumerate() {
                    let s0 = i * b;
                    let e = (s0 + b).min(nn.norms.len()) - 1;
                    let id = format!("layer.{}.ffn.n.{s0}-{e}", nn.layer);
                    self.fine_stats.insert(id.clone(), *v as f64);
                    if max > 0.0 {
                        self.fine_importance.insert(id, Importance { score: v / max, source: "stats: mean neuron norm relative to the layer's largest bucket".into() });
                    }
                }
            }
            for e in &s.experts {
                let max = e.norms.iter().cloned().fold(0f32, f32::max);
                for (i, v) in e.norms.iter().enumerate() {
                    let id = format!("layer.{}.moe.expert.{i}", e.layer);
                    self.fine_stats.insert(id.clone(), *v as f64);
                    if max > 0.0 {
                        self.fine_importance.insert(id, Importance { score: v / max, source: "stats: expert norm relative to the layer's largest".into() });
                    }
                }
            }
        }
        // Trace: hidden-state change per layer, entropy per head, top neurons.
        if let Some(tr) = inputs.trace {
            let max_change = tr.layers.iter().filter_map(|l| l.cos_prev).map(|c| 1.0 - c).fold(0f32, f32::max);
            for lt in &tr.layers {
                let lid = format!("layer.{}", lt.index);
                if let Some(n) = self.nodes.get_mut(&lid) {
                    n.activation = Some(json!({
                        "hidden_norm": lt.hidden_norm,
                        "cos_in_out": lt.cos_prev,
                        "logit_lens_top": lt.logit_lens.first().map(|t| json!({ "text": t.text, "prob": t.prob })),
                        "samples": tr.samples,
                    }));
                    if let (Some(c), true) = (lt.cos_prev, max_change > 0.0) {
                        n.importance = Importance { score: ((1.0 - c) / max_change).clamp(0.0, 1.0), source: format!("trace: 1 - cos(in,out), {} sample(s)", tr.samples) };
                    }
                }
                for (h, e) in lt.head_entropy.iter().enumerate() {
                    self.fine_activation.insert(format!("layer.{}.attn.head.{h}", lt.index), json!({ "attention_entropy": e }));
                }
                if let Some(d) = self.dims.get(&lt.index) {
                    let b = d.bucket.max(1) as u32;
                    for (neuron, v) in &lt.top_neurons {
                        let s0 = neuron / b * b;
                        let e = (s0 + b).min(d.n_ff as u32) - 1;
                        let id = format!("layer.{}.ffn.n.{s0}-{e}", lt.index);
                        let entry = self.fine_activation.entry(id).or_insert_with(|| json!({ "top_neurons": [] }));
                        if let Some(a) = entry["top_neurons"].as_array_mut() {
                            a.push(json!({ "neuron": neuron, "mean_abs": v }));
                        }
                    }
                }
            }
            notes.push(format!("activations from the latest trace ({} sample(s))", tr.samples));
        }
        // Sweeps: measured damage when removed. The strongest signal.
        for sw in &inputs.sweeps {
            let max = sw.damage.values().cloned().fold(0f64, f64::max);
            for (id, d) in &sw.damage {
                let imp = Importance { score: if max > 0.0 { (d / max) as f32 } else { 0.0 }, source: format!("sweep: KL {d:.4} when removed (relative to the worst)") };
                if let Some(n) = self.nodes.get_mut(id) {
                    n.importance = imp;
                } else {
                    self.fine_importance.insert(id.clone(), imp);
                }
            }
        }
        if !inputs.sweeps.is_empty() {
            notes.push("importance from ablation sweeps where available".into());
        }
        // Blocks and tensors inherit the layer's importance unless they have their own.
        let layer_imp: HashMap<u32, Importance> =
            self.nodes.values().filter(|n| n.kind == NodeKind::Layer).filter_map(|n| n.layer.map(|l| (l, n.importance.clone()))).collect();
        for n in self.nodes.values_mut() {
            if matches!(n.kind, NodeKind::Block | NodeKind::Tensor) && n.removable != Removable::BreaksModel && n.importance.source.starts_with("structural: not") {
                if let Some(li) = n.layer.and_then(|l| layer_imp.get(&l)).filter(|li| !li.source.starts_with("structural: not")) {
                    n.importance = Importance { score: li.score, source: format!("inherited from layer: {}", li.source) };
                }
            }
        }
    }

    pub(super) fn fill_contains(&mut self) {
        let mut kids: HashMap<String, Vec<ChildSummary>> = HashMap::new();
        for n in self.nodes.values() {
            if let Some(p) = &n.parent {
                kids.entry(p.clone()).or_default().push(ChildSummary { id: n.id.clone(), kind: n.kind, label: n.label.clone(), params: n.params, bytes: n.bytes });
            }
        }
        for n in self.nodes.values_mut() {
            n.contains = kids.remove(&n.id).unwrap_or_default();
        }
    }

    // --- fine level --------------------------------------------------------------------

}
