//! Lazy fine level (heads, neuron buckets, experts) and id resolution.

use super::*;

impl ModelGraph {
    /// Heads (in GQA groups), neuron buckets or experts of a block, generated on demand.
    pub fn fine_children(&self, block_id: &str) -> Vec<Node> {
        let Some(b) = self.nodes.get(block_id) else { return vec![] };
        if !b.lazy {
            return vec![];
        }
        let Some(layer) = b.layer else { return vec![] };
        let Some(d) = self.dims.get(&layer) else { return vec![] };
        let tensors: Vec<&Node> = self.nodes.values().filter(|n| n.parent.as_deref() == Some(block_id) && n.kind == NodeKind::Tensor).collect();
        let role_sum = |pred: &dyn Fn(&str) -> bool| -> (u64, u64) {
            tensors
                .iter()
                .filter(|t| pred(t.info["role"].as_str().unwrap_or("")))
                .fold((0, 0), |(p, by), t| (p + t.params, by + t.bytes))
        };
        let mk = |id: String, kind: NodeKind, label: String, parent: &str, params: u64, bytes: u64, info: Value| -> Node {
            let mut n = new_node(&id, kind, &label, Some(parent));
            n.layer = Some(layer);
            n.block = b.block.clone();
            n.layer_kind = b.layer_kind.clone();
            n.params = params;
            n.bytes = bytes;
            n.quant = b.quant.clone();
            n.info = info;
            n.removable = Removable::Risky;
            n.removable_reason = "zeroed in place: shapes stay the same, so llama.cpp still loads it".into();
            n.importance = self
                .fine_importance
                .get(&id)
                .cloned()
                .unwrap_or_else(|| Importance { score: b.importance.score, source: format!("inherited from block: {}", b.importance.source) });
            n.activation = self.fine_activation.get(&id).cloned();
            n.stats = self.fine_stats.get(&id).map(|v| StatsSummary { mean: None, std: None, l2: *v, sparsity: None, outlier_fraction: None });
            n
        };
        let mut out = Vec::new();
        match b.block.as_deref() {
            Some("attention") if d.n_head > 0 => {
                let n_kv = d.n_kv.max(1);
                let group = (d.n_head / n_kv).max(1);
                let (qp, qb) = role_sum(&|r| matches!(r, "attn_q" | "attn_output"));
                let (kp, kb) = role_sum(&|r| matches!(r, "attn_k" | "attn_v"));
                let (fp, fb) = role_sum(&|r| r == "attn_qkv");
                let q_frac = d.n_head as f64 / (d.n_head + 2 * n_kv) as f64;
                let head_p = qp / d.n_head + (fp as f64 * q_frac) as u64 / d.n_head;
                let head_b = qb / d.n_head + (fb as f64 * q_frac) as u64 / d.n_head;
                let kv_p = kp / n_kv + (fp as f64 * (1.0 - q_frac)) as u64 / n_kv;
                let kv_b = kb / n_kv + (fb as f64 * (1.0 - q_frac)) as u64 / n_kv;
                for kv in 0..n_kv {
                    let gid = format!("{block_id}.kv.{kv}");
                    let heads: Vec<u64> = (kv * group..(kv + 1) * group).collect();
                    let mut gn = mk(
                        gid.clone(),
                        NodeKind::KvGroup,
                        format!("KV group {kv} (heads {}-{})", heads[0], heads[heads.len() - 1]),
                        block_id,
                        kv_p + head_p * group,
                        kv_b + head_b * group,
                        json!({ "kv_head": kv, "heads": heads, "head_dim": d.head_dim }),
                    );
                    gn.contains = heads
                        .iter()
                        .map(|h| ChildSummary { id: format!("{block_id}.head.{h}"), kind: NodeKind::Head, label: format!("head {h}"), params: head_p, bytes: head_b })
                        .collect();
                    out.push(gn);
                    for h in heads {
                        out.push(mk(
                            format!("{block_id}.head.{h}"),
                            NodeKind::Head,
                            format!("head {h}"),
                            &gid,
                            head_p,
                            head_b,
                            json!({ "head": h, "kv_head": kv, "head_dim": d.head_dim, "columns_of_attn_output": [h * d.head_dim, (h + 1) * d.head_dim] }),
                        ));
                    }
                }
            }
            Some("ffn") if d.n_ff > 0 => {
                let per = b.params as f64 / d.n_ff as f64;
                let per_b = b.bytes as f64 / d.n_ff as f64;
                let mut s0 = 0;
                while s0 < d.n_ff {
                    let e = (s0 + d.bucket).min(d.n_ff);
                    let n = e - s0;
                    out.push(mk(
                        format!("{block_id}.n.{s0}-{}", e - 1),
                        NodeKind::NeuronBucket,
                        format!("neurons {s0}-{}", e - 1),
                        block_id,
                        (per * n as f64) as u64,
                        (per_b * n as f64) as u64,
                        json!({ "start": s0, "end": e - 1, "count": n }),
                    ));
                    s0 = e;
                }
            }
            Some("moe") if d.n_expert > 0 => {
                let (ep, eb) = role_sum(&|r| r.contains("_exps"));
                for e in 0..d.n_expert {
                    out.push(mk(
                        format!("{block_id}.expert.{e}"),
                        NodeKind::Expert,
                        format!("expert {e}"),
                        block_id,
                        ep / d.n_expert,
                        eb / d.n_expert,
                        json!({ "expert": e }),
                    ));
                }
            }
            _ => {}
        }
        out
    }

    /// Accepts exact ids plus shorthands: `token_embd`, `blk.7`, `L7`, `7`, `layer.7.attention`, `vocab:10-20`.
    pub fn resolve_id(&self, q: &str) -> Option<String> {
        let q = q.trim();
        if q.starts_with("vocab:") {
            return Some(q.to_string());
        }
        if self.nodes.contains_key(q) {
            return Some(q.to_string());
        }
        for c in [format!("t.{q}"), format!("t.{q}.weight")] {
            if self.nodes.contains_key(&c) {
                return Some(c);
            }
        }
        let layer_alias = q
            .strip_prefix('L')
            .or_else(|| q.strip_prefix("layer"))
            .filter(|r| !r.starts_with('.'))
            .and_then(|r| r.parse::<u32>().ok())
            .or_else(|| q.parse::<u32>().ok());
        if let Some(l) = layer_alias {
            let id = format!("layer.{l}");
            return self.nodes.contains_key(&id).then_some(id);
        }
        if let Some((l, tail)) = parse_layer_prefix(q) {
            let mut parts: Vec<&str> = tail.split('.').filter(|s| !s.is_empty()).collect();
            if let Some(first) = parts.first_mut() {
                if let Some(b) = block_from_short(first) {
                    *first = block_short(b);
                }
            }
            let id = if parts.is_empty() { format!("layer.{l}") } else { format!("layer.{l}.{}", parts.join(".")) };
            if self.nodes.contains_key(&id) {
                return Some(id);
            }
            if self.fine_node(&id).is_some() {
                return Some(id);
            }
            // blk.N.attn_q → tensor
            for c in [format!("t.blk.{l}.{tail}"), format!("t.blk.{l}.{tail}.weight")] {
                if self.nodes.contains_key(&c) {
                    return Some(c);
                }
            }
        }
        None
    }

    /// The block id that owns a fine node id, if the id has a fine shape.
    pub fn fine_parent_block(id: &str) -> Option<String> {
        let (l, tail) = parse_layer_prefix(id)?;
        let mut it = tail.split('.');
        let b = it.next()?;
        let kind = it.next()?;
        matches!(kind, "kv" | "head" | "n" | "expert").then(|| format!("layer.{l}.{b}"))
    }

    pub fn fine_node(&self, id: &str) -> Option<Node> {
        let block = Self::fine_parent_block(id)?;
        self.fine_children(&block).into_iter().find(|n| n.id == id)
    }

}
