//! `compile`: enabled edits → an `Intervention` (masks, scales, sampling, prompt changes).

use super::*;

/// Enabled edits turned into what the forward pass and the prompt builder use.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Compiled {
    pub intervention: Intervention,
    /// Sampling for the edited side (last `sampling` edit) plus forced tokens.
    pub sampling: Option<Sampling>,
    pub system: Option<String>,
    pub template: Option<String>,
    /// Graph node ids changed by the edits (highlighted in the graph views).
    pub touched: Vec<String>,
    pub notes: Vec<String>,
    pub active: usize,
}

impl Compiled {
    /// No edit changes anything.
    pub fn is_noop(&self) -> bool {
        self.intervention.is_empty() && self.sampling.is_none() && self.system.is_none() && self.template.is_none()
    }

    pub fn prompt_edited(&self) -> bool {
        self.system.is_some() || self.template.is_some()
    }
}

/// `layer.L.ffn.neuron.N` (a single FFN neuron).
pub(super) fn parse_neuron(id: &str) -> Option<(u32, u32)> {
    let rest = id.strip_prefix("layer.")?;
    let (l, tail) = rest.split_once('.')?;
    let n = tail.strip_prefix("ffn.neuron.")?;
    Some((l.parse().ok()?, n.parse().ok()?))
}

pub(super) fn scale_node(graph: &ModelGraph, vocab: Option<&Vocab>, target: &str, factor: f32, c: &mut Compiled) -> Result<()> {
    let a = &mut c.intervention;
    if let Some((l, n)) = parse_neuron(target) {
        let n_ff = graph.dims.get(&l).map_or(0, |d| d.n_ff);
        if n as u64 >= n_ff {
            bail!("layer {l} has {n_ff} FFN neurons; {target} does not exist");
        }
        a.scale_neurons.push((l, n, n, factor));
        c.touched.push(format!("layer.{l}.ffn"));
        return Ok(());
    }
    let id = graph.resolve_id(target).ok_or_else(|| anyhow!("unknown node {target:?}"))?;
    let n = graph.get(&id).ok_or_else(|| anyhow!("unknown node {target:?}"))?;
    match n.kind {
        NodeKind::Layer => {
            let l = n.layer.ok_or_else(|| anyhow!("{id} has no layer index"))?;
            if factor == 0.0 {
                a.skip_layers.push(l);
            } else {
                a.scale_layers.push((l, factor));
            }
        }
        NodeKind::Block => {
            let (Some(l), Some(b)) = (n.layer, n.block.clone()) else { bail!("{id}: not a layer block") };
            if !matches!(b.as_str(), "attention" | "ssm" | "ffn" | "moe") {
                bail!("{id}: only attention, SSM, FFN and MoE blocks can be scaled");
            }
            a.scale_blocks.push((l, b, factor));
        }
        NodeKind::KvGroup => {
            let l = n.layer.unwrap_or(0);
            for h in n.info["heads"].as_array().into_iter().flatten().filter_map(|h| h.as_u64()) {
                a.scale_heads.push((l, h as u32, factor));
            }
        }
        NodeKind::Head => {
            let (Some(l), Some(h)) = (n.layer, n.info["head"].as_u64()) else { bail!("{id}: not a head") };
            a.scale_heads.push((l, h as u32, factor));
        }
        NodeKind::NeuronBucket => {
            let (Some(l), Some(s), Some(e)) = (n.layer, n.info["start"].as_u64(), n.info["end"].as_u64()) else { bail!("{id}: not a bucket") };
            a.scale_neurons.push((l, s as u32, e as u32, factor));
        }
        NodeKind::Expert => {
            let (Some(l), Some(e)) = (n.layer, n.info["expert"].as_u64()) else { bail!("{id}: not an expert") };
            a.scale_experts.push((l, e as u32, factor));
            c.notes.push(format!("{id}: the runtime forward pass has no MoE support; expert edits only apply when baked"));
        }
        _ if factor == 0.0 => {
            let (targets, unknown, _) = resolve_cut(graph, vocab, &[id.clone()]);
            if !unknown.is_empty() {
                bail!("cannot zero {}", unknown.join(", "));
            }
            let n_vocab = vocab.map_or(0, |v| v.tokens.len() as u32);
            a.merge(ablation_for(graph, &targets, n_vocab));
        }
        other => bail!("{id} is a {}: scale works on layers, blocks, KV groups, heads, neuron buckets, neurons and experts (tensors and sections can only be zeroed)", other.as_str()),
    }
    c.touched.push(id);
    Ok(())
}

/// Compiles the enabled edits (in order) for a model.
pub fn compile(graph: &ModelGraph, vocab: Option<&Vocab>, edits: &[Edit]) -> Result<Compiled> {
    let mut c = Compiled::default();
    let n_vocab = vocab.map_or(u32::MAX, |v| v.tokens.len() as u32);
    let check_token = |t: u32| -> Result<()> {
        if t >= n_vocab {
            bail!("token id {t} is outside the vocabulary ({n_vocab} tokens)");
        }
        Ok(())
    };
    let check_layer = |l: u32| -> Result<()> {
        if l >= graph.n_layers {
            bail!("layer {l} does not exist ({} layers)", graph.n_layers);
        }
        Ok(())
    };
    for e in edits.iter().filter(|e| e.enabled) {
        c.active += 1;
        let r: Result<()> = (|| {
            match &e.op {
                EditOp::Scale { target, factor } => scale_node(graph, vocab, target, *factor, &mut c)?,
                EditOp::Zero { target } => scale_node(graph, vocab, target, 0.0, &mut c)?,
                EditOp::Skip { layer } => {
                    check_layer(*layer)?;
                    c.intervention.skip_layers.push(*layer);
                    c.touched.push(format!("layer.{layer}"));
                }
                EditOp::Ablate { cut } => {
                    let (targets, unknown, _) = resolve_cut(graph, vocab, cut);
                    if !unknown.is_empty() {
                        bail!("unknown id(s) {}", unknown.join(", "));
                    }
                    let nv = vocab.map_or(0, |v| v.tokens.len() as u32);
                    c.intervention.merge(ablation_for(graph, &targets, nv));
                    c.touched.extend(targets.into_iter().map(|t| t.id));
                }
                EditOp::Embed { token, scale, replace_with } => {
                    check_token(*token)?;
                    if let Some(r) = replace_with {
                        check_token(*r)?;
                    }
                    c.intervention.embed_edits.push(EmbedEdit { token: *token, scale: *scale, replace_with: *replace_with });
                    c.touched.push("t.token_embd.weight".into());
                }
                EditOp::Ban { token } => {
                    check_token(*token)?;
                    c.intervention.mask_tokens.push((*token, *token));
                    c.touched.push("output".into());
                }
                EditOp::Boost { token, bias } => {
                    check_token(*token)?;
                    if !bias.is_finite() {
                        bail!("boost bias must be finite (use `ban` to forbid a token)");
                    }
                    c.intervention.logit_bias.push((*token, *bias));
                    c.touched.push("output".into());
                }
                EditOp::Patch { layer, head, source, all_positions } => {
                    check_layer(*layer)?;
                    if source.trim().is_empty() {
                        bail!("patch needs a source prompt");
                    }
                    c.intervention.patch.push(Patch { layer: *layer, head: *head, source: source.clone(), all_positions: *all_positions });
                    c.touched.push(match head {
                        Some(h) => format!("layer.{layer}.attn.head.{h}"),
                        None => format!("layer.{layer}"),
                    });
                }
                EditOp::Steer { layer, a, b, strength } => {
                    check_layer(*layer)?;
                    c.intervention.steer.push(Steer { layer: *layer, a: a.clone(), b: b.clone(), strength: *strength });
                    c.touched.push(format!("layer.{layer}"));
                }
                EditOp::Sampling { temperature, top_k, top_p, seed } => {
                    let forced = c.sampling.take().map(|s| s.forced).unwrap_or_default();
                    c.sampling = Some(Sampling { temperature: temperature.max(0.0), top_k: *top_k, top_p: *top_p, seed: *seed, forced });
                }
                EditOp::Force { step, token } => {
                    check_token(*token)?;
                    c.sampling.get_or_insert_with(Sampling::default).forced.push((*step, *token));
                }
                EditOp::System { text } => c.system = Some(text.clone()),
                EditOp::Template { text } => {
                    crate::chat::render_template(text, Some("s"), "p").context("chat template edit")?;
                    c.template = Some(text.clone());
                }
            }
            Ok(())
        })();
        r.with_context(|| format!("edit #{} ({})", e.id, e.label))?;
    }
    let ab = &mut c.intervention;
    ab.skip_layers.sort_unstable();
    ab.skip_layers.dedup();
    let mut seen = HashSet::new();
    c.touched.retain(|t| seen.insert(t.clone()));
    Ok(c)
}

// --- bake ------------------------------------------------------------------------------
