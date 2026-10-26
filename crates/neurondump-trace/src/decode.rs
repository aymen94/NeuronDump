//! Building masks, generation (greedy / sampled / step by step), decision breakdown, per-position log-probs.

use super::*;

impl<'a> Model<'a> {
    /// Masks for an intervention, including steering vectors and patching sources
    /// (computed with clean runs of the prompts involved).
    pub(super) fn masks(&self, enc: &Encoder, a: &Ablation, max_tokens: usize, p: &dyn Progress) -> Result<Masks> {
        let mut m = Masks::from(a);
        if a.steer.is_empty() && a.patch.is_empty() {
            return Ok(m);
        }
        let clean = Masks::default();
        let encode = |t: &str| -> Vec<u32> {
            let mut ids = enc.encode(t, true, None).ids;
            ids.truncate(max_tokens.max(1));
            ids
        };
        let keep = FwdOpt { keep: true, ..Default::default() };
        for s in &a.steer {
            let l = s.layer as usize;
            if l >= self.layers.len() {
                bail!("steer: no layer {l}");
            }
            p.set(3.0, "steering vector");
            let mean_at = |text: &str| -> Result<Vec<f32>> {
                let ids = encode(text);
                if ids.is_empty() {
                    bail!("steer: empty prompt");
                }
                let f = self.run(&ids, &clean, keep, p)?;
                let rows = &f.hidden[l];
                Ok((0..self.n_embd).map(|j| rows.iter().map(|r| r[j]).sum::<f32>() / rows.len() as f32).collect())
            };
            let (va, vb) = (mean_at(&s.a)?, mean_at(&s.b)?);
            let v: Vec<f32> = va.iter().zip(&vb).map(|(x, y)| s.strength * (x - y)).collect();
            let t = Tensor::from_vec(v, (1, self.n_embd), &self.w.dev)?;
            let t = match m.steer.remove(&l) {
                Some(prev) => (prev + t)?,
                None => t,
            };
            m.steer.insert(l, t);
        }
        for pt in &a.patch {
            let l = pt.layer as usize;
            if l >= self.layers.len() {
                bail!("patch: no layer {l}");
            }
            p.set(3.0, "patching source");
            let ids = encode(&pt.source);
            if ids.is_empty() {
                bail!("patch: empty source prompt");
            }
            let f = self.run(&ids, &clean, keep, p)?;
            match pt.head {
                Some(hh) => {
                    if f.zs[l].is_empty() {
                        bail!("patch: layer {l} has no attention output to patch");
                    }
                    m.patch_z.entry(l).or_default().push((hh as usize, f.zs[l].clone(), pt.all_positions));
                }
                None => {
                    m.patch_h.insert(l, (f.hidden[l].clone(), pt.all_positions));
                }
            }
        }
        Ok(m)
    }

    /// Greedy continuation; returns (ids, text, tokens/s).
    pub(super) fn generate(&self, ids: &[u32], masks: &Masks, n: usize, eos: Option<u32>, p: &dyn Progress) -> Result<(Vec<u32>, String, f64)> {
        self.generate_sampled(ids, masks, n, eos, &Sampling::default(), p)
    }

    pub(super) fn generate_sampled(&self, ids: &[u32], masks: &Masks, n: usize, eos: Option<u32>, s: &Sampling, p: &dyn Progress) -> Result<(Vec<u32>, String, f64)> {
        let t = Instant::now();
        let steps = self.steps(ids, masks, n, eos, s, false, p)?;
        let out: Vec<u32> = steps.iter().map(|st| st.token.id).collect();
        let secs = t.elapsed().as_secs_f64().max(1e-9);
        let text = self.detok(&out);
        let tps = if out.is_empty() { 0.0 } else { out.len() as f64 / secs };
        Ok((out, text, tps))
    }

    pub(super) fn detok(&self, ids: &[u32]) -> String {
        ids.iter().map(|i| self.vocab.tokens.get(*i as usize).map(|t| t.text.as_str()).unwrap_or("")).collect()
    }

    pub(super) fn lens_token(&self, id: usize, prob: f32) -> LensToken {
        LensToken { id: id as u32, text: self.vocab.tokens.get(id).map(|t| t.text.clone()).unwrap_or_default(), prob }
    }

    /// Generation with a per-step record (chosen token, entropy, alternatives, decision).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn steps(&self, ids: &[u32], masks: &Masks, n: usize, eos: Option<u32>, s: &Sampling, explain: bool, p: &dyn Progress) -> Result<Vec<GenStep>> {
        let mut cur = ids.to_vec();
        let mut out = Vec::new();
        let mut rng = s.seed.unwrap_or(0x9E37_79B9_7F4A_7C15).max(1);
        for step in 0..n {
            check_cancel(p)?;
            p.set(10.0 + 88.0 * step as f32 / n.max(1) as f32, &format!("token {}/{n}", step + 1));
            let pos = cur.len() - 1;
            let f = self.run(&cur, masks, FwdOpt { record: explain.then_some(pos), ..Default::default() }, p)?;
            if f.last_logits.is_empty() {
                break;
            }
            let lp = log_softmax(&f.last_logits);
            let probs: Vec<f32> = lp.iter().map(|x| x.exp() as f32).collect();
            let forced = s.forced.iter().find(|(st, _)| *st == step).map(|(_, t)| *t as usize);
            let next = forced.unwrap_or_else(|| sample_token(&f.last_logits, s, &mut rng));
            let alternatives: Vec<LensToken> = self.top_tokens(&probs).into_iter().filter(|t| t.id as usize != next).collect();
            let decision = match (&f.record, explain) {
                (Some(r), true) => self.decision(r, &f.last_logits, next as u32)?,
                _ => None,
            };
            out.push(GenStep {
                index: step,
                position: pos,
                token: self.lens_token(next, probs.get(next).copied().unwrap_or(0.0)),
                entropy: entropy(&lp),
                alternatives,
                forced: forced.is_some(),
                decision,
            });
            cur.push(next as u32);
            if Some(next as u32) == eos {
                break;
            }
        }
        Ok(out)
    }

    /// Direct logit attribution at the recorded position: how much the embedding, every
    /// layer's mixer and FFN, every head and the top neurons push `chosen` against the
    /// runner-up. The final norm is linearized (its scale is taken from the actual run).
    pub(super) fn decision(&self, r: &Record, logits: &[f32], chosen: u32) -> Result<Option<Decision>> {
        let Some(head) = &self.lm_head else { return Ok(None) };
        if self.spec.encoder || logits.is_empty() {
            return Ok(None);
        }
        let lp = log_softmax(logits);
        let probs: Vec<f32> = lp.iter().map(|x| x.exp() as f32).collect();
        let c = chosen as usize;
        let mut ru = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, x) in logits.iter().enumerate() {
            if i != c && *x > best {
                best = *x;
                ru = i;
            }
        }
        let final_h = r.layers.last().map(|l| l.h.clone()).unwrap_or_else(|| r.emb.clone());
        let w = self.w.rows(&head.name, &[chosen, ru as u32])?.to_vec2::<f32>()?;
        let (g, eps) = match &self.out_norm {
            Some(n) => (n.w.to_vec1::<f32>()?, n.eps),
            None => (vec![1f32; final_h.len()], 1e-6),
        };
        let scale = 1.0 / rms(&final_h, eps);
        let u: Vec<f32> = (0..final_h.len()).map(|j| (w[0][j] - w[1][j]) * g.get(j).copied().unwrap_or(1.0) * scale).collect();
        let dotu = |v: &[f32]| -> f32 { if v.is_empty() { 0.0 } else { v.iter().zip(&u).map(|(a, b)| a * b).sum() } };
        let embedding = dotu(&r.emb);
        let mut layers = Vec::new();
        let mut heads = Vec::new();
        let mut neurons = Vec::new();
        for (li, (l, rec)) in self.layers.iter().zip(&r.layers).enumerate() {
            layers.push(LayerAttribution { layer: li as u32, attn: dotu(&rec.attn_out), ffn: dotu(&rec.ffn_out) });
            if rec.skipped {
                continue;
            }
            // Per-head outputs: the projection applied to each head's slice alone.
            if !rec.z.is_empty() {
                let width = l.n_head * l.head_dim;
                let mut zm = vec![0f32; l.n_head * width];
                for hh in 0..l.n_head {
                    zm[hh * width + hh * l.head_dim..hh * width + (hh + 1) * l.head_dim].copy_from_slice(&rec.z[hh * l.head_dim..(hh + 1) * l.head_dim]);
                }
                let o = l.o.w.forward(&Tensor::from_vec(zm, (l.n_head, width), &self.w.dev)?)?.to_vec2::<f32>()?;
                let post = match (&l.attn_post_norm, rec.attn_pre_rms) {
                    (Some(n), Some(pr)) => Some((n.w.to_vec1::<f32>()?, pr)),
                    _ => None,
                };
                for (hh, oh) in o.iter().enumerate() {
                    let v: f32 = match &post {
                        Some((pw, pr)) => oh.iter().zip(pw).zip(&u).map(|((a, wv), b)| a * wv / pr * b).sum(),
                        None => dotu(oh),
                    };
                    heads.push(HeadAttribution { layer: li as u32, head: hh as u32, value: v * rec.attn_factor });
                }
            }
            // Top neurons by |activation|: their columns of ffn_down alone.
            if !rec.inner.is_empty() {
                let n_ff = rec.inner.len();
                let mut order: Vec<usize> = (0..n_ff).collect();
                let k = 8.min(n_ff);
                order.select_nth_unstable_by(k - 1, |a, b| rec.inner[*b].abs().partial_cmp(&rec.inner[*a].abs()).unwrap_or(std::cmp::Ordering::Equal));
                let top = &order[..k];
                let mut im = vec![0f32; k * n_ff];
                for (i, &nn) in top.iter().enumerate() {
                    im[i * n_ff + nn] = rec.inner[nn];
                }
                let o = l.down.w.forward(&Tensor::from_vec(im, (k, n_ff), &self.w.dev)?)?.to_vec2::<f32>()?;
                let post = match (&l.ffn_post_norm, rec.ffn_pre_rms) {
                    (Some(n), Some(pr)) => Some((n.w.to_vec1::<f32>()?, pr)),
                    _ => None,
                };
                for (i, &nn) in top.iter().enumerate() {
                    let v: f32 = match &post {
                        Some((pw, pr)) => o[i].iter().zip(pw).zip(&u).map(|((a, wv), b)| a * wv / pr * b).sum(),
                        None => dotu(&o[i]),
                    };
                    neurons.push(NeuronAttribution { layer: li as u32, neuron: nn as u32, activation: rec.inner[nn], value: v * rec.ffn_factor });
                }
            }
        }
        let attributed = embedding + layers.iter().map(|l| l.attn + l.ffn).sum::<f32>();
        heads.sort_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap_or(std::cmp::Ordering::Equal));
        heads.truncate(16);
        neurons.sort_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap_or(std::cmp::Ordering::Equal));
        neurons.truncate(16);
        Ok(Some(Decision {
            position: r.pos,
            candidates: self.top_tokens(&probs),
            chosen: self.lens_token(c, probs[c]),
            runner_up: self.lens_token(ru, probs[ru]),
            logit_diff: logits[c] - logits[ru],
            attributed,
            embedding,
            layers,
            top_heads: heads,
            top_neurons: neurons,
        }))
    }

    /// Per-position log-probabilities [seq-1][vocab] predicting the next token.
    pub(super) fn position_logprobs(&self, f: &FwdOut, masks: &Masks) -> Result<Vec<Vec<f64>>> {
        let Some(lg) = &f.all_logits else { return Ok(vec![]) };
        let rows = lg.to_vec2::<f32>()?;
        let n = rows.len().saturating_sub(1);
        Ok(rows
            .into_iter()
            .take(n)
            .map(|mut r| {
                masks.apply_logits(&mut r);
                log_softmax(&r)
            })
            .collect())
    }
}
