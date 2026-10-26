//! `journey`: one token through every layer (logit lens, attention, FFN neurons, decision breakdown).

use super::*;

impl CandleBackend {
    pub(super) fn journey_impl(&self, m: &ModelHandle, req: &JourneyRequest, p: &dyn Progress) -> Result<JourneyReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let top_k = req.top_k.max(1);
        let max_tokens = if req.max_tokens == 0 { 256 } else { req.max_tokens };
        let mut warnings: Vec<String> = enc.warnings.clone();
        let mut ids = enc.encode(&req.text, true, None).ids;
        if ids.len() > max_tokens {
            warnings.push(format!("input truncated from {} to {max_tokens} tokens", ids.len()));
            ids.truncate(max_tokens);
        }
        ids.extend(&req.append);
        if ids.is_empty() {
            bail!("the prompt produced no tokens");
        }
        let n = ids.len() as i64;
        let pos = if req.token < 0 { n + req.token } else { req.token };
        if pos < 0 || pos >= n {
            bail!("token index {} is out of range (the sequence has {n} tokens: 0..{}, or -1 for the last)", req.token, n - 1);
        }
        let pos = pos as usize;
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, top_k, p)?;
        let masks = model.masks(&enc, &req.ablation, max_tokens, p)?;
        p.set(50.0, "forward pass");
        let f = model.run(&ids[..=pos.max(if model.spec.encoder { ids.len() - 1 } else { pos })], &masks, FwdOpt { record: Some(pos), ..Default::default() }, p)?;
        let rec = f.record.ok_or_else(|| anyhow!("nothing recorded"))?;
        let token = |id: u32| JourneyToken {
            id,
            text: vocab.tokens.get(id as usize).map(|t| t.text.clone()).unwrap_or_default(),
            special: vocab.tokens.get(id as usize).is_some_and(|t| matches!(t.token_type, TokenType::Control | TokenType::UserDefined)),
        };
        p.set(80.0, "logit lens");
        let final_h = rec.layers.last().map(|l| l.h.clone()).unwrap_or_else(|| rec.emb.clone());
        let dim = final_h.len();
        let (prediction, final_probs) = if model.lm_head.is_some() {
            model.lens_full(&Tensor::from_vec(final_h.clone(), (1, dim), &model.w.dev)?, &masks)?
        } else {
            (vec![], vec![])
        };
        let next = ids.get(pos + 1).map(|&i| token(i));
        let target = next.as_ref().map(|t| t.id as usize).or_else(|| prediction.first().map(|t| t.id as usize));
        let bucket_of = |l: u32, n_ff: usize| -> usize {
            req.buckets.iter().find(|(x, _)| *x == l).map(|(_, b)| *b as usize).unwrap_or_else(|| {
                let mut b = 64;
                while n_ff / b > 256 {
                    b *= 2;
                }
                b
            })
        };
        let mut layers = Vec::new();
        let (mut first_top1, mut first_top5) = (None, None);
        for (li, r) in rec.layers.iter().enumerate() {
            check_cancel(p)?;
            let (lens, probs) = if model.lm_head.is_some() {
                model.lens_full(&Tensor::from_vec(r.h.clone(), (1, dim), &model.w.dev)?, &masks)?
            } else {
                (vec![], vec![])
            };
            let (target_prob, target_rank) = match target {
                Some(t) if !probs.is_empty() => {
                    let pt = probs[t];
                    (pt, probs.iter().filter(|x| **x > pt).count() as u32)
                }
                _ => (0.0, 0),
            };
            if target.is_some() && !probs.is_empty() {
                if target_rank == 0 && first_top1.is_none() {
                    first_top1 = Some(li as u32);
                }
                if target_rank < 5 && first_top5.is_none() {
                    first_top5 = Some(li as u32);
                }
            }
            let keys = r.attn.first().map_or(0, Vec::len);
            let mut attended: Vec<(usize, f32)> = (0..keys).map(|k| (k, r.attn.iter().map(|h| h[k]).sum::<f32>() / r.attn.len().max(1) as f32)).collect();
            attended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            attended.truncate(5);
            let n_ff = r.inner.len();
            let mut ffn_buckets = Vec::new();
            let mut top_neurons = Vec::new();
            if n_ff > 0 {
                let b = bucket_of(li as u32, n_ff).max(1);
                let mut s0 = 0;
                while s0 < n_ff {
                    let e = (s0 + b).min(n_ff);
                    let mean = r.inner[s0..e].iter().map(|x| x.abs()).sum::<f32>() / (e - s0) as f32;
                    ffn_buckets.push((format!("layer.{li}.ffn.n.{s0}-{}", e - 1), mean));
                    s0 = e;
                }
                ffn_buckets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                ffn_buckets.truncate(5);
                let mut order: Vec<usize> = (0..n_ff).collect();
                order.sort_by(|a, b| r.inner[*b].abs().partial_cmp(&r.inner[*a].abs()).unwrap_or(std::cmp::Ordering::Equal));
                top_neurons = order.iter().take(top_k).map(|&i| (i as u32, r.inner[i])).collect();
            }
            layers.push(JourneyLayer {
                index: li as u32,
                residual_norm: r.h.iter().map(|x| x * x).sum::<f32>().sqrt(),
                lens,
                target_prob,
                target_rank,
                attention: r.attn.clone(),
                attended,
                ffn_buckets,
                top_neurons,
                skipped: r.skipped,
            });
        }
        let decision = match (target, final_probs.is_empty()) {
            (Some(t), false) => {
                let mut logits = model.head_logits(&Tensor::from_vec(final_h.clone(), (1, dim), &model.w.dev)?)?.ok_or_else(|| anyhow!("no output head"))?.squeeze(0)?.to_vec1::<f32>()?;
                masks.apply_logits(&mut logits);
                model.decision(&rec, &logits, t as u32)?
            }
            _ => None,
        };
        let mut highlight = Vec::new();
        if let Some(d) = &decision {
            highlight.extend(d.top_heads.iter().take(5).map(|h| format!("layer.{}.attn.head.{}", h.layer, h.head)));
            let mut ls: Vec<&LayerAttribution> = d.layers.iter().collect();
            ls.sort_by(|a, b| (b.attn.abs() + b.ffn.abs()).partial_cmp(&(a.attn.abs() + a.ffn.abs())).unwrap_or(std::cmp::Ordering::Equal));
            for l in ls.iter().take(3) {
                highlight.push(format!("layer.{}", l.layer));
                highlight.push(format!("layer.{}.{}", l.layer, if l.attn.abs() >= l.ffn.abs() { "attn" } else { "ffn" }));
                if let Some((b, _)) = layers.get(l.layer as usize).and_then(|x| x.ffn_buckets.first()) {
                    highlight.push(b.clone());
                }
            }
        }
        if model.spec.encoder {
            warnings.push("embedding model: there is no next-token prediction, so the logit lens and the decision breakdown are empty".into());
        }
        if m.architecture() == "gemma2" {
            warnings.push("gemma2 logit/attention soft-capping is not applied".into());
        }
        p.set(100.0, "done");
        Ok(JourneyReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            backend: self.name().into(),
            edited: !req.ablation.is_empty(),
            text: req.text.clone(),
            tokens: ids.iter().map(|&i| token(i)).collect(),
            position: pos,
            next,
            layers,
            prediction,
            first_top1,
            first_top5,
            decision,
            highlight,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}
