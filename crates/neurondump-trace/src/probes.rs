//! `causal` (restore clean states in a corrupted run), `head_roles` (attention patterns on
//! repeated random tokens) and `similarity` (linear CKA between layers on a corpus).

use super::*;
use neurondump_core::trace::{
    centered_gram, cka_from_grams, CausalComponent, CausalPeak, CausalReport, CausalRequest, HeadRolesReport, HeadRolesRequest, HeadScore, LayerChange,
    SimilarityReport, SimilarityRequest,
};

/// Keeps the last `max` tokens (the prediction is made at the end).
fn keep_last(ids: &mut Vec<u32>, max: usize, what: &str, warnings: &mut Vec<String>) {
    if ids.len() > max {
        warnings.push(format!("{what} truncated from {} to its last {max} tokens", ids.len()));
        ids.drain(..ids.len() - max);
    }
}

fn prob_of(logits: &[f32], id: usize) -> f32 {
    log_softmax(logits).get(id).map_or(0.0, |x| x.exp() as f32)
}

impl CandleBackend {
    pub(super) fn causal_impl(&self, m: &ModelHandle, req: &CausalRequest, p: &dyn Progress) -> Result<CausalReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let max_tokens = req.max_tokens.clamp(2, 512);
        let mut warnings: Vec<String> = enc.warnings.clone();
        let mut clean_ids = enc.encode(&req.clean, true, None).ids;
        let mut corrupt_ids = enc.encode(&req.corrupt, true, None).ids;
        keep_last(&mut clean_ids, max_tokens, "clean prompt", &mut warnings);
        keep_last(&mut corrupt_ids, max_tokens, "corrupted prompt", &mut warnings);
        if clean_ids.is_empty() || corrupt_ids.is_empty() {
            bail!("both prompts must produce tokens");
        }
        if clean_ids.len() != corrupt_ids.len() {
            warnings.push(format!(
                "the prompts have {} and {} tokens: positions are aligned from the end",
                clean_ids.len(),
                corrupt_ids.len()
            ));
        }
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, 5, p)?;
        if model.lm_head.is_none() {
            bail!("causal tracing measures next-token probabilities; this model has no output head (embedding model)");
        }
        let mut masks = model.masks(&enc, &req.ablation, max_tokens, p)?;
        let blocks = req.component != CausalComponent::Residual;
        p.set(46.0, "clean run");
        let clean = model.run(&clean_ids, &masks, FwdOpt { keep: true, keep_blocks: blocks, ..Default::default() }, p)?;
        let target = req.target.map(|t| t as usize).unwrap_or_else(|| argmax(&clean.last_logits));
        if target >= clean.last_logits.len() {
            bail!("target token {target} is outside the vocabulary");
        }
        p.set(48.0, "corrupted run");
        let corrupt = model.run(&corrupt_ids, &masks, FwdOpt { keep: true, ..Default::default() }, p)?;
        // Restoring at layer l leaves the layers before it as in the corrupted run: resume there.
        let dims = (corrupt_ids.len(), corrupt.input.first().map_or(0, Vec::len));
        let state = |rows: &[Vec<f32>]| Tensor::from_vec(rows.concat(), dims, &model.w.dev);
        let clean_prob = prob_of(&clean.last_logits, target);
        let corrupt_prob = prob_of(&corrupt.last_logits, target);
        let gap = clean_prob - corrupt_prob;
        if gap.abs() < 1e-4 {
            warnings.push("the corruption barely changes the target probability: the effects are not meaningful".into());
        }
        let corrupt_probs: Vec<f32> = log_softmax(&corrupt.last_logits).iter().map(|x| x.exp() as f32).collect();
        let corrupt_top = model.top_tokens(&corrupt_probs);

        let n_layers = model.layers.len();
        let seq = corrupt_ids.len();
        let off = clean_ids.len() as i64 - seq as i64;
        let source = |l: usize, j: usize| -> Option<Vec<f32>> {
            let rows = match req.component {
                CausalComponent::Residual => clean.hidden.get(l),
                CausalComponent::Attn => clean.attn_rows.get(l),
                CausalComponent::Ffn => clean.ffn_rows.get(l),
            }?;
            rows.get(j).cloned()
        };
        let total = (n_layers * seq) as f32;
        let mut probs = vec![vec![None; seq]; n_layers];
        let mut effect = vec![vec![None; seq]; n_layers];
        for l in 0..n_layers {
            for i in 0..seq {
                check_cancel(p)?;
                let j = i as i64 + off;
                if j < 0 {
                    continue;
                }
                let Some(row) = source(l, j as usize) else { continue };
                p.set(50.0 + 49.0 * (l * seq + i) as f32 / total, &format!("layer {l}, position {i}"));
                let r = match req.component {
                    CausalComponent::Residual => {
                        // The residual after layer l with row i restored, resumed at layer l + 1.
                        let mut rows = corrupt.hidden[l].clone();
                        rows[i] = row;
                        masks.start = Some((l + 1, state(&rows)?));
                        model.run(&corrupt_ids, &masks, FwdOpt::default(), p)
                    }
                    c => {
                        let before = if l == 0 { &corrupt.input } else { &corrupt.hidden[l - 1] };
                        masks.start = Some((l, state(before)?));
                        let map = if c == CausalComponent::Attn { &mut masks.restore_attn } else { &mut masks.restore_ffn };
                        map.insert(l, vec![(i, row)]);
                        let r = model.run(&corrupt_ids, &masks, FwdOpt::default(), p);
                        masks.restore_attn.remove(&l);
                        masks.restore_ffn.remove(&l);
                        r
                    }
                };
                masks.start = None;
                let pr = prob_of(&r?.last_logits, target);
                probs[l][i] = Some(pr);
                effect[l][i] = Some(if gap.abs() > 1e-6 { (pr - corrupt_prob) / gap } else { 0.0 });
            }
        }
        let mut peaks: Vec<CausalPeak> = effect
            .iter()
            .enumerate()
            .flat_map(|(l, row)| row.iter().enumerate().filter_map(move |(i, e)| e.map(|e| CausalPeak { layer: l as u32, position: i, effect: e })))
            .collect();
        peaks.sort_by(|a, b| b.effect.total_cmp(&a.effect));
        peaks.truncate(8);
        let token = |id: u32| JourneyToken {
            id,
            text: vocab.tokens.get(id as usize).map(|t| t.text.clone()).unwrap_or_default(),
            special: vocab.tokens.get(id as usize).is_some_and(|t| matches!(t.token_type, TokenType::Control | TokenType::UserDefined)),
        };
        p.set(100.0, "done");
        Ok(CausalReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            component: req.component,
            clean_tokens: clean_ids.iter().map(|&i| token(i)).collect(),
            corrupt_tokens: corrupt_ids.iter().map(|&i| token(i)).collect(),
            target: model.lens_token(target, clean_prob),
            clean_prob,
            corrupt_prob,
            corrupt_top,
            probs,
            effect,
            peaks,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    pub(super) fn head_roles_impl(&self, m: &ModelHandle, req: &HeadRolesRequest, p: &dyn Progress) -> Result<HeadRolesReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let warnings: Vec<String> = enc.warnings.clone();
        // Ordinary tokens only, so the sequences carry no structure of their own.
        let pool: Vec<u32> = vocab
            .tokens
            .iter()
            .filter(|t| t.token_type == TokenType::Normal && t.text.trim().chars().count() >= 2)
            .map(|t| t.id)
            .collect();
        if pool.len() < req.length * 2 {
            bail!("the vocabulary has too few ordinary tokens for synthetic sequences");
        }
        let prefix = enc.encode("", true, None).ids;
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, 5, p)?;
        let masks = model.masks(&enc, &req.ablation, 2 * req.length + prefix.len(), p)?;
        let mut rng = req.seed.max(1);
        let mut sums: Vec<Vec<[f32; 5]>> = model.layers.iter().map(|l| vec![[0.0; 5]; l.n_head]).collect();
        let mut counts: Vec<f32> = vec![0.0; model.layers.len()];
        for s in 0..req.samples {
            check_cancel(p)?;
            p.set(50.0 + 49.0 * s as f32 / req.samples as f32, &format!("sequence {}/{}", s + 1, req.samples));
            let r: Vec<u32> = (0..req.length).map(|_| pool[(next_rand(&mut rng) * pool.len() as f64) as usize % pool.len()]).collect();
            let mut ids = prefix.clone();
            ids.extend(&r);
            ids.extend(&r);
            let f = model.run(&ids, &masks, FwdOpt { head_probe: Some((prefix.len(), req.length)), ..Default::default() }, p)?;
            for (l, scores) in f.head_scores.iter().enumerate() {
                if scores.is_empty() {
                    continue;
                }
                counts[l] += 1.0;
                for (h, sc) in scores.iter().enumerate() {
                    for k in 0..5 {
                        sums[l][h][k] += sc[k];
                    }
                }
            }
        }
        let mut heads = Vec::new();
        for (l, hs) in sums.iter().enumerate() {
            if counts[l] == 0.0 {
                continue;
            }
            for (h, s) in hs.iter().enumerate() {
                let c = counts[l];
                let mut hsc = HeadScore {
                    layer: l as u32,
                    head: h as u32,
                    prev_token: s[0] / c,
                    induction: s[1] / c,
                    duplicate: s[2] / c,
                    sink: s[3] / c,
                    self_attn: s[4] / c,
                    role: String::new(),
                };
                hsc.classify();
                heads.push(hsc);
            }
        }
        let mut counts: Vec<(String, usize)> = Vec::new();
        for h in &heads {
            match counts.iter_mut().find(|(r, _)| *r == h.role) {
                Some(c) => c.1 += 1,
                None => counts.push((h.role.clone(), 1)),
            }
        }
        counts.sort_by_key(|c| std::cmp::Reverse(c.1));
        p.set(100.0, "done");
        Ok(HeadRolesReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            samples: req.samples,
            length: req.length,
            heads,
            counts,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    pub(super) fn similarity_impl(&self, m: &ModelHandle, req: &SimilarityRequest, p: &dyn Progress) -> Result<SimilarityReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let mut warnings: Vec<String> = enc.warnings.clone();
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, 5, p)?;
        let masks = model.masks(&enc, &req.ablation, req.max_tokens, p)?;
        let n_layers = model.layers.len();
        // points[0] = embeddings, points[l + 1] = after layer l; rows are token positions.
        let mut points: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers + 1];
        let mut samples = 0;
        for (si, t) in req.corpus.iter().enumerate() {
            check_cancel(p)?;
            let have = points[0].len();
            if have >= req.max_positions {
                warnings.push(format!("stopped after {si} samples: {} positions reached", req.max_positions));
                break;
            }
            let mut ids = enc.encode(t, true, None).ids;
            if ids.len() > req.max_tokens {
                ids.truncate(req.max_tokens);
            }
            if ids.len() < 2 {
                continue;
            }
            p.set(50.0 + 30.0 * si as f32 / req.corpus.len() as f32, &format!("sample {}/{}", si + 1, req.corpus.len()));
            let f = model.run(&ids, &masks, FwdOpt { keep: true, ..Default::default() }, p)?;
            // Position 0 (often BOS, an attention sink with a huge norm) would dominate the kernels.
            let take = (ids.len() - 1).min(req.max_positions - have);
            points[0].extend(f.input[1..=take].iter().cloned());
            for (l, rows) in f.hidden.iter().enumerate() {
                points[l + 1].extend(rows[1..=take].iter().cloned());
            }
            samples += 1;
        }
        let n = points[0].len();
        if n < 4 {
            bail!("the corpus gave only {n} positions; add text");
        }
        p.set(82.0, "kernels");
        let grams: Vec<Vec<f64>> = points.iter().map(|x| centered_gram(x)).collect();
        check_cancel(p)?;
        p.set(92.0, "CKA");
        let k = grams.len();
        let mut cka = vec![vec![0f32; k]; k];
        for i in 0..k {
            cka[i][i] = 1.0;
            for j in i + 1..k {
                let v = cka_from_grams(&grams[i], &grams[j]) as f32;
                cka[i][j] = v;
                cka[j][i] = v;
            }
        }
        let layers: Vec<LayerChange> = (0..n_layers)
            .map(|l| {
                let (a, b) = (&points[l], &points[l + 1]);
                let cos = a.iter().zip(b).map(|(x, y)| cosine(x, y)).sum::<f32>() / n as f32;
                let rel = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| {
                        let d: f32 = x.iter().zip(y).map(|(u, v)| (v - u) * (v - u)).sum::<f32>().sqrt();
                        let nx: f32 = x.iter().map(|u| u * u).sum::<f32>().sqrt();
                        if nx > 0.0 { d / nx } else { 0.0 }
                    })
                    .sum::<f32>()
                    / n as f32;
                LayerChange { layer: l as u32, cka_in_out: cka[l][l + 1], cos_in_out: cos, rel_update: rel }
            })
            .collect();
        let mut redundant: Vec<&LayerChange> = layers.iter().filter(|c| c.cka_in_out >= 0.99).collect();
        redundant.sort_by(|a, b| b.cka_in_out.total_cmp(&a.cka_in_out));
        let mut labels = vec!["embeddings".to_string()];
        labels.extend((0..n_layers).map(|l| format!("layer {l}")));
        warnings.dedup();
        p.set(100.0, "done");
        Ok(SimilarityReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            samples,
            positions: n,
            labels,
            cka,
            redundant: redundant.iter().map(|c| c.layer).collect(),
            layers,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}
