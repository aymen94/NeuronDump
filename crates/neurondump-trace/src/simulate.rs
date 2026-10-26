//! `simulate`: baseline vs edited runs over prompts and a corpus (KL, perplexity, generations, lens diff).

use super::*;

impl CandleBackend {
    pub(super) fn simulate_impl(&self, m: &ModelHandle, req: &SimRequest, p: &dyn Progress) -> Result<SimReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let top_k = if req.top_k == 0 { 5 } else { req.top_k };
        let max_tokens = if req.max_tokens == 0 { 128 } else { req.max_tokens };
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, top_k, p)?;
        let base = Masks::default();
        let abl = model.masks(&enc, &req.ablation, max_tokens, p)?;
        let mut warnings: Vec<String> = enc.warnings.clone();
        if !req.ablation.zero_experts.is_empty() || !req.ablation.scale_experts.is_empty() {
            warnings.push("expert edits are not supported by this forward pass; ignored".into());
        }
        let eos = m.gguf.get_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        let encode = |t: &str, warnings: &mut Vec<String>| -> Vec<u32> {
            let mut ids = enc.encode(t, true, None).ids;
            if ids.len() > max_tokens {
                warnings.push(format!("input truncated from {} to {max_tokens} tokens", ids.len()));
                ids.truncate(max_tokens);
            }
            ids
        };
        let n_steps = (req.prompts.len() * 2 + req.corpus.len()).max(1) as f32;
        let mut step = 0f32;
        let mut prompts = Vec::new();
        for (pi, text) in req.prompts.iter().enumerate() {
            check_cancel(p)?;
            let ids = encode(text, &mut warnings);
            if ids.is_empty() {
                continue;
            }
            let edited_text = req.edited_prompts.get(pi).filter(|t| *t != text).cloned();
            let eids = match &edited_text {
                Some(t) => encode(t, &mut warnings),
                None => ids.clone(),
            };
            if eids.is_empty() {
                continue;
            }
            p.set(10.0 + 85.0 * step / n_steps, "baseline");
            step += 1.0;
            let opt = |n: usize| FwdOpt { measure: true, lens_probs: true, record: req.decisions.then(|| n - 1), ..Default::default() };
            let b = model.run(&ids, &base, opt(ids.len()), p)?;
            p.set(10.0 + 85.0 * step / n_steps, "edited");
            step += 1.0;
            let a = model.run(&eids, &abl, opt(eids.len()), p)?;
            let (kl_v, top1_changed, btop, atop) = if b.last_logits.is_empty() {
                (None, false, vec![], vec![])
            } else {
                let lb = log_softmax(&b.last_logits);
                let la = log_softmax(&a.last_logits);
                let pb: Vec<f32> = lb.iter().map(|x| x.exp() as f32).collect();
                let pa: Vec<f32> = la.iter().map(|x| x.exp() as f32).collect();
                (Some(kl(&lb, &la)), argmax(&b.last_logits) != argmax(&a.last_logits), model.top_tokens(&pb), model.top_tokens(&pa))
            };
            let target = btop.first().map(|t| t.id as usize);
            let lens: Vec<LensDiff> = (0..b.sample.layers.len())
                .map(|li| {
                    let tp = |v: &Vec<Vec<f32>>| target.and_then(|t| v.get(li).and_then(|r| r.get(t).copied())).unwrap_or(0.0);
                    LensDiff {
                        layer: li as u32,
                        baseline: b.sample.layers[li].logit_lens.first().cloned(),
                        ablated: a.sample.layers.get(li).and_then(|l| l.logit_lens.first().cloned()),
                        baseline_target_prob: tp(&b.lens_probs),
                        ablated_target_prob: tp(&a.lens_probs),
                        skipped: abl.skip.contains(&li),
                    }
                })
                .collect();
            let decide = |f: &FwdOut| -> Result<Option<Decision>> {
                match (&f.record, f.last_logits.is_empty()) {
                    (Some(r), false) => model.decision(r, &f.last_logits, argmax(&f.last_logits) as u32),
                    _ => Ok(None),
                }
            };
            let (db, da) = (decide(&b)?, decide(&a)?);
            let (bg, ag) = if req.gen_tokens > 0 && !b.last_logits.is_empty() {
                p.set(10.0 + 85.0 * step / n_steps, "generating");
                (model.generate(&ids, &base, req.gen_tokens, eos, p)?, model.generate_sampled(&eids, &abl, req.gen_tokens, eos, &req.sampling, p)?)
            } else {
                let t0 = Instant::now();
                model.run(&ids, &base, FwdOpt::default(), p)?;
                let tb = ids.len() as f64 / t0.elapsed().as_secs_f64().max(1e-9);
                let t1 = Instant::now();
                model.run(&eids, &abl, FwdOpt::default(), p)?;
                let ta = eids.len() as f64 / t1.elapsed().as_secs_f64().max(1e-9);
                ((vec![], String::new(), tb), (vec![], String::new(), ta))
            };
            let norm = |v: &Option<Vec<f32>>| v.as_ref().map(|x| x.iter().map(|y| y * y).sum::<f32>().sqrt());
            prompts.push(SimPrompt {
                prompt: text.clone(),
                n_tokens: ids.len(),
                baseline: SimSide { top_k: btop, generated: bg.1, generated_tokens: bg.0.len(), tokens_per_s: bg.2, embedding_norm: norm(&b.pooled) },
                ablated: SimSide { top_k: atop, generated: ag.1, generated_tokens: ag.0.len(), tokens_per_s: ag.2, embedding_norm: norm(&a.pooled) },
                kl: kl_v,
                top1_changed,
                embedding_cosine: match (&b.pooled, &a.pooled) {
                    (Some(x), Some(y)) => Some(cosine64(x, y)),
                    _ => None,
                },
                lens,
                edited_prompt: edited_text,
                decision_baseline: db,
                decision_edited: da,
            });
        }
        let mut corpus = None;
        if !req.corpus.is_empty() && model.lm_head.is_some() {
            let mut base_acc = EvalAcc::default();
            let mut acc = EvalAcc::default();
            let mut n = 0;
            for text in &req.corpus {
                check_cancel(p)?;
                p.set(10.0 + 85.0 * step / n_steps, "corpus");
                step += 1.0;
                let ids = encode(text, &mut warnings);
                if ids.len() < 2 {
                    continue;
                }
                let opt = FwdOpt { all_logits: true, ..Default::default() };
                let b = model.run(&ids, &base, opt, p)?;
                let bl = model.position_logprobs(&b, &base)?;
                let a = model.run(&ids, &abl, opt, p)?;
                let al = model.position_logprobs(&a, &abl)?;
                base_acc.add(&ids, &bl, &bl);
                acc.add(&ids, &bl, &al);
                n += 1;
            }
            corpus = Some(CorpusEval {
                samples: n,
                tokens: acc.positions,
                baseline_ppl: base_acc.ppl(),
                ablated_ppl: acc.ppl(),
                mean_kl: (acc.positions > 0).then(|| acc.mean_kl()),
                top1_agreement: acc.agreement(),
            });
        }
        let total: u64 = m.gguf.tensors.iter().map(|t| t.n_bytes).sum();
        let skipped: u64 = m
            .gguf
            .tensors
            .iter()
            .filter(|t| neurondump_core::arch::parse_tensor_name(&t.name).layer.is_some_and(|l| abl.skip.contains(&(l as usize))))
            .map(|t| t.n_bytes)
            .sum();
        p.set(100.0, "done");
        warnings.push("in-memory masks: process memory is the same for both runs; weight bytes are what a slimmed file would hold".into());
        if req.gen_tokens > 0 {
            warnings.push("generation recomputes the whole sequence for every token (no KV cache, CPU): tokens/s is only for comparing baseline and ablated".into());
        }
        if !req.sampling.is_greedy() {
            warnings.push(format!("the edited side samples (temperature {}): its text changes with the seed", req.sampling.temperature));
        }
        Ok(SimReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            backend: self.name().into(),
            cut: req.cut.clone(),
            ablation: req.ablation.clone(),
            prompts,
            corpus,
            weights_bytes_baseline: total,
            weights_bytes_ablated: total - skipped,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}
