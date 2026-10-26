//! `sweep`: ablate each layer or head in turn and rank by damage on a corpus.

use super::*;

impl CandleBackend {
    pub(super) fn sweep_impl(&self, m: &ModelHandle, req: &SweepRequest, p: &dyn Progress) -> Result<SweepReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let max_tokens = req.max_tokens.unwrap_or(64).max(2);
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, 5, p)?;
        if model.lm_head.is_none() {
            bail!("sweeps measure next-token predictions; this model has no output head (embedding model)");
        }
        let mut warnings: Vec<String> = enc.warnings.clone();
        let masks: Vec<Masks> = req.targets.iter().map(|t| Masks::from(&t.ablation)).collect();
        let base = Masks::default();
        let mut base_acc = EvalAcc::default();
        let mut accs = vec![EvalAcc::default(); req.targets.len()];
        let texts: Vec<Vec<u32>> = req
            .corpus
            .iter()
            .map(|t| {
                let mut ids = enc.encode(t, true, None).ids;
                if ids.len() > max_tokens {
                    warnings.push(format!("sample truncated from {} to {max_tokens} tokens", ids.len()));
                    ids.truncate(max_tokens);
                }
                ids
            })
            .filter(|ids| ids.len() >= 2)
            .collect();
        if texts.is_empty() {
            bail!("the corpus produced no samples of at least 2 tokens");
        }
        let total = (texts.len() * (req.targets.len() + 1)) as f32;
        let mut done = 0f32;
        let opt = FwdOpt { all_logits: true, ..Default::default() };
        for ids in &texts {
            check_cancel(p)?;
            p.set(5.0 + 94.0 * done / total, "baseline");
            let b = model.run(ids, &base, opt, p)?;
            let bl = model.position_logprobs(&b, &base)?;
            base_acc.add(ids, &bl, &bl);
            done += 1.0;
            for (i, mk) in masks.iter().enumerate() {
                check_cancel(p)?;
                p.set(5.0 + 94.0 * done / total, &format!("{} ({}/{})", req.targets[i].label, i + 1, masks.len()));
                let a = model.run(ids, mk, opt, p)?;
                let al = model.position_logprobs(&a, mk)?;
                accs[i].add(ids, &bl, &al);
                done += 1.0;
            }
        }
        let base_ppl = base_acc.ppl();
        let mut entries: Vec<SweepEntry> = req
            .targets
            .iter()
            .zip(&accs)
            .map(|(t, a)| SweepEntry {
                id: t.id.clone(),
                label: t.label.clone(),
                mean_kl: a.mean_kl(),
                ppl: a.ppl(),
                ppl_increase: match (a.ppl(), base_ppl) {
                    (Some(x), Some(b)) if b > 0.0 => Some(x / b - 1.0),
                    _ => None,
                },
                top1_agreement: a.agreement(),
                rank: 0,
            })
            .collect();
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|a, b| entries[*b].mean_kl.partial_cmp(&entries[*a].mean_kl).unwrap_or(std::cmp::Ordering::Equal));
        for (r, &i) in order.iter().enumerate() {
            entries[i].rank = r + 1;
        }
        p.set(100.0, "done");
        warnings.dedup();
        Ok(SweepReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            digest: m.entry.digest.clone(),
            architecture: m.architecture().to_string(),
            level: req.level.clone(),
            layer: req.layer,
            samples: texts.len(),
            tokens: base_acc.positions,
            baseline_ppl: base_ppl,
            ranking: order.iter().map(|&i| entries[i].id.clone()).collect(),
            entries,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}
