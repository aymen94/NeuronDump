//! `CandleBackend`: the `TraceBackend` implementation (trace inline, the other operations delegate).

use super::*;

pub struct CandleBackend;

impl CandleBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceBackend for CandleBackend {
    fn name(&self) -> &str {
        "candle-cpu"
    }

    fn supported_architectures(&self) -> Vec<String> {
        SUPPORTED.iter().map(|s| s.to_string()).collect()
    }

    fn trace(&self, m: &ModelHandle, req: &TraceRequest, p: &dyn Progress) -> Result<TraceReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let top_k = req.top_k.unwrap_or(5).max(1);
        let max_tokens = req.max_tokens.unwrap_or(128).max(1);
        let mut texts: Vec<String> = req.prompt.iter().cloned().collect();
        texts.extend(req.corpus.iter().cloned());
        if texts.is_empty() {
            bail!("nothing to trace");
        }
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, top_k, p)?;
        let mut warnings: Vec<String> = enc.warnings.clone();
        if m.architecture() == "gemma2" {
            warnings.push("gemma2 logit/attention soft-capping is not applied".into());
        }

        let mut samples = Vec::new();
        for (i, t) in texts.iter().enumerate() {
            check_cancel(p)?;
            let mut ids = enc.encode(t, true, None).ids;
            if ids.is_empty() {
                continue;
            }
            if ids.len() > max_tokens {
                warnings.push(format!("sample {i}: truncated from {} to {max_tokens} tokens", ids.len()));
                ids.truncate(max_tokens);
            }
            samples.push(model.trace_tokens(&ids, p)?);
        }
        let Some(last) = samples.last() else { bail!("inputs produced no tokens") };
        let n = samples.len();
        // Aggregate: mean of layer metrics over samples; lens/prediction from the last one.
        let mut layers = last.layers.clone();
        if n > 1 {
            for (li, l) in layers.iter_mut().enumerate() {
                l.hidden_norm = samples.iter().map(|s| s.layers[li].hidden_norm).sum::<f32>() / n as f32;
                l.cos_prev = Some(samples.iter().filter_map(|s| s.layers[li].cos_prev).sum::<f32>() / n as f32);
                for (h, e) in l.head_entropy.iter_mut().enumerate() {
                    *e = samples.iter().map(|s| s.layers[li].head_entropy.get(h).copied().unwrap_or(0.0)).sum::<f32>() / n as f32;
                }
                let dim = last.neuron_means[li].len();
                let agg: Vec<f32> = (0..dim).map(|j| samples.iter().map(|s| s.neuron_means[li][j]).sum::<f32>() / n as f32).collect();
                let mut order: Vec<usize> = (0..dim).collect();
                order.sort_by(|a, b| agg[*b].partial_cmp(&agg[*a]).unwrap_or(std::cmp::Ordering::Equal));
                l.top_neurons = order.iter().take(top_k).map(|&i| (i as u32, agg[i])).collect();
            }
        }
        let mut hm: HashMap<u32, String> = HashMap::new();
        for &id in &last.tokens {
            hm.entry(id).or_insert_with(|| vocab.tokens.get(id as usize).map(|t| t.text.clone()).unwrap_or_default());
        }
        p.set(100.0, "done");
        Ok(TraceReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            backend: self.name().into(),
            supported: true,
            tokens: last.tokens.iter().map(|id| (*id, hm[id].clone())).collect(),
            layers,
            prediction: last.prediction.clone(),
            embedding_contributions: last.contributions.clone(),
            samples: n,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn simulate(&self, m: &ModelHandle, req: &SimRequest, p: &dyn Progress) -> Result<SimReport> {
        self.simulate_impl(m, req, p)
    }

    fn journey(&self, m: &ModelHandle, req: &JourneyRequest, p: &dyn Progress) -> Result<JourneyReport> {
        self.journey_impl(m, req, p)
    }

    fn generate(&self, m: &ModelHandle, req: &GenerateRequest, p: &dyn Progress) -> Result<GenerateReport> {
        self.generate_impl(m, req, p)
    }

    fn sweep(&self, m: &ModelHandle, req: &SweepRequest, p: &dyn Progress) -> Result<SweepReport> {
        self.sweep_impl(m, req, p)
    }
}
