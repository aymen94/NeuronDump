//! `generate`: step-by-step generation with optional explanations.

use super::*;

impl CandleBackend {
    pub(super) fn generate_impl(&self, m: &ModelHandle, req: &GenerateRequest, p: &dyn Progress) -> Result<GenerateReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let max_tokens = if req.max_tokens == 0 { 512 } else { req.max_tokens };
        let mut warnings: Vec<String> = enc.warnings.clone();
        let mut ids = enc.encode(&req.text, true, None).ids;
        if ids.len() > max_tokens {
            warnings.push(format!("input truncated from {} to {max_tokens} tokens", ids.len()));
            ids.truncate(max_tokens);
        }
        if ids.is_empty() {
            bail!("the prompt produced no tokens");
        }
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, req.top_k.max(2), p)?;
        if model.lm_head.is_none() {
            bail!("this model has no output head (embedding model): it cannot generate");
        }
        let masks = model.masks(&enc, &req.ablation, max_tokens, p)?;
        let eos = m.gguf.get_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        let t = Instant::now();
        let steps = model.steps(&ids, &masks, req.steps.max(1), eos, &req.sampling, req.explain, p)?;
        let gen: Vec<u32> = steps.iter().map(|s| s.token.id).collect();
        let tps = gen.len() as f64 / t.elapsed().as_secs_f64().max(1e-9);
        warnings.push("no KV cache: every step reruns the whole sequence on the CPU".into());
        let token = |id: u32| JourneyToken {
            id,
            text: vocab.tokens.get(id as usize).map(|t| t.text.clone()).unwrap_or_default(),
            special: vocab.tokens.get(id as usize).is_some_and(|t| matches!(t.token_type, TokenType::Control | TokenType::UserDefined)),
        };
        p.set(100.0, "done");
        Ok(GenerateReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            backend: self.name().into(),
            edited: !req.ablation.is_empty() || !req.sampling.is_greedy() || !req.sampling.forced.is_empty(),
            text: req.text.clone(),
            prompt_tokens: ids.iter().map(|&i| token(i)).collect(),
            generated: model.detok(&gen),
            generated_ids: gen,
            steps,
            sampling: req.sampling.clone(),
            tokens_per_s: tps,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}
