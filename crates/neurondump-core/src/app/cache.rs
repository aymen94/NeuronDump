//! Cached per-model products: vocab, stats, graph, latest traces / sweeps, impact, plans, embedding map.

use super::*;

impl App {
    pub fn vocab(&self, m: &ModelHandle) -> Result<Arc<Vocab>> {
        if let Some(v) = self.vocabs.lock().unwrap().get(&m.entry.id) {
            return Ok(v.clone());
        }
        let v = Arc::new(Vocab::from_gguf(&m.gguf)?);
        self.vocabs.lock().unwrap().insert(m.entry.id.clone(), v.clone());
        Ok(v)
    }

    pub(super) fn cache_get(&self, m: &ModelHandle, kind: &str, params: &Value) -> Option<Value> {
        let p = cache_path(&self.config.output_dir, &m.entry.digest, kind, params);
        let bytes = std::fs::read(p).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    pub(super) fn cache_put(&self, m: &ModelHandle, kind: &str, params: &Value, v: &Value) {
        let p = cache_path(&self.config.output_dir, &m.entry.digest, kind, params);
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Err(e) = std::fs::write(&p, serde_json::to_vec(v).unwrap_or_default()) {
            tracing::warn!("cache write {}: {e}", p.display());
        }
    }

    /// Weight statistics for the default options, from cache or computed.
    pub fn stats(&self, m: &ModelHandle, compute: bool, p: &dyn Progress) -> Result<Option<StatsReport>> {
        let params = serde_json::to_value(StatsOptions::default())?;
        if let Some(v) = self.cache_get(m, "stats", &params) {
            return Ok(serde_json::from_value(v).ok());
        }
        if !compute {
            return Ok(None);
        }
        let s = compute_stats(&m.gguf, &m.entry.name, &m.entry.digest, &StatsOptions::default(), p)?;
        self.cache_put(m, "stats", &params, &serde_json::to_value(&s)?);
        Ok(Some(s))
    }

    /// Latest corpus trace for the model, if any (feeds layer-redundancy scores).
    pub fn latest_corpus_trace(&self, m: &ModelHandle) -> Option<TraceReport> {
        self.cache_get(m, "trace-corpus-latest", &Value::Null).and_then(|v| serde_json::from_value(v).ok())
    }

    /// Structural graph with cached measurements (stats, corpus trace, sweeps); built once
    /// per model and rebuilt after a job that adds measurements.
    pub fn graph(&self, m: &ModelHandle) -> Result<Arc<ModelGraph>> {
        if let Some(g) = self.graphs.lock().unwrap().get(&m.entry.id) {
            return Ok(g.clone());
        }
        let vocab = self.vocab(m).ok();
        let stats = self.stats(m, false, &crate::util::NoProgress)?;
        let trace = self.latest_corpus_trace(m);
        let mut sweeps = Vec::new();
        if let Some(sw) = self.latest_sweep(m, "layer", None) {
            sweeps.push(ablate::sweep_importance(&sw));
        }
        let n_layers = crate::arch::analyze_architecture(&m.gguf).layers.len() as u32;
        for l in 0..n_layers {
            if let Some(sw) = self.latest_sweep(m, "head", Some(l)) {
                sweeps.push(ablate::sweep_importance(&sw));
            }
        }
        let inputs = GraphInputs { stats: stats.as_ref(), trace: trace.as_ref(), sweeps };
        let g = Arc::new(ModelGraph::build(m, vocab.as_deref(), &inputs));
        self.graphs.lock().unwrap().insert(m.entry.id.clone(), g.clone());
        Ok(g)
    }

    pub(super) fn invalidate_graph(&self, m: &ModelHandle) {
        self.graphs.lock().unwrap().remove(&m.entry.id);
    }

    pub(super) fn sweep_key(level: &str, layer: Option<u32>) -> Value {
        serde_json::json!({ "level": if level.starts_with("head") { "head" } else { "layer" }, "layer": layer })
    }

    /// Latest sweep for a level (and layer, for head sweeps).
    pub fn latest_sweep(&self, m: &ModelHandle, level: &str, layer: Option<u32>) -> Option<SweepReport> {
        self.cache_get(m, "sweep-latest", &Self::sweep_key(level, layer)).and_then(|v| serde_json::from_value(v).ok())
    }

    /// Static impact of cutting these node ids (instant, any architecture).
    pub fn impact(&self, m: &ModelHandle, cut: &[String], force: bool) -> Result<StaticImpact> {
        let g = self.graph(m)?;
        let v = self.vocab(m).ok();
        Ok(ablate::static_impact(m, &g, v.as_deref(), cut, force))
    }

    /// Prune plan for a cut (`slim` can apply it).
    pub fn plan_from_cut(&self, m: &ModelHandle, cut: &[String], force: bool) -> Result<PrunePlan> {
        let g = self.graph(m)?;
        let i = self.impact(m, cut, force)?;
        ablate::plan_from_impact(m, &g, &i, force)
    }

    /// Prune plan from the `top` least damaging entries of a sweep.
    pub fn plan_from_sweep(&self, m: &ModelHandle, sw: &SweepReport, top: usize, force: bool) -> Result<(PrunePlan, StaticImpact)> {
        let g = self.graph(m)?;
        let v = self.vocab(m).ok();
        ablate::plan_from_sweep(m, &g, v.as_deref(), sw, top, force)
    }

}

impl App {
    /// The PCA embedding map, from memory, the disk cache or computed.
    pub fn embedding_map(&self, m: &ModelHandle, p: &dyn Progress) -> Result<Arc<EmbeddingMap>> {
        if let Some(e) = self.maps.lock().unwrap().get(&m.entry.id) {
            return Ok(e.clone());
        }
        let path = self.config.output_dir.join("cache").join(&m.entry.digest).join("embedmap-v1.bin");
        let map = match EmbeddingMap::load(&path) {
            Ok(e) => e,
            Err(_) => {
                let v = self.vocab(m).ok();
                let e = crate::tokens::embedding_map(m, v.as_deref(), p)?;
                if let Err(err) = e.save(&path) {
                    tracing::warn!("embedding map cache {}: {err}", path.display());
                }
                e
            }
        };
        let map = Arc::new(map);
        self.maps.lock().unwrap().insert(m.entry.id.clone(), map.clone());
        Ok(map)
    }

    /// Parses a stored plan (CLI `slim --plan`).
    pub fn load_plan(path: &std::path::Path) -> Result<PrunePlan> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading plan {}", path.display()))?;
        serde_json::from_str(&text).context("invalid prune plan JSON")
    }
}
