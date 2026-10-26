//! `App::run` (every job kind, with result caching) and `App::submit` (background job threads).

use super::*;

impl App {
    /// Runs a job synchronously (CLI) — the job system calls the same function.
    /// Returns the result and whether it came from the cache.
    pub fn run(&self, kind: JobKind, model: &str, params: Value, p: &dyn Progress) -> Result<(Value, bool)> {
        let m = self.open(model)?;
        let mut params = if params.is_null() { Value::Object(Default::default()) } else { params };
        if matches!(kind, JobKind::Journey | JobKind::Generate | JobKind::Compare | JobKind::Bake) {
            // Jobs see the edit stack as it was when they started (and are cached on it).
            if let Some(o) = params.as_object_mut() {
                if o.get("edits").is_none_or(Value::is_null) {
                    o.insert("edits".into(), serde_json::to_value(self.edit_stack(&m).enabled())?);
                }
            }
        }
        let mut other = None;
        if kind == JobKind::Diff {
            let q = params.get("other").and_then(Value::as_str).context("diff params need `other` (the second model)")?.to_string();
            let b = self.open(&q)?;
            // The cache key must change when either file does (e.g. a re-slim to the same path).
            let stamp = |p: &std::path::Path| {
                std::fs::metadata(p).map(|md| {
                    let t = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
                    format!("{}:{t}", md.len())
                })
            };
            if let Some(o) = params.as_object_mut() {
                o.insert("other".into(), Value::String(b.entry.id.clone()));
                o.insert("files".into(), serde_json::json!([b.entry.digest, stamp(&m.entry.path).ok(), stamp(&b.entry.path).ok()]));
            }
            other = Some(b);
        }
        let cacheable = matches!(
            kind,
            JobKind::Inspect
                | JobKind::Stats
                | JobKind::Trace
                | JobKind::Ablate
                | JobKind::Sweep
                | JobKind::Journey
                | JobKind::Generate
                | JobKind::Compare
                | JobKind::Diff
        );
        if cacheable {
            if let Some(v) = self.cache_get(&m, kind.as_str(), &params) {
                p.set(100.0, "cached");
                return Ok((v, true));
            }
        }
        let v = match kind {
            JobKind::Inspect => {
                let o: InspectOptions = serde_json::from_value(params.clone()).context("inspect params")?;
                let r = inspect(&m, &o, p)?;
                if o.stats {
                    if let Some(s) = &r.stats {
                        self.cache_put(&m, "stats", &serde_json::to_value(StatsOptions::default())?, &serde_json::to_value(s)?);
                    }
                }
                serde_json::to_value(r)?
            }
            JobKind::Stats => {
                let o: StatsOptions = serde_json::from_value(params.clone()).context("stats params")?;
                serde_json::to_value(compute_stats(&m.gguf, &m.entry.name, &m.entry.digest, &o, p)?)?
            }
            JobKind::Analyze => {
                let compute_stats = params.get("compute_stats").and_then(Value::as_bool).unwrap_or(false);
                let o: AnalyzeOptions = serde_json::from_value(params.clone()).context("analyze params")?;
                let stats = self.stats(&m, compute_stats, p)?;
                let tr = self.latest_corpus_trace(&m);
                serde_json::to_value(analyze(&m, &o, stats.as_ref(), tr.as_ref(), p)?)?
            }
            JobKind::Slim => {
                let o: SlimOptions = serde_json::from_value(params.clone()).context("slim params (need plan + out)")?;
                serde_json::to_value(slim(&m, &o, &self.config.output_dir, p)?)?
            }
            JobKind::Trace => {
                let o: TraceRequest = serde_json::from_value(params.clone()).context("trace params")?;
                let r = trace(&m, &o, p)?;
                if r.samples > 1 {
                    self.cache_put(&m, "trace-corpus-latest", &Value::Null, &serde_json::to_value(&r)?);
                    self.invalidate_graph(&m);
                }
                serde_json::to_value(r)?
            }
            JobKind::Ablate => {
                let o: AblateParams = serde_json::from_value(params.clone()).context("ablate params")?;
                p.set(1.0, "static impact");
                let impact = self.impact(&m, &o.cut, o.force)?;
                let simulation = if o.static_only {
                    None
                } else {
                    let mut corpus = o.corpus.clone();
                    if let Some(f) = &o.corpus_file {
                        corpus.extend(crate::trace::read_corpus(f)?);
                    }
                    let mut prompts = o.prompts.clone();
                    prompts.extend(o.prompt.clone());
                    let req = SimRequest {
                        prompts,
                        corpus,
                        gen_tokens: o.tokens.unwrap_or(16),
                        top_k: o.top_k.unwrap_or(5),
                        max_tokens: o.max_tokens.unwrap_or(128),
                        ..Default::default()
                    };
                    Some(ablate::simulate(&m, &impact, req, p)?)
                };
                serde_json::to_value(AblateResult { impact, simulation })?
            }
            JobKind::Sweep => {
                let o: SweepRequest = serde_json::from_value(params.clone()).context("sweep params")?;
                let g = self.graph(&m)?;
                let r = ablate::sweep(&m, &g, o, p)?;
                self.cache_put(&m, "sweep-latest", &Self::sweep_key(&r.level, r.layer), &serde_json::to_value(&r)?);
                self.invalidate_graph(&m);
                serde_json::to_value(r)?
            }
            JobKind::Journey => {
                let o: JourneyParams = serde_json::from_value(params.clone()).context("journey params")?;
                let c = self.compile_for(&m, o.edits.as_deref().unwrap_or_default(), o.baseline)?;
                let g = self.graph(&m)?;
                let req = JourneyRequest {
                    text: self.prompt_text(&m, &o.prompt, &c)?,
                    append: o.append.clone(),
                    token: o.token.unwrap_or(-1),
                    top_k: o.top_k.unwrap_or(5),
                    max_tokens: o.max_tokens.unwrap_or(256),
                    ablation: c.intervention.clone(),
                    buckets: g.dims.iter().map(|(l, d)| (*l, d.bucket)).collect(),
                };
                serde_json::to_value(crate::trace::journey(&m, &req, p)?)?
            }
            JobKind::Generate => {
                let o: GenerateParams = serde_json::from_value(params.clone()).context("generate params")?;
                let c = self.compile_for(&m, o.edits.as_deref().unwrap_or_default(), o.baseline)?;
                let mut sampling = c.sampling.clone().unwrap_or_default();
                if let Some(t) = o.temperature {
                    sampling.temperature = t.max(0.0);
                }
                if o.sample_top_k.is_some() {
                    sampling.top_k = o.sample_top_k;
                }
                if o.top_p.is_some() {
                    sampling.top_p = o.top_p;
                }
                if o.seed.is_some() {
                    sampling.seed = o.seed;
                }
                let req = GenerateRequest {
                    text: self.prompt_text(&m, &o.prompt, &c)?,
                    steps: o.steps.unwrap_or(16).clamp(1, 512),
                    explain: o.explain,
                    sampling,
                    top_k: o.top_k.unwrap_or(5),
                    max_tokens: o.max_tokens.unwrap_or(512),
                    ablation: c.intervention.clone(),
                };
                serde_json::to_value(crate::trace::generate(&m, &req, p)?)?
            }
            JobKind::Compare => {
                let o: CompareParams = serde_json::from_value(params.clone()).context("compare params")?;
                let edits = o.edits.clone().unwrap_or_default();
                let c = self.compile_for(&m, &edits, false)?;
                let mut corpus = o.corpus.clone();
                if let Some(f) = &o.corpus_file {
                    corpus.extend(crate::trace::read_corpus(f)?);
                }
                let mut texts = o.prompts.clone();
                texts.extend(o.prompt.clone());
                if texts.is_empty() && corpus.is_empty() {
                    texts.push("The name of the star that Earth orbits is".into());
                }
                // A prompt edit only means something in chat format: use it on both sides.
                let chat = o.chat || c.prompt_edited();
                let base = Compiled::default();
                let mut prompts = Vec::new();
                let mut edited_prompts = Vec::new();
                for t in &texts {
                    let pp = PromptParams { prompt: t.clone(), chat, ..Default::default() };
                    prompts.push(self.prompt_text(&m, &pp, &base)?);
                    edited_prompts.push(self.prompt_text(&m, &pp, &c)?);
                }
                let req = SimRequest {
                    ablation: c.intervention.clone(),
                    cut: c.touched.clone(),
                    prompts,
                    edited_prompts,
                    sampling: c.sampling.clone().unwrap_or_default(),
                    decisions: true,
                    corpus,
                    gen_tokens: o.tokens.unwrap_or(16),
                    top_k: o.top_k.unwrap_or(5),
                    max_tokens: o.max_tokens.unwrap_or(256),
                };
                let sim = crate::trace::require_backend(&m)?.simulate(&m, &req, p)?;
                serde_json::to_value(CompareResult { edits, compiled: c, simulation: sim })?
            }
            JobKind::Bake => {
                let o: BakeParams = serde_json::from_value(params.clone()).context("bake params (need out)")?;
                let edits = o.edits.clone().unwrap_or_default();
                if edits.iter().all(|e| !e.enabled) {
                    anyhow::bail!("nothing to bake: the edit stack is empty or every edit is off");
                }
                let out = match crate::slim::safe_file_name(&o.out) {
                    Ok(n) => {
                        let n = if n.to_lowercase().ends_with(".gguf") { n.to_string() } else { format!("{n}.gguf") };
                        self.config.output_dir.join("bake").join(n)
                    }
                    Err(_) => PathBuf::from(&o.out),
                };
                let g = self.graph(&m)?;
                let v = self.vocab(&m).ok();
                serde_json::to_value(bake(&m, &g, v.as_deref(), &edits, &out, p)?)?
            }
            JobKind::Diff => {
                let o: crate::diff::DiffOptions = serde_json::from_value(params.clone()).context("diff params")?;
                let b = other.expect("opened above");
                let g = self.graph(&m).ok();
                serde_json::to_value(crate::diff::diff(&m, &b, g.as_deref(), &o, p)?)?
            }
        };

        if matches!(kind, JobKind::Stats | JobKind::Inspect) {
            self.invalidate_graph(&m);
        }
        if cacheable {
            self.cache_put(&m, kind.as_str(), &params, &v);
        }
        Ok((v, false))
    }

    /// Queues a job on a background thread; returns its id immediately.
    pub fn submit(self: &Arc<Self>, kind: JobKind, model: &str, params: Value) -> Result<String> {
        let entry = self.resolve(model)?;
        let (id, cancel) = self.jobs.create(kind, &entry.name, params.clone());
        let app = self.clone();
        let jid = id.clone();
        let model_id = entry.id.clone();
        std::thread::Builder::new().name(format!("job-{id}")).spawn(move || {
            app.jobs.update(&jid, |j| {
                j.status = JobStatus::Running;
                j.started = Some(now_unix());
                j.step = "starting".into();
            });
            let progress = app.jobs.progress(&jid, cancel);
            let res = app.run(kind, &model_id, params, &progress);
            match res {
                Ok((v, cached)) => {
                    let path = app.jobs.store_result(&jid, &v).ok();
                    app.jobs.update(&jid, |j| {
                        j.status = JobStatus::Done;
                        j.progress = 100.0;
                        j.step = "done".into();
                        j.cached = cached;
                        j.result_path = path;
                        j.finished = Some(now_unix());
                    });
                }
                Err(e) => {
                    let cancelled = e.downcast_ref::<Cancelled>().is_some();
                    app.jobs.update(&jid, |j| {
                        j.status = if cancelled { JobStatus::Cancelled } else { JobStatus::Failed };
                        j.error = Some(format!("{e:#}"));
                        j.finished = Some(now_unix());
                    });
                }
            }
        })?;
        Ok(id)
    }

    // --- edits ---------------------------------------------------------------------------

}
