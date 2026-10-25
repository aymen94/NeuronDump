//! The facade both front ends talk to: library, opened-model cache, result cache, jobs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::analyze::{analyze, AnalyzeOptions, PrunePlan};
use crate::config::Config;
use crate::discovery::{DiscoveryOptions, Library, ModelEntry};
use crate::inspect::{inspect, InspectOptions};
use crate::jobs::{JobKind, JobManager, JobStatus};
use crate::model::ModelHandle;
use crate::slim::{slim, SlimOptions};
use crate::stats::{compute_stats, StatsOptions, StatsReport};
use crate::tokenizer::Vocab;
use crate::trace::{trace, TraceReport, TraceRequest};
use crate::util::{fnv1a, now_unix, Cancelled, Progress};

pub struct App {
    pub config: Config,
    library: RwLock<Library>,
    handles: Mutex<HashMap<String, Arc<ModelHandle>>>,
    vocabs: Mutex<HashMap<String, Arc<Vocab>>>,
    pub jobs: JobManager,
}

/// On-disk result cache keyed by model digest + job kind + parameters.
fn cache_path(output_dir: &std::path::Path, digest: &str, kind: &str, params: &Value) -> PathBuf {
    let key = fnv1a(params.to_string().as_bytes());
    output_dir.join("cache").join(digest).join(format!("{kind}-{key:016x}.json"))
}

impl App {
    pub fn new(config: Config, discovery: DiscoveryOptions) -> Arc<Self> {
        let mut discovery = discovery;
        discovery.extra_dirs.extend(config.scan_dirs.iter().cloned());
        let library = Library::discover(discovery);
        let jobs = JobManager::new(config.output_dir.join("jobs"));
        Arc::new(Self { config, library: RwLock::new(library), handles: Mutex::new(HashMap::new()), vocabs: Mutex::new(HashMap::new()), jobs })
    }

    pub fn library(&self) -> RwLockReadGuard<'_, Library> {
        self.library.read().unwrap()
    }

    pub fn rescan(&self) {
        self.library.write().unwrap().rescan();
        self.handles.lock().unwrap().clear();
        self.vocabs.lock().unwrap().clear();
    }

    pub fn refresh_if_changed(&self) -> bool {
        let changed = self.library.write().unwrap().refresh_if_changed();
        if changed {
            self.handles.lock().unwrap().clear();
            self.vocabs.lock().unwrap().clear();
        }
        changed
    }

    pub fn resolve(&self, query: &str) -> Result<ModelEntry> {
        self.library.write().unwrap().resolve(query)
    }

    /// Opens (or reuses) a model by name, id or path.
    pub fn open(&self, query: &str) -> Result<Arc<ModelHandle>> {
        let entry = self.resolve(query)?;
        if let Some(h) = self.handles.lock().unwrap().get(&entry.id) {
            return Ok(h.clone());
        }
        let h = Arc::new(ModelHandle::open(entry.clone())?);
        self.handles.lock().unwrap().insert(entry.id.clone(), h.clone());
        Ok(h)
    }

    /// Opens a model by opaque id only (web API: never by path).
    pub fn open_id(&self, id: &str) -> Result<Arc<ModelHandle>> {
        let exists = self.library().by_id(id).is_some();
        if !exists {
            anyhow::bail!("unknown model id {id:?}");
        }
        self.open(id)
    }

    pub fn vocab(&self, m: &ModelHandle) -> Result<Arc<Vocab>> {
        if let Some(v) = self.vocabs.lock().unwrap().get(&m.entry.id) {
            return Ok(v.clone());
        }
        let v = Arc::new(Vocab::from_gguf(&m.gguf)?);
        self.vocabs.lock().unwrap().insert(m.entry.id.clone(), v.clone());
        Ok(v)
    }

    fn cache_get(&self, m: &ModelHandle, kind: &str, params: &Value) -> Option<Value> {
        let p = cache_path(&self.config.output_dir, &m.entry.digest, kind, params);
        let bytes = std::fs::read(p).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    fn cache_put(&self, m: &ModelHandle, kind: &str, params: &Value, v: &Value) {
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

    /// Runs a job synchronously (CLI) — the job system calls the same function.
    /// Returns the result and whether it came from the cache.
    pub fn run(&self, kind: JobKind, model: &str, params: Value, p: &dyn Progress) -> Result<(Value, bool)> {
        let m = self.open(model)?;
        let params = if params.is_null() { Value::Object(Default::default()) } else { params };
        let cacheable = matches!(kind, JobKind::Inspect | JobKind::Stats | JobKind::Trace);
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
                }
                serde_json::to_value(r)?
            }
        };
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

    /// Parses a stored plan (CLI `slim --plan`).
    pub fn load_plan(path: &std::path::Path) -> Result<PrunePlan> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading plan {}", path.display()))?;
        serde_json::from_str(&text).context("invalid prune plan JSON")
    }
}
