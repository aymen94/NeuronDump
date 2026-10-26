//! The facade both front ends talk to: library, opened-model cache, result cache, jobs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::ablate::{self, AblateResult, StaticImpact};
use crate::analyze::{analyze, AnalyzeOptions, PrunePlan};
use crate::graph::{GraphInputs, ModelGraph};
use crate::config::Config;
use crate::discovery::{DiscoveryOptions, Library, ModelEntry};
use crate::inspect::{inspect, InspectOptions};
use crate::jobs::{JobKind, JobManager, JobStatus};
use crate::model::ModelHandle;
use crate::slim::{slim, SlimOptions};
use crate::stats::{compute_stats, StatsOptions, StatsReport};
use crate::tokenizer::Vocab;
use crate::chat::{render_chat, ChatRequest};
use crate::edits::{bake, compile, Compiled, Edit, EditOp, EditStack};
use crate::tokens::EmbeddingMap;
use crate::trace::{trace, GenerateRequest, JourneyRequest, Sampling, SimRequest, SweepReport, SweepRequest, TraceReport, TraceRequest};
use crate::util::{fnv1a, now_unix, Cancelled, Progress};

mod params;
mod cache;
mod jobs;
mod edits;
pub use params::*;

pub struct App {
    pub config: Config,
    library: RwLock<Library>,
    handles: Mutex<HashMap<String, Arc<ModelHandle>>>,
    vocabs: Mutex<HashMap<String, Arc<Vocab>>>,
    graphs: Mutex<HashMap<String, Arc<ModelGraph>>>,
    edits: Mutex<HashMap<String, EditStack>>,
    maps: Mutex<HashMap<String, Arc<EmbeddingMap>>>,
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
        Arc::new(Self {
            config,
            library: RwLock::new(library),
            handles: Mutex::new(HashMap::new()),
            vocabs: Mutex::new(HashMap::new()),
            graphs: Mutex::new(HashMap::new()),
            edits: Mutex::new(HashMap::new()),
            maps: Mutex::new(HashMap::new()),
            jobs,
        })
    }

    pub fn library(&self) -> RwLockReadGuard<'_, Library> {
        self.library.read().unwrap()
    }

    pub fn rescan(&self) {
        self.library.write().unwrap().rescan();
        self.handles.lock().unwrap().clear();
        self.vocabs.lock().unwrap().clear();
        self.graphs.lock().unwrap().clear();
    }

    pub fn refresh_if_changed(&self) -> bool {
        let changed = self.library.write().unwrap().refresh_if_changed();
        if changed {
            self.handles.lock().unwrap().clear();
            self.vocabs.lock().unwrap().clear();
            self.graphs.lock().unwrap().clear();
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

}
