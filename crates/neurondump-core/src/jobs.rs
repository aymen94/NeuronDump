//! Long-running operations as jobs: status, progress, cancellation, on-disk results.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::util::{now_unix, Progress};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobKind {
    Inspect,
    Stats,
    Analyze,
    Slim,
    Trace,
    Ablate,
    Sweep,
}

impl JobKind {
    pub fn as_str(self) -> &'static str {
        match self {
            JobKind::Inspect => "inspect",
            JobKind::Stats => "stats",
            JobKind::Analyze => "analyze",
            JobKind::Slim => "slim",
            JobKind::Trace => "trace",
            JobKind::Ablate => "ablate",
            JobKind::Sweep => "sweep",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "inspect" => JobKind::Inspect,
            "stats" => JobKind::Stats,
            "analyze" => JobKind::Analyze,
            "slim" => JobKind::Slim,
            "trace" => JobKind::Trace,
            "ablate" => JobKind::Ablate,
            "sweep" => JobKind::Sweep,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_finished(self) -> bool {
        matches!(self, JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobInfo {
    pub id: String,
    pub kind: JobKind,
    pub model: String,
    pub params: serde_json::Value,
    pub status: JobStatus,
    pub progress: f32,
    pub step: String,
    pub created: u64,
    pub started: Option<u64>,
    pub finished: Option<u64>,
    pub error: Option<String>,
    pub result_path: Option<PathBuf>,
    pub cached: bool,
    /// Incremented on every change (used by SSE to detect updates).
    pub version: u64,
}

struct Slot {
    info: JobInfo,
    cancel: Arc<AtomicBool>,
}

pub struct JobManager {
    jobs: Mutex<IndexMap<String, Slot>>,
    dir: PathBuf,
    seq: AtomicU64,
}

impl JobManager {
    pub fn new(dir: PathBuf) -> Self {
        Self { jobs: Mutex::new(IndexMap::new()), dir, seq: AtomicU64::new(1) }
    }

    pub fn results_dir(&self) -> &PathBuf {
        &self.dir
    }

    pub fn create(&self, kind: JobKind, model: &str, params: serde_json::Value) -> (String, Arc<AtomicBool>) {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        let id = format!("{}-{:x}{:04x}", kind.as_str(), now_unix() & 0xffffff, n);
        let cancel = Arc::new(AtomicBool::new(false));
        let info = JobInfo {
            id: id.clone(),
            kind,
            model: model.into(),
            params,
            status: JobStatus::Queued,
            progress: 0.0,
            step: "queued".into(),
            created: now_unix(),
            started: None,
            finished: None,
            error: None,
            result_path: None,
            cached: false,
            version: 0,
        };
        self.jobs.lock().unwrap().insert(id.clone(), Slot { info, cancel: cancel.clone() });
        (id, cancel)
    }

    pub fn update(&self, id: &str, f: impl FnOnce(&mut JobInfo)) {
        if let Some(s) = self.jobs.lock().unwrap().get_mut(id) {
            f(&mut s.info);
            s.info.version += 1;
        }
    }

    pub fn get(&self, id: &str) -> Option<JobInfo> {
        self.jobs.lock().unwrap().get(id).map(|s| s.info.clone())
    }

    pub fn list(&self) -> Vec<JobInfo> {
        self.jobs.lock().unwrap().values().map(|s| s.info.clone()).collect()
    }

    /// Requests cancellation; returns false for unknown or finished jobs.
    pub fn cancel(&self, id: &str) -> bool {
        let jobs = self.jobs.lock().unwrap();
        match jobs.get(id) {
            Some(s) if !s.info.status.is_finished() => {
                s.cancel.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    pub fn store_result(&self, id: &str, value: &serde_json::Value) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(&self.dir)?;
        let p = self.dir.join(format!("{id}.json"));
        std::fs::write(&p, serde_json::to_vec_pretty(value).unwrap_or_default())?;
        Ok(p)
    }

    pub fn result(&self, id: &str) -> Option<serde_json::Value> {
        let info = self.get(id)?;
        let p = info.result_path?;
        let bytes = std::fs::read(p).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    pub fn progress<'a>(&'a self, id: &str, cancel: Arc<AtomicBool>) -> JobProgress<'a> {
        JobProgress { mgr: self, id: id.to_string(), cancel, last: Mutex::new((-1.0, String::new())) }
    }
}

pub struct JobProgress<'a> {
    mgr: &'a JobManager,
    id: String,
    cancel: Arc<AtomicBool>,
    last: Mutex<(f32, String)>,
}

impl Progress for JobProgress<'_> {
    fn set(&self, percent: f32, step: &str) {
        let mut last = self.last.lock().unwrap();
        // Throttle: only publish on ≥0.5% change or a new step name.
        if (percent - last.0).abs() < 0.5 && last.1 == step {
            return;
        }
        *last = (percent, step.to_string());
        drop(last);
        let step = step.to_string();
        self.mgr.update(&self.id, |j| {
            j.progress = percent.clamp(0.0, 100.0);
            j.step = step;
        });
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}
