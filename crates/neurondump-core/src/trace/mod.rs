//! Runtime tracing types and the pluggable backend. The candle-based implementation
//! lives in `neurondump-trace` and is registered by the binary when built with the
//! `trace` feature, so core stays free of inference dependencies.

use std::sync::OnceLock;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::model::ModelHandle;
use crate::util::Progress;

mod report;
mod intervention;
mod journey;
mod simulate;
pub use report::*;
pub use intervention::*;
pub use journey::*;
pub use simulate::*;

pub trait TraceBackend: Send + Sync {
    fn name(&self) -> &str;
    fn supported_architectures(&self) -> Vec<String>;
    fn trace(&self, model: &ModelHandle, req: &TraceRequest, progress: &dyn Progress) -> Result<TraceReport>;
    fn simulate(&self, _model: &ModelHandle, _req: &SimRequest, _progress: &dyn Progress) -> Result<SimReport> {
        bail!("this trace backend cannot simulate ablations")
    }
    fn sweep(&self, _model: &ModelHandle, _req: &SweepRequest, _progress: &dyn Progress) -> Result<SweepReport> {
        bail!("this trace backend cannot run sweeps")
    }
    fn journey(&self, _model: &ModelHandle, _req: &JourneyRequest, _progress: &dyn Progress) -> Result<JourneyReport> {
        bail!("this trace backend cannot follow tokens")
    }
    fn generate(&self, _model: &ModelHandle, _req: &GenerateRequest, _progress: &dyn Progress) -> Result<GenerateReport> {
        bail!("this trace backend cannot generate")
    }
}

/// Checks that runtime simulation is available for this model.
pub fn require_backend(m: &ModelHandle) -> Result<&'static dyn TraceBackend> {
    let Some(b) = backend() else {
        bail!("simulation needs the trace backend; rebuild with `cargo build --features trace` (static impact still works with --static)");
    };
    let arch = m.architecture().to_string();
    if !b.supported_architectures().iter().any(|a| *a == arch) {
        bail!(
            "simulation unsupported for architecture {arch:?} (supported: {}); static impact still works with --static",
            b.supported_architectures().join(", ")
        );
    }
    Ok(b)
}

/// Splits a corpus file into texts (blank-line separated), like `trace --corpus`.
pub fn read_corpus(path: &std::path::Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading corpus {}: {e}", path.display()))?;
    Ok(text.split("\n\n").map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect())
}

static BACKEND: OnceLock<Box<dyn TraceBackend>> = OnceLock::new();

pub fn register_backend(b: Box<dyn TraceBackend>) {
    let _ = BACKEND.set(b);
}

pub fn backend() -> Option<&'static dyn TraceBackend> {
    BACKEND.get().map(|b| b.as_ref())
}

pub fn supported_architectures() -> Vec<String> {
    backend().map(|b| b.supported_architectures()).unwrap_or_default()
}

pub fn trace(m: &ModelHandle, req: &TraceRequest, p: &dyn Progress) -> Result<TraceReport> {
    let mut req = req.clone();
    if let Some(f) = &req.corpus_file {
        let text = std::fs::read_to_string(f)?;
        req.corpus.extend(text.split("\n\n").map(str::trim).filter(|s| !s.is_empty()).map(String::from));
    }
    if req.prompt.is_none() && req.corpus.is_empty() {
        bail!("trace needs a prompt or a corpus");
    }
    let Some(b) = backend() else {
        bail!("trace support is not compiled in; rebuild with `cargo build --features trace`");
    };
    let arch = m.architecture().to_string();
    if !b.supported_architectures().iter().any(|a| *a == arch) {
        bail!(
            "trace unsupported for architecture {arch:?} (supported: {}); static analysis is still available",
            b.supported_architectures().join(", ")
        );
    }
    b.trace(m, &req, p)
}

/// Follows one token through the model (feature `trace`, supported architectures only).
pub fn journey(m: &ModelHandle, req: &JourneyRequest, p: &dyn Progress) -> Result<JourneyReport> {
    require_backend(m)?.journey(m, req, p)
}

/// Generates tokens with a per-step timeline (feature `trace`, supported architectures only).
pub fn generate(m: &ModelHandle, req: &GenerateRequest, p: &dyn Progress) -> Result<GenerateReport> {
    require_backend(m)?.generate(m, req, p)
}
