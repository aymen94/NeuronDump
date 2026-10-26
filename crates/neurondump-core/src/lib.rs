//! NeuronDump core: the only place analysis logic lives. Front ends (CLI, web) call this
//! API and render its serializable results.
//!
//! The `native` feature (default) adds everything that needs a filesystem, threads or the
//! trace runtime: the app / job system, config, analysis, slimming, edits, ablation and diff.
//! Without it the static inspection part (GGUF metadata, tensors, architecture, vocabulary,
//! chat template, graph, weight statistics) builds for `wasm32-unknown-unknown`.

#[cfg(feature = "native")]
pub mod ablate;
#[cfg(feature = "native")]
pub mod analyze;
pub mod api;
#[cfg(feature = "native")]
pub mod app;
pub mod arch;
pub mod chat;
pub mod compare;
#[cfg(feature = "native")]
pub mod config;
#[cfg(feature = "native")]
pub mod diff;
pub mod discovery;
#[cfg(feature = "native")]
pub mod edits;
pub mod export;
pub mod graph;
pub mod inspect;
#[cfg(feature = "native")]
pub mod jobs;
pub mod model;
pub mod search;
#[cfg(feature = "native")]
pub mod slim;
pub mod stats;
pub mod template;
pub mod tokenizer;
pub mod tokens;
pub mod trace;
pub mod util;

#[cfg(feature = "native")]
pub use analyze::{AnalyzeOptions, Candidate, CandidateKind, PrunePlan, Risk};
#[cfg(feature = "native")]
pub use app::App;
#[cfg(feature = "native")]
pub use config::{Config, UiMode};
pub use discovery::{DiscoveryOptions, Library, ModelEntry, ModelSummary, SourceKind};
pub use inspect::{InspectOptions, InspectReport};
#[cfg(feature = "native")]
pub use jobs::{JobInfo, JobKind, JobManager, JobStatus};
pub use model::ModelHandle;
pub use neurondump_gguf as gguf;
#[cfg(feature = "native")]
pub use slim::{SlimOptions, SlimReport};
pub use stats::{StatsOptions, StatsReport};
pub use trace::{TraceReport, TraceRequest};
pub use util::{NoProgress, Progress};
