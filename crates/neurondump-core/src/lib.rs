//! NeuronDump core: the only place analysis logic lives. Front ends (CLI, web) call this
//! API and render its serializable results.

pub mod ablate;
pub mod analyze;
pub mod app;
pub mod arch;
pub mod chat;
pub mod compare;
pub mod config;
pub mod discovery;
pub mod edits;
pub mod export;
pub mod graph;
pub mod inspect;
pub mod jobs;
pub mod model;
pub mod search;
pub mod slim;
pub mod stats;
pub mod template;
pub mod tokenizer;
pub mod tokens;
pub mod trace;
pub mod util;

pub use analyze::{AnalyzeOptions, Candidate, CandidateKind, PrunePlan, Risk};
pub use app::App;
pub use config::{Config, UiMode};
pub use discovery::{DiscoveryOptions, Library, ModelEntry, ModelSummary, SourceKind};
pub use inspect::{InspectOptions, InspectReport};
pub use jobs::{JobInfo, JobKind, JobManager, JobStatus};
pub use model::ModelHandle;
pub use neurondump_gguf as gguf;
pub use slim::{SlimOptions, SlimReport};
pub use stats::{StatsOptions, StatsReport};
pub use trace::{TraceReport, TraceRequest};
pub use util::{NoProgress, Progress};
