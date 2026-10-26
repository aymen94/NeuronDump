//! "What happens if I cut this?": static impact of removing or zeroing graph nodes, the
//! matching in-memory [`Ablation`] for simulation, sweeps, and conversion to a [`PrunePlan`].
//!
//! Every cut is either a **remove** (layers, tokens, the projector, the chat template,
//! optional tensors: the file gets smaller) or a **zero** (blocks, heads, neuron buckets,
//! experts, single tensors: weights are set to 0 in place, so every shape stays the same
//! and llama.cpp still loads the file, but it does not get smaller).

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use neurondump_gguf::{GgmlType, GgufFile};

use crate::analyze::{row_bytes, vocab_tensors, Candidate, CandidateKind, PrunePlan, Risk};
use crate::graph::{EdgeKind, ModelGraph, Node, NodeKind, Removable};
use crate::model::ModelHandle;
use crate::tokenizer::Vocab;
use crate::trace::{Ablation, SimReport, SimRequest, SweepReport, SweepRequest, SweepTarget};
use crate::util::{human_bytes, Progress, SCHEMA_VERSION};

mod cut;
mod zero;
mod impact;
mod run;
mod plan;
pub use cut::*;
pub use zero::*;
pub use impact::*;
pub use run::*;
pub use plan::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CutMode {
    /// Deleted from the file (smaller file).
    Remove,
    /// Weights set to zero in place (same shapes, same size).
    Zero,
    /// Cannot be cut (residual stream ops).
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutTarget {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub mode: CutMode,
    pub params: u64,
    pub bytes: u64,
    pub removable: Removable,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticImpact {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub cut: Vec<CutTarget>,
    /// Ids that matched nothing.
    pub unknown: Vec<String>,
    /// Ids dropped because an ancestor is also cut.
    pub redundant: Vec<String>,
    pub total_params: u64,
    pub params_removed: u64,
    pub params_zeroed: u64,
    pub percent_of_model: f64,
    pub original_size: u64,
    pub bytes_removed: u64,
    /// Size change from requantizing tensors whose columns are zeroed (Q8_0), usually > 0.
    pub requant_bytes_delta: i64,
    pub new_size: u64,
    /// Bytes saved outside the main GGUF (vision projector file).
    pub other_files_removed: u64,
    pub verdict: Removable,
    pub reasons: Vec<String>,
    /// Nodes left without a path from the input to the output.
    pub disconnected: Vec<String>,
    pub dependencies: Vec<String>,
    pub llama_cpp_loads: bool,
    pub legality: Vec<String>,
    /// (old, new) indices of layers that move.
    pub layer_renumbering: Vec<(u32, u32)>,
    /// What `slim` would do.
    pub file_ops: Vec<String>,
    /// Refused: verdict is breaks_model and `force` was not given.
    pub refused: bool,
    pub ablation: Ablation,
    pub simulation_supported: bool,
    pub simulation_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblateResult {
    pub impact: StaticImpact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simulation: Option<SimReport>,
}
