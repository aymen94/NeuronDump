//! Weight and metadata diff of two models. Tensors are matched by name and dequantized in
//! bounded row chunks from both files at once, so any pair of types (Q4_K vs Q8_0 vs F16)
//! compares without materializing either model. Results aggregate onto model A's graph
//! (sections, layers, blocks, experts) so the web UI can color it like a sweep.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use neurondump_gguf::{can_dequantize, GgufFile, TensorInfo};

use crate::arch::parse_tensor_name;
use crate::discovery::quantization_label;
use crate::graph::{ModelGraph, NodeKind};
use crate::model::ModelHandle;
use crate::util::{check_cancel, Progress, SCHEMA_VERSION, Instant};

mod meta;
mod weights;
use meta::*;
pub use weights::*;

/// Original (pre-slim) numbers of the layers a slim removed; written by `slim`.
pub const REMOVED_LAYERS_KEY: &str = "neurondump.removed_layers";
const CHUNK_ELEMS: u64 = 1 << 20;
const MAX_EXAMPLES: usize = 20;

pub fn removed_layers(g: &GgufFile) -> Vec<u32> {
    match g.get(REMOVED_LAYERS_KEY).and_then(|v| v.as_array()) {
        Some(neurondump_gguf::Array::U32(v)) => v.clone(),
        _ => Vec::new(),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffOptions {
    /// Keep only the N most-changed tensors in the report (all when unset).
    #[serde(default)]
    pub top: Option<usize>,
    /// Compare metadata, vocabulary and tensor lists only; skip reading weights.
    #[serde(default)]
    pub metadata_only: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Metrics {
    /// ||a - b|| / ||a||.
    pub rel_l2: f64,
    pub cosine: f64,
    pub max_abs: f32,
    /// Values compared.
    pub n: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDiff {
    /// Name in model A.
    pub name: String,
    /// Graph node id in model A (`t.<name>`).
    pub id: String,
    pub layer: Option<u32>,
    pub dims: Vec<u64>,
    pub type_a: String,
    pub type_b: String,
    #[serde(flatten)]
    pub metrics: Metrics,
}

/// Aggregate over a node of model A's graph (section, layer, block or MoE expert).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupDiff {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub layer: Option<u32>,
    pub block: Option<String>,
    pub tensors: usize,
    #[serde(flatten)]
    pub metrics: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeMismatch {
    pub name: String,
    pub dims_a: Vec<u64>,
    pub dims_b: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueDiff {
    pub key: String,
    pub a: Option<String>,
    pub b: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VocabDiff {
    pub size_a: usize,
    pub size_b: usize,
    /// Ids present in both whose token text differs.
    pub changed: usize,
    /// First changed / added / removed ids: (id, text in A, text in B).
    pub examples: Vec<(usize, Option<String>, Option<String>)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemplateDiff {
    pub changed: bool,
    pub chars_a: usize,
    pub chars_b: usize,
    /// First differing line (1-based) with both versions.
    pub first_difference: Option<(usize, String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSide {
    pub name: String,
    pub path: String,
    pub architecture: String,
    pub quantization: Option<String>,
    pub tensors: usize,
    pub file_size: u64,
    /// Layers removed by a NeuronDump slim (original numbering), used to realign names.
    pub removed_layers: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffReport {
    pub schema_version: u32,
    pub a: DiffSide,
    pub b: DiffSide,
    pub metadata: Vec<ValueDiff>,
    pub vocab: VocabDiff,
    pub chat_template: TemplateDiff,
    pub only_in_a: Vec<String>,
    pub only_in_b: Vec<String>,
    pub shape_mismatch: Vec<ShapeMismatch>,
    /// Matched tensors that could not be compared, with the reason.
    pub skipped: Vec<(String, String)>,
    pub compared: usize,
    pub identical: usize,
    pub total: Option<Metrics>,
    /// Graph-node aggregates of model A (sections, layers, blocks, experts), graph order.
    pub groups: Vec<GroupDiff>,
    /// Compared tensors, most changed (highest relative L2) first.
    pub tensors: Vec<TensorDiff>,
    pub elapsed_ms: u64,
}

/// `blk.N.` names in the original numbering, undoing a slim's layer renumbering.
fn canonical_name(name: &str, removed: &[u32]) -> String {
    if removed.is_empty() {
        return name.to_string();
    }
    let Some((n, rest)) = name.strip_prefix("blk.").and_then(|r| r.split_once('.')) else { return name.to_string() };
    let Ok(n) = n.parse::<usize>() else { return name.to_string() };
    let orig = (0u32..).filter(|l| !removed.contains(l)).nth(n).expect("unbounded");
    format!("blk.{orig}.{rest}")
}

#[cfg(test)]
mod tests {
    use super::canonical_name;

    #[test]
    fn realigns_slimmed_layers() {
        // Layers 1 and 3 removed: slimmed blk.1 was blk.2, blk.2 was blk.4.
        assert_eq!(canonical_name("blk.0.ffn_up.weight", &[1, 3]), "blk.0.ffn_up.weight");
        assert_eq!(canonical_name("blk.1.ffn_up.weight", &[1, 3]), "blk.2.ffn_up.weight");
        assert_eq!(canonical_name("blk.2.attn_q.weight", &[1, 3]), "blk.4.attn_q.weight");
        assert_eq!(canonical_name("token_embd.weight", &[1, 3]), "token_embd.weight");
    }
}
