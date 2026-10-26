//! `analyze`: find removable or compressible parts and produce a [`PrunePlan`].

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use neurondump_gguf::{GgmlType, GgufFile, TensorInfo};

use crate::arch::{analyze_architecture, parse_tensor_name, ModelKind};
use crate::model::ModelHandle;
use crate::stats::{is_float_type, tensor_cosine, StatsReport};
use crate::tokenizer::{Encoder, TokenClass, TokenType, Vocab};
use crate::trace::TraceReport;
use crate::util::{check_cancel, human_bytes, Progress, SCHEMA_VERSION};

mod find;
pub use find::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Risk {
    Low,
    Medium,
    High,
    Experimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateKind {
    VocabTrim,
    ReservedTokens,
    TiedEmbeddings,
    LowNormHeads,
    LowNormNeurons,
    LowNormExperts,
    LayerRedundancy,
    QuantHeadroom,
    UnusedProjector,
    LargeMetadata,
    /// Weights set to zero in place (ablation): same shapes, same size.
    ZeroWeights,
    /// Optional tensors dropped from the file (MTP head, in-file vision encoder).
    DropTensors,
    StripTemplate,
}

impl CandidateKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CandidateKind::VocabTrim => "vocab_trim",
            CandidateKind::ReservedTokens => "reserved_tokens",
            CandidateKind::TiedEmbeddings => "tied_embeddings",
            CandidateKind::LowNormHeads => "low_norm_heads",
            CandidateKind::LowNormNeurons => "low_norm_neurons",
            CandidateKind::LowNormExperts => "low_norm_experts",
            CandidateKind::LayerRedundancy => "layer_redundancy",
            CandidateKind::QuantHeadroom => "quant_headroom",
            CandidateKind::UnusedProjector => "unused_projector",
            CandidateKind::LargeMetadata => "large_metadata",
            CandidateKind::ZeroWeights => "zero_weights",
            CandidateKind::DropTensors => "drop_tensors",
            CandidateKind::StripTemplate => "strip_template",
        }
    }

    /// Group names accepted by `slim --only` (`vocab`, `layers`, ...).
    pub fn group(self) -> &'static str {
        match self {
            CandidateKind::VocabTrim | CandidateKind::ReservedTokens => "vocab",
            CandidateKind::LayerRedundancy => "layers",
            CandidateKind::TiedEmbeddings => "tied",
            CandidateKind::QuantHeadroom => "quant",
            CandidateKind::UnusedProjector => "projector",
            CandidateKind::LargeMetadata => "metadata",
            CandidateKind::LowNormHeads => "heads",
            CandidateKind::LowNormNeurons => "neurons",
            CandidateKind::LowNormExperts => "experts",
            CandidateKind::ZeroWeights => "zero",
            CandidateKind::DropTensors => "drop",
            CandidateKind::StripTemplate => "template",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub kind: CandidateKind,
    pub title: String,
    pub detail: String,
    pub risk: Risk,
    pub bytes_saved: u64,
    /// Whether `slim` can apply it.
    pub applicable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Selected by default when applying the plan.
    pub selected: bool,
    /// Kind-specific payload (token ids, layer index, tensor names, ...).
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrunePlan {
    pub schema_version: u32,
    pub model: String,
    pub digest: String,
    pub architecture: String,
    pub original_size: u64,
    pub candidates: Vec<Candidate>,
    pub notes: Vec<String>,
}

impl PrunePlan {
    pub fn total_selected_bytes(&self) -> u64 {
        self.candidates.iter().filter(|c| c.selected && c.applicable).map(|c| c.bytes_saved).sum()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyzeOptions {
    /// Corpus files for vocabulary trimming.
    #[serde(default)]
    pub corpus_files: Vec<PathBuf>,
    /// Inline corpus text (web).
    #[serde(default)]
    pub corpus_text: Option<String>,
    /// Allow proposing first/last layer removal.
    #[serde(default)]
    pub include_edge_layers: bool,
    /// Max number of layer-removal candidates.
    #[serde(default)]
    pub max_layer_candidates: Option<usize>,
    /// Skip the (slow) weight-similarity pass.
    #[serde(default)]
    pub skip_layer_similarity: bool,
}

pub fn row_bytes(t: &TensorInfo) -> u64 {
    t.n_bytes / t.n_rows().max(1)
}

/// Tensors indexed by vocabulary id along their row dimension.
pub fn vocab_tensors(g: &GgufFile, n_vocab: u64) -> Vec<&TensorInfo> {
    g.tensors
        .iter()
        .filter(|t| matches!(t.name.as_str(), "token_embd.weight" | "output.weight") && t.n_rows() == n_vocab && t.dims.len() == 2)
        .collect()
}

/// Ids that must survive any vocabulary trim.
pub fn protected_ids(g: &GgufFile, vocab: &Vocab) -> BTreeSet<u32> {
    let mut keep = BTreeSet::new();
    for k in crate::tokenizer::special_token_keys(g) {
        if let Some(id) = g.get_u64(&k) {
            keep.insert(id as u32);
        }
    }
    for t in &vocab.tokens {
        let single_char = t.raw.chars().count() == 1;
        if matches!(t.token_type, TokenType::Control | TokenType::Unknown | TokenType::Byte)
            || t.class == TokenClass::Byte
            || single_char
        {
            keep.insert(t.id);
        }
    }
    keep
}

fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}
