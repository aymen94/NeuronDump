//! `slim`: apply (a subset of) a [`PrunePlan`] and write a new, validated GGUF.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use neurondump_gguf::{dequantize_into, quantize_q8_0, Array, GgmlType, GgufFile, GgufWriter, Value, WriterTensor};

use crate::ablate::{needs_requant, ZeroOp};

use crate::analyze::{protected_ids, row_bytes, vocab_tensors, Candidate, CandidateKind, PrunePlan};
use crate::arch::parse_tensor_name;
use crate::model::ModelHandle;
use crate::tokenizer::Vocab;
use crate::util::{check_cancel, Progress, SCHEMA_VERSION};

mod write;
mod tensors;
mod check;
pub use write::*;
pub(crate) use tensors::*;
pub use check::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlimOptions {
    pub plan: PrunePlan,
    /// Candidate ids or groups (`vocab`, `layers`, `tied`, `quant`, `projector`, `metadata`)
    /// to apply. Empty = the plan's `selected` candidates.
    #[serde(default)]
    pub only: Vec<String>,
    /// Output file name (written inside `<output_dir>/slim/`).
    pub out: String,
    /// Corpus for an optional `llama-perplexity` before/after comparison.
    #[serde(default)]
    pub perplexity_corpus: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validation {
    pub ok: bool,
    pub n_tensors: usize,
    pub n_vocab: Option<u64>,
    pub block_count: Option<u64>,
    pub problems: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perplexity {
    pub original: Option<f64>,
    pub slim: Option<f64>,
    pub log: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlimReport {
    pub schema_version: u32,
    pub model: String,
    pub output: PathBuf,
    pub modelfile: PathBuf,
    pub original_size: u64,
    pub new_size: u64,
    pub applied: Vec<String>,
    pub refused: Vec<(String, String)>,
    pub removed_tokens: usize,
    pub removed_layers: Vec<u32>,
    pub requantized: Vec<String>,
    /// Tensors with weights set to zero in place.
    #[serde(default)]
    pub zeroed: Vec<String>,
    pub validation: Validation,
    pub perplexity: Option<Perplexity>,
}

/// Rejects anything but a plain file name.
pub fn safe_file_name(name: &str) -> Result<&str> {
    let n = name.trim();
    if n.is_empty()
        || n.contains(['/', '\\', ':', '\0'])
        || n == "."
        || n == ".."
        || n.starts_with('.')
        || Path::new(n).components().count() != 1
    {
        bail!("output must be a plain file name (no directories or '..'): {name:?}");
    }
    Ok(n)
}

fn chosen<'a>(plan: &'a PrunePlan, only: &[String]) -> Vec<&'a Candidate> {
    plan.candidates
        .iter()
        .filter(|c| {
            if only.is_empty() {
                c.selected
            } else {
                only.iter().any(|o| o == &c.id || o == c.kind.group() || o == c.kind.as_str())
            }
        })
        .collect()
}

fn keep_only_was_corpus(applied: &[String]) -> bool {
    applied.iter().any(|a| a == "vocab_trim")
}
