//! Side-by-side comparison of several models.

use std::collections::{BTreeSet, HashSet};

use serde::Serialize;

use crate::arch::{analyze_architecture, ModelKind};
use crate::discovery::quantization_label;
use crate::model::ModelHandle;
use crate::util::SCHEMA_VERSION;

#[derive(Debug, Clone, Serialize)]
pub struct ModelBrief {
    pub name: String,
    pub architecture: String,
    pub kind: ModelKind,
    pub params: u64,
    pub quantization: Option<String>,
    pub file_size: u64,
    pub layers: usize,
    pub vocab_size: Option<u64>,
    pub tokenizer: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub expert_count: Option<u64>,
    pub layer_kinds: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KeyDiff {
    pub key: String,
    pub values: Vec<Option<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompareReport {
    pub schema_version: u32,
    pub models: Vec<ModelBrief>,
    /// Metadata keys whose values differ (tokenizer arrays excluded).
    pub metadata_diff: Vec<KeyDiff>,
    /// Pairwise Jaccard similarity of vocabularies (`[i][j]`).
    pub vocab_overlap: Vec<Vec<f64>>,
    pub shared_tensor_names: usize,
    /// Tensors present in all models but with different shapes or types.
    pub tensor_diffs: Vec<KeyDiff>,
}

pub fn compare(models: &[&ModelHandle]) -> CompareReport {
    let mut briefs = Vec::new();
    let mut vocabs: Vec<HashSet<&str>> = Vec::new();
    for m in models {
        let a = analyze_architecture(&m.gguf);
        let hp = &a.hyperparams;
        briefs.push(ModelBrief {
            name: m.entry.name.clone(),
            architecture: a.name.clone(),
            kind: a.kind,
            params: a.total_params,
            quantization: quantization_label(&m.gguf),
            file_size: m.gguf.file_size,
            layers: a.layers.len(),
            vocab_size: hp.vocab_size,
            tokenizer: m.gguf.get_str("tokenizer.ggml.model").map(String::from),
            context_length: hp.context_length,
            embedding_length: hp.embedding_length,
            head_count: hp.head_count,
            head_count_kv: hp.head_count_kv,
            feed_forward_length: hp.feed_forward_length,
            expert_count: hp.expert_count,
            layer_kinds: a.layer_kinds.iter().map(|(k, v)| format!("{k} x{}", v.len())).collect(),
        });
        vocabs.push(
            m.gguf
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .and_then(|a| a.as_strings())
                .map(|s| s.iter().map(String::as_str).collect())
                .unwrap_or_default(),
        );
    }

    let keys: BTreeSet<&String> = models.iter().flat_map(|m| m.gguf.metadata.keys()).collect();
    let mut metadata_diff = Vec::new();
    for k in keys {
        if k.starts_with("tokenizer.ggml.") && !k.ends_with("_id") && k != "tokenizer.ggml.model" && k != "tokenizer.ggml.pre" {
            continue;
        }
        let vals: Vec<Option<String>> = models
            .iter()
            .map(|m| m.gguf.get(k).map(|v| {
                let s = v.to_string();
                if s.len() > 120 { format!("{}… ({} chars)", &s[..s.char_indices().nth(100).map_or(s.len(), |c| c.0)], s.len()) } else { s }
            }))
            .collect();
        if vals.windows(2).any(|w| w[0] != w[1]) {
            metadata_diff.push(KeyDiff { key: k.clone(), values: vals });
        }
    }

    let n = models.len();
    let mut vocab_overlap = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let inter = vocabs[i].intersection(&vocabs[j]).count() as f64;
            let uni = vocabs[i].union(&vocabs[j]).count() as f64;
            vocab_overlap[i][j] = if uni > 0.0 { inter / uni } else { 0.0 };
        }
    }

    let mut shared: Option<BTreeSet<&str>> = None;
    for m in models {
        let names: BTreeSet<&str> = m.gguf.tensors.iter().map(|t| t.name.as_str()).collect();
        shared = Some(match shared {
            None => names,
            Some(s) => s.intersection(&names).copied().collect(),
        });
    }
    let shared = shared.unwrap_or_default();
    let mut tensor_diffs = Vec::new();
    for name in &shared {
        let vals: Vec<Option<String>> = models
            .iter()
            .map(|m| m.gguf.tensor(name).map(|t| format!("{:?} {}", t.dims, t.ggml_type)))
            .collect();
        if vals.windows(2).any(|w| w[0] != w[1]) {
            tensor_diffs.push(KeyDiff { key: name.to_string(), values: vals });
        }
    }

    CompareReport {
        schema_version: SCHEMA_VERSION,
        models: briefs,
        metadata_diff,
        vocab_overlap,
        shared_tensor_names: shared.len(),
        tensor_diffs,
    }
}
