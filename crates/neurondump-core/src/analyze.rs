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

pub fn analyze(
    m: &ModelHandle,
    opts: &AnalyzeOptions,
    stats: Option<&StatsReport>,
    trace: Option<&TraceReport>,
    p: &dyn Progress,
) -> Result<PrunePlan> {
    let g = &m.gguf;
    let arch = analyze_architecture(g);
    let mut c: Vec<Candidate> = Vec::new();
    let mut notes = Vec::new();
    let vocab = Vocab::from_gguf(g).ok();
    let n_vocab = vocab.as_ref().map_or(0, |v| v.tokens.len() as u64);
    let vt = vocab_tensors(g, n_vocab);
    let per_token_bytes: u64 = vt.iter().map(|t| row_bytes(t)).sum();
    if n_vocab > 0 && vt.is_empty() {
        notes.push("no token_embd/output tensor with vocabulary rows; vocabulary trimming saves metadata only".into());
    }

    // --- Vocabulary from corpus ------------------------------------------------------
    p.set(5.0, "vocabulary");
    if let Some(vocab) = &vocab {
        let mut corpus = String::new();
        for f in &opts.corpus_files {
            corpus.push_str(&std::fs::read_to_string(f).with_context(|| format!("reading corpus {}", f.display()))?);
            corpus.push('\n');
        }
        if let Some(t) = &opts.corpus_text {
            corpus.push_str(t);
        }
        if !corpus.is_empty() {
            match Encoder::new(vocab) {
                Ok(enc) => {
                    notes.extend(enc.warnings.iter().cloned());
                    let mut used: HashSet<u32> = HashSet::new();
                    let mut n_tokens = 0usize;
                    for (i, line) in corpus.lines().enumerate() {
                        if i % 256 == 0 {
                            check_cancel(p)?;
                        }
                        n_tokens += enc.encode(line, false, Some(&mut used)).ids.len();
                        // Newlines are encoded across line boundaries too.
                        enc.encode("\n", false, Some(&mut used));
                    }
                    let n_used = used.len();
                    let mut keep: BTreeSet<u32> = used.into_iter().collect();
                    keep.extend(protected_ids(g, vocab));
                    let removed = n_vocab - keep.len() as u64;
                    let meta_bytes: u64 = vocab
                        .tokens
                        .iter()
                        .filter(|t| !keep.contains(&t.id))
                        .map(|t| t.raw.len() as u64 + 8 + 4 + 4)
                        .sum();
                    c.push(Candidate {
                        id: "vocab_trim".into(),
                        kind: CandidateKind::VocabTrim,
                        title: format!("Trim vocabulary to the {} tokens used by the corpus", keep.len()),
                        detail: format!(
                            "Corpus of {} chars encodes to {n_tokens} tokens touching {n_used} distinct ids (incl. BPE intermediates); with protected special/byte/single-char tokens {} are kept. Removes {removed} of {n_vocab} tokens; the model can no longer produce text outside this domain.",
                            corpus.len(),
                            keep.len(),
                        ),
                        risk: Risk::High,
                        bytes_saved: removed * per_token_bytes + meta_bytes,
                        applicable: vocab.model != crate::tokenizer::TokenizerModel::Other,
                        reason: None,
                        selected: false,
                        data: serde_json::json!({ "keep_ids": keep.iter().collect::<Vec<_>>() }),
                    });
                }
                Err(e) => notes.push(format!("corpus tokenization unavailable: {e}")),
            }
        }

        // --- Reserved / unused tokens -------------------------------------------------
        let protected: BTreeSet<u32> = crate::tokenizer::special_token_keys(g).iter().filter_map(|k| g.get_u64(k)).map(|v| v as u32).collect();
        let reserved: Vec<u32> =
            vocab.tokens.iter().filter(|t| t.class == TokenClass::Reserved && !protected.contains(&t.id)).map(|t| t.id).collect();
        if !reserved.is_empty() {
            c.push(Candidate {
                id: "reserved_tokens".into(),
                kind: CandidateKind::ReservedTokens,
                title: format!("Remove {} reserved/unused tokens", reserved.len()),
                detail: "Placeholder tokens (`<|reserved_special_token_N|>`, `[unusedN]`, token type UNUSED) never produced by the tokenizer. Fine-tunes that assigned meaning to them would break.".into(),
                risk: Risk::Medium,
                bytes_saved: reserved.len() as u64 * per_token_bytes,
                applicable: true,
                reason: None,
                selected: false,
                data: serde_json::json!({ "remove_ids": reserved }),
            });
        }
    }

    // --- Tied embeddings --------------------------------------------------------------
    p.set(15.0, "embeddings");
    match (g.tensor("token_embd.weight"), g.tensor("output.weight")) {
        (Some(_), None) => notes.push("embeddings are already tied (no separate output.weight)".into()),
        (Some(e), Some(o)) => {
            let identical = e.dims == o.dims && e.ggml_type == o.ggml_type && g.info_bytes(e).ok() == g.info_bytes(o).ok();
            if identical {
                c.push(Candidate {
                    id: "tied_embeddings".into(),
                    kind: CandidateKind::TiedEmbeddings,
                    title: "Drop duplicated output.weight (identical to token_embd.weight)".into(),
                    detail: "The output projection is a byte-for-byte copy of the input embeddings; llama.cpp falls back to token_embd when output.weight is absent.".into(),
                    risk: Risk::Low,
                    bytes_saved: o.n_bytes,
                    applicable: true,
                    reason: None,
                    selected: true,
                    data: serde_json::json!({ "tensor": "output.weight" }),
                });
            } else {
                notes.push(format!(
                    "output.weight ({}, {}) differs from token_embd.weight ({}, {}): embeddings are untied",
                    o.ggml_type,
                    human_bytes(o.n_bytes),
                    e.ggml_type,
                    human_bytes(e.n_bytes)
                ));
            }
        }
        _ => {}
    }

    // --- Static weight heuristics ------------------------------------------------------
    p.set(25.0, "weight heuristics");
    let experimental_reason =
        "structured pruning changes per-layer shapes; no architecture is confirmed to load non-uniform shapes in llama.cpp, so slim refuses it".to_string();
    if let Some(s) = stats {
        let mut heads = Vec::new();
        for h in &s.heads {
            let score: Vec<f32> = if !h.o.is_empty() && h.o.len() == h.q.len() {
                h.q.iter().zip(&h.o).map(|(a, b)| a * b).collect()
            } else if !h.q.is_empty() {
                h.q.clone()
            } else {
                h.o.clone()
            };
            let med = median(&score);
            for (i, v) in score.iter().enumerate() {
                if med > 0.0 && *v < 0.5 * med {
                    heads.push(serde_json::json!({ "layer": h.layer, "head": i, "score": v, "layer_median": med }));
                }
            }
        }
        if !heads.is_empty() {
            let hd = s.heads.first().map_or(0, |h| h.head_dim);
            let emb = arch.hyperparams.embedding_length.unwrap_or(0);
            c.push(Candidate {
                id: "low_norm_heads".into(),
                kind: CandidateKind::LowNormHeads,
                title: format!("{} attention heads with < 50% of their layer's median |Q|·|O| norm", heads.len()),
                detail: "Static estimate: small projection norms suggest small contributions. Validate with `trace` before trusting it.".into(),
                risk: Risk::Experimental,
                bytes_saved: heads.len() as u64 * hd * emb * 2,
                applicable: false,
                reason: Some(experimental_reason.clone()),
                selected: false,
                data: serde_json::json!({ "heads": heads }),
            });
        }
        let mut neurons = 0u64;
        let mut per_layer = Vec::new();
        for n in &s.ffn_neurons {
            let med = median(&n.norms);
            let low: Vec<usize> = n.norms.iter().enumerate().filter(|(_, v)| med > 0.0 && **v < 0.1 * med).map(|(i, _)| i).collect();
            neurons += low.len() as u64;
            if !low.is_empty() {
                per_layer.push(serde_json::json!({ "layer": n.layer, "neurons": low }));
            }
        }
        if neurons > 0 {
            let per_neuron: u64 = ["ffn_up", "ffn_gate", "ffn_down"]
                .iter()
                .filter_map(|r| g.tensors.iter().find(|t| parse_tensor_name(&t.name).role == *r))
                .map(|t| t.n_bytes / arch.hyperparams.feed_forward_length.unwrap_or(1).max(1))
                .sum();
            c.push(Candidate {
                id: "low_norm_neurons".into(),
                kind: CandidateKind::LowNormNeurons,
                title: format!("{neurons} FFN neurons with < 10% of their layer's median norm"),
                detail: "Neurons whose up/gate rows and down column are all near zero ('dead' neurons).".into(),
                risk: Risk::Experimental,
                bytes_saved: neurons * per_neuron,
                applicable: false,
                reason: Some(experimental_reason.clone()),
                selected: false,
                data: serde_json::json!({ "layers": per_layer }),
            });
        }
        let mut experts = Vec::new();
        for e in &s.experts {
            let med = median(&e.norms);
            for (i, v) in e.norms.iter().enumerate() {
                if med > 0.0 && *v < 0.5 * med {
                    experts.push(serde_json::json!({ "layer": e.layer, "expert": i, "norm": v }));
                }
            }
        }
        if !experts.is_empty() {
            c.push(Candidate {
                id: "low_norm_experts".into(),
                kind: CandidateKind::LowNormExperts,
                title: format!("{} MoE experts with < 50% of their layer's median norm", experts.len()),
                detail: "Router usage (from `trace --corpus`) is a better signal than weight norms.".into(),
                risk: Risk::Experimental,
                bytes_saved: 0,
                applicable: false,
                reason: Some(experimental_reason.clone()),
                selected: false,
                data: serde_json::json!({ "experts": experts }),
            });
        }
    } else {
        notes.push("no weight statistics available: run `inspect --stats` (or the stats job) for head/neuron/expert heuristics".into());
    }

    // --- Layer redundancy ----------------------------------------------------------------
    p.set(35.0, "layer redundancy");
    let n_layers = arch.layers.len();
    let layer_bytes = |i: u32| arch.layers.iter().find(|l| l.index == i).map_or(0, |l| l.bytes);
    let hybrid_pattern = g.arch_key("full_attention_interval").is_some() || arch.kind == ModelKind::Hybrid;
    let mut layer_scores: Vec<(u32, f64, &str)> = Vec::new();
    if let Some(tr) = trace.filter(|t| !t.layers.is_empty()) {
        for l in &tr.layers {
            if let Some(cs) = l.cos_prev {
                layer_scores.push((l.index, cs as f64, "trace: cosine(hidden_in, hidden_out)"));
            }
        }
    } else if !opts.skip_layer_similarity && n_layers > 2 {
        let roles = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_gate", "ffn_down"];
        for i in 1..n_layers as u32 {
            check_cancel(p)?;
            p.set(35.0 + 50.0 * i as f32 / n_layers as f32, &format!("layer similarity {}/{}", i, n_layers - 1));
            let mut sims = Vec::new();
            for r in roles {
                let a = g.tensor(&format!("blk.{}.{r}.weight", i - 1));
                let b = g.tensor(&format!("blk.{i}.{r}.weight"));
                if let (Some(a), Some(b)) = (a, b) {
                    if let Some(s) = tensor_cosine(g, a, b)? {
                        sims.push(s);
                    }
                }
            }
            if !sims.is_empty() {
                layer_scores.push((i, sims.iter().sum::<f64>() / sims.len() as f64, "static: mean weight cosine with previous layer"));
            }
        }
        if !layer_scores.is_empty() {
            notes.push("layer redundancy is a static weight-similarity estimate; run `trace --corpus` for hidden-state similarity".into());
        }
    }
    let mut ranked: Vec<_> = layer_scores
        .iter()
        .filter(|(i, _, _)| opts.include_edge_layers || (*i > 0 && (*i as usize) + 1 < n_layers))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, score, how) in ranked.into_iter().take(opts.max_layer_candidates.unwrap_or(3)) {
        c.push(Candidate {
            id: format!("layer_{i}"),
            kind: CandidateKind::LayerRedundancy,
            title: format!("Remove layer {i} (similarity {score:.4})"),
            detail: format!("{how}. Removing a layer shifts every later layer; quality loss must be measured (perplexity)."),
            risk: Risk::High,
            bytes_saved: layer_bytes(*i),
            applicable: !hybrid_pattern,
            reason: hybrid_pattern.then(|| "layer kinds follow an index pattern (hybrid model); removing a layer would misalign it".to_string()),
            selected: false,
            data: serde_json::json!({ "layer": i, "score": score }),
        });
    }
    if layer_scores.is_empty() && n_layers > 2 {
        notes.push("layer similarity not computed".into());
    }

    // --- Quantization headroom -----------------------------------------------------------
    p.set(88.0, "quantization headroom");
    let mut quant = Vec::new();
    let mut saved = 0;
    for t in &g.tensors {
        if is_float_type(t.ggml_type) && t.dims.len() >= 2 && t.n_elements >= 1 << 20 && t.row_len() % 32 == 0 {
            let q = GgmlType::Q8_0.bytes_for(t.n_elements).unwrap_or(t.n_bytes);
            if q < t.n_bytes {
                saved += t.n_bytes - q;
                quant.push(t.name.clone());
            }
        }
    }
    if !quant.is_empty() {
        c.push(Candidate {
            id: "quant_headroom".into(),
            kind: CandidateKind::QuantHeadroom,
            title: format!("Requantize {} float tensors to Q8_0", quant.len()),
            detail: "Q8_0 is near-lossless for weight matrices. For lower bit-widths use `llama-quantize`.".into(),
            risk: Risk::Low,
            bytes_saved: saved,
            applicable: true,
            reason: None,
            selected: false,
            data: serde_json::json!({ "tensors": quant, "target": "Q8_0" }),
        });
    }
    let low_bits: u64 = g.tensors.iter().filter(|t| t.ggml_type == GgmlType::Q8_0 && t.dims.len() >= 2).map(|t| t.n_bytes).sum();
    if low_bits > 0 {
        notes.push(format!(
            "{} of Q8_0 weights could go to Q4_K_M with `llama-quantize --allow-requantize` (~45% smaller, small quality loss)",
            human_bytes(low_bits)
        ));
    }

    // --- Unused extras ---------------------------------------------------------------------
    if let Some(pp) = &m.entry.projector {
        let size = std::fs::metadata(pp).map(|m| m.len()).unwrap_or(0);
        c.push(Candidate {
            id: "unused_projector".into(),
            kind: CandidateKind::UnusedProjector,
            title: "Omit the vision projector for text-only use".into(),
            detail: "The generated Modelfile will not reference the projector GGUF; image input stops working.".into(),
            risk: Risk::Low,
            bytes_saved: size,
            applicable: true,
            reason: None,
            selected: false,
            data: serde_json::json!({ "projector": pp }),
        });
    }
    let arch_prefix = format!("{}.", arch.name);
    let big: Vec<(String, u64)> = g
        .metadata
        .iter()
        .filter(|(k, _)| !k.starts_with("tokenizer.ggml.") && !k.starts_with(&arch_prefix) && *k != "tokenizer.chat_template")
        .filter_map(|(k, v)| {
            let size = match v {
                neurondump_gguf::Value::String(s) => s.len() as u64,
                neurondump_gguf::Value::Array(a) => a.as_strings().map_or(a.len() * 4, |s| s.iter().map(|x| x.len() as u64 + 8).sum()),
                _ => 0,
            };
            (size > 16 * 1024).then(|| (k.clone(), size))
        })
        .collect();
    if !big.is_empty() {
        c.push(Candidate {
            id: "large_metadata".into(),
            kind: CandidateKind::LargeMetadata,
            title: format!("Strip {} large non-essential metadata entries", big.len()),
            detail: big.iter().map(|(k, s)| format!("{k} ({})", human_bytes(*s))).collect::<Vec<_>>().join(", "),
            risk: Risk::Low,
            bytes_saved: big.iter().map(|(_, s)| s).sum(),
            applicable: true,
            reason: None,
            selected: false,
            data: serde_json::json!({ "keys": big.iter().map(|(k, _)| k).collect::<Vec<_>>() }),
        });
    }

    p.set(100.0, "done");
    Ok(PrunePlan {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        digest: m.entry.digest.clone(),
        architecture: arch.name,
        original_size: g.file_size,
        candidates: c,
        notes,
    })
}
