//! One query across models, and (for one model) tokens, tensors and metadata keys.

use serde::Serialize;

use crate::discovery::ModelSummary;
use crate::model::ModelHandle;
use crate::tokenizer::{Vocab, VocabQuery};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HitKind {
    Model,
    Token,
    Tensor,
    Metadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub kind: HitKind,
    pub model_id: String,
    pub model_name: String,
    /// What matched: model name, token text, tensor name or metadata key.
    pub label: String,
    /// Short extra information (architecture, token id, shape, value…).
    pub detail: String,
    /// Token id, tensor name or metadata key, used to link to the right view.
    pub target: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResults {
    pub query: String,
    pub hits: Vec<SearchHit>,
    /// Total matches per kind before the limit (models, tokens, tensors, metadata).
    pub totals: [usize; 4],
}

fn contains(hay: &str, needle_lower: &str) -> bool {
    hay.to_lowercase().contains(needle_lower)
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() > n {
        format!("{}…", s.chars().take(n).collect::<String>())
    } else {
        s.to_string()
    }
}

/// Searches `models`, and the inside of `model` when one is given. `limit` applies per kind.
pub fn search(q: &str, models: &[ModelSummary], model: Option<(&ModelHandle, Option<&Vocab>)>, limit: usize) -> SearchResults {
    let needle = q.trim().to_lowercase();
    let mut hits = Vec::new();
    let mut totals = [0usize; 4];
    if needle.is_empty() {
        return SearchResults { query: q.to_string(), hits, totals };
    }

    for m in models {
        let arch = m.architecture.clone().unwrap_or_default();
        let quant = m.quantization.clone().unwrap_or_default();
        if contains(&m.name, &needle) || contains(&arch, &needle) || contains(&quant, &needle) || m.id == needle {
            totals[0] += 1;
            if totals[0] <= limit {
                hits.push(SearchHit {
                    kind: HitKind::Model,
                    model_id: m.id.clone(),
                    model_name: m.name.clone(),
                    label: m.name.clone(),
                    detail: format!("{arch} · {quant} · {}", crate::util::human_bytes(m.size)),
                    target: m.id.clone(),
                });
            }
        }
    }

    if let Some((h, vocab)) = model {
        let (id, name) = (h.entry.id.clone(), h.entry.name.clone());
        let hit = |kind, label: String, detail: String, target: String| SearchHit {
            kind,
            model_id: id.clone(),
            model_name: name.clone(),
            label,
            detail,
            target,
        };

        if let Some(Ok(page)) = vocab.map(|v| v.search(&VocabQuery { q: Some(q.trim().to_string()), limit: Some(limit), ..Default::default() })) {
            totals[1] = page.total;
            for t in page.tokens {
                hits.push(hit(HitKind::Token, format!("{:?}", t.text), format!("id {} · {}", t.id, t.class.as_str()), t.id.to_string()));
            }
        }

        for t in &h.gguf.tensors {
            if contains(&t.name, &needle) {
                totals[2] += 1;
                if totals[2] <= limit {
                    let dims = t.dims.iter().map(u64::to_string).collect::<Vec<_>>().join("×");
                    hits.push(hit(HitKind::Tensor, t.name.clone(), format!("{dims} · {}", t.ggml_type), t.name.clone()));
                }
            }
        }

        for (k, v) in &h.gguf.metadata {
            let value = match v {
                neurondump_gguf::Value::Array(a) => a.type_description(),
                v => v.to_string(),
            };
            let in_value = !matches!(v, neurondump_gguf::Value::Array(_)) && contains(&value, &needle);
            if contains(k, &needle) || in_value {
                totals[3] += 1;
                if totals[3] <= limit {
                    hits.push(hit(HitKind::Metadata, k.clone(), short(&value, 80), k.clone()));
                }
            }
        }
    }
    SearchResults { query: q.to_string(), hits, totals }
}
