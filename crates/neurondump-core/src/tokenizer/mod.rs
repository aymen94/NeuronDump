//! Vocabulary extraction: readable token text, classification, special tokens, search.

mod encode;
mod keywords;

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use anyhow::{bail, Result};
use regex::Regex;
use serde::Serialize;

use neurondump_gguf::GgufFile;

pub use encode::{Encoder, Encoding};

mod types;
mod text;
mod classify;
mod search;
pub use types::*;
pub use text::*;
pub use classify::*;
pub use search::*;

pub struct Vocab {
    pub model_name: String,
    pub model: TokenizerModel,
    pub tokens: Vec<Token>,
    pub summary: VocabSummary,
    pub merges: Vec<String>,
    pub by_raw: HashMap<String, u32>,
}

impl Vocab {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model_name = g.get_str("tokenizer.ggml.model").unwrap_or("none").to_string();
        let model = match model_name.as_str() {
            "gpt2" => TokenizerModel::Bpe,
            "llama" => TokenizerModel::Spm,
            "bert" => TokenizerModel::Wpm,
            _ => TokenizerModel::Other,
        };
        let Some(raws) = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).and_then(|a| a.as_strings()) else {
            bail!("model has no tokenizer.ggml.tokens vocabulary");
        };
        let scores = g.get("tokenizer.ggml.scores").and_then(|v| v.as_array()).and_then(|a| a.as_f32s());
        let types = g.get("tokenizer.ggml.token_type").and_then(|v| v.as_array());
        let merges: Vec<String> = g
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_strings())
            .map(|m| m.to_vec())
            .unwrap_or_default();
        let mut warnings = Vec::new();
        if model == TokenizerModel::Other {
            warnings.push(format!("tokenizer model {model_name:?}: text decoding uses generic rules"));
        }

        let tokens: Vec<Token> = {
            use rayon::prelude::*;
            raws.par_iter()
                .enumerate()
                .map(|(i, raw)| {
                    let ty = types.and_then(|t| t.get_i64(i)).map_or(TokenType::Undefined, TokenType::from_id);
                    let text = decode_raw(model, raw);
                    let (class, script, code_langs, word_start) = classify(model, raw, &text, ty);
                    Token {
                        id: i as u32,
                        raw: raw.clone(),
                        text,
                        score: scores.and_then(|s| s.get(i).copied()),
                        token_type: ty,
                        class,
                        script,
                        code_langs,
                        word_start,
                    }
                })
                .collect()
        };

        let mut class_counts = BTreeMap::new();
        let mut script_counts = BTreeMap::new();
        let mut code_lang_counts = BTreeMap::new();
        let mut type_counts = BTreeMap::new();
        let mut control_tokens = Vec::new();
        for t in &tokens {
            *class_counts.entry(t.class.as_str().to_string()).or_insert(0) += 1;
            if let Some(s) = t.script {
                *script_counts.entry(s.to_string()).or_insert(0) += 1;
            }
            for l in &t.code_langs {
                *code_lang_counts.entry(l.to_string()).or_insert(0) += 1;
            }
            let tn = serde_json::to_value(t.token_type).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
            *type_counts.entry(tn).or_insert(0) += 1;
            if t.class == TokenClass::Special {
                let role = if t.raw.to_lowercase().contains("tool") || t.raw.contains("python_tag") {
                    "tool"
                } else if t.raw.contains("fim") || t.raw.contains("FIM") {
                    "fim"
                } else if t.raw.contains("think") {
                    "reasoning"
                } else if t.raw.contains("image") || t.raw.contains("vision") || t.raw.contains("video") || t.raw.contains("audio") {
                    "multimodal"
                } else {
                    "control"
                };
                control_tokens.push(SpecialToken { role: role.into(), id: t.id, text: Some(t.raw.clone()) });
            }
        }

        let mut specials = Vec::new();
        for (role, key) in SPECIAL_KEYS {
            if let Some(id) = g.get_u64(key) {
                if specials.iter().any(|s: &SpecialToken| s.role == *role && s.id as u64 == id) {
                    continue;
                }
                let text = tokens.get(id as usize).map(|t| t.raw.clone());
                if text.is_none() {
                    warnings.push(format!("{key} = {id} is outside the vocabulary"));
                }
                specials.push(SpecialToken { role: role.to_string(), id: id as u32, text });
            }
        }

        let by_raw = tokens.iter().map(|t| (t.raw.clone(), t.id)).collect();
        let summary = VocabSummary {
            model: model_name.clone(),
            pre: g.get_str("tokenizer.ggml.pre").map(str::to_string),
            size: tokens.len(),
            merges: merges.len(),
            add_bos: g.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()),
            add_eos: g.get("tokenizer.ggml.add_eos_token").and_then(|v| v.as_bool()),
            add_space_prefix: g.get("tokenizer.ggml.add_space_prefix").and_then(|v| v.as_bool()),
            class_counts,
            script_counts,
            code_lang_counts,
            type_counts,
            specials,
            control_tokens,
            warnings,
        };
        Ok(Self { model_name, model, tokens, summary, merges, by_raw })
    }

}
