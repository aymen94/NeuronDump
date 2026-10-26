//! Chat templates as decision trees: Jinja (`tokenizer.chat_template*`) and Ollama's
//! Go `text/template` layer. Not an evaluator — a structural parser that extracts the
//! branches, the variables they read and the special tokens they emit.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use regex::Regex;
use serde::Serialize;

use neurondump_gguf::GgufFile;

use crate::model::ModelHandle;

mod lex;
mod builder;
mod parse;
use lex::*;
use builder::*;
pub use parse::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Syntax {
    Jinja,
    GoTemplate,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemplateNode {
    /// root, if, branch, for, range, with, set, macro, call, define, template, emit, filter, ...
    pub kind: String,
    /// For branches: `if`, `elif`, `else` (Go: `if`, `else if`, `else`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub vars: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub emits: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<TemplateNode>,
}

impl TemplateNode {
    fn new(kind: &str) -> Self {
        Self { kind: kind.into(), branch: None, condition: None, vars: vec![], emits: vec![], text: None, children: vec![] }
    }

    pub fn count_decisions(&self) -> usize {
        let own = usize::from(matches!(self.kind.as_str(), "branch" | "for" | "range" | "with"));
        own + self.children.iter().map(TemplateNode::count_decisions).sum::<usize>()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TemplateTree {
    pub name: String,
    pub syntax: Syntax,
    pub source: String,
    pub root: TemplateNode,
    pub variables: BTreeSet<String>,
    pub special_tokens: BTreeSet<String>,
    pub decisions: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TemplateReport {
    pub templates: Vec<TemplateTree>,
    pub ollama_system: Option<String>,
    pub ollama_params: Option<serde_json::Value>,
}

/// Raw special-token strings from the vocabulary (control + user-defined).
pub fn vocab_specials(g: &GgufFile) -> Vec<String> {
    let Some(tokens) = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).and_then(|a| a.as_strings()) else {
        return vec![];
    };
    let types = g.get("tokenizer.ggml.token_type").and_then(|v| v.as_array());
    let mut out: Vec<String> = tokens
        .iter()
        .enumerate()
        .filter(|(i, t)| types.and_then(|ty| ty.get_i64(*i)).is_some_and(|ty| ty == 3 || ty == 4) && t.len() > 2)
        .map(|(_, t)| t.clone())
        .collect();
    out.sort_by_key(|s| std::cmp::Reverse(s.len()));
    out
}

pub fn template_report(m: &ModelHandle) -> TemplateReport {
    let specials = vocab_specials(&m.gguf);
    let mut templates = Vec::new();
    for (k, v) in &m.gguf.metadata {
        if k == "tokenizer.chat_template" || k.starts_with("tokenizer.chat_template.") {
            if let Some(src) = v.as_str() {
                let name = k.strip_prefix("tokenizer.chat_template.").unwrap_or("default");
                templates.push(parse_jinja(name, src, &specials));
            }
        }
    }
    if let Some(t) = &m.ollama.template {
        templates.push(parse_go("ollama", t, &specials));
    }
    TemplateReport { templates, ollama_system: m.ollama.system.clone(), ollama_params: m.ollama.params.clone() }
}
