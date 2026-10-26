//! Chat prompt rendering: a small evaluator for the subset of Go `text/template` that
//! Ollama templates use (`if`/`else if`/`range`/`with`, variables, `eq`/`ne`/`and`/`or`/
//! `not`/`len`/`slice`/`index`, trim markers), plus built-in templates for the common
//! families when a model has no Ollama template (raw GGUF files only carry Jinja).

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use crate::model::ModelHandle;
use crate::tokenizer::Vocab;

mod ast;
mod parse;
mod eval;
use ast::*;
use parse::*;
use eval::*;

/// What the chat formatting is built from.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    /// Overrides the model's default system prompt (empty string = none).
    #[serde(default)]
    pub system: Option<String>,
    /// Go template text replacing the model's template.
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRender {
    /// `ollama`, `builtin:<family>`, `custom` or `none`.
    pub source: String,
    pub template: String,
    pub system: Option<String>,
    pub text: String,
    /// Byte range of the user prompt inside `text`.
    pub prompt_span: Option<(usize, usize)>,
}

const CHATML: &str = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n";
const LLAMA3: &str = "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
const GEMMA: &str = "<start_of_turn>user\n{{ if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}<end_of_turn>\n<start_of_turn>model\n";
const PHI3: &str = "{{ if .System }}<|system|>\n{{ .System }}<|end|>\n{{ end }}<|user|>\n{{ .Prompt }}<|end|>\n<|assistant|>\n";

/// Built-in template for the model's family, detected from its special tokens.
pub fn builtin_template(vocab: &Vocab) -> Option<(&'static str, &'static str)> {
    let has = |s: &str| vocab.by_raw.contains_key(s);
    if has("<|start_header_id|>") {
        Some(("llama3", LLAMA3))
    } else if has("<|im_start|>") {
        Some(("chatml", CHATML))
    } else if has("<start_of_turn>") {
        Some(("gemma", GEMMA))
    } else if has("<|user|>") && has("<|assistant|>") {
        Some(("phi3", PHI3))
    } else {
        None
    }
}

/// Renders the prompt as the model's chat format.
pub fn render_chat(m: &ModelHandle, vocab: Option<&Vocab>, req: &ChatRequest) -> Result<ChatRender> {
    let (source, template) = match &req.template {
        Some(t) => ("custom".to_string(), t.clone()),
        None => match (&m.ollama.template, vocab.and_then(builtin_template)) {
            (Some(t), _) if !t.trim().is_empty() => ("ollama".to_string(), t.clone()),
            (_, Some((name, t))) => (format!("builtin:{name}"), t.to_string()),
            _ => ("none".to_string(), "{{ if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}".to_string()),
        },
    };
    let system = match &req.system {
        Some(s) if s.is_empty() => None,
        Some(s) => Some(s.clone()),
        None => m.ollama.system.clone(),
    };
    let text = render_template(&template, system.as_deref(), &req.prompt)?;
    let prompt_span = (!req.prompt.is_empty()).then(|| text.rfind(&req.prompt).map(|i| (i, i + req.prompt.len()))).flatten();
    Ok(ChatRender { source, template, system, text, prompt_span })
}

/// Executes a Go template with Ollama's values for a single user turn.
pub fn render_template(template: &str, system: Option<&str>, prompt: &str) -> Result<String> {
    let msg = |role: &str, content: &str| {
        let mut m = BTreeMap::new();
        m.insert("Role".into(), V::Str(role.into()));
        m.insert("Content".into(), V::Str(content.into()));
        V::Map(m)
    };
    let mut messages = Vec::new();
    if let Some(s) = system {
        messages.push(msg("system", s));
    }
    messages.push(msg("user", prompt));
    let mut root = BTreeMap::new();
    root.insert("System".into(), V::Str(system.unwrap_or("").into()));
    root.insert("Prompt".into(), V::Str(prompt.into()));
    root.insert("Response".into(), V::Str(String::new()));
    root.insert("Suffix".into(), V::Str(String::new()));
    root.insert("Messages".into(), V::List(messages));
    let nodes = parse(template)?;
    let mut out = String::new();
    let root = V::Map(root);
    let mut ev = Eval { root: &root, vars: vec![BTreeMap::new()] };
    ev.exec(&nodes, &root, &mut out)?;
    Ok(out)
}

// --- values -----------------------------------------------------------------------------
