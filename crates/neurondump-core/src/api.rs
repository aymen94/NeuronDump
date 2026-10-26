//! JSON responses of the read-only inspection endpoints, shared by the HTTP server and the
//! in-browser (wasm) demo so both front ends see exactly the same shapes.

use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::graph::{GroupBy, Level, ModelGraph, ViewOptions};
use crate::inspect::{tensor_rows, InspectReport};
use crate::tokenizer::{TokenClass, Vocab, VocabQuery};
use neurondump_gguf::GgufFile;

/// `GET /models/{id}`: the overview (inspect report without the big tables).
pub fn model(r: &InspectReport) -> Value {
    json!({
        "model": r.model,
        "file": r.file,
        "architecture": {
            "name": r.architecture.name,
            "kind": r.architecture.kind,
            "hyperparams": r.architecture.hyperparams,
            "keys": r.architecture.keys,
            "total_params": r.architecture.total_params,
            "params_by_component": r.architecture.params_by_component,
            "bytes_by_component": r.architecture.bytes_by_component,
            "types": r.architecture.types,
            "layer_kinds": r.architecture.layer_kinds,
            "unknown_roles": r.architecture.unknown_roles,
            "tied_embeddings": r.architecture.tied_embeddings,
        },
        "tokenizer": r.tokenizer,
        "ollama": r.ollama,
        "projector": r.projector.as_ref().map(|p| json!({
            "path": p.path, "architecture": p.architecture.name, "kind": p.architecture.kind,
            "params": p.architecture.total_params, "tensors": p.file.n_tensors, "size": p.file.file_size,
        })),
        "warnings": r.warnings,
        "timings": r.timings,
    })
}

/// `GET /models/{id}/layers`.
pub fn layers(r: &InspectReport) -> Value {
    json!({ "layers": r.architecture.layers, "layer_kinds": r.architecture.layer_kinds })
}

#[derive(Debug, Default, Deserialize)]
pub struct TensorQuery {
    pub layer: Option<u32>,
    pub role: Option<String>,
    pub q: Option<String>,
    pub sort: Option<String>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

/// `GET /models/{id}/tensors`: filtered, sorted, paged tensor table.
pub fn tensors(g: &GgufFile, q: &TensorQuery) -> Value {
    let mut sel: Vec<_> = tensor_rows(g)
        .into_iter()
        .filter(|r| q.layer.is_none_or(|l| r.layer == Some(l)))
        .filter(|r| q.role.as_ref().is_none_or(|ro| ro.is_empty() || r.role.contains(ro.as_str())))
        .filter(|r| q.q.as_ref().is_none_or(|n| n.is_empty() || r.name.contains(n.as_str())))
        .collect();
    match q.sort.as_deref().unwrap_or("offset") {
        "size" => sel.sort_by(|a, b| b.n_bytes.cmp(&a.n_bytes)),
        "type" => sel.sort_by(|a, b| a.ggml_type.cmp(&b.ggml_type).then(a.name.cmp(&b.name))),
        "layer" => sel.sort_by(|a, b| a.layer.cmp(&b.layer).then(a.name.cmp(&b.name))),
        "name" => sel.sort_by(|a, b| a.name.cmp(&b.name)),
        _ => sel.sort_by_key(|r| r.offset),
    }
    let total = sel.len();
    let offset = q.offset.unwrap_or(0).min(total);
    let limit = q.limit.unwrap_or(200).min(2000);
    let page: Vec<_> = sel.into_iter().skip(offset).take(limit).collect();
    json!({ "total": total, "offset": offset, "limit": limit, "tensors": page })
}

/// `GET /models/{id}/vocab`.
pub fn vocab(v: &Vocab, q: &VocabQuery) -> Result<Value> {
    let mut q = q.clone();
    q.limit = Some(q.limit.unwrap_or(100).min(1000));
    let page = v.search(&q)?;
    Ok(json!({ "summary": v.summary, "page": page }))
}

/// `GET /models/{id}/vocab/texts`.
pub fn vocab_texts(v: &Vocab) -> Value {
    json!({
        "texts": v.tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>(),
        "classes": TokenClass::ALL.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
    })
}

#[derive(Debug, Default, Deserialize)]
pub struct GraphQuery {
    pub level: Option<String>,
    pub group_by: Option<String>,
    /// Comma-separated node ids to expand (lazy fine level).
    pub expand: Option<String>,
}

/// `GET /models/{id}/graph`.
pub fn graph(g: &ModelGraph, q: &GraphQuery) -> Result<Value> {
    let level = match q.level.as_deref().filter(|l| !l.is_empty()) {
        Some(l) => Level::parse(l).ok_or_else(|| anyhow!("level must be section, layer, block, tensor or fine"))?,
        None => Level::Layer,
    };
    let group_by = match q.group_by.as_deref().filter(|l| !l.is_empty()) {
        Some(l) => GroupBy::parse(l).ok_or_else(|| anyhow!("group_by must be section, layer, block, quant, kind or importance"))?,
        None => GroupBy::Section,
    };
    let expand = q.expand.as_deref().unwrap_or_default().split(',').map(str::trim).filter(|x| !x.is_empty()).map(String::from).collect();
    Ok(serde_json::to_value(g.view(&ViewOptions { level, group_by, expand }))?)
}

/// `GET /models/{id}/nodes/{node}`.
pub fn node(g: &ModelGraph, node: &str) -> Result<Value> {
    let rid = g.resolve_id(node).ok_or_else(|| anyhow!("no node {node:?}"))?;
    let d = g.detail(&rid).ok_or_else(|| anyhow!("no node {node:?}"))?;
    Ok(serde_json::to_value(d)?)
}
