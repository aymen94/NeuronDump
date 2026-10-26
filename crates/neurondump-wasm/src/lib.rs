//! In-browser NeuronDump: static inspection of a GGUF file read lazily with `Blob.slice`.
//! JavaScript reads the header (growing the prefix while [`Session::new`] reports
//! `NEED_MORE`), then answers the web UI's API calls through [`Session::api`]. Tensor data is
//! only read for one tensor at a time, when its statistics are requested.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use neurondump_core::api;
use neurondump_core::discovery::quantization_label;
use neurondump_core::graph::{GraphInputs, ModelGraph};
use neurondump_core::inspect::{inspect, metadata_entries, InspectOptions, InspectReport};
use neurondump_core::stats::tensor_stats;
use neurondump_core::template::template_report;
use neurondump_core::tokenizer::Vocab;
use neurondump_core::util::SCHEMA_VERSION;
use neurondump_core::{ModelEntry, ModelHandle, ModelSummary, NoProgress, SourceKind};
use neurondump_gguf::{GgufError, GgufFile, ParseOptions};

/// Prefix of the error message telling JavaScript to read more header bytes and retry.
const NEED_MORE: &str = "NEED_MORE";

#[wasm_bindgen]
pub struct Session {
    m: ModelHandle,
    vocab: Option<Vocab>,
    graph: ModelGraph,
    report: InspectReport,
}

fn js(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

#[wasm_bindgen]
impl Session {
    /// Parses the header from the first bytes of the file. Throws an error starting with
    /// `NEED_MORE` when `prefix` ends inside the header.
    #[wasm_bindgen(constructor)]
    pub fn new(prefix: Vec<u8>, file_size: f64, name: String) -> Result<Session, JsError> {
        let size = file_size as u64;
        let mut g = match GgufFile::from_prefix(prefix, size, ParseOptions::default()) {
            Ok(g) => g,
            Err(GgufError::Truncated { .. }) => return Err(JsError::new(NEED_MORE)),
            Err(e) => return Err(JsError::new(&e.to_string())),
        };
        // Everything needed is decoded now; free the header bytes.
        g.set_window(0, Vec::new());
        let mut entry = ModelEntry::from_file(Path::new(&name), SourceKind::File, name.clone());
        entry.size = size;
        let m = ModelHandle::from_gguf(entry, g);
        let vocab = Vocab::from_gguf(&m.gguf).ok();
        let graph = ModelGraph::build(&m, vocab.as_ref(), &GraphInputs { stats: None, trace: None, sweeps: Vec::new() });
        let report = inspect(&m, &InspectOptions::default(), &NoProgress).map_err(js)?;
        Ok(Session { m, vocab, graph, report })
    }

    pub fn id(&self) -> String {
        self.m.entry.id.clone()
    }

    /// Answers `GET /api<path>?<query>` of the web UI (read-only endpoints) as JSON text.
    pub fn api(&self, path: &str, query: &str) -> Result<String, JsError> {
        self.route(path, query).map(|v| v.to_string()).map_err(js)
    }

    /// Absolute `[start, end)` byte range of a tensor's data, to read with `Blob.slice`.
    pub fn tensor_range(&self, name: &str) -> Result<Vec<f64>, JsError> {
        let t = self.m.gguf.tensor(name).ok_or_else(|| JsError::new(&format!("no tensor {name:?}")))?;
        let (s, e) = self.m.gguf.data_range(t);
        Ok(vec![s as f64, e as f64])
    }

    /// Weight statistics of one tensor from its raw bytes (`bytes` = the `tensor_range`).
    pub fn tensor_stats(&mut self, name: &str, bytes: Vec<u8>) -> Result<String, JsError> {
        let g = std::sync::Arc::get_mut(&mut self.m.gguf).ok_or_else(|| JsError::new("model is busy"))?;
        let t = g.tensor(name).ok_or_else(|| JsError::new(&format!("no tensor {name:?}")))?.clone();
        g.set_window(g.data_range(&t).0, bytes);
        let r = tensor_stats(g, &t, &NoProgress, &|_| {});
        g.set_window(0, Vec::new());
        Ok(serde_json::to_string(&r.map_err(js)?)?)
    }
}

impl Session {
    fn summary(&self) -> ModelSummary {
        let (e, g) = (&self.m.entry, &self.m.gguf);
        ModelSummary {
            id: e.id.clone(),
            name: e.name.clone(),
            source: e.source,
            architecture: g.architecture().map(str::to_string),
            params: Some(g.total_params()),
            quantization: quantization_label(g),
            context_length: g.arch_key("context_length").and_then(|v| v.as_u64()),
            size: e.size,
            path: e.path.clone(),
            has_projector: false,
            error: None,
        }
    }

    fn route(&self, path: &str, query: &str) -> Result<Value> {
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        match parts.as_slice() {
            ["info"] => Ok(json!({
                "version": env!("CARGO_PKG_VERSION"),
                "schema_version": SCHEMA_VERSION,
                "trace_architectures": [],
                "job_kinds": [],
                "demo": true,
            })),
            ["models"] => Ok(json!({ "models": [self.summary()], "warnings": [] })),
            ["search"] => {
                #[derive(serde::Deserialize)]
                struct S {
                    q: String,
                    limit: Option<usize>,
                }
                let s: S = serde_urlencoded::from_str(query)?;
                let models = [self.summary()];
                Ok(serde_json::to_value(neurondump_core::search::search(
                    &s.q,
                    &models,
                    Some((&self.m, self.vocab.as_ref())),
                    s.limit.unwrap_or(10).min(200),
                ))?)
            }
            ["models", id, rest @ ..] => {
                if *id != self.m.entry.id {
                    bail!("model not found");
                }
                match rest {
                    [] => Ok(api::model(&self.report)),
                    ["metadata"] => {
                        let full = query.split('&').any(|kv| kv == "full=true" || kv == "full=1");
                        Ok(if full {
                            serde_json::to_value(metadata_entries(&self.m.gguf, true))?
                        } else {
                            serde_json::to_value(&self.report.metadata)?
                        })
                    }
                    ["vocab"] => api::vocab(self.vocab()?, &serde_urlencoded::from_str(query)?),
                    ["vocab", "texts"] => Ok(api::vocab_texts(self.vocab()?)),
                    ["tensors"] => Ok(api::tensors(&self.m.gguf, &serde_urlencoded::from_str(query)?)),
                    ["layers"] => Ok(api::layers(&self.report)),
                    ["template"] => Ok(serde_json::to_value(template_report(&self.m))?),
                    ["graph"] => api::graph(&self.graph, &serde_urlencoded::from_str(query)?),
                    ["nodes", node] => api::node(&self.graph, node),
                    _ => Err(not_here()),
                }
            }
            _ => Err(not_here()),
        }
    }

    fn vocab(&self) -> Result<&Vocab> {
        self.vocab.as_ref().ok_or_else(|| anyhow!("this file has no vocabulary"))
    }
}

fn not_here() -> anyhow::Error {
    anyhow!("not available in the browser demo: download NeuronDump for traces, edits, analysis and slimming")
}
