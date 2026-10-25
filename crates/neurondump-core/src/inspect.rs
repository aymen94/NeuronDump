//! `inspect`: everything inside a model file, as one serializable report.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use neurondump_gguf::GgufFile;

use crate::arch::{analyze_architecture, parse_tensor_name, ArchitectureReport};
use crate::discovery::{quantization_label, OllamaInfo, SourceKind};
use crate::model::ModelHandle;
use crate::stats::{compute_stats, StatsOptions, StatsReport};
use crate::template::{template_report, TemplateReport};
use crate::tokenizer::{Token, Vocab, VocabSummary};
use crate::util::{value_json, Progress, SCHEMA_VERSION};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InspectOptions {
    /// Do not summarize large metadata arrays; include the full vocabulary.
    #[serde(default)]
    pub full: bool,
    /// Compute weight statistics (slow: reads all tensor data).
    #[serde(default)]
    pub stats: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub source: SourceKind,
    pub path: PathBuf,
    pub size: u64,
    pub digest: String,
    pub quantization: Option<String>,
    pub ollama: Option<OllamaInfo>,
    pub shares_weights_with: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileInfo {
    pub version: u32,
    pub endianness: neurondump_gguf::Endianness,
    pub alignment: u64,
    pub header_size: u64,
    pub data_offset: u64,
    pub file_size: u64,
    pub tensor_data_bytes: u64,
    pub n_kv: usize,
    pub n_tensors: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetaEntry {
    pub key: String,
    #[serde(rename = "type")]
    pub value_type: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorRow {
    pub name: String,
    pub layer: Option<u32>,
    pub role: String,
    pub component: String,
    pub dims: Vec<u64>,
    pub ggml_type: String,
    pub n_elements: u64,
    pub n_bytes: u64,
    pub offset: u64,
    pub bits_per_weight: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaSummary {
    pub template: Option<String>,
    pub system: Option<String>,
    pub params: Option<serde_json::Value>,
    pub licenses: Vec<String>,
    pub adapters: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectorReport {
    pub path: PathBuf,
    pub file: FileInfo,
    pub architecture: ArchitectureReport,
    pub metadata: Vec<MetaEntry>,
    pub tensors: Vec<TensorRow>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Timings {
    pub parse_ms: u64,
    pub architecture_ms: u64,
    pub tokenizer_ms: u64,
    pub template_ms: u64,
    pub stats_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectReport {
    pub schema_version: u32,
    pub model: ModelInfo,
    pub file: FileInfo,
    pub metadata: Vec<MetaEntry>,
    pub architecture: ArchitectureReport,
    pub tokenizer: Option<VocabSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab: Option<Vec<Token>>,
    pub templates: TemplateReport,
    pub tensors: Vec<TensorRow>,
    pub ollama: Option<OllamaSummary>,
    pub projector: Option<ProjectorReport>,
    pub stats: Option<StatsReport>,
    pub warnings: Vec<String>,
    pub timings: Timings,
}

pub fn file_info(g: &GgufFile) -> FileInfo {
    FileInfo {
        version: g.version,
        endianness: g.endianness,
        alignment: g.alignment,
        header_size: g.header_size,
        data_offset: g.data_offset,
        file_size: g.file_size,
        tensor_data_bytes: g.tensor_data_bytes(),
        n_kv: g.metadata.len(),
        n_tensors: g.tensors.len(),
    }
}

pub fn metadata_entries(g: &GgufFile, full: bool) -> Vec<MetaEntry> {
    g.metadata
        .iter()
        .map(|(k, v)| MetaEntry {
            key: k.clone(),
            value_type: v.type_description(),
            value: value_json(v, if full { None } else { Some(16) }),
        })
        .collect()
}

pub fn tensor_rows(g: &GgufFile) -> Vec<TensorRow> {
    g.tensors
        .iter()
        .map(|t| {
            let r = parse_tensor_name(&t.name);
            TensorRow {
                name: t.name.clone(),
                layer: r.layer,
                role: r.role,
                component: r.component.as_str().into(),
                dims: t.dims.clone(),
                ggml_type: t.ggml_type.name().into_owned(),
                n_elements: t.n_elements,
                n_bytes: t.n_bytes,
                offset: t.offset,
                bits_per_weight: t.ggml_type.bits_per_weight(),
            }
        })
        .collect()
}

pub fn model_info(m: &ModelHandle) -> ModelInfo {
    ModelInfo {
        id: m.entry.id.clone(),
        name: m.entry.name.clone(),
        source: m.entry.source,
        path: m.entry.path.clone(),
        size: m.entry.size,
        digest: m.entry.digest.clone(),
        quantization: quantization_label(&m.gguf),
        ollama: m.entry.ollama.clone(),
        shares_weights_with: m.entry.shares_weights_with.clone(),
    }
}

pub fn inspect(m: &ModelHandle, opts: &InspectOptions, p: &dyn Progress) -> Result<InspectReport> {
    let t0 = Instant::now();
    let g = &m.gguf;
    let mut timings = Timings::default();
    let mut warnings: Vec<String> = g.warnings.clone();
    warnings.extend(m.entry.warnings.iter().cloned());

    p.set(5.0, "architecture");
    let t = Instant::now();
    let architecture = analyze_architecture(g);
    warnings.extend(architecture.warnings.iter().cloned());
    for r in &architecture.unknown_roles {
        warnings.push(format!("unrecognized tensor role {r}"));
    }
    timings.architecture_ms = t.elapsed().as_millis() as u64;

    p.set(20.0, "tokenizer");
    let t = Instant::now();
    let (tokenizer, vocab) = match Vocab::from_gguf(g) {
        Ok(v) => {
            warnings.extend(v.summary.warnings.iter().cloned());
            (Some(v.summary.clone()), opts.full.then(|| v.tokens.clone()))
        }
        Err(e) => {
            warnings.push(format!("tokenizer: {e}"));
            (None, None)
        }
    };
    timings.tokenizer_ms = t.elapsed().as_millis() as u64;

    p.set(40.0, "templates");
    let t = Instant::now();
    let templates = template_report(m);
    for tt in &templates.templates {
        for e in &tt.errors {
            warnings.push(format!("template {}: {e}", tt.name));
        }
    }
    timings.template_ms = t.elapsed().as_millis() as u64;

    let ollama = m.entry.ollama.as_ref().map(|_| OllamaSummary {
        template: m.ollama.template.clone(),
        system: m.ollama.system.clone(),
        params: m.ollama.params.clone(),
        licenses: m.ollama.licenses.iter().map(|l| l.lines().find(|x| !x.trim().is_empty()).unwrap_or("").trim().to_string()).collect(),
        adapters: m.ollama.adapters.clone(),
    });

    let projector = match m.open_projector() {
        Some(Ok(pg)) => Some(ProjectorReport {
            path: m.entry.projector.clone().unwrap_or_default(),
            file: file_info(&pg),
            architecture: analyze_architecture(&pg),
            metadata: metadata_entries(&pg, opts.full),
            tensors: tensor_rows(&pg),
        }),
        Some(Err(e)) => {
            warnings.push(format!("vision projector: {e}"));
            None
        }
        None => None,
    };

    let stats = if opts.stats {
        let t = Instant::now();
        let s = compute_stats(g, &m.entry.name, &m.entry.digest, &StatsOptions::default(), p)?;
        timings.stats_ms = t.elapsed().as_millis() as u64;
        Some(s)
    } else {
        None
    };

    timings.total_ms = t0.elapsed().as_millis() as u64;
    p.set(100.0, "done");
    Ok(InspectReport {
        schema_version: SCHEMA_VERSION,
        model: model_info(m),
        file: file_info(g),
        metadata: metadata_entries(g, opts.full),
        architecture,
        tokenizer,
        vocab,
        templates,
        tensors: tensor_rows(g),
        ollama,
        projector,
        stats,
        warnings,
        timings,
    })
}
