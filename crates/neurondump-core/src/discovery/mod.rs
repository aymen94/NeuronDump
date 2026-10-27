//! Finding local GGUF models: Ollama stores, llama.cpp / Hugging Face / LM Studio / Jan
//! caches and user-configured directories (`--dir`, `scan_dirs`, `NEURONDUMP_MODEL_DIRS`).

mod files;
mod ollama;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{bail, Result};
use serde::Serialize;

use neurondump_gguf::{file_type_name, GgufFile};

use crate::util::{edit_distance, short_id};

pub use ollama::{ollama_model_dirs, OllamaInfo, OllamaLayer};

mod library;
#[cfg(feature = "native")]
pub use library::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Ollama,
    LlamaCpp,
    HuggingFace,
    LmStudio,
    Jan,
    Directory,
    File,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::Ollama => "ollama",
            SourceKind::LlamaCpp => "llama.cpp",
            SourceKind::HuggingFace => "huggingface",
            SourceKind::LmStudio => "lmstudio",
            SourceKind::Jan => "jan",
            SourceKind::Directory => "dir",
            SourceKind::File => "file",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    /// Opaque, stable id (used by the web API instead of paths).
    pub id: String,
    pub name: String,
    pub source: SourceKind,
    /// Path of the main weights GGUF.
    pub path: PathBuf,
    pub size: u64,
    /// Content digest when known (Ollama blob sha256), else a path hash.
    pub digest: String,
    pub ollama: Option<OllamaInfo>,
    pub projector: Option<PathBuf>,
    /// Other entries pointing at the same weights file.
    pub shares_weights_with: Vec<String>,
    pub warnings: Vec<String>,
}

impl ModelEntry {
    pub fn from_file(path: &Path, source: SourceKind, name: String) -> Self {
        let canon = canonical(path);
        let size = std::fs::metadata(&canon).map(|m| m.len()).unwrap_or(0);
        let key = canon.to_string_lossy().to_string();
        Self {
            id: short_id(&format!("file:{key}")),
            name,
            source,
            digest: format!("path-{:016x}", crate::util::fnv1a(key.as_bytes())),
            path: canon,
            size,
            ollama: None,
            projector: None,
            shares_weights_with: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

/// `list` row: entry plus a header-only read.
#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub id: String,
    pub name: String,
    pub source: SourceKind,
    pub architecture: Option<String>,
    pub params: Option<u64>,
    pub quantization: Option<String>,
    pub context_length: Option<u64>,
    pub size: u64,
    pub path: PathBuf,
    pub has_projector: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DiscoveryOptions {
    pub extra_dirs: Vec<PathBuf>,
    /// Skip the platform default locations (only scan `extra_dirs`).
    pub no_defaults: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Library {
    pub entries: Vec<ModelEntry>,
    pub warnings: Vec<String>,
    #[serde(skip)]
    options: DiscoveryOptions,
    #[serde(skip)]
    fingerprint: u64,
}

pub(crate) fn canonical(p: &Path) -> PathBuf {
    let c = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    // Strip the Windows verbatim prefix so paths stay readable.
    let mut s = c.to_string_lossy().to_string();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        s = format!(r"\\{rest}");
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        s = rest.to_string();
    }
    if s.starts_with(r"\\wsl$\") {
        s = s.replacen(r"\\wsl$\", r"\\wsl.localhost\", 1);
    }
    PathBuf::from(s)
}

/// Quantization label: llama.cpp file type if declared, else the dominant tensor type.
pub fn quantization_label(g: &GgufFile) -> Option<String> {
    if let Some(ft) = g.get_u64("general.file_type") {
        if let Some(n) = file_type_name(ft) {
            return Some(n.to_string());
        }
    }
    let mut by_type: HashMap<String, u64> = HashMap::new();
    for t in &g.tensors {
        if t.dims.len() >= 2 {
            *by_type.entry(t.ggml_type.name().into_owned()).or_default() += t.n_bytes;
        }
    }
    by_type.into_iter().max_by_key(|(_, b)| *b).map(|(n, _)| n)
}

/// Directories from `NEURONDUMP_MODEL_DIRS` (OS path-list separator: `;` on Windows, `:` elsewhere).
pub fn env_model_dirs() -> Vec<PathBuf> {
    std::env::var_os("NEURONDUMP_MODEL_DIRS")
        .map(|v| std::env::split_paths(&v).filter(|p| !p.as_os_str().is_empty()).collect())
        .unwrap_or_default()
}
