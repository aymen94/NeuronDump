//! Finding local GGUF models: Ollama stores, llama.cpp / Hugging Face / LM Studio caches
//! and user-configured directories.

mod files;
mod ollama;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::Serialize;

use neurondump_gguf::{file_type_name, GgufFile, ParseOptions};

use crate::util::{edit_distance, short_id};

pub use ollama::{ollama_model_dirs, OllamaInfo, OllamaLayer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Ollama,
    LlamaCpp,
    HuggingFace,
    LmStudio,
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
    let s = c.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => c,
    }
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

impl Library {
    pub fn discover(options: DiscoveryOptions) -> Self {
        let mut entries = Vec::new();
        let mut warnings = Vec::new();
        if !options.no_defaults {
            for dir in ollama_model_dirs() {
                ollama::scan(&dir, &mut entries, &mut warnings);
            }
            files::scan_default_caches(&mut entries, &mut warnings);
        }
        for dir in &options.extra_dirs {
            if dir.join("manifests").is_dir() && dir.join("blobs").is_dir() {
                ollama::scan(dir, &mut entries, &mut warnings);
            } else {
                files::scan_dir(dir, SourceKind::Directory, &mut entries, &mut warnings);
            }
        }
        dedup(&mut entries);
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let fingerprint = Self::compute_fingerprint(&options);
        Self { entries, warnings, options, fingerprint }
    }

    pub fn rescan(&mut self) {
        let extra: Vec<ModelEntry> =
            self.entries.iter().filter(|e| e.source == SourceKind::File).cloned().collect();
        *self = Self::discover(self.options.clone());
        for e in extra {
            if !self.entries.iter().any(|x| x.path == e.path) {
                self.entries.push(e);
            }
        }
    }

    /// Cheap change detector over Ollama manifest dirs and scan dirs (count + newest mtime).
    fn compute_fingerprint(options: &DiscoveryOptions) -> u64 {
        let mut dirs: Vec<PathBuf> = Vec::new();
        if !options.no_defaults {
            for d in ollama_model_dirs() {
                dirs.push(d.join("manifests"));
                dirs.push(d.join("manifests-v2"));
            }
        }
        dirs.extend(options.extra_dirs.iter().cloned());
        let mut count = 0u64;
        let mut newest = 0u64;
        for d in dirs {
            walk_mtimes(&d, 0, &mut count, &mut newest);
        }
        count.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ newest
    }

    /// Rescans if anything changed on disk since the last scan. Returns true if it did.
    pub fn refresh_if_changed(&mut self) -> bool {
        if Self::compute_fingerprint(&self.options) != self.fingerprint {
            self.rescan();
            true
        } else {
            false
        }
    }

    pub fn by_id(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Resolves `llama3.2:1b`, `ns/model:tag`, `host/ns/model:tag`, an id, a file
    /// stem, or a path to a `.gguf` (which is added to the library).
    pub fn resolve(&mut self, query: &str) -> Result<ModelEntry> {
        let q = query.trim();
        if let Some(e) = self.find(q) {
            return Ok(e.clone());
        }
        let p = Path::new(q);
        if p.is_file() {
            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| q.to_string());
            let e = ModelEntry::from_file(p, SourceKind::File, name);
            if let Some(existing) = self.entries.iter().find(|x| x.path == e.path) {
                return Ok(existing.clone());
            }
            self.entries.push(e.clone());
            return Ok(e);
        }
        let mut suggestions: Vec<(usize, &str)> = self
            .entries
            .iter()
            .map(|e| (edit_distance(&e.name.to_lowercase(), &q.to_lowercase()), e.name.as_str()))
            .collect();
        suggestions.sort();
        let close: Vec<&str> = suggestions.iter().filter(|(d, _)| *d <= 4.max(q.len() / 3)).take(3).map(|s| s.1).collect();
        if close.is_empty() {
            bail!("no model named {q:?} (run `neurondump list` to see available models)");
        }
        bail!("no model named {q:?}; did you mean: {}?", close.join(", "))
    }

    fn find(&self, q: &str) -> Option<&ModelEntry> {
        if let Some(e) = self.entries.iter().find(|e| e.id == q || e.name == q) {
            return Some(e);
        }
        let norm = ollama::normalize_name(q);
        if let Some(e) = self
            .entries
            .iter()
            .find(|e| e.ollama.as_ref().is_some_and(|o| o.full_name() == norm))
        {
            return Some(e);
        }
        let lower = q.to_lowercase();
        self.entries.iter().find(|e| {
            e.name.to_lowercase() == lower
                || e.path.file_stem().is_some_and(|s| s.to_string_lossy().to_lowercase() == lower)
        })
    }

    /// Header-only summaries of all entries, read in parallel.
    pub fn summaries(&self) -> Vec<ModelSummary> {
        self.entries.par_iter().map(summarize).collect()
    }
}

pub fn summarize(e: &ModelEntry) -> ModelSummary {
    let mut s = ModelSummary {
        id: e.id.clone(),
        name: e.name.clone(),
        source: e.source,
        architecture: None,
        params: None,
        quantization: None,
        context_length: None,
        size: e.size,
        path: e.path.clone(),
        has_projector: e.projector.is_some(),
        error: None,
    };
    match GgufFile::open_with(&e.path, ParseOptions::header_only()) {
        Ok(g) => {
            s.architecture = g.architecture().map(str::to_string);
            s.params = Some(g.total_params());
            s.quantization = quantization_label(&g);
            s.context_length = g.arch_key("context_length").and_then(|v| v.as_u64());
        }
        Err(err) => s.error = Some(err.to_string()),
    }
    s
}

fn walk_mtimes(dir: &Path, depth: usize, count: &mut u64, newest: &mut u64) {
    if depth > 6 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for ent in rd.flatten() {
        let Ok(md) = ent.metadata() else { continue };
        *count += 1;
        if let Ok(m) = md.modified() {
            let secs = m.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            *newest = (*newest).max(secs);
        }
        if md.is_dir() {
            walk_mtimes(&ent.path(), depth + 1, count, newest);
        }
    }
}

/// Removes duplicate discoveries of the same entry and links entries sharing weights.
fn dedup(entries: &mut Vec<ModelEntry>) {
    let mut seen: HashMap<(String, PathBuf), usize> = HashMap::new();
    let mut keep = Vec::with_capacity(entries.len());
    for e in entries.drain(..) {
        let key = (e.name.clone(), e.path.clone());
        if seen.contains_key(&key) {
            continue;
        }
        // Plain files that are already known through another source (e.g. Ollama blob).
        if e.ollama.is_none() && keep.iter().any(|k: &ModelEntry| k.path == e.path) {
            continue;
        }
        seen.insert(key, keep.len());
        keep.push(e);
    }
    let mut by_path: HashMap<PathBuf, Vec<String>> = HashMap::new();
    for e in &keep {
        by_path.entry(e.path.clone()).or_default().push(e.name.clone());
    }
    for e in &mut keep {
        if let Some(names) = by_path.get(&e.path) {
            e.shares_weights_with = names.iter().filter(|n| **n != e.name).cloned().collect();
        }
    }
    *entries = keep;
}
