//! Plain `.gguf` files: llama.cpp cache, Hugging Face hub cache, LM Studio, Jan, scan dirs.

use std::path::{Path, PathBuf};

use super::{ModelEntry, SourceKind};

const MAX_DEPTH: usize = 10;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
/// `~/.cache/huggingface/hub` (same on every OS, per the huggingface_hub docs).
fn hf_hub_dir() -> Option<PathBuf> {
    if let Some(p) = env_path("HF_HUB_CACHE") {
        return Some(p);
    }
    if let Some(p) = env_path("HF_HOME") {
        return Some(p.join("hub"));
    }
    if let Some(p) = env_path("XDG_CACHE_HOME") {
        return Some(p.join("huggingface").join("hub"));
    }
    dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub"))
}

/// LM Studio home is `~/.lmstudio` on every OS unless `~/.lmstudio-home-pointer` names
/// another folder; pre-0.3 versions used `~/.cache/lm-studio`.
fn lmstudio_dirs() -> Vec<PathBuf> {
    let Some(h) = dirs::home_dir() else { return Vec::new() };
    let mut v = Vec::new();
    if let Ok(p) = std::fs::read_to_string(h.join(".lmstudio-home-pointer")) {
        if !p.trim().is_empty() {
            v.push(PathBuf::from(p.trim()).join("models"));
        }
    }
    v.push(h.join(".lmstudio").join("models"));
    v.push(h.join(".cache").join("lm-studio").join("models"));
    v
}

/// Jan data folder: `%APPDATA%\Jan\data`, `~/Library/Application Support/Jan/data`,
/// `~/.local/share/Jan/data` (some Linux builds use `~/.config/Jan/data`). Models live in
/// `llamacpp/models` (Jan >= 0.6) or `models` (older); `~/jan/models` is the legacy layout.
fn jan_dirs() -> Vec<PathBuf> {
    let mut data = Vec::new();
    if let Some(d) = dirs::data_dir() {
        data.push(d.join("Jan").join("data"));
    }
    if cfg!(target_os = "linux") {
        if let Some(d) = dirs::config_dir() {
            data.push(d.join("Jan").join("data"));
        }
    }
    let mut v: Vec<PathBuf> =
        data.iter().flat_map(|d| [d.join("llamacpp").join("models"), d.join("models")]).collect();
    if let Some(h) = dirs::home_dir() {
        v.push(h.join("jan").join("models"));
    }
    v
}

fn llama_cache_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = env_path("LLAMA_CACHE") {
        v.push(p);
    }
    if let Some(h) = dirs::home_dir() {
        v.push(h.join(".cache").join("llama.cpp"));
    }
    if let Some(c) = dirs::cache_dir() {
        v.push(c.join("llama.cpp"));
    }
    v
}

pub(super) fn scan_default_caches(entries: &mut Vec<ModelEntry>, warnings: &mut Vec<String>) {
    for d in llama_cache_dirs() {
        scan_dir(&d, SourceKind::LlamaCpp, entries, warnings);
    }
    if let Some(d) = hf_hub_dir() {
        scan_dir(&d, SourceKind::HuggingFace, entries, warnings);
    }
    for d in lmstudio_dirs() {
        scan_dir(&d, SourceKind::LmStudio, entries, warnings);
    }
    for d in jan_dirs() {
        scan_dir(&d, SourceKind::Jan, entries, warnings);
    }
}

/// Display name for a file found under `root`.
fn name_for(root: &Path, path: &Path, source: SourceKind) -> String {
    let file = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    if source == SourceKind::HuggingFace {
        // hub/models--org--repo/snapshots/<rev>/<file>
        if let Ok(rel) = path.strip_prefix(root) {
            if let Some(first) = rel.components().next() {
                let repo = first.as_os_str().to_string_lossy();
                if let Some(r) = repo.strip_prefix("models--") {
                    return format!("hf:{}/{file}", r.replace("--", "/"));
                }
            }
        }
    }
    match path.strip_prefix(root) {
        Ok(rel) if rel.components().count() > 1 => rel.to_string_lossy().replace('\\', "/"),
        _ => file,
    }
}

pub(super) fn scan_dir(root: &Path, source: SourceKind, entries: &mut Vec<ModelEntry>, warnings: &mut Vec<String>) {
    if !root.is_dir() {
        if source == SourceKind::Directory {
            warnings.push(format!("scan dir {} does not exist", root.display()));
        }
        return;
    }
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let first = entries.len();
        let mut projector = None;
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                // Never treat an Ollama blob store as loose files.
                if depth < MAX_DEPTH && p.file_name().is_none_or(|n| n != "blobs" || source != SourceKind::Directory) {
                    stack.push((p, depth + 1));
                }
            } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
                if p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().starts_with("mmproj")) {
                    projector = Some(p);
                    continue;
                }
                let mut e = ModelEntry::from_file(&p, source, name_for(root, &p, source));
                if let Some(stem) = p.file_stem().map(|s| s.to_string_lossy().to_string()) {
                    if stem.contains("-00001-of-") {
                        e.warnings.push("split GGUF: only the first shard is analyzed".into());
                    }
                }
                entries.push(e);
            }
        }
        // A vision projector next to the weights (LM Studio / Jan / HF layout) belongs to them;
        // a lone one is still listed so it can be inspected.
        if let Some(p) = projector {
            if entries.len() > first {
                for e in &mut entries[first..] {
                    e.projector = Some(super::canonical(&p));
                }
            } else {
                entries.push(ModelEntry::from_file(&p, source, name_for(root, &p, source)));
            }
        }
    }
}
