//! Plain `.gguf` files: llama.cpp cache, Hugging Face hub cache, LM Studio, scan dirs.

use std::path::{Path, PathBuf};

use super::{ModelEntry, SourceKind};

const MAX_DEPTH: usize = 10;

fn hf_hub_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(p).join("hub"));
    }
    dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub"))
}

fn llama_cache_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("LLAMA_CACHE") {
        v.push(PathBuf::from(p));
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
    if let Some(h) = dirs::home_dir() {
        scan_dir(&h.join(".lmstudio").join("models"), SourceKind::LmStudio, entries, warnings);
        scan_dir(&h.join(".cache").join("lm-studio").join("models"), SourceKind::LmStudio, entries, warnings);
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
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                // Never treat an Ollama blob store as loose files.
                if depth < MAX_DEPTH && p.file_name().is_none_or(|n| n != "blobs" || source != SourceKind::Directory) {
                    stack.push((p, depth + 1));
                }
            } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
                let mut e = ModelEntry::from_file(&p, source, name_for(root, &p, source));
                if let Some(stem) = p.file_stem().map(|s| s.to_string_lossy().to_string()) {
                    if stem.contains("-00001-of-") {
                        e.warnings.push("split GGUF: only the first shard is analyzed".into());
                    }
                }
                entries.push(e);
            }
        }
    }
}
