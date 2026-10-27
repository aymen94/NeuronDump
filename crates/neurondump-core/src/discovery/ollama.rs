//! Ollama model store: `manifests[-v2]/<registry>/<namespace>/<model>/<tag>` + `blobs/`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::{canonical, ModelEntry, SourceKind};
use crate::util::short_id;

pub const DEFAULT_REGISTRY: &str = "registry.ollama.ai";
const REGISTRY_ALIASES: &[&str] = &["registry.ollama.ai", "ollama.com", "www.ollama.com"];

#[derive(Debug, Clone, Serialize)]
pub struct OllamaLayer {
    pub media_type: String,
    /// Short kind: model, projector, template, system, params, license, adapter, messages, ...
    pub kind: String,
    pub digest: String,
    pub size: u64,
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaInfo {
    pub registry: String,
    pub namespace: String,
    pub model: String,
    pub tag: String,
    pub manifest_path: PathBuf,
    pub models_dir: PathBuf,
    pub layers: Vec<OllamaLayer>,
}

impl OllamaInfo {
    /// Canonical lowercase `registry/namespace/model:tag`.
    pub fn full_name(&self) -> String {
        format!("{}/{}/{}:{}", self.registry, self.namespace, self.model, self.tag).to_lowercase()
    }

    /// Short display name in Ollama's style (`llama3.2:1b`, `ns/model:tag`, `host/ns/model:tag`).
    pub fn display_name(&self) -> String {
        if self.registry == DEFAULT_REGISTRY {
            if self.namespace == "library" {
                format!("{}:{}", self.model, self.tag)
            } else {
                format!("{}/{}:{}", self.namespace, self.model, self.tag)
            }
        } else {
            format!("{}/{}/{}:{}", self.registry, self.namespace, self.model, self.tag)
        }
    }

    pub fn layer(&self, kind: &str) -> Option<&OllamaLayer> {
        self.layers.iter().find(|l| l.kind == kind)
    }

    pub fn layers_of(&self, kind: &str) -> impl Iterator<Item = &OllamaLayer> {
        let kind = kind.to_string();
        self.layers.iter().filter(move |l| l.kind == kind)
    }
}

fn normalize_registry(r: &str) -> String {
    let r = r.to_lowercase();
    if REGISTRY_ALIASES.contains(&r.as_str()) {
        DEFAULT_REGISTRY.to_string()
    } else {
        r
    }
}

/// Normalizes a user query to the canonical form used by [`OllamaInfo::full_name`].
pub fn normalize_name(q: &str) -> String {
    let q = q.trim().trim_start_matches("https://").trim_start_matches("http://");
    let (base, tag) = match q.rfind(':') {
        Some(i) if !q[i..].contains('/') => (&q[..i], &q[i + 1..]),
        _ => (q, "latest"),
    };
    let parts: Vec<&str> = base.split('/').filter(|p| !p.is_empty()).collect();
    let (registry, namespace, model) = match parts.as_slice() {
        [m] => (DEFAULT_REGISTRY.to_string(), "library".to_string(), m.to_string()),
        [ns, m] => (DEFAULT_REGISTRY.to_string(), ns.to_string(), m.to_string()),
        [host @ .., ns, m] => (normalize_registry(&host.join("/")), ns.to_string(), m.to_string()),
        [] => (DEFAULT_REGISTRY.to_string(), "library".to_string(), String::new()),
    };
    format!("{registry}/{namespace}/{model}:{tag}").to_lowercase()
}

fn inspect_and_push_volume(path: &Path, out: &mut Vec<PathBuf>) -> bool {
    let m1 = path.join("_data").join("models");
    let m2 = path.join("_data");
    if m1.join("manifests").is_dir() || m1.join("manifests-v2").is_dir() {
        out.push(m1);
        true
    } else if m2.join("manifests").is_dir() || m2.join("manifests-v2").is_dir() {
        out.push(m2);
        true
    } else {
        false
    }
}

fn add_docker_ollama_dirs(out: &mut Vec<PathBuf>) {
    let custom_vol = std::env::var("OLLAMA_DOCKER_VOLUME").ok();

    #[cfg(target_os = "windows")]
    {
        let docker_volume_roots = [
            r"\\wsl.localhost\docker-desktop\mnt\docker-desktop-disk\data\docker\volumes",
            r"\\wsl$\docker-desktop\mnt\docker-desktop-disk\data\docker\volumes",
            r"\\wsl.localhost\docker-desktop-data\mnt\docker-desktop-disk\data\docker\volumes",
            r"\\wsl$\docker-desktop-data\mnt\docker-desktop-disk\data\docker\volumes",
            r"\\wsl.localhost\docker-desktop-data\version-pack-data\community\docker\volumes",
            r"\\wsl$\docker-desktop-data\version-pack-data\community\docker\volumes",
        ];

        for root_str in docker_volume_roots {
            let root = Path::new(root_str);
            if !root.is_dir() {
                continue;
            }
            if let Some(ref vol) = custom_vol {
                out.push(root.join(vol).join("_data").join("models"));
                out.push(root.join(vol).join("_data"));
            }

            let mut matched = false;
            if let Ok(entries) = std::fs::read_dir(root) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        // Check if this volume actually contains Ollama models, regardless of its name
                        if inspect_and_push_volume(&path, out) {
                            matched = true;
                        } else {
                            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                            if name.eq_ignore_ascii_case("ollama") || name.to_lowercase().contains("ollama") {
                                out.push(path.join("_data").join("models"));
                                out.push(path.join("_data"));
                                matched = true;
                            }
                        }
                    }
                }
            }
            if !matched {
                out.push(root.join("ollama").join("_data").join("models"));
                out.push(root.join("ollama").join("_data"));
            }
            // Once we find a working Docker volume root, stop checking aliases (e.g. \\wsl$ vs \\wsl.localhost)
            break;
        }
    }

    #[cfg(target_os = "linux")]
    {
        let root = Path::new("/var/lib/docker/volumes");
        if root.is_dir() {
            if let Some(ref vol) = custom_vol {
                out.push(root.join(vol).join("_data").join("models"));
                out.push(root.join(vol).join("_data"));
            }
            let mut matched = false;
            if let Ok(entries) = std::fs::read_dir(root) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if inspect_and_push_volume(&path, out) {
                            matched = true;
                        } else {
                            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                            if name.eq_ignore_ascii_case("ollama") || name.to_lowercase().contains("ollama") {
                                out.push(path.join("_data").join("models"));
                                out.push(path.join("_data"));
                                matched = true;
                            }
                        }
                    }
                }
            }
            if !matched {
                out.push(root.join("ollama").join("_data").join("models"));
                out.push(root.join("ollama").join("_data"));
            }
        }
    }
}

/// All Ollama model directories that exist on this machine.
pub fn ollama_model_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("OLLAMA_MODELS") {
        if !p.is_empty() {
            out.push(PathBuf::from(p));
        }
    }
    if let Some(home) = dirs::home_dir() {
        out.push(home.join(".ollama").join("models"));
    }
    if cfg!(target_os = "linux") {
        out.push(PathBuf::from("/usr/share/ollama/.ollama/models"));
        out.push(PathBuf::from("/var/lib/ollama/.ollama/models"));
    }
    add_docker_ollama_dirs(&mut out);

    let mut seen = HashSet::new();
    out.into_iter()
        .filter(|p| p.join("manifests").is_dir() || p.join("manifests-v2").is_dir())
        .filter(|p| seen.insert(canonical(p)))
        .collect()
}

fn layer_kind(media_type: &str) -> String {
    match media_type.strip_prefix("application/vnd.ollama.image.") {
        Some(k) => k.to_string(),
        None if media_type.contains("manifest") => "manifest-ref".to_string(),
        None => format!("other:{media_type}"),
    }
}

fn collect_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 8 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for ent in rd.flatten() {
        let p = ent.path();
        if p.is_dir() {
            collect_files(&p, depth + 1, out);
        } else {
            out.push(p);
        }
    }
}

pub(super) fn scan(models_dir: &Path, entries: &mut Vec<ModelEntry>, warnings: &mut Vec<String>) {
    let blobs = models_dir.join("blobs");
    let mut seen_names: HashSet<String> = HashSet::new();
    // Newer Ollama writes manifests-v2 (host ollama.com) and keeps v1 for compatibility.
    for sub in ["manifests-v2", "manifests"] {
        let root = models_dir.join(sub);
        let mut files = Vec::new();
        collect_files(&root, 0, &mut files);
        files.sort();
        for mf in files {
            let Ok(rel) = mf.strip_prefix(&root) else { continue };
            let comps: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
            let (registry, namespace, model, tag) = match comps.as_slice() {
                [m, t] => (DEFAULT_REGISTRY.to_string(), "library".to_string(), m.clone(), t.clone()),
                [ns, m, t] => (DEFAULT_REGISTRY.to_string(), ns.clone(), m.clone(), t.clone()),
                [host @ .., ns, m, t] => (normalize_registry(&host.join("/")), ns.clone(), m.clone(), t.clone()),
                _ => continue,
            };
            let mut info = OllamaInfo {
                registry,
                namespace,
                model,
                tag,
                manifest_path: mf.clone(),
                models_dir: models_dir.to_path_buf(),
                layers: Vec::new(),
            };
            if seen_names.contains(&info.full_name()) {
                continue;
            }
            let text = match std::fs::read_to_string(&mf) {
                Ok(t) => t,
                Err(e) => {
                    warnings.push(format!("cannot read manifest {}: {e}", mf.display()));
                    continue;
                }
            };
            let mut json: serde_json::Value = match serde_json::from_str(&text) {
                Ok(j) => j,
                Err(e) => {
                    warnings.push(format!("invalid manifest {}: {e}", mf.display()));
                    continue;
                }
            };
            // Manifest lists (e.g. MLX + GGUF variants): follow the GGUF / llama.cpp one.
            if let Some(list) = json["manifests"].as_array() {
                let pick = list
                    .iter()
                    .find(|m| m["format"].as_str() == Some("gguf") || m["runner"].as_str() == Some("llamacpp"))
                    .and_then(|m| m["digest"].as_str())
                    .map(|d| blobs.join(d.replace(':', "-")));
                match pick.and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()) {
                    Some(inner) => json = inner,
                    None => {
                        warnings.push(format!("{}: manifest list has no local GGUF variant", info.display_name()));
                        continue;
                    }
                }
            }
            let mut entry_warnings = Vec::new();
            for l in json["layers"].as_array().into_iter().flatten() {
                let media_type = l["mediaType"].as_str().unwrap_or_default().to_string();
                let digest = l["digest"].as_str().unwrap_or_default().to_string();
                let blob = blobs.join(digest.replace(':', "-"));
                let path = if blob.is_file() { Some(blob) } else { None };
                let kind = layer_kind(&media_type);
                if path.is_none() && !kind.starts_with("manifest") {
                    entry_warnings.push(format!("missing blob for {kind} layer {digest}"));
                }
                info.layers.push(OllamaLayer { media_type, kind, digest, size: l["size"].as_u64().unwrap_or(0), path });
            }
            let Some(model_layer) = info.layer("model").cloned() else {
                warnings.push(format!("{}: manifest has no model layer", info.display_name()));
                continue;
            };
            let Some(path) = model_layer.path.clone() else {
                warnings.push(format!("{}: weights blob {} is missing", info.display_name(), model_layer.digest));
                continue;
            };
            let name = info.display_name();
            seen_names.insert(info.full_name());
            entries.push(ModelEntry {
                id: short_id(&format!("ollama:{}", info.full_name())),
                name,
                source: SourceKind::Ollama,
                size: model_layer.size,
                digest: model_layer.digest.replace("sha256:", "sha256-"),
                path: canonical(&path),
                projector: info.layer("projector").and_then(|l| l.path.clone()),
                ollama: Some(info),
                shares_weights_with: Vec::new(),
                warnings: entry_warnings,
            });
        }
    }
}
