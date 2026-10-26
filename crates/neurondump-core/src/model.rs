//! An opened model: weights GGUF plus Ollama side layers.

use std::sync::Arc;

use anyhow::Result;
use serde::Serialize;

use neurondump_gguf::GgufFile;

use crate::discovery::ModelEntry;

#[derive(Debug, Clone, Default, Serialize)]
pub struct OllamaExtras {
    pub template: Option<String>,
    pub system: Option<String>,
    pub params: Option<serde_json::Value>,
    pub licenses: Vec<String>,
    pub messages: Option<serde_json::Value>,
    pub adapters: Vec<String>,
}

pub struct ModelHandle {
    pub entry: ModelEntry,
    pub gguf: Arc<GgufFile>,
    pub ollama: OllamaExtras,
}

#[cfg(feature = "native")]
fn read_small(path: &std::path::Path, limit: u64) -> Option<String> {
    let md = std::fs::metadata(path).ok()?;
    if md.len() > limit {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

impl ModelHandle {
    /// A handle over an already parsed file (e.g. read lazily in the browser).
    pub fn from_gguf(entry: ModelEntry, gguf: GgufFile) -> Self {
        Self { entry, gguf: Arc::new(gguf), ollama: OllamaExtras::default() }
    }

    #[cfg(feature = "native")]
    pub fn open(entry: ModelEntry) -> Result<Self> {
        use anyhow::Context;
        let gguf = GgufFile::open(&entry.path).with_context(|| format!("parsing {}", entry.path.display()))?;
        let mut ollama = OllamaExtras::default();
        if let Some(info) = &entry.ollama {
            for l in &info.layers {
                let Some(p) = &l.path else { continue };
                match l.kind.as_str() {
                    "template" => ollama.template = read_small(p, 1 << 20),
                    "system" => ollama.system = read_small(p, 1 << 20),
                    "params" => ollama.params = read_small(p, 1 << 20).and_then(|s| serde_json::from_str(&s).ok()),
                    "messages" => ollama.messages = read_small(p, 4 << 20).and_then(|s| serde_json::from_str(&s).ok()),
                    "license" => ollama.licenses.extend(read_small(p, 4 << 20)),
                    "adapter" => ollama.adapters.push(p.to_string_lossy().to_string()),
                    _ => {}
                }
            }
        }
        Ok(Self { entry, gguf: Arc::new(gguf), ollama })
    }

    pub fn architecture(&self) -> &str {
        self.gguf.architecture().unwrap_or("unknown")
    }

    pub fn open_projector(&self) -> Option<Result<GgufFile>> {
        #[cfg(feature = "native")]
        return self.entry.projector.as_ref().map(|p| GgufFile::open(p).map_err(Into::into));
        #[cfg(not(feature = "native"))]
        None
    }
}
