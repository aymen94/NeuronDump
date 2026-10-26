//! Trace request and report: per-layer logit lens, norms and token contributions.

use super::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceRequest {
    #[serde(default)]
    pub prompt: Option<String>,
    /// Corpus mode: each entry is traced and statistics are aggregated.
    #[serde(default)]
    pub corpus: Vec<String>,
    /// Corpus file (read by core before tracing).
    #[serde(default)]
    pub corpus_file: Option<std::path::PathBuf>,
    /// Number of logit-lens tokens / top neurons per layer.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Truncate inputs to this many tokens.
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensToken {
    pub id: u32,
    pub text: String,
    pub prob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTrace {
    pub index: u32,
    /// Mean L2 norm of the residual stream after this layer (over positions).
    pub hidden_norm: f32,
    /// Cosine similarity between the layer's input and output hidden states
    /// (≈1 means the layer barely changes the residual stream).
    pub cos_prev: Option<f32>,
    /// Mean attention entropy per head (nats), last position or averaged.
    pub head_entropy: Vec<f32>,
    /// Top activated FFN neurons (index, mean |activation|).
    pub top_neurons: Vec<(u32, f32)>,
    /// MoE router: experts chosen for the last position.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub router: Vec<u32>,
    /// Logit lens at the last position.
    pub logit_lens: Vec<LensToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenContribution {
    pub id: u32,
    pub text: String,
    /// Cosine similarity of the token's final hidden state with the pooled embedding.
    pub contribution: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub backend: String,
    pub supported: bool,
    pub tokens: Vec<(u32, String)>,
    pub layers: Vec<LayerTrace>,
    /// Final next-token prediction (decoders).
    pub prediction: Vec<LensToken>,
    /// Embedding models: per-token contribution to the pooled embedding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding_contributions: Vec<TokenContribution>,
    /// Corpus mode: number of texts aggregated (layer values are means).
    pub samples: usize,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}
