//! Token inspector, static part (no inference): tokenization with the chat template,
//! per-token embedding row statistics and nearest neighbours, and the 3D embedding map
//! (token_embd rows projected with PCA).

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use neurondump_gguf::{GgufFile, TensorInfo};

use crate::chat::{render_chat, ChatRender, ChatRequest};
use crate::model::ModelHandle;
use crate::tokenizer::{Encoder, TokenClass, TokenType, Vocab};
use crate::util::{check_cancel, Progress, SCHEMA_VERSION, Instant};

mod neighbors;
mod tokenize;
mod embedding_map;
pub use neighbors::*;
pub use tokenize::*;
pub use embedding_map::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
    /// Wrap the text in the model's chat template first.
    #[serde(default)]
    pub chat: bool,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    /// Nearest neighbours per token (0 = skip the embedding scan).
    #[serde(default = "default_neighbors")]
    pub neighbors: usize,
}

fn default_neighbors() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbor {
    pub id: u32,
    pub text: String,
    pub cosine: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenRow {
    pub index: usize,
    pub id: u32,
    pub raw: String,
    pub text: String,
    pub class: TokenClass,
    pub token_type: TokenType,
    /// Control / user-defined token (chat markers, BOS, ...).
    pub special: bool,
    /// Added by the chat template (outside the user's text) or by the tokenizer (BOS/EOS).
    pub added: bool,
    /// Byte range in `full_text` when it could be aligned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<(usize, usize)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_norm: Option<f32>,
    /// Norm of the token's row in the output head (the tied embedding row when tied).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_norm: Option<f32>,
    /// The model can predict this token (an output row exists and it is not banned).
    pub in_output: bool,
    pub banned: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub neighbors: Vec<Neighbor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizeReport {
    pub schema_version: u32,
    pub model: String,
    pub tokenizer: String,
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat: Option<ChatRender>,
    pub full_text: String,
    pub tokens: Vec<TokenRow>,
    pub tied_embeddings: bool,
    pub has_output_head: bool,
    pub embedding_tensor: Option<String>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}
