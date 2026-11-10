//! Job parameter types (prompt, journey, generate, compare, bake, ablate) and the edit view.

use super::*;

/// The prompt part shared by journey / generate / compare jobs.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PromptParams {
    #[serde(default)]
    pub prompt: String,
    /// Wrap the prompt in the model's chat template.
    #[serde(default)]
    pub chat: bool,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct JourneyParams {
    #[serde(flatten)]
    pub prompt: PromptParams,
    /// Position to inspect; negative counts from the end (default -1).
    #[serde(default)]
    pub token: Option<i64>,
    /// Token ids appended after the prompt (generated tokens from a timeline).
    #[serde(default)]
    pub append: Vec<u32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Edits to apply (default: the model's current edit stack).
    #[serde(default)]
    pub edits: Option<Vec<Edit>>,
    /// Ignore the edits.
    #[serde(default)]
    pub baseline: bool,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GenerateParams {
    #[serde(flatten)]
    pub prompt: PromptParams,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub explain: bool,
    /// Sampling overrides (otherwise greedy, or the stack's `sampling` edit).
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub sample_top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Alternatives recorded per step.
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub edits: Option<Vec<Edit>>,
    #[serde(default)]
    pub baseline: bool,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CompareParams {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub prompts: Vec<String>,
    #[serde(default)]
    pub chat: bool,
    #[serde(default)]
    pub corpus: Vec<String>,
    #[serde(default)]
    pub corpus_file: Option<PathBuf>,
    /// Tokens generated per prompt on both sides.
    #[serde(default)]
    pub tokens: Option<usize>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub edits: Option<Vec<Edit>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompareResult {
    pub edits: Vec<Edit>,
    pub compiled: Compiled,
    pub simulation: crate::trace::SimReport,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BakeParams {
    #[serde(default)]
    pub edits: Option<Vec<Edit>>,
    /// Plain file name (written to `<output_dir>/bake/`) or, from the CLI, a path.
    pub out: String,
}

/// The edit stack as the front ends show it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EditView {
    pub model: String,
    pub digest: String,
    pub edits: Vec<Edit>,
    pub can_undo: bool,
    pub can_redo: bool,
    pub active: usize,
    /// Compilation of the enabled edits (None when it failed, see `error`).
    pub compiled: Option<Compiled>,
    pub error: Option<String>,
    pub snapshots: Vec<String>,
}

/// Parameters of a `causal` job: the clean prompt, a corrupted one, the measured token.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CausalParams {
    #[serde(flatten)]
    pub prompt: PromptParams,
    /// The corrupted prompt (templated like the clean one).
    pub corrupt: String,
    /// Token text whose probability is measured (its first token); default: the clean top-1.
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub component: crate::trace::CausalComponent,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub edits: Option<Vec<Edit>>,
    #[serde(default)]
    pub baseline: bool,
}

/// Parameters of a `heads` job.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HeadsParams {
    #[serde(default)]
    pub samples: Option<usize>,
    #[serde(default)]
    pub length: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub edits: Option<Vec<Edit>>,
    #[serde(default)]
    pub baseline: bool,
}

/// Parameters of a `similarity` job.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SimilarityParams {
    #[serde(default)]
    pub corpus: Vec<String>,
    #[serde(default)]
    pub corpus_file: Option<PathBuf>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_positions: Option<usize>,
    #[serde(default)]
    pub edits: Option<Vec<Edit>>,
    #[serde(default)]
    pub baseline: bool,
}

/// Parameters of an `ablate` job.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AblateParams {
    #[serde(default)]
    pub cut: Vec<String>,
    #[serde(default)]
    pub force: bool,
    /// Static impact only (no inference).
    #[serde(default, rename = "static")]
    pub static_only: bool,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub prompts: Vec<String>,
    #[serde(default)]
    pub corpus: Vec<String>,
    #[serde(default)]
    pub corpus_file: Option<PathBuf>,
    /// Greedy tokens to generate per prompt.
    #[serde(default)]
    pub tokens: Option<usize>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}
