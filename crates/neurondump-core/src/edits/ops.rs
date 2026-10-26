//! Edit operations, edits and the portable edits.json file; resolving a token argument.

use super::*;

/// One change. Node ids are graph ids (`layer.7`, `layer.7.attn.head.3`,
/// `layer.7.ffn.n.0-63`, `layer.7.ffn.neuron.42`, `layer.7.moe.expert.3`, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum EditOp {
    /// Multiply a node's contribution to the residual stream.
    Scale { target: String, factor: f32 },
    /// Zero a node (tensors, sections and token ranges too).
    Zero { target: String },
    /// Replace a layer by the identity.
    Skip { layer: u32 },
    /// The cut of `neurondump ablate` (any ids it accepts).
    Ablate { cut: Vec<String> },
    /// Scale a token's embedding row, or replace it with another token's row.
    Embed {
        token: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scale: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replace_with: Option<u32>,
    },
    /// The token can no longer be predicted.
    Ban { token: u32 },
    /// Add `bias` to the token's logit (negative to discourage).
    Boost { token: u32, bias: f32 },
    /// Activation patching from another prompt.
    Patch {
        layer: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        head: Option<u32>,
        source: String,
        #[serde(default)]
        all_positions: bool,
    },
    /// Steering vector `strength * (h(a) - h(b))` added after `layer`.
    Steer { layer: u32, a: String, b: String, strength: f32 },
    Sampling {
        #[serde(default)]
        temperature: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_k: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        top_p: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
    },
    /// Force the token chosen at a generation step.
    Force { step: usize, token: u32 },
    /// Replace the system prompt (empty = none).
    System { text: String },
    /// Replace the chat template (Go template syntax, as in Ollama).
    Template { text: String },
}

impl EditOp {
    /// Short human description.
    pub fn describe(&self) -> String {
        match self {
            EditOp::Scale { target, factor } => format!("scale {target} x{factor}"),
            EditOp::Zero { target } => format!("zero {target}"),
            EditOp::Skip { layer } => format!("skip layer {layer}"),
            EditOp::Ablate { cut } => format!("ablate {}", cut.join(", ")),
            EditOp::Embed { token, scale, replace_with } => match (scale, replace_with) {
                (_, Some(r)) => format!("embedding of token {token} := row of {r}{}", scale.map(|s| format!(" x{s}")).unwrap_or_default()),
                (Some(s), None) => format!("embedding of token {token} x{s}"),
                _ => format!("embedding of token {token}"),
            },
            EditOp::Ban { token } => format!("ban token {token}"),
            EditOp::Boost { token, bias } => format!("boost token {token} {bias:+}"),
            EditOp::Patch { layer, head, source, all_positions } => format!(
                "patch layer {layer}{} from \"{source}\"{}",
                head.map(|h| format!(" head {h}")).unwrap_or_default(),
                if *all_positions { " (all positions)" } else { "" }
            ),
            EditOp::Steer { layer, a, b, strength } => format!("steer layer {layer}: \"{a}\" - \"{b}\" x{strength}"),
            EditOp::Sampling { temperature, top_k, top_p, seed } => format!(
                "sampling temperature {temperature}{}{}{}",
                top_k.map(|k| format!(", top-k {k}")).unwrap_or_default(),
                top_p.map(|p| format!(", top-p {p}")).unwrap_or_default(),
                seed.map(|s| format!(", seed {s}")).unwrap_or_default()
            ),
            EditOp::Force { step, token } => format!("force token {token} at step {step}"),
            EditOp::System { text } => format!("system prompt \"{}\"", text.chars().take(40).collect::<String>()),
            EditOp::Template { text } => format!("chat template ({} chars)", text.len()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edit {
    pub id: u64,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub created: u64,
    #[serde(flatten)]
    pub op: EditOp,
}

pub(super) fn yes() -> bool {
    true
}

/// The portable `edits.json` patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditsFile {
    pub schema_version: u32,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub name: Option<String>,
    pub edits: Vec<Edit>,
}

impl EditsFile {
    pub fn new(m: &ModelHandle, name: Option<String>, edits: Vec<Edit>) -> Self {
        Self { schema_version: SCHEMA_VERSION, model: m.entry.name.clone(), digest: m.entry.digest.clone(), name, edits }
    }

    /// Accepts the file format, a bare list of edits or a list of ops.
    pub fn parse(text: &str) -> Result<Self> {
        if let Ok(f) = serde_json::from_str::<EditsFile>(text) {
            return Ok(f);
        }
        if let Ok(edits) = serde_json::from_str::<Vec<Edit>>(text) {
            return Ok(Self { schema_version: SCHEMA_VERSION, model: String::new(), digest: String::new(), name: None, edits });
        }
        let ops: Vec<EditOp> = serde_json::from_str(text).context("edits JSON: expected {\"edits\": [...]} or a list of edits")?;
        let edits = ops.into_iter().enumerate().map(|(i, op)| Edit { id: i as u64 + 1, enabled: true, label: op.describe(), created: 0, op }).collect();
        Ok(Self { schema_version: SCHEMA_VERSION, model: String::new(), digest: String::new(), name: None, edits })
    }
}

/// Resolves a token given as `#123`, `123`, the raw vocabulary string or its decoded text.
pub fn resolve_token(vocab: &Vocab, s: &str) -> Result<u32> {
    let t = s.trim();
    if let Some(n) = t.strip_prefix('#') {
        return n.parse().map_err(|_| anyhow!("bad token id {t:?}"));
    }
    if let Ok(n) = t.parse::<u32>() {
        if (n as usize) < vocab.tokens.len() {
            return Ok(n);
        }
    }
    if let Some(&id) = vocab.by_raw.get(s) {
        return Ok(id);
    }
    if let Some(tok) = vocab.tokens.iter().find(|x| x.text == s) {
        return Ok(tok.id);
    }
    let enc = crate::tokenizer::Encoder::new(vocab)?;
    let ids = enc.encode(s, false, None).ids;
    match ids.as_slice() {
        [one] => Ok(*one),
        [] => bail!("{s:?} is not a token"),
        many => bail!("{s:?} is {} tokens ({:?}); give one token (or #id)", many.len(), many),
    }
}
