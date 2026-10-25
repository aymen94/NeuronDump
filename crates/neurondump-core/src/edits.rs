//! Live edits: an in-memory overlay on top of a model. The original GGUF is never
//! modified. An edit stack holds the edits (each can be toggled), with undo / redo /
//! restore and named snapshots; [`compile`] turns the enabled edits into an
//! [`Intervention`] for the forward pass, and [`bake`] writes them into a **new** GGUF.
//!
//! Ablations are edits too: `zero`, `skip` and `ablate` compile to the same masks as
//! `neurondump ablate`.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use neurondump_gguf::{dequantize_into, quantize_q8_0, GgmlType, GgufWriter, Value, WriterTensor};

use crate::ablate::{ablation_for, block_out_tensors, resolve_cut, zero_ops, ZeroOp};
use crate::graph::{ModelGraph, NodeKind};
use crate::model::ModelHandle;
use crate::slim::{apply_zero, validate, Validation};
use crate::tokenizer::Vocab;
use crate::trace::{EmbedEdit, Intervention, Patch, Sampling, Steer};
use crate::util::{check_cancel, now_unix, Progress, SCHEMA_VERSION};

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

fn yes() -> bool {
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

/// Edit stack with undo / redo (whole-stack checkpoints).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditStack {
    pub edits: Vec<Edit>,
    #[serde(skip)]
    undo: Vec<Vec<Edit>>,
    #[serde(skip)]
    redo: Vec<Vec<Edit>>,
    next_id: u64,
}

impl EditStack {
    fn checkpoint(&mut self) {
        self.undo.push(self.edits.clone());
        if self.undo.len() > 200 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn push(&mut self, op: EditOp, label: Option<String>) -> u64 {
        self.checkpoint();
        self.next_id = self.next_id.max(self.edits.iter().map(|e| e.id).max().unwrap_or(0)) + 1;
        let id = self.next_id;
        let label = label.filter(|l| !l.is_empty()).unwrap_or_else(|| op.describe());
        self.edits.push(Edit { id, enabled: true, label, created: now_unix(), op });
        id
    }

    pub fn toggle(&mut self, id: u64) -> Result<bool> {
        let i = self.edits.iter().position(|e| e.id == id).ok_or_else(|| anyhow!("no edit #{id}"))?;
        self.checkpoint();
        self.edits[i].enabled = !self.edits[i].enabled;
        Ok(self.edits[i].enabled)
    }

    pub fn remove(&mut self, id: u64) -> Result<()> {
        let i = self.edits.iter().position(|e| e.id == id).ok_or_else(|| anyhow!("no edit #{id}"))?;
        self.checkpoint();
        self.edits.remove(i);
        Ok(())
    }

    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.edits, prev));
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(std::mem::replace(&mut self.edits, next));
                true
            }
            None => false,
        }
    }

    /// Back to the original model (undoable).
    pub fn restore(&mut self) {
        if !self.edits.is_empty() {
            self.checkpoint();
            self.edits.clear();
        }
    }

    /// Replaces the whole stack (load / import), renumbering ids (undoable).
    pub fn replace(&mut self, edits: Vec<Edit>) {
        self.checkpoint();
        self.edits = edits
            .into_iter()
            .enumerate()
            .map(|(i, mut e)| {
                e.id = i as u64 + 1;
                if e.label.is_empty() {
                    e.label = e.op.describe();
                }
                e
            })
            .collect();
        self.next_id = self.edits.len() as u64;
    }

    pub fn enabled(&self) -> Vec<Edit> {
        self.edits.iter().filter(|e| e.enabled).cloned().collect()
    }
}

/// Enabled edits turned into what the forward pass and the prompt builder use.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Compiled {
    pub intervention: Intervention,
    /// Sampling for the edited side (last `sampling` edit) plus forced tokens.
    pub sampling: Option<Sampling>,
    pub system: Option<String>,
    pub template: Option<String>,
    /// Graph node ids changed by the edits (highlighted in the graph views).
    pub touched: Vec<String>,
    pub notes: Vec<String>,
    pub active: usize,
}

impl Compiled {
    /// No edit changes anything.
    pub fn is_noop(&self) -> bool {
        self.intervention.is_empty() && self.sampling.is_none() && self.system.is_none() && self.template.is_none()
    }

    pub fn prompt_edited(&self) -> bool {
        self.system.is_some() || self.template.is_some()
    }
}

/// `layer.L.ffn.neuron.N` (a single FFN neuron).
fn parse_neuron(id: &str) -> Option<(u32, u32)> {
    let rest = id.strip_prefix("layer.")?;
    let (l, tail) = rest.split_once('.')?;
    let n = tail.strip_prefix("ffn.neuron.")?;
    Some((l.parse().ok()?, n.parse().ok()?))
}

fn scale_node(graph: &ModelGraph, vocab: Option<&Vocab>, target: &str, factor: f32, c: &mut Compiled) -> Result<()> {
    let a = &mut c.intervention;
    if let Some((l, n)) = parse_neuron(target) {
        let n_ff = graph.dims.get(&l).map_or(0, |d| d.n_ff);
        if n as u64 >= n_ff {
            bail!("layer {l} has {n_ff} FFN neurons; {target} does not exist");
        }
        a.scale_neurons.push((l, n, n, factor));
        c.touched.push(format!("layer.{l}.ffn"));
        return Ok(());
    }
    let id = graph.resolve_id(target).ok_or_else(|| anyhow!("unknown node {target:?}"))?;
    let n = graph.get(&id).ok_or_else(|| anyhow!("unknown node {target:?}"))?;
    match n.kind {
        NodeKind::Layer => {
            let l = n.layer.ok_or_else(|| anyhow!("{id} has no layer index"))?;
            if factor == 0.0 {
                a.skip_layers.push(l);
            } else {
                a.scale_layers.push((l, factor));
            }
        }
        NodeKind::Block => {
            let (Some(l), Some(b)) = (n.layer, n.block.clone()) else { bail!("{id}: not a layer block") };
            if !matches!(b.as_str(), "attention" | "ssm" | "ffn" | "moe") {
                bail!("{id}: only attention, SSM, FFN and MoE blocks can be scaled");
            }
            a.scale_blocks.push((l, b, factor));
        }
        NodeKind::KvGroup => {
            let l = n.layer.unwrap_or(0);
            for h in n.info["heads"].as_array().into_iter().flatten().filter_map(|h| h.as_u64()) {
                a.scale_heads.push((l, h as u32, factor));
            }
        }
        NodeKind::Head => {
            let (Some(l), Some(h)) = (n.layer, n.info["head"].as_u64()) else { bail!("{id}: not a head") };
            a.scale_heads.push((l, h as u32, factor));
        }
        NodeKind::NeuronBucket => {
            let (Some(l), Some(s), Some(e)) = (n.layer, n.info["start"].as_u64(), n.info["end"].as_u64()) else { bail!("{id}: not a bucket") };
            a.scale_neurons.push((l, s as u32, e as u32, factor));
        }
        NodeKind::Expert => {
            let (Some(l), Some(e)) = (n.layer, n.info["expert"].as_u64()) else { bail!("{id}: not an expert") };
            a.scale_experts.push((l, e as u32, factor));
            c.notes.push(format!("{id}: the runtime forward pass has no MoE support; expert edits only apply when baked"));
        }
        _ if factor == 0.0 => {
            let (targets, unknown, _) = resolve_cut(graph, vocab, &[id.clone()]);
            if !unknown.is_empty() {
                bail!("cannot zero {}", unknown.join(", "));
            }
            let n_vocab = vocab.map_or(0, |v| v.tokens.len() as u32);
            a.merge(ablation_for(graph, &targets, n_vocab));
        }
        other => bail!("{id} is a {}: scale works on layers, blocks, KV groups, heads, neuron buckets, neurons and experts (tensors and sections can only be zeroed)", other.as_str()),
    }
    c.touched.push(id);
    Ok(())
}

/// Compiles the enabled edits (in order) for a model.
pub fn compile(graph: &ModelGraph, vocab: Option<&Vocab>, edits: &[Edit]) -> Result<Compiled> {
    let mut c = Compiled::default();
    let n_vocab = vocab.map_or(u32::MAX, |v| v.tokens.len() as u32);
    let check_token = |t: u32| -> Result<()> {
        if t >= n_vocab {
            bail!("token id {t} is outside the vocabulary ({n_vocab} tokens)");
        }
        Ok(())
    };
    let check_layer = |l: u32| -> Result<()> {
        if l >= graph.n_layers {
            bail!("layer {l} does not exist ({} layers)", graph.n_layers);
        }
        Ok(())
    };
    for e in edits.iter().filter(|e| e.enabled) {
        c.active += 1;
        let r: Result<()> = (|| {
            match &e.op {
                EditOp::Scale { target, factor } => scale_node(graph, vocab, target, *factor, &mut c)?,
                EditOp::Zero { target } => scale_node(graph, vocab, target, 0.0, &mut c)?,
                EditOp::Skip { layer } => {
                    check_layer(*layer)?;
                    c.intervention.skip_layers.push(*layer);
                    c.touched.push(format!("layer.{layer}"));
                }
                EditOp::Ablate { cut } => {
                    let (targets, unknown, _) = resolve_cut(graph, vocab, cut);
                    if !unknown.is_empty() {
                        bail!("unknown id(s) {}", unknown.join(", "));
                    }
                    let nv = vocab.map_or(0, |v| v.tokens.len() as u32);
                    c.intervention.merge(ablation_for(graph, &targets, nv));
                    c.touched.extend(targets.into_iter().map(|t| t.id));
                }
                EditOp::Embed { token, scale, replace_with } => {
                    check_token(*token)?;
                    if let Some(r) = replace_with {
                        check_token(*r)?;
                    }
                    c.intervention.embed_edits.push(EmbedEdit { token: *token, scale: *scale, replace_with: *replace_with });
                    c.touched.push("t.token_embd.weight".into());
                }
                EditOp::Ban { token } => {
                    check_token(*token)?;
                    c.intervention.mask_tokens.push((*token, *token));
                    c.touched.push("output".into());
                }
                EditOp::Boost { token, bias } => {
                    check_token(*token)?;
                    if !bias.is_finite() {
                        bail!("boost bias must be finite (use `ban` to forbid a token)");
                    }
                    c.intervention.logit_bias.push((*token, *bias));
                    c.touched.push("output".into());
                }
                EditOp::Patch { layer, head, source, all_positions } => {
                    check_layer(*layer)?;
                    if source.trim().is_empty() {
                        bail!("patch needs a source prompt");
                    }
                    c.intervention.patch.push(Patch { layer: *layer, head: *head, source: source.clone(), all_positions: *all_positions });
                    c.touched.push(match head {
                        Some(h) => format!("layer.{layer}.attn.head.{h}"),
                        None => format!("layer.{layer}"),
                    });
                }
                EditOp::Steer { layer, a, b, strength } => {
                    check_layer(*layer)?;
                    c.intervention.steer.push(Steer { layer: *layer, a: a.clone(), b: b.clone(), strength: *strength });
                    c.touched.push(format!("layer.{layer}"));
                }
                EditOp::Sampling { temperature, top_k, top_p, seed } => {
                    let forced = c.sampling.take().map(|s| s.forced).unwrap_or_default();
                    c.sampling = Some(Sampling { temperature: temperature.max(0.0), top_k: *top_k, top_p: *top_p, seed: *seed, forced });
                }
                EditOp::Force { step, token } => {
                    check_token(*token)?;
                    c.sampling.get_or_insert_with(Sampling::default).forced.push((*step, *token));
                }
                EditOp::System { text } => c.system = Some(text.clone()),
                EditOp::Template { text } => {
                    crate::chat::render_template(text, Some("s"), "p").context("chat template edit")?;
                    c.template = Some(text.clone());
                }
            }
            Ok(())
        })();
        r.with_context(|| format!("edit #{} ({})", e.id, e.label))?;
    }
    let ab = &mut c.intervention;
    ab.skip_layers.sort_unstable();
    ab.skip_layers.dedup();
    let mut seen = HashSet::new();
    c.touched.retain(|t| seen.insert(t.clone()));
    Ok(c)
}

// --- bake ------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BakeReport {
    pub schema_version: u32,
    pub model: String,
    pub output: PathBuf,
    pub modelfile: PathBuf,
    pub original_size: u64,
    pub new_size: u64,
    /// Edits written into the weights or the Modelfile.
    pub applied: Vec<String>,
    /// (edit, reason) for edits that cannot be baked.
    pub skipped: Vec<(String, String)>,
    pub tensors_changed: Vec<String>,
    /// Tensors re-encoded as Q8_0 (quantized types without an exact in-place scale).
    pub requantized: Vec<String>,
    pub notes: Vec<String>,
    pub validation: Validation,
}

/// Weight change on a dequantized tensor (element ranges are half-open).
#[derive(Debug, Clone)]
enum WOp {
    Zero(ZeroOp),
    /// Scale everything.
    All(f32),
    /// Columns (inside every row) [a, b).
    Cols(u64, u64, f32),
    /// Slices along dims[2] [a, b).
    Slices(u64, u64, f32),
    /// Row r := row `from` of the original tensor, then times the factor.
    Row { row: u64, from: Option<u64>, factor: f32 },
}

fn encode_f32(ty: GgmlType, f: &[f32], row_len: u64) -> (GgmlType, Vec<u8>) {
    match ty {
        GgmlType::F32 => (ty, f.iter().flat_map(|x| x.to_le_bytes()).collect()),
        GgmlType::F16 => (ty, f.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()),
        GgmlType::BF16 => (ty, f.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect()),
        _ if row_len % 32 == 0 => (GgmlType::Q8_0, quantize_q8_0(f)),
        _ => (GgmlType::F16, f.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()),
    }
}

/// Writes the enabled edits into a new GGUF (and a Modelfile next to it).
pub fn bake(m: &ModelHandle, graph: &ModelGraph, vocab: Option<&Vocab>, edits: &[Edit], out: &Path, p: &dyn Progress) -> Result<BakeReport> {
    if out.canonicalize().ok().is_some_and(|o| m.entry.path.canonicalize().ok() == Some(o)) {
        bail!("refusing to bake into the original model file; bake always writes a new GGUF");
    }
    let g = &m.gguf;
    let c = compile(graph, vocab, edits)?;
    let a = &c.intervention;
    let mut applied = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut notes = c.notes.clone();
    for e in edits.iter().filter(|e| e.enabled) {
        let why = match &e.op {
            EditOp::Ban { .. } | EditOp::Boost { .. } => Some("logit edits are applied at sampling time; a GGUF has no logit bias"),
            EditOp::Patch { .. } => Some("activation patching depends on a second prompt at run time"),
            EditOp::Steer { .. } => Some("steering adds a run-time vector (no bias tensor to hold it in most architectures)"),
            EditOp::Force { .. } => Some("forced tokens only exist during one generation"),
            _ => None,
        };
        match why {
            Some(w) => skipped.push((e.label.clone(), w.into())),
            None => applied.push(e.label.clone()),
        }
    }

    // --- weight operations per tensor ------------------------------------------------------
    let mut ops: BTreeMap<String, Vec<WOp>> = BTreeMap::new();
    let mut add = |name: String, op: WOp| {
        if g.tensor(&name).is_some() {
            ops.entry(name).or_default().push(op);
        }
    };
    // Skipped layers = identity: zero every block output of the layer.
    let mut zero_part = a.clone();
    for l in &a.skip_layers {
        for b in ["attention", "ssm", "ffn", "moe"] {
            zero_part.zero_blocks.push((*l, b.to_string()));
        }
    }
    zero_part.skip_layers.clear();
    for o in zero_ops(g, graph, &zero_part) {
        add(o.tensor.clone(), WOp::Zero(o));
    }
    let block_short = |b: &str| match b {
        "attention" => "attn",
        other => other,
    }
    .to_string();
    let post_norm = |l: u32, b: &str| -> Option<String> {
        let n = match b {
            "attention" | "ssm" => format!("blk.{l}.post_attention_norm.weight"),
            _ => format!("blk.{l}.post_ffw_norm.weight"),
        };
        g.tensor(&n).map(|_| n)
    };
    let scale_block = |l: u32, b: &str, f: f32, add: &mut dyn FnMut(String, WOp)| {
        // With a post-norm the norm's output is what reaches the residual stream.
        if let Some(n) = post_norm(l, b) {
            add(n, WOp::All(f));
            return;
        }
        for name in block_out_tensors(graph, &format!("layer.{l}.{}", block_short(b))) {
            add(name.trim_end_matches(".weight").to_string() + ".bias", WOp::All(f));
            add(name, WOp::All(f));
        }
    };
    for (l, f) in &a.scale_layers {
        for b in ["attention", "ssm", "ffn", "moe"] {
            if graph.nodes.contains_key(&format!("layer.{l}.{}", block_short(b))) {
                scale_block(*l, b, *f, &mut add);
            }
        }
    }
    for (l, b, f) in &a.scale_blocks {
        scale_block(*l, b, *f, &mut add);
    }
    for (l, h, f) in &a.scale_heads {
        let hd = graph.dims.get(l).map_or(0, |d| d.head_dim);
        if post_norm(*l, "attention").is_some() {
            notes.push(format!("layer {l} has a post-attention norm: a baked head scale is renormalized, so it is weaker than at run time"));
        }
        add(format!("blk.{l}.attn_output.weight"), WOp::Cols(*h as u64 * hd, (*h as u64 + 1) * hd, *f));
    }
    for (l, s, e, f) in &a.scale_neurons {
        add(format!("blk.{l}.ffn_down.weight"), WOp::Cols(*s as u64, *e as u64 + 1, *f));
    }
    for (l, e, f) in &a.scale_experts {
        add(format!("blk.{l}.ffn_down_exps.weight"), WOp::Slices(*e as u64, *e as u64 + 1, *f));
    }
    for ed in &a.embed_edits {
        add("token_embd.weight".into(), WOp::Row { row: ed.token as u64, from: ed.replace_with.map(|r| r as u64), factor: ed.scale.unwrap_or(1.0) });
    }
    if !a.embed_edits.is_empty() && g.tensor("output.weight").is_none() {
        notes.push("embeddings are tied: the edited token_embd rows also change the output head (at run time only the input side changes)".into());
    }
    if !a.mask_tokens.is_empty() {
        notes.push("banned tokens are not baked: use `slim` with a vocabulary cut to remove tokens".into());
    }

    // --- write ------------------------------------------------------------------------------
    let mut w = GgufWriter::new();
    w.metadata = g.metadata.clone();
    w.metadata.insert("general.edited_by".into(), Value::String(format!("neurondump {}", env!("CARGO_PKG_VERSION"))));
    w.metadata.insert("neurondump.edits".into(), Value::String(serde_json::to_string(&edits.iter().filter(|e| e.enabled).collect::<Vec<_>>())?));
    if let Some(t) = &c.template {
        // Jinja templates in the file would disagree with the new (Go) template.
        w.metadata.shift_remove("tokenizer.chat_template");
        notes.push("tokenizer.chat_template removed: the Modelfile carries the edited template".into());
        let _ = t;
    }
    let mut changed = Vec::new();
    let mut requantized = Vec::new();
    let total = g.tensors.len().max(1);
    for (i, t) in g.tensors.iter().enumerate() {
        check_cancel(p)?;
        p.set(5.0 + 45.0 * i as f32 / total as f32, &format!("preparing {}", t.name));
        let src = g.info_bytes(t)?;
        let Some(list) = ops.get(&t.name) else {
            w.tensors.push(WriterTensor { name: t.name.clone(), dims: t.dims.clone(), ggml_type: t.ggml_type, data: Cow::Borrowed(src) });
            continue;
        };
        changed.push(t.name.clone());
        let mut ty = t.ggml_type;
        let data: Vec<u8> = if list.iter().all(|o| matches!(o, WOp::Zero(_))) {
            let mut buf = src.to_vec();
            for o in list {
                if let WOp::Zero(z) = o {
                    if let Some(nt) = apply_zero(z, &t.dims, ty, &mut buf)? {
                        ty = nt;
                        requantized.push(t.name.clone());
                    }
                }
            }
            buf
        } else {
            let row_len = t.dims.first().copied().unwrap_or(1).max(1);
            let n_rows = t.dims.iter().skip(1).product::<u64>().max(1);
            let mut f = vec![0f32; (row_len * n_rows) as usize];
            dequantize_into(t.ggml_type, src, &mut f).with_context(|| format!("dequantizing {}", t.name))?;
            let orig = if list.iter().any(|o| matches!(o, WOp::Row { from: Some(_), .. })) { Some(f.clone()) } else { None };
            let rl = row_len as usize;
            for o in list {
                match o {
                    WOp::Zero(z) => {
                        if z.all {
                            f.fill(0.0);
                        }
                        for (s, e) in &z.rows {
                            let (s, e) = ((*s).min(n_rows) as usize, (*e).min(n_rows) as usize);
                            f[s * rl..e * rl].fill(0.0);
                        }
                        for (s, e) in &z.cols {
                            for r in 0..n_rows as usize {
                                f[r * rl + (*s).min(row_len) as usize..r * rl + (*e).min(row_len) as usize].fill(0.0);
                            }
                        }
                        if let Some(&ns) = t.dims.get(2) {
                            let per = f.len() / ns.max(1) as usize;
                            for (s, e) in &z.slices {
                                f[(*s).min(ns) as usize * per..(*e).min(ns) as usize * per].fill(0.0);
                            }
                        }
                    }
                    WOp::All(k) => f.iter_mut().for_each(|x| *x *= k),
                    WOp::Cols(s, e, k) => {
                        for r in 0..n_rows as usize {
                            for x in &mut f[r * rl + (*s).min(row_len) as usize..r * rl + (*e).min(row_len) as usize] {
                                *x *= k;
                            }
                        }
                    }
                    WOp::Slices(s, e, k) => {
                        if let Some(&ns) = t.dims.get(2) {
                            let per = f.len() / ns.max(1) as usize;
                            for x in &mut f[(*s).min(ns) as usize * per..(*e).min(ns) as usize * per] {
                                *x *= k;
                            }
                        }
                    }
                    WOp::Row { row, from, factor } => {
                        if *row >= n_rows {
                            continue;
                        }
                        let r = *row as usize;
                        if let (Some(fr), Some(o)) = (from, &orig) {
                            let fr = (*fr).min(n_rows - 1) as usize;
                            f[r * rl..(r + 1) * rl].copy_from_slice(&o[fr * rl..(fr + 1) * rl]);
                        }
                        for x in &mut f[r * rl..(r + 1) * rl] {
                            *x *= factor;
                        }
                    }
                }
            }
            let (nt, bytes) = encode_f32(t.ggml_type, &f, row_len);
            if nt != t.ggml_type {
                requantized.push(t.name.clone());
            }
            ty = nt;
            bytes
        };
        w.tensors.push(WriterTensor { name: t.name.clone(), dims: t.dims.clone(), ggml_type: ty, data: Cow::Owned(data) });
    }
    if let Some(d) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
    }
    p.set(50.0, "writing");
    let new_size = w
        .write_to(out, |done, total| p.set(50.0 + 45.0 * done as f32 / total.max(1) as f32, "writing"))
        .with_context(|| format!("writing {}", out.display()))?;
    drop(w);
    p.set(96.0, "validating");
    let validation = validate(out);
    let name = out.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "baked.gguf".into());
    let modelfile = out.with_file_name(format!("{}.Modelfile", name.trim_end_matches(".gguf")));
    std::fs::write(&modelfile, modelfile_text(m, &name, &c)).with_context(|| format!("writing {}", modelfile.display()))?;
    if c.sampling.is_some() || c.system.is_some() || c.template.is_some() {
        notes.push("sampling, system prompt and template edits are written to the Modelfile".into());
    }
    p.set(100.0, "done");
    Ok(BakeReport {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        output: out.to_path_buf(),
        modelfile,
        original_size: g.file_size,
        new_size,
        applied,
        skipped,
        tensors_changed: changed,
        requantized,
        notes,
        validation,
    })
}

/// Modelfile for a baked model, with the edited template / system prompt / sampling.
fn modelfile_text(m: &ModelHandle, gguf_name: &str, c: &Compiled) -> String {
    let mut s = format!("# Generated by NeuronDump (bake) from {}\nFROM ./{gguf_name}\n", m.entry.name);
    if let Some(p) = &m.entry.projector {
        s.push_str(&format!("FROM {}\n", p.display()));
    }
    if let Some(t) = c.template.as_ref().or(m.ollama.template.as_ref()) {
        s.push_str(&format!("TEMPLATE \"\"\"{t}\"\"\"\n"));
    }
    match &c.system {
        Some(sys) if !sys.is_empty() => s.push_str(&format!("SYSTEM \"\"\"{sys}\"\"\"\n")),
        Some(_) => s.push_str("SYSTEM \"\"\"\"\"\"\n"),
        None => {
            if let Some(sys) = &m.ollama.system {
                s.push_str(&format!("SYSTEM \"\"\"{sys}\"\"\"\n"));
            }
        }
    }
    let mut overridden: Vec<&str> = Vec::new();
    if let Some(sp) = &c.sampling {
        s.push_str(&format!("PARAMETER temperature {}\n", sp.temperature));
        overridden.push("temperature");
        if let Some(k) = sp.top_k {
            s.push_str(&format!("PARAMETER top_k {k}\n"));
            overridden.push("top_k");
        }
        if let Some(pp) = sp.top_p {
            s.push_str(&format!("PARAMETER top_p {pp}\n"));
            overridden.push("top_p");
        }
        if let Some(seed) = sp.seed {
            s.push_str(&format!("PARAMETER seed {seed}\n"));
            overridden.push("seed");
        }
    }
    if let Some(serde_json::Value::Object(params)) = &m.ollama.params {
        for (k, v) in params.iter().filter(|(k, _)| !overridden.contains(&k.as_str())) {
            let vals: Vec<&serde_json::Value> = match v {
                serde_json::Value::Array(a) => a.iter().collect(),
                v => vec![v],
            };
            for v in vals {
                match v {
                    serde_json::Value::String(x) => s.push_str(&format!("PARAMETER {k} {}\n", serde_json::to_string(x).unwrap_or_default())),
                    other => s.push_str(&format!("PARAMETER {k} {other}\n")),
                }
            }
        }
    }
    for lic in &m.ollama.licenses {
        s.push_str(&format!("LICENSE \"\"\"{lic}\"\"\"\n"));
    }
    s
}

/// Snapshot files live in `<output_dir>/edits/<digest>/<name>.json`.
pub fn snapshot_dir(output_dir: &Path, m: &ModelHandle) -> PathBuf {
    output_dir.join("edits").join(&m.entry.digest)
}

pub fn snapshot_name(name: &str) -> Result<String> {
    let n = name.trim();
    if n.is_empty() || n.len() > 80 || !n.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) || n.starts_with('.') {
        bail!("snapshot names use letters, digits, '-', '_' and '.' only: {name:?}");
    }
    Ok(n.trim_end_matches(".json").to_string())
}

pub fn save_snapshot(output_dir: &Path, m: &ModelHandle, name: &str, edits: &[Edit]) -> Result<PathBuf> {
    let name = snapshot_name(name)?;
    let dir = snapshot_dir(output_dir, m);
    std::fs::create_dir_all(&dir)?;
    let p = dir.join(format!("{name}.json"));
    std::fs::write(&p, serde_json::to_vec_pretty(&EditsFile::new(m, Some(name), edits.to_vec()))?)?;
    Ok(p)
}

pub fn load_snapshot(output_dir: &Path, m: &ModelHandle, name: &str) -> Result<EditsFile> {
    let name = snapshot_name(name)?;
    let p = snapshot_dir(output_dir, m).join(format!("{name}.json"));
    let text = std::fs::read_to_string(&p).with_context(|| format!("no snapshot {name:?} ({})", p.display()))?;
    EditsFile::parse(&text)
}

pub fn list_snapshots(output_dir: &Path, m: &ModelHandle) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(snapshot_dir(output_dir, m))
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".json")).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
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
