//! Structural graph of a model: Model → Sections → Layers → Blocks → Tensors → (lazy)
//! heads / neuron buckets / experts, plus dataflow, tied and uses edges.
//!
//! Built only from architecture metadata and tensor roles (same rules as `arch`), so it
//! works for any model. Dataflow edges are stored at the finest structural level and
//! lifted to the nearest visible ancestor when a coarser view is requested.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::arch::{analyze_architecture, parse_tensor_name, Component, LayerInfo, ModelKind};
use crate::model::ModelHandle;
use crate::stats::StatsReport;
use crate::tokenizer::Vocab;
use crate::trace::TraceReport;
use crate::util::{human_bytes, human_count, SCHEMA_VERSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Model,
    Section,
    Layer,
    Block,
    Op,
    Tensor,
    Token,
    KvGroup,
    Head,
    NeuronBucket,
    Expert,
    /// Synthetic grouping node of a `group_by` view.
    Group,
}

impl NodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeKind::Model => "model",
            NodeKind::Section => "section",
            NodeKind::Layer => "layer",
            NodeKind::Block => "block",
            NodeKind::Op => "op",
            NodeKind::Tensor => "tensor",
            NodeKind::Token => "token",
            NodeKind::KvGroup => "kv_group",
            NodeKind::Head => "head",
            NodeKind::NeuronBucket => "neuron_bucket",
            NodeKind::Expert => "expert",
            NodeKind::Group => "group",
        }
    }

    pub fn is_fine(self) -> bool {
        matches!(self, NodeKind::KvGroup | NodeKind::Head | NodeKind::NeuronBucket | NodeKind::Expert)
    }
}

/// What removing (or zeroing) a node does to a model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Removable {
    Safe,
    Risky,
    BreaksModel,
}

impl Removable {
    pub fn as_str(self) -> &'static str {
        match self {
            Removable::Safe => "safe",
            Removable::Risky => "risky",
            Removable::BreaksModel => "breaks_model",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Dataflow,
    Contains,
    Tied,
    Uses,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildSummary {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub params: u64,
    pub bytes: u64,
}

/// Weight statistics summary (from the stats cache; never computed by the graph).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSummary {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub std: Option<f64>,
    pub l2: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparsity: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outlier_fraction: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Importance {
    /// 0 = removing it changes nothing, 1 = essential (relative within its level).
    pub score: f32,
    /// `sweep`, `trace`, `stats` or `structural` (with a short explanation).
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    /// Block type for blocks and everything inside them: attention, ssm, ffn, moe, mtp, other.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<String>,
    /// Layer kind (`full_attention+dense`, ...) for layers and their contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_kind: Option<String>,
    pub params: u64,
    pub bytes: u64,
    pub quant: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<StatsSummary>,
    pub importance: Importance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation: Option<Value>,
    pub contains: Vec<ChildSummary>,
    pub removable: Removable,
    pub removable_reason: String,
    /// Has fine-level children generated on expand (heads, neuron buckets, experts).
    pub lazy: bool,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub info: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Section,
    Layer,
    Block,
    Tensor,
    Fine,
}

impl Level {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "section" | "sections" => Level::Section,
            "layer" | "layers" => Level::Layer,
            "block" | "blocks" => Level::Block,
            "tensor" | "tensors" => Level::Tensor,
            "fine" | "head" | "heads" => Level::Fine,
            _ => return None,
        })
    }

    fn rank(self) -> u8 {
        match self {
            Level::Section => 1,
            Level::Layer => 2,
            Level::Block => 3,
            Level::Tensor => 4,
            Level::Fine => 5,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Level::Section => "section",
            Level::Layer => "layer",
            Level::Block => "block",
            Level::Tensor => "tensor",
            Level::Fine => "fine",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupBy {
    Section,
    Layer,
    Block,
    Quant,
    Kind,
    Importance,
}

impl GroupBy {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "section" => GroupBy::Section,
            "layer" => GroupBy::Layer,
            "block" => GroupBy::Block,
            "quant" => GroupBy::Quant,
            "kind" => GroupBy::Kind,
            "importance" => GroupBy::Importance,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            GroupBy::Section => "section",
            GroupBy::Layer => "layer",
            GroupBy::Block => "block",
            GroupBy::Quant => "quant",
            GroupBy::Kind => "kind",
            GroupBy::Importance => "importance",
        }
    }
}

/// One measured result of an ablation sweep, used as importance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SweepImportance {
    /// node id -> damage (mean KL divergence per token when removed).
    pub damage: BTreeMap<String, f64>,
}

#[derive(Default)]
pub struct GraphInputs<'a> {
    pub stats: Option<&'a StatsReport>,
    pub trace: Option<&'a TraceReport>,
    pub sweeps: Vec<SweepImportance>,
}

/// Per-layer attention / FFN / expert dimensions used for the fine level.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerDims {
    pub n_head: u64,
    pub n_kv: u64,
    pub head_dim: u64,
    pub n_ff: u64,
    pub n_expert: u64,
    pub bucket: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGraph {
    pub schema_version: u32,
    pub model: String,
    pub digest: String,
    pub architecture: String,
    pub kind: ModelKind,
    pub total_params: u64,
    pub file_size: u64,
    pub tied_embeddings: bool,
    /// `full_attention_interval` for hybrid models (layer kinds repeat with this period).
    pub pattern_period: Option<u64>,
    pub n_layers: u32,
    pub nodes: IndexMap<String, Node>,
    pub edges: Vec<Edge>,
    pub dims: BTreeMap<u32, LayerDims>,
    /// Weight stats per fine node id (head/neuron/expert norms), filled lazily.
    #[serde(skip)]
    fine_stats: HashMap<String, f64>,
    #[serde(skip)]
    fine_activation: HashMap<String, Value>,
    #[serde(skip)]
    fine_importance: HashMap<String, Importance>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewOptions {
    pub level: Level,
    pub group_by: GroupBy,
    #[serde(default)]
    pub expand: Vec<String>,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self { level: Level::Layer, group_by: GroupBy::Section, expand: vec![] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphView {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub kind: ModelKind,
    pub level: Level,
    pub group_by: GroupBy,
    pub expand: Vec<String>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDetail {
    pub node: Node,
    pub ancestors: Vec<ChildSummary>,
    pub children: Vec<Node>,
    pub incoming: Vec<Edge>,
    pub outgoing: Vec<Edge>,
}

// --- helpers -------------------------------------------------------------------------

fn tensor_id(name: &str) -> String {
    format!("t.{name}")
}

fn block_label(b: &str) -> &'static str {
    match b {
        "attention" => "attention",
        "ssm" => "linear attention (SSM)",
        "ffn" => "feed-forward",
        "moe" => "mixture of experts",
        "mtp" => "next-token prediction head",
        _ => "other",
    }
}

fn block_short(b: &str) -> &'static str {
    match b {
        "attention" => "attn",
        "ssm" => "ssm",
        "ffn" => "ffn",
        "moe" => "moe",
        "mtp" => "mtp",
        _ => "other",
    }
}

fn block_from_short(s: &str) -> Option<&'static str> {
    Some(match s {
        "attn" | "attention" => "attention",
        "ssm" => "ssm",
        "ffn" => "ffn",
        "moe" => "moe",
        "mtp" => "mtp",
        "other" => "other",
        _ => return None,
    })
}

/// Which block of a layer a tensor role belongs to.
fn block_of(role: &str, comp: Component, has_ssm: bool, has_ffn: bool, has_moe: bool) -> &'static str {
    let r = role.strip_prefix("unknown:").unwrap_or(role);
    if r.starts_with("nextn.") || r.starts_with("mtp.") || comp == Component::Mtp {
        return "mtp";
    }
    if comp == Component::Moe {
        return "moe";
    }
    if comp == Component::LinearAttention || r.starts_with("ssm_") {
        return "ssm";
    }
    let mixer = if has_ssm { "ssm" } else { "attention" };
    let ffn = if has_ffn || !has_moe { "ffn" } else { "moe" };
    if r.starts_with("attn_") || r == "post_attention_norm" {
        // BERT-style post-norms belong to the residual stream after the attention add.
        return mixer;
    }
    if r.starts_with("ffn_") || matches!(r, "post_ffw_norm" | "layer_output_norm" | "layer_out_norm") {
        return ffn;
    }
    "other"
}

/// Position of a tensor inside its block's dataflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Pre,
    In,
    Out,
    Post,
    /// Norm applied to the residual stream after the add (BERT post-norm).
    PostAdd,
}

fn stage_of(role: &str) -> Stage {
    let r = role.strip_prefix("unknown:").unwrap_or(role);
    match r {
        "attn_norm" | "ffn_norm" | "attn_norm_2" => Stage::Pre,
        "attn_output_norm" | "layer_output_norm" | "layer_out_norm" => Stage::PostAdd,
        "post_attention_norm" | "post_ffw_norm" | "attn_post_norm" | "ffn_post_norm" => Stage::Post,
        "attn_output" | "attn_out" | "ffn_down" | "ffn_down_exps" | "ffn_down_shexp" | "ssm_out" => Stage::Out,
        _ => Stage::In,
    }
}

fn ancestors_of(nodes: &IndexMap<String, Node>, id: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = nodes.get(id).and_then(|n| n.parent.clone());
    while let Some(p) = cur {
        cur = nodes.get(&p).and_then(|n| n.parent.clone());
        out.push(p);
    }
    out
}

fn new_node(id: &str, kind: NodeKind, label: &str, parent: Option<&str>) -> Node {
    Node {
        id: id.to_string(),
        kind,
        label: label.to_string(),
        parent: parent.map(String::from),
        layer: None,
        block: None,
        layer_kind: None,
        params: 0,
        bytes: 0,
        quant: vec![],
        stats: None,
        importance: Importance { score: 0.5, source: "structural: not measured (run `sweep`)".into() },
        activation: None,
        contains: vec![],
        removable: Removable::Risky,
        removable_reason: String::new(),
        lazy: false,
        info: Value::Null,
    }
}

fn enum_str<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default()
}

fn layer_kind_label(l: &LayerInfo) -> String {
    format!("{}+{}", enum_str(&l.mixer), enum_str(&l.ffn))
}

/// Parses `layer.N...` / `blk.N` / `LN` style prefixes.
fn parse_layer_prefix(s: &str) -> Option<(u32, &str)> {
    let rest = s.strip_prefix("layer.").or_else(|| s.strip_prefix("blk."))?;
    let (n, tail) = match rest.split_once('.') {
        Some((n, t)) => (n, t),
        None => (rest, ""),
    };
    Some((n.parse().ok()?, tail))
}

// --- build -----------------------------------------------------------------------------

impl ModelGraph {
    pub fn build(m: &ModelHandle, vocab: Option<&Vocab>, inputs: &GraphInputs) -> ModelGraph {
        let g = &m.gguf;
        let arch = analyze_architecture(g);
        let mut nodes: IndexMap<String, Node> = IndexMap::new();
        let mut edges: Vec<Edge> = Vec::new();
        let mut notes = Vec::new();
        let encoder = arch.kind == ModelKind::Encoder;
        let pattern_period = g.arch_key("full_attention_interval").and_then(|v| v.as_u64()).filter(|&p| p > 1);
        let hybrid = arch.kind == ModelKind::Hybrid || pattern_period.is_some();
        let n_layers = arch.layers.len() as u32;
        let nextn_layers = g.arch_key("nextn_predict_layers").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let main_layers = n_layers.saturating_sub(nextn_layers);

        let mut root = new_node("model", NodeKind::Model, &m.entry.name, None);
        root.removable = Removable::BreaksModel;
        root.removable_reason = "the whole model".into();
        nodes.insert("model".into(), root);

        let section = |nodes: &mut IndexMap<String, Node>, id: &str, label: &str, rem: Removable, why: &str| {
            let mut n = new_node(id, NodeKind::Section, label, Some("model"));
            n.removable = rem;
            n.removable_reason = why.into();
            nodes.insert(id.into(), n);
        };
        section(&mut nodes, "input", "Input", Removable::BreaksModel, "token embeddings are the model's only input");
        section(&mut nodes, "body", "Body", Removable::BreaksModel, "removing every layer leaves no model");
        section(
            &mut nodes,
            "output",
            if encoder { "Output (pooling)" } else { "Output" },
            Removable::BreaksModel,
            "the output norm and head produce the predictions",
        );

        // --- top-level tensors ---------------------------------------------------------
        let roles: Vec<_> = g.tensors.iter().map(|t| parse_tensor_name(&t.name)).collect();
        let mut has_mtp_section = false;
        let mut has_vision_section = false;
        let mut has_extra = false;
        for (t, r) in g.tensors.iter().zip(&roles) {
            if r.layer.is_some() {
                continue;
            }
            let role = r.role.strip_prefix("unknown:").unwrap_or(&r.role);
            let vision_prefix = role.starts_with("v.") || role.starts_with("mm.") || role.starts_with("a.");
            let sec = match r.component {
                _ if vision_prefix => "vision",
                Component::Embedding => "input",
                Component::Output => "output",
                Component::Mtp => "mtp",
                Component::Vision => "vision",
                Component::Norm if role.starts_with("output") => "output",
                Component::Norm if role.starts_with("token_embd") => "input",
                _ if role.starts_with("rope") => "input",
                _ if role.starts_with("output") || role.starts_with("cls") => "output",
                _ => "extra",
            };
            match sec {
                "mtp" if !has_mtp_section => {
                    has_mtp_section = true;
                    section(&mut nodes, "mtp", "Multi-token prediction", Removable::Safe, "speculative-decoding head; normal generation does not use it");
                }
                "vision" if !has_vision_section => {
                    has_vision_section = true;
                    section(&mut nodes, "vision", "Vision encoder (in file)", Removable::Risky, "image input stops working; llama.cpp may expect these tensors");
                }
                "extra" if !has_extra => {
                    has_extra = true;
                    section(&mut nodes, "extra", "Other tensors", Removable::Risky, "unrecognized top-level tensors");
                }
                _ => {}
            }
            let id = tensor_id(&t.name);
            let mut n = new_node(&id, NodeKind::Tensor, &t.name, Some(sec));
            n.params = t.n_elements;
            n.bytes = t.n_bytes;
            n.quant = vec![t.ggml_type.to_string()];
            n.info = json!({ "name": t.name, "dims": t.dims, "role": r.role, "type": t.ggml_type.to_string() });
            let (rem, why) = match (sec, role) {
                ("input", "token_embd") => (Removable::BreaksModel, "maps token ids to vectors; nothing runs without it"),
                ("input", r) if r.starts_with("rope") => (Removable::BreaksModel, "rotary position frequencies used by every attention layer"),
                ("input", "pos_embd") => (Removable::BreaksModel, "position embeddings: the model loses word order"),
                ("input", _) => (Removable::BreaksModel, "part of the input embedding path"),
                ("output", "output_norm") => (Removable::BreaksModel, "final normalization before the output head"),
                ("output", _) => (Removable::BreaksModel, "produces the predictions"),
                ("mtp", _) => (Removable::Safe, "speculative-decoding head only"),
                _ => (Removable::Risky, "top-level tensor"),
            };
            n.removable = rem;
            n.removable_reason = why.into();
            nodes.insert(id, n);
        }

        // Tied output head: a virtual node pointing at the embedding matrix.
        let tied = arch.tied_embeddings;
        if tied {
            let mut n = new_node("output.head", NodeKind::Tensor, "output head (tied to token_embd)", Some("output"));
            n.removable = Removable::BreaksModel;
            n.removable_reason = "shares its weights with token_embd".into();
            n.info = json!({ "virtual": true, "tied_to": "token_embd.weight" });
            nodes.insert("output.head".into(), n);
            edges.push(Edge { from: "output.head".into(), to: tensor_id("token_embd.weight"), kind: EdgeKind::Tied, label: Some("shared weights".into()) });
        }
        let pooling = g.arch_key("pooling_type").and_then(|v| v.as_u64()).unwrap_or(0);
        let sink_id = if encoder || pooling > 0 { "output.embedding" } else { "output.logits" };
        {
            let label = if sink_id == "output.embedding" { "pooled embedding" } else { "next-token logits" };
            let mut n = new_node(sink_id, NodeKind::Op, label, Some("output"));
            n.removable = Removable::BreaksModel;
            n.removable_reason = "the model's result".into();
            nodes.insert(sink_id.into(), n);
        }

        // --- layers --------------------------------------------------------------------
        let mut dims: BTreeMap<u32, LayerDims> = BTreeMap::new();
        let emb = arch.hyperparams.embedding_length.unwrap_or(0);
        let mut layer_tensors: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, r) in roles.iter().enumerate() {
            if let Some(l) = r.layer {
                layer_tensors.entry(l).or_default().push(i);
            }
        }
        for l in &arch.layers {
            let lid = format!("layer.{}", l.index);
            let mut n = new_node(&lid, NodeKind::Layer, &format!("layer {}", l.index), Some("body"));
            n.layer = Some(l.index);
            n.layer_kind = Some(layer_kind_label(l));
            n.params = l.params;
            n.bytes = l.bytes;
            let is_nextn = l.index >= main_layers && nextn_layers > 0;
            let (rem, why) = if is_nextn {
                (Removable::Risky, "next-n prediction layer: only used for speculative decoding, but its metadata must be updated too".to_string())
            } else if hybrid {
                (
                    Removable::BreaksModel,
                    format!(
                        "hybrid model: layer kinds follow a fixed pattern{}; removing one layer misaligns every later layer",
                        pattern_period.map(|p| format!(" (every {p}th layer is full attention)")).unwrap_or_default()
                    ),
                )
            } else if n_layers <= 1 {
                (Removable::BreaksModel, "the only layer".into())
            } else if l.index == 0 || l.index + 1 == main_layers {
                (Removable::Risky, "edge layer: llama.cpp loads the file after renumbering, but first/last layers usually matter most".into())
            } else {
                (Removable::Risky, "llama.cpp loads it after renumbering the layers; quality loss must be measured".into())
            };
            n.removable = rem;
            n.removable_reason = why;
            nodes.insert(lid.clone(), n);

            let idx = layer_tensors.get(&l.index).cloned().unwrap_or_default();
            let comp_has = |c: Component| idx.iter().any(|&i| roles[i].component == c);
            let has_ssm = comp_has(Component::LinearAttention);
            let has_moe = comp_has(Component::Moe);
            let has_ffn = idx.iter().any(|&i| {
                let r = roles[i].role.as_str();
                matches!(r, "ffn_up" | "ffn_down" | "ffn_gate" | "ffn_gate_up")
            });

            // Dimensions for the fine level.
            let n_head = l.head_count.unwrap_or(0);
            let n_kv = l.head_count_kv.unwrap_or(n_head);
            let head_dim = arch.hyperparams.key_length.unwrap_or(if n_head > 0 { emb / n_head } else { 0 });
            let n_ff = g
                .tensor(&format!("blk.{}.ffn_down.weight", l.index))
                .map(|t| t.row_len())
                .or(l.ffn_length)
                .unwrap_or(0);
            let n_expert = g
                .tensor(&format!("blk.{}.ffn_down_exps.weight", l.index))
                .or_else(|| g.tensor(&format!("blk.{}.ffn_gate_exps.weight", l.index)))
                .and_then(|t| t.dims.get(2).copied())
                .unwrap_or(0);
            let mut bucket = 64u64;
            while n_ff / bucket > 256 {
                bucket *= 2;
            }
            dims.insert(l.index, LayerDims { n_head, n_kv, head_dim, n_ff, n_expert, bucket });

            // Blocks.
            let mut blocks: IndexMap<&'static str, Vec<usize>> = IndexMap::new();
            for &i in &idx {
                let b = block_of(&roles[i].role, roles[i].component, has_ssm, has_ffn, has_moe);
                blocks.entry(b).or_default().push(i);
            }
            let order = ["attention", "ssm", "ffn", "moe", "mtp", "other"];
            blocks.sort_by_key(|k, _| order.iter().position(|o| o == k).unwrap_or(9));
            for (b, ts) in &blocks {
                let bid = format!("{lid}.{}", block_short(b));
                let mut bn = new_node(&bid, NodeKind::Block, block_label(b), Some(&lid));
                bn.layer = Some(l.index);
                bn.block = Some(b.to_string());
                bn.layer_kind = Some(layer_kind_label(l));
                let (rem, why) = match *b {
                    "mtp" => (Removable::Safe, "speculative-decoding head only"),
                    "other" => (Removable::Risky, "unrecognized tensors"),
                    _ if hybrid && *b == "ssm" => (Removable::Risky, "zeroed in place: shapes stay, the layer stops mixing tokens"),
                    _ => (Removable::Risky, "zeroed in place (output projection set to 0): shapes stay, so llama.cpp still loads it"),
                };
                bn.removable = rem;
                bn.removable_reason = why.into();
                bn.lazy = match *b {
                    "attention" => n_head > 0,
                    "ffn" => n_ff > 0,
                    "moe" => n_expert > 0,
                    _ => false,
                };
                bn.info = match *b {
                    "attention" => json!({ "heads": n_head, "kv_heads": n_kv, "head_dim": head_dim, "gqa_group": if n_kv > 0 { n_head / n_kv } else { 0 } }),
                    "ffn" => json!({ "neurons": n_ff, "bucket": bucket }),
                    "moe" => json!({ "experts": n_expert }),
                    _ => Value::Null,
                };
                let mut quants = BTreeSet::new();
                for &i in ts {
                    let t = &g.tensors[i];
                    bn.params += t.n_elements;
                    bn.bytes += t.n_bytes;
                    quants.insert(t.ggml_type.to_string());
                    let tid = tensor_id(&t.name);
                    let short = t.name.strip_prefix(&format!("blk.{}.", l.index)).unwrap_or(&t.name);
                    let mut tn = new_node(&tid, NodeKind::Tensor, short, Some(&bid));
                    tn.layer = Some(l.index);
                    tn.block = Some(b.to_string());
                    tn.layer_kind = Some(layer_kind_label(l));
                    tn.params = t.n_elements;
                    tn.bytes = t.n_bytes;
                    tn.quant = vec![t.ggml_type.to_string()];
                    tn.info = json!({ "name": t.name, "dims": t.dims, "role": roles[i].role, "type": t.ggml_type.to_string() });
                    tn.removable = if *b == "mtp" { Removable::Safe } else { Removable::Risky };
                    tn.removable_reason = "zeroed in place: llama.cpp still loads it, its contribution becomes 0".into();
                    nodes.insert(tid, tn);
                }
                bn.quant = quants.into_iter().collect();
                nodes.insert(bid, bn);
            }
            if let Some(n) = nodes.get_mut(&lid) {
                let mut q: BTreeSet<String> = BTreeSet::new();
                for &i in &idx {
                    q.insert(g.tensors[i].ggml_type.to_string());
                }
                n.quant = q.into_iter().collect();
            }
            for (op, label) in [("add_mix", "residual add (mixer)"), ("add_ffn", "residual add (FFN)")] {
                let oid = format!("{lid}.{op}");
                let mut on = new_node(&oid, NodeKind::Op, label, Some(&lid));
                on.layer = Some(l.index);
                on.removable = Removable::BreaksModel;
                on.removable_reason = "the residual stream itself".into();
                nodes.insert(oid, on);
            }
        }

        // --- dataflow ------------------------------------------------------------------
        let flow = |edges: &mut Vec<Edge>, a: &str, b: &str, label: Option<&str>| {
            edges.push(Edge { from: a.into(), to: b.into(), kind: EdgeKind::Dataflow, label: label.map(String::from) });
        };
        let mut cur = tensor_id("token_embd.weight");
        if !nodes.contains_key(&cur) {
            cur = "input".into();
        }
        for extra in ["pos_embd.weight", "token_types.weight"] {
            let id = tensor_id(extra);
            if nodes.contains_key(&id) {
                flow(&mut edges, &cur, &id, Some("add"));
                cur = id;
            }
        }
        let en = tensor_id("token_embd_norm.weight");
        if nodes.contains_key(&en) {
            flow(&mut edges, &cur, &en, None);
            cur = en;
        }
        let mut last_main = cur.clone();
        for l in &arch.layers {
            let lid = format!("layer.{}", l.index);
            let is_nextn = l.index >= main_layers && nextn_layers > 0;
            let start = if is_nextn { last_main.clone() } else { cur.clone() };
            let mut residual = start.clone();
            let blocks: Vec<(String, String)> = nodes
                .values()
                .filter(|n| n.kind == NodeKind::Block && n.parent.as_deref() == Some(lid.as_str()))
                .map(|n| (n.id.clone(), n.block.clone().unwrap_or_default()))
                .collect();
            let mixer = blocks.iter().find(|(_, b)| b == "attention" || b == "ssm").cloned();
            let ffnb = blocks.iter().find(|(_, b)| b == "ffn" || b == "moe").cloned();
            let extra_mtp = blocks.iter().find(|(_, b)| b == "mtp").cloned();
            for (stage_block, add) in [(mixer, "add_mix"), (ffnb, "add_ffn")] {
                let Some((bid, _)) = stage_block else { continue };
                let add_id = format!("{lid}.{add}");
                let ts: Vec<(String, Stage)> = nodes
                    .values()
                    .filter(|n| n.parent.as_deref() == Some(bid.as_str()))
                    .map(|n| (n.id.clone(), stage_of(n.info["role"].as_str().unwrap_or(""))))
                    .collect();
                let of = |s: Stage| -> Vec<String> { ts.iter().filter(|(_, st)| *st == s).map(|(i, _)| i.clone()).collect() };
                let (pre, inn, out, post, post_add) = (of(Stage::Pre), of(Stage::In), of(Stage::Out), of(Stage::Post), of(Stage::PostAdd));
                let mut frontier = vec![residual.clone()];
                for stage in [&pre, &inn, &out, &post] {
                    if stage.is_empty() {
                        continue;
                    }
                    for f in &frontier {
                        for s in stage {
                            flow(&mut edges, f, s, None);
                        }
                    }
                    frontier = stage.clone();
                }
                for f in &frontier {
                    flow(&mut edges, f, &add_id, None);
                }
                flow(&mut edges, &residual, &add_id, Some("residual"));
                residual = add_id;
                for pa in &post_add {
                    flow(&mut edges, &residual, pa, None);
                    residual = pa.clone();
                }
            }
            if let Some((bid, _)) = extra_mtp {
                flow(&mut edges, &residual, &bid, Some("speculative"));
            }
            if !is_nextn {
                cur = residual;
                last_main = cur.clone();
            }
        }
        let on = tensor_id("output_norm.weight");
        if nodes.contains_key(&on) {
            flow(&mut edges, &cur, &on, None);
            cur = on;
        }
        let head = if nodes.contains_key(&tensor_id("output.weight")) {
            Some(tensor_id("output.weight"))
        } else if nodes.contains_key("output.head") {
            Some("output.head".to_string())
        } else {
            None
        };
        if let Some(h) = head.filter(|_| sink_id == "output.logits") {
            flow(&mut edges, &cur, &h, None);
            cur = h;
        }
        flow(&mut edges, &cur, sink_id, None);
        if has_mtp_section {
            flow(&mut edges, &last_main, "mtp", Some("speculative"));
        }
        if has_vision_section {
            edges.push(Edge { from: "vision".into(), to: "input".into(), kind: EdgeKind::Uses, label: Some("image embeddings".into()) });
        }

        // --- tokenizer, template, projector ----------------------------------------------
        if let Some(v) = vocab {
            section(&mut nodes, "tokenizer", &format!("Tokenizer ({})", v.model_name), Removable::BreaksModel, "turns text into token ids");
            if let Some(n) = nodes.get_mut("tokenizer") {
                n.params = v.tokens.len() as u64;
                n.bytes = v.tokens.iter().map(|t| t.raw.len() as u64 + 12).sum::<u64>() + v.merges.iter().map(|x| x.len() as u64 + 8).sum::<u64>();
                n.info = json!({ "tokens": v.tokens.len(), "merges": v.merges.len(), "model": v.model_name, "classes": v.summary.class_counts });
            }
            if nodes.contains_key(&tensor_id("token_embd.weight")) {
                edges.push(Edge { from: "tokenizer".into(), to: tensor_id("token_embd.weight"), kind: EdgeKind::Uses, label: Some("token ids".into()) });
            }
            let report = crate::template::template_report(m);
            let mut used: BTreeSet<String> = BTreeSet::new();
            for t in &report.templates {
                used.extend(t.special_tokens.iter().cloned());
            }
            let mut ids: BTreeMap<u32, Vec<String>> = BTreeMap::new();
            for s in &v.summary.specials {
                ids.entry(s.id).or_default().push(s.role.clone());
            }
            for s in &used {
                if let Some(&id) = v.by_raw.get(s) {
                    ids.entry(id).or_default().push("template".into());
                }
            }
            for (id, roles_) in &ids {
                let Some(tok) = v.tokens.get(*id as usize) else { continue };
                let nid = format!("token.{id}");
                let mut n = new_node(&nid, NodeKind::Token, &format!("{:?}", tok.raw), Some("tokenizer"));
                n.params = 1;
                n.info = json!({ "id": id, "text": tok.raw, "roles": roles_ });
                let structural = roles_.iter().any(|r| r == "bos" || r == "eos");
                n.removable = if structural { Removable::BreaksModel } else { Removable::Risky };
                n.removable_reason = if structural {
                    "referenced by the tokenizer metadata (bos/eos)".into()
                } else {
                    "used by the chat template".into()
                };
                nodes.insert(nid, n);
            }
            if !report.templates.is_empty() {
                section(&mut nodes, "template", "Chat template", Removable::Risky, "raw-text inference still works, chat formatting is lost");
                if let Some(n) = nodes.get_mut("template") {
                    n.info = json!({
                        "templates": report.templates.iter().map(|t| json!({ "name": t.name, "decisions": t.decisions, "variables": t.variables.len() })).collect::<Vec<_>>(),
                        "special_tokens": used,
                    });
                    n.params = report.templates.iter().map(|t| t.decisions as u64).sum();
                    n.bytes = report.templates.iter().map(|t| t.source.len() as u64).sum();
                }
                for s in &used {
                    if let Some(&id) = v.by_raw.get(s) {
                        edges.push(Edge { from: "template".into(), to: format!("token.{id}"), kind: EdgeKind::Uses, label: None });
                    }
                }
            }
        }
        if let Some(p) = &m.entry.projector {
            section(&mut nodes, "projector", "Vision projector (separate file)", Removable::Safe, "text-only use does not need it; image input stops working");
            if let Some(n) = nodes.get_mut("projector") {
                n.bytes = std::fs::metadata(p).map(|md| md.len()).unwrap_or(0);
                if let Some(Ok(pg)) = m.open_projector() {
                    n.params = pg.total_params();
                    let mut q: BTreeSet<String> = BTreeSet::new();
                    for t in &pg.tensors {
                        q.insert(t.ggml_type.to_string());
                    }
                    n.quant = q.into_iter().collect();
                    n.info = json!({ "file": p, "tensors": pg.tensors.len(), "architecture": pg.architecture() });
                }
            }
            edges.push(Edge { from: "projector".into(), to: "input".into(), kind: EdgeKind::Uses, label: Some("image embeddings".into()) });
        }

        // --- aggregate sizes for sections ------------------------------------------------
        for sec in ["input", "output", "body", "mtp", "vision", "extra"] {
            let (p, b, q) = nodes.values().filter(|n| n.kind == NodeKind::Tensor).filter(|n| ancestors_of(&nodes, &n.id).iter().any(|a| a == sec)).fold(
                (0u64, 0u64, BTreeSet::new()),
                |(p, b, mut q), n| {
                    q.extend(n.quant.iter().cloned());
                    (p + n.params, b + n.bytes, q)
                },
            );
            if let Some(n) = nodes.get_mut(sec) {
                n.params = p;
                n.bytes = b;
                n.quant = q.into_iter().collect();
            }
        }
        if let Some(n) = nodes.get_mut("model") {
            n.params = g.total_params();
            n.bytes = g.file_size;
        }

        let mut graph = ModelGraph {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            digest: m.entry.digest.clone(),
            architecture: arch.name.clone(),
            kind: arch.kind,
            total_params: g.total_params(),
            file_size: g.file_size,
            tied_embeddings: tied,
            pattern_period,
            n_layers,
            nodes,
            edges,
            dims,
            fine_stats: HashMap::new(),
            fine_activation: HashMap::new(),
            fine_importance: HashMap::new(),
            notes: vec![],
        };
        graph.apply_measurements(inputs, &mut notes);
        graph.fill_contains();
        graph.notes = notes;
        graph
    }

    fn apply_measurements(&mut self, inputs: &GraphInputs, notes: &mut Vec<String>) {
        // Structural importance first; measurements override it.
        let last_main = self.n_layers.saturating_sub(1);
        for n in self.nodes.values_mut() {
            if n.removable == Removable::BreaksModel {
                n.importance = Importance { score: 1.0, source: "structural: required".into() };
            } else if n.kind == NodeKind::Layer && (n.layer == Some(0) || n.layer == Some(last_main)) {
                n.importance = Importance { score: 0.9, source: "structural: edge layer (not measured)".into() };
            } else if n.removable == Removable::Safe {
                n.importance = Importance { score: 0.0, source: "structural: optional".into() };
            }
        }
        // Weight stats.
        if let Some(s) = inputs.stats {
            let by_name: HashMap<&str, &crate::stats::TensorStats> = s.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
            for n in self.nodes.values_mut().filter(|n| n.kind == NodeKind::Tensor) {
                if let Some(t) = n.info["name"].as_str().and_then(|nm| by_name.get(nm)) {
                    n.stats = Some(StatsSummary {
                        mean: Some(t.mean),
                        std: Some(t.std),
                        l2: t.l2,
                        sparsity: Some(t.sparsity),
                        outlier_fraction: Some(t.outlier_fraction),
                    });
                }
            }
            // Blocks / layers: combined L2 norm.
            let ids: Vec<String> = self.nodes.values().filter(|n| matches!(n.kind, NodeKind::Block | NodeKind::Layer)).map(|n| n.id.clone()).collect();
            for id in ids {
                let prefix_children: Vec<f64> = self
                    .nodes
                    .values()
                    .filter(|c| c.kind == NodeKind::Tensor && ancestors_of(&self.nodes, &c.id).contains(&id))
                    .filter_map(|c| c.stats.as_ref().map(|s| s.l2 * s.l2))
                    .collect();
                if !prefix_children.is_empty() {
                    if let Some(n) = self.nodes.get_mut(&id) {
                        n.stats = Some(StatsSummary { mean: None, std: None, l2: prefix_children.iter().sum::<f64>().sqrt(), sparsity: None, outlier_fraction: None });
                    }
                }
            }
            for h in &s.heads {
                let score: Vec<f32> = if !h.o.is_empty() && h.o.len() == h.q.len() { h.q.iter().zip(&h.o).map(|(a, b)| a * b).collect() } else { h.q.clone() };
                let max = score.iter().cloned().fold(0f32, f32::max);
                for (i, v) in score.iter().enumerate() {
                    let id = format!("layer.{}.attn.head.{i}", h.layer);
                    self.fine_stats.insert(id.clone(), *v as f64);
                    if max > 0.0 {
                        self.fine_importance.insert(id, Importance { score: v / max, source: "stats: |Q|·|O| norm relative to the layer's largest head".into() });
                    }
                }
            }
            for nn in &s.ffn_neurons {
                let Some(d) = self.dims.get(&nn.layer) else { continue };
                let b = d.bucket.max(1) as usize;
                let means: Vec<f32> = nn.norms.chunks(b).map(|c| c.iter().sum::<f32>() / c.len().max(1) as f32).collect();
                let max = means.iter().cloned().fold(0f32, f32::max);
                for (i, v) in means.iter().enumerate() {
                    let s0 = i * b;
                    let e = (s0 + b).min(nn.norms.len()) - 1;
                    let id = format!("layer.{}.ffn.n.{s0}-{e}", nn.layer);
                    self.fine_stats.insert(id.clone(), *v as f64);
                    if max > 0.0 {
                        self.fine_importance.insert(id, Importance { score: v / max, source: "stats: mean neuron norm relative to the layer's largest bucket".into() });
                    }
                }
            }
            for e in &s.experts {
                let max = e.norms.iter().cloned().fold(0f32, f32::max);
                for (i, v) in e.norms.iter().enumerate() {
                    let id = format!("layer.{}.moe.expert.{i}", e.layer);
                    self.fine_stats.insert(id.clone(), *v as f64);
                    if max > 0.0 {
                        self.fine_importance.insert(id, Importance { score: v / max, source: "stats: expert norm relative to the layer's largest".into() });
                    }
                }
            }
        }
        // Trace: hidden-state change per layer, entropy per head, top neurons.
        if let Some(tr) = inputs.trace {
            let max_change = tr.layers.iter().filter_map(|l| l.cos_prev).map(|c| 1.0 - c).fold(0f32, f32::max);
            for lt in &tr.layers {
                let lid = format!("layer.{}", lt.index);
                if let Some(n) = self.nodes.get_mut(&lid) {
                    n.activation = Some(json!({
                        "hidden_norm": lt.hidden_norm,
                        "cos_in_out": lt.cos_prev,
                        "logit_lens_top": lt.logit_lens.first().map(|t| json!({ "text": t.text, "prob": t.prob })),
                        "samples": tr.samples,
                    }));
                    if let (Some(c), true) = (lt.cos_prev, max_change > 0.0) {
                        n.importance = Importance { score: ((1.0 - c) / max_change).clamp(0.0, 1.0), source: format!("trace: 1 - cos(in,out), {} sample(s)", tr.samples) };
                    }
                }
                for (h, e) in lt.head_entropy.iter().enumerate() {
                    self.fine_activation.insert(format!("layer.{}.attn.head.{h}", lt.index), json!({ "attention_entropy": e }));
                }
                if let Some(d) = self.dims.get(&lt.index) {
                    let b = d.bucket.max(1) as u32;
                    for (neuron, v) in &lt.top_neurons {
                        let s0 = neuron / b * b;
                        let e = (s0 + b).min(d.n_ff as u32) - 1;
                        let id = format!("layer.{}.ffn.n.{s0}-{e}", lt.index);
                        let entry = self.fine_activation.entry(id).or_insert_with(|| json!({ "top_neurons": [] }));
                        if let Some(a) = entry["top_neurons"].as_array_mut() {
                            a.push(json!({ "neuron": neuron, "mean_abs": v }));
                        }
                    }
                }
            }
            notes.push(format!("activations from the latest trace ({} sample(s))", tr.samples));
        }
        // Sweeps: measured damage when removed. The strongest signal.
        for sw in &inputs.sweeps {
            let max = sw.damage.values().cloned().fold(0f64, f64::max);
            for (id, d) in &sw.damage {
                let imp = Importance { score: if max > 0.0 { (d / max) as f32 } else { 0.0 }, source: format!("sweep: KL {d:.4} when removed (relative to the worst)") };
                if let Some(n) = self.nodes.get_mut(id) {
                    n.importance = imp;
                } else {
                    self.fine_importance.insert(id.clone(), imp);
                }
            }
        }
        if !inputs.sweeps.is_empty() {
            notes.push("importance from ablation sweeps where available".into());
        }
        // Blocks and tensors inherit the layer's importance unless they have their own.
        let layer_imp: HashMap<u32, Importance> =
            self.nodes.values().filter(|n| n.kind == NodeKind::Layer).filter_map(|n| n.layer.map(|l| (l, n.importance.clone()))).collect();
        for n in self.nodes.values_mut() {
            if matches!(n.kind, NodeKind::Block | NodeKind::Tensor) && n.removable != Removable::BreaksModel && n.importance.source.starts_with("structural: not") {
                if let Some(li) = n.layer.and_then(|l| layer_imp.get(&l)).filter(|li| !li.source.starts_with("structural: not")) {
                    n.importance = Importance { score: li.score, source: format!("inherited from layer: {}", li.source) };
                }
            }
        }
    }

    fn fill_contains(&mut self) {
        let mut kids: HashMap<String, Vec<ChildSummary>> = HashMap::new();
        for n in self.nodes.values() {
            if let Some(p) = &n.parent {
                kids.entry(p.clone()).or_default().push(ChildSummary { id: n.id.clone(), kind: n.kind, label: n.label.clone(), params: n.params, bytes: n.bytes });
            }
        }
        for n in self.nodes.values_mut() {
            n.contains = kids.remove(&n.id).unwrap_or_default();
        }
    }

    // --- fine level --------------------------------------------------------------------

    /// Heads (in GQA groups), neuron buckets or experts of a block, generated on demand.
    pub fn fine_children(&self, block_id: &str) -> Vec<Node> {
        let Some(b) = self.nodes.get(block_id) else { return vec![] };
        if !b.lazy {
            return vec![];
        }
        let Some(layer) = b.layer else { return vec![] };
        let Some(d) = self.dims.get(&layer) else { return vec![] };
        let tensors: Vec<&Node> = self.nodes.values().filter(|n| n.parent.as_deref() == Some(block_id) && n.kind == NodeKind::Tensor).collect();
        let role_sum = |pred: &dyn Fn(&str) -> bool| -> (u64, u64) {
            tensors
                .iter()
                .filter(|t| pred(t.info["role"].as_str().unwrap_or("")))
                .fold((0, 0), |(p, by), t| (p + t.params, by + t.bytes))
        };
        let mk = |id: String, kind: NodeKind, label: String, parent: &str, params: u64, bytes: u64, info: Value| -> Node {
            let mut n = new_node(&id, kind, &label, Some(parent));
            n.layer = Some(layer);
            n.block = b.block.clone();
            n.layer_kind = b.layer_kind.clone();
            n.params = params;
            n.bytes = bytes;
            n.quant = b.quant.clone();
            n.info = info;
            n.removable = Removable::Risky;
            n.removable_reason = "zeroed in place: shapes stay the same, so llama.cpp still loads it".into();
            n.importance = self
                .fine_importance
                .get(&id)
                .cloned()
                .unwrap_or_else(|| Importance { score: b.importance.score, source: format!("inherited from block: {}", b.importance.source) });
            n.activation = self.fine_activation.get(&id).cloned();
            n.stats = self.fine_stats.get(&id).map(|v| StatsSummary { mean: None, std: None, l2: *v, sparsity: None, outlier_fraction: None });
            n
        };
        let mut out = Vec::new();
        match b.block.as_deref() {
            Some("attention") if d.n_head > 0 => {
                let n_kv = d.n_kv.max(1);
                let group = (d.n_head / n_kv).max(1);
                let (qp, qb) = role_sum(&|r| matches!(r, "attn_q" | "attn_output"));
                let (kp, kb) = role_sum(&|r| matches!(r, "attn_k" | "attn_v"));
                let (fp, fb) = role_sum(&|r| r == "attn_qkv");
                let q_frac = d.n_head as f64 / (d.n_head + 2 * n_kv) as f64;
                let head_p = qp / d.n_head + (fp as f64 * q_frac) as u64 / d.n_head;
                let head_b = qb / d.n_head + (fb as f64 * q_frac) as u64 / d.n_head;
                let kv_p = kp / n_kv + (fp as f64 * (1.0 - q_frac)) as u64 / n_kv;
                let kv_b = kb / n_kv + (fb as f64 * (1.0 - q_frac)) as u64 / n_kv;
                for kv in 0..n_kv {
                    let gid = format!("{block_id}.kv.{kv}");
                    let heads: Vec<u64> = (kv * group..(kv + 1) * group).collect();
                    let mut gn = mk(
                        gid.clone(),
                        NodeKind::KvGroup,
                        format!("KV group {kv} (heads {}-{})", heads[0], heads[heads.len() - 1]),
                        block_id,
                        kv_p + head_p * group,
                        kv_b + head_b * group,
                        json!({ "kv_head": kv, "heads": heads, "head_dim": d.head_dim }),
                    );
                    gn.contains = heads
                        .iter()
                        .map(|h| ChildSummary { id: format!("{block_id}.head.{h}"), kind: NodeKind::Head, label: format!("head {h}"), params: head_p, bytes: head_b })
                        .collect();
                    out.push(gn);
                    for h in heads {
                        out.push(mk(
                            format!("{block_id}.head.{h}"),
                            NodeKind::Head,
                            format!("head {h}"),
                            &gid,
                            head_p,
                            head_b,
                            json!({ "head": h, "kv_head": kv, "head_dim": d.head_dim, "columns_of_attn_output": [h * d.head_dim, (h + 1) * d.head_dim] }),
                        ));
                    }
                }
            }
            Some("ffn") if d.n_ff > 0 => {
                let per = b.params as f64 / d.n_ff as f64;
                let per_b = b.bytes as f64 / d.n_ff as f64;
                let mut s0 = 0;
                while s0 < d.n_ff {
                    let e = (s0 + d.bucket).min(d.n_ff);
                    let n = e - s0;
                    out.push(mk(
                        format!("{block_id}.n.{s0}-{}", e - 1),
                        NodeKind::NeuronBucket,
                        format!("neurons {s0}-{}", e - 1),
                        block_id,
                        (per * n as f64) as u64,
                        (per_b * n as f64) as u64,
                        json!({ "start": s0, "end": e - 1, "count": n }),
                    ));
                    s0 = e;
                }
            }
            Some("moe") if d.n_expert > 0 => {
                let (ep, eb) = role_sum(&|r| r.contains("_exps"));
                for e in 0..d.n_expert {
                    out.push(mk(
                        format!("{block_id}.expert.{e}"),
                        NodeKind::Expert,
                        format!("expert {e}"),
                        block_id,
                        ep / d.n_expert,
                        eb / d.n_expert,
                        json!({ "expert": e }),
                    ));
                }
            }
            _ => {}
        }
        out
    }

    /// Accepts exact ids plus shorthands: `token_embd`, `blk.7`, `L7`, `7`, `layer.7.attention`, `vocab:10-20`.
    pub fn resolve_id(&self, q: &str) -> Option<String> {
        let q = q.trim();
        if q.starts_with("vocab:") {
            return Some(q.to_string());
        }
        if self.nodes.contains_key(q) {
            return Some(q.to_string());
        }
        for c in [format!("t.{q}"), format!("t.{q}.weight")] {
            if self.nodes.contains_key(&c) {
                return Some(c);
            }
        }
        let layer_alias = q
            .strip_prefix('L')
            .or_else(|| q.strip_prefix("layer"))
            .filter(|r| !r.starts_with('.'))
            .and_then(|r| r.parse::<u32>().ok())
            .or_else(|| q.parse::<u32>().ok());
        if let Some(l) = layer_alias {
            let id = format!("layer.{l}");
            return self.nodes.contains_key(&id).then_some(id);
        }
        if let Some((l, tail)) = parse_layer_prefix(q) {
            let mut parts: Vec<&str> = tail.split('.').filter(|s| !s.is_empty()).collect();
            if let Some(first) = parts.first_mut() {
                if let Some(b) = block_from_short(first) {
                    *first = block_short(b);
                }
            }
            let id = if parts.is_empty() { format!("layer.{l}") } else { format!("layer.{l}.{}", parts.join(".")) };
            if self.nodes.contains_key(&id) {
                return Some(id);
            }
            if self.fine_node(&id).is_some() {
                return Some(id);
            }
            // blk.N.attn_q → tensor
            for c in [format!("t.blk.{l}.{tail}"), format!("t.blk.{l}.{tail}.weight")] {
                if self.nodes.contains_key(&c) {
                    return Some(c);
                }
            }
        }
        None
    }

    /// The block id that owns a fine node id, if the id has a fine shape.
    pub fn fine_parent_block(id: &str) -> Option<String> {
        let (l, tail) = parse_layer_prefix(id)?;
        let mut it = tail.split('.');
        let b = it.next()?;
        let kind = it.next()?;
        matches!(kind, "kv" | "head" | "n" | "expert").then(|| format!("layer.{l}.{b}"))
    }

    pub fn fine_node(&self, id: &str) -> Option<Node> {
        let block = Self::fine_parent_block(id)?;
        self.fine_children(&block).into_iter().find(|n| n.id == id)
    }

    /// Structural or fine node.
    pub fn get(&self, id: &str) -> Option<Node> {
        self.nodes.get(id).cloned().or_else(|| self.fine_node(id))
    }

    /// Children of a node, including generated fine children.
    pub fn children(&self, id: &str) -> Vec<Node> {
        let mut out: Vec<Node> = self.nodes.values().filter(|n| n.parent.as_deref() == Some(id)).cloned().collect();
        if let Some(n) = self.nodes.get(id) {
            if n.lazy {
                out.extend(self.fine_children(id).into_iter().filter(|c| c.parent.as_deref() == Some(id)));
            }
        } else if Self::fine_parent_block(id).is_some() {
            let block = Self::fine_parent_block(id).unwrap_or_default();
            out.extend(self.fine_children(&block).into_iter().filter(|c| c.parent.as_deref() == Some(id)));
        }
        out
    }

    /// Ancestor chain (closest first), for structural and fine nodes.
    pub fn ancestors(&self, id: &str) -> Vec<String> {
        if let Some(n) = self.fine_node(id) {
            let mut out = vec![];
            let mut cur = n.parent.clone();
            while let Some(p) = cur {
                cur = self.get(&p).and_then(|x| x.parent);
                out.push(p);
            }
            return out;
        }
        ancestors_of(&self.nodes, id)
    }

    /// Every tensor node under `id` (or `id` itself when it is a tensor).
    pub fn tensors_under(&self, id: &str) -> Vec<&Node> {
        self.nodes
            .values()
            .filter(|n| n.kind == NodeKind::Tensor && (n.id == id || ancestors_of(&self.nodes, &n.id).iter().any(|a| a == id)))
            .collect()
    }

    pub fn detail(&self, id: &str) -> Option<NodeDetail> {
        let node = self.get(id)?;
        let mut node = node;
        let children = self.children(id);
        if node.lazy {
            node.contains = children.iter().filter(|c| c.kind.is_fine()).map(|c| ChildSummary { id: c.id.clone(), kind: c.kind, label: c.label.clone(), params: c.params, bytes: c.bytes }).collect::<Vec<_>>();
            node.contains.extend(self.nodes.get(id).map(|n| n.contains.clone()).unwrap_or_default());
        }
        let ancestors = self
            .ancestors(id)
            .iter()
            .filter_map(|a| self.get(a))
            .map(|a| ChildSummary { id: a.id.clone(), kind: a.kind, label: a.label.clone(), params: a.params, bytes: a.bytes })
            .collect();
        // Connections: base edges with exactly one endpoint inside the subtree, the other
        // end lifted to the ancestor at the node's own depth.
        let inside = |x: &str| x == id || self.ancestors(x).iter().any(|a| a == id);
        let depth = self.ancestors(id).len();
        let lift = |x: &str| -> String {
            let anc = self.ancestors(x);
            if anc.len() <= depth {
                return x.to_string();
            }
            let mut chain = vec![x.to_string()];
            chain.extend(anc);
            // chain[k] has depth len-1-k
            let target = chain.len() - 1 - depth;
            chain[target].clone()
        };
        let mut incoming = BTreeSet::new();
        let mut outgoing = BTreeSet::new();
        for e in &self.edges {
            let (fi, ti) = (inside(&e.from), inside(&e.to));
            if fi && !ti {
                outgoing.insert((id.to_string(), lift(&e.to), e.kind as u8, e.label.clone()));
            } else if ti && !fi {
                incoming.insert((lift(&e.from), id.to_string(), e.kind as u8, e.label.clone()));
            }
        }
        let to_edge = |(f, t, k, l): (String, String, u8, Option<String>)| Edge {
            from: f,
            to: t,
            kind: match k {
                0 => EdgeKind::Dataflow,
                1 => EdgeKind::Contains,
                2 => EdgeKind::Tied,
                _ => EdgeKind::Uses,
            },
            label: l,
        };
        Some(NodeDetail {
            node,
            ancestors,
            children,
            incoming: incoming.into_iter().map(to_edge).collect(),
            outgoing: outgoing.into_iter().map(to_edge).collect(),
        })
    }

    /// All node ids down to tensor level (for completion).
    pub fn ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    // --- views -------------------------------------------------------------------------

    fn rank(&self, n: &Node) -> u8 {
        match n.kind {
            NodeKind::Model => 0,
            NodeKind::Section => 1,
            NodeKind::Layer => 2,
            NodeKind::Tensor | NodeKind::Token if n.layer.is_none() => {
                // Input/output tensors are few and part of the main path; side sections
                // (vision encoder, MTP head, unknown tensors) can hold hundreds.
                if n.kind == NodeKind::Token || !matches!(n.parent.as_deref(), Some("input" | "output")) {
                    4
                } else {
                    2
                }
            }
            NodeKind::Block | NodeKind::Op if n.layer.is_some() => 3,
            NodeKind::Op => 2,
            NodeKind::Block => 3,
            NodeKind::Tensor | NodeKind::Token => 4,
            NodeKind::Group => 1,
            _ => 5,
        }
    }

    pub fn view(&self, o: &ViewOptions) -> GraphView {
        let lr = o.level.rank();
        let mut visible: IndexMap<String, Node> = IndexMap::new();
        for n in self.nodes.values() {
            if n.kind != NodeKind::Model && self.rank(n) <= lr {
                visible.insert(n.id.clone(), n.clone());
            }
        }
        if o.level == Level::Fine {
            let blocks: Vec<String> = self.nodes.values().filter(|n| n.lazy).map(|n| n.id.clone()).collect();
            for b in blocks {
                for c in self.fine_children(&b) {
                    visible.insert(c.id.clone(), c);
                }
            }
        }
        let mut expanded = Vec::new();
        for e in &o.expand {
            let Some(id) = self.resolve_id(e) else { continue };
            expanded.push(id.clone());
            for a in self.ancestors(&id) {
                if a != "model" {
                    if let Some(n) = self.get(&a) {
                        visible.entry(a).or_insert(n);
                    }
                }
            }
            if let Some(n) = self.get(&id) {
                visible.entry(id.clone()).or_insert(n);
            }
            for c in self.children(&id) {
                visible.entry(c.id.clone()).or_insert(c);
            }
        }

        let mut nodes: Vec<Node>;
        let mut parent_of: HashMap<String, Option<String>> = HashMap::new();
        if o.group_by == GroupBy::Section {
            nodes = visible.values().cloned().collect();
            for n in &mut nodes {
                if n.parent.as_deref() == Some("model") {
                    n.parent = None;
                }
                parent_of.insert(n.id.clone(), n.parent.clone());
            }
        } else {
            // Units: the deepest visible kind of the requested level; everything structural
            // above them is replaced by synthetic group nodes.
            let unit_rank = visible.values().map(|n| self.rank(n)).max().unwrap_or(1).min(lr.max(2));
            let mut groups: IndexMap<String, Node> = IndexMap::new();
            nodes = Vec::new();
            for n in visible.values() {
                let r = self.rank(n);
                let is_unit = r == unit_rank || (r < unit_rank && n.kind == NodeKind::Section && !visible.values().any(|c| c.parent.as_deref() == Some(&n.id)));
                let side = n.layer.is_none() && n.kind != NodeKind::Group && !matches!(o.group_by, GroupBy::Quant | GroupBy::Importance);
                if !is_unit && !(r > unit_rank) {
                    continue;
                }
                let mut c = n.clone();
                if r > unit_rank {
                    // Deeper nodes from `expand` keep their structural parent.
                    parent_of.insert(c.id.clone(), c.parent.clone());
                    nodes.push(c);
                    continue;
                }
                let key: Option<(String, String)> = match o.group_by {
                    GroupBy::Section => None,
                    GroupBy::Layer => n.layer.map(|l| (format!("group.layer.{l}"), format!("layer {l}"))),
                    GroupBy::Block => n.block.clone().map(|b| (format!("group.block.{b}"), format!("all {}", block_label(&b)))),
                    GroupBy::Quant => Some({
                        let q = if n.quant.is_empty() { "none".to_string() } else { n.quant.join("+") };
                        (format!("group.quant.{q}"), format!("quant {q}"))
                    }),
                    GroupBy::Kind => n.layer_kind.clone().map(|k| (format!("group.kind.{k}"), k)),
                    GroupBy::Importance => Some({
                        let s = n.importance.score;
                        let b = if n.importance.source.starts_with("structural") && n.removable != Removable::BreaksModel {
                            "unmeasured"
                        } else if s >= 0.66 {
                            "high"
                        } else if s >= 0.33 {
                            "medium"
                        } else {
                            "low"
                        };
                        (format!("group.importance.{b}"), format!("importance {b}"))
                    }),
                };
                match key {
                    Some((gid, glabel)) => {
                        let gn = groups.entry(gid.clone()).or_insert_with(|| {
                            let mut g = new_node(&gid, NodeKind::Group, &glabel, None);
                            g.removable = Removable::Safe;
                            g.importance = Importance { score: 0.0, source: "group".into() };
                            g
                        });
                        gn.params += n.params;
                        gn.bytes += n.bytes;
                        gn.removable = gn.removable.max(n.removable);
                        gn.contains.push(ChildSummary { id: n.id.clone(), kind: n.kind, label: n.label.clone(), params: n.params, bytes: n.bytes });
                        c.parent = Some(gid);
                    }
                    None => {
                        c.parent = if side { n.parent.clone().filter(|p| p != "model" && visible.contains_key(p)) } else { None };
                        if let Some(p) = &c.parent {
                            if !nodes.iter().any(|x| &x.id == p) {
                                if let Some(pn) = visible.get(p) {
                                    let mut pn = pn.clone();
                                    pn.parent = None;
                                    parent_of.insert(pn.id.clone(), None);
                                    nodes.push(pn);
                                }
                            }
                        }
                    }
                }
                parent_of.insert(c.id.clone(), c.parent.clone());
                nodes.push(c);
            }
            for (gid, g) in groups {
                parent_of.insert(gid, None);
                nodes.insert(0, g);
            }
        }
        let vis: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();

        // Lift edges to the nearest visible ancestor.
        let lift = |x: &str| -> Option<String> {
            if vis.contains(x) {
                return Some(x.to_string());
            }
            self.ancestors(x).into_iter().find(|a| vis.contains(a))
        };
        let mut seen: HashSet<(String, String, EdgeKind)> = HashSet::new();
        let mut edges = Vec::new();
        for e in &self.edges {
            let (Some(f), Some(t)) = (lift(&e.from), lift(&e.to)) else { continue };
            if f == t {
                continue;
            }
            // Edges into a node's own ancestor are containment, not flow.
            if self.ancestors(&f).contains(&t) || self.ancestors(&t).contains(&f) {
                continue;
            }
            if seen.insert((f.clone(), t.clone(), e.kind)) {
                let label = if e.from == f && e.to == t { e.label.clone() } else { e.label.clone().filter(|l| l != "residual") };
                edges.push(Edge { from: f, to: t, kind: e.kind, label });
            }
        }
        for n in &nodes {
            if let Some(Some(p)) = parent_of.get(&n.id) {
                if vis.contains(p) {
                    edges.push(Edge { from: p.clone(), to: n.id.clone(), kind: EdgeKind::Contains, label: None });
                }
            }
        }
        GraphView {
            schema_version: SCHEMA_VERSION,
            model: self.model.clone(),
            architecture: self.architecture.clone(),
            kind: self.kind,
            level: o.level,
            group_by: o.group_by,
            expand: expanded,
            nodes,
            edges,
            notes: self.notes.clone(),
        }
    }
}

// --- exports -----------------------------------------------------------------------------

fn removable_color(r: Removable) -> &'static str {
    match r {
        Removable::Safe => "#c8f0c8",
        Removable::Risky => "#fde9b6",
        Removable::BreaksModel => "#f7c4c0",
    }
}

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn short_size(n: &Node) -> String {
    if n.params == 0 && n.bytes == 0 {
        String::new()
    } else {
        format!("{} · {}", human_count(n.params), human_bytes(n.bytes))
    }
}

impl GraphView {
    fn children_map(&self) -> (Vec<&Node>, HashMap<&str, Vec<&Node>>) {
        let ids: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        let mut kids: HashMap<&str, Vec<&Node>> = HashMap::new();
        let mut roots = Vec::new();
        for n in &self.nodes {
            match n.parent.as_deref().filter(|p| ids.contains(p)) {
                Some(p) => kids.entry(p).or_default().push(n),
                None => roots.push(n),
            }
        }
        (roots, kids)
    }

    pub fn to_tree(&self) -> String {
        let (roots, kids) = self.children_map();
        let mut s = format!(
            "{} ({}, {}) · level {} · group by {}\n",
            self.model,
            self.architecture,
            enum_str(&self.kind),
            self.level.as_str(),
            self.group_by.as_str()
        );
        fn walk(s: &mut String, n: &Node, kids: &HashMap<&str, Vec<&Node>>, depth: usize) {
            let imp = if n.importance.source.starts_with("structural: not") || n.kind == NodeKind::Group { String::new() } else { format!("  imp {:.2}", n.importance.score) };
            let q = if n.quant.is_empty() { String::new() } else { format!("  [{}]", n.quant.join(",")) };
            let _ = writeln!(
                s,
                "{}{} {}  ({}){}{}  {}  #{}",
                "  ".repeat(depth),
                if kids.contains_key(n.id.as_str()) { "▾" } else if n.lazy { "▸" } else { "•" },
                n.label,
                n.kind.as_str(),
                if short_size(n).is_empty() { String::new() } else { format!("  {}", short_size(n)) },
                q,
                n.removable.as_str(),
                n.id
            );
            let _ = imp.is_empty() || {
                s.pop();
                let _ = writeln!(s, "{imp}");
                true
            };
            if let Some(cs) = kids.get(n.id.as_str()) {
                for c in cs {
                    walk(s, c, kids, depth + 1);
                }
            }
        }
        for r in roots {
            walk(&mut s, r, &kids, 0);
        }
        let flows = self.edges.iter().filter(|e| e.kind != EdgeKind::Contains).count();
        let _ = writeln!(s, "{} nodes, {} connections (▸ = expandable with --expand <id>)", self.nodes.len(), flows);
        s
    }

    pub fn to_dot(&self) -> String {
        let (roots, kids) = self.children_map();
        let mut s = String::from("digraph neurondump {\n  rankdir=TB;\n  compound=true;\n  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10];\n  edge [fontname=\"Helvetica\", fontsize=8];\n");
        let _ = writeln!(s, "  label=\"{} ({})\";", dot_escape(&self.model), dot_escape(&self.architecture));
        let mut idx: HashMap<&str, usize> = HashMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            idx.insert(&n.id, i);
        }
        fn emit(s: &mut String, n: &Node, kids: &HashMap<&str, Vec<&Node>>, idx: &HashMap<&str, usize>, depth: usize) {
            let pad = "  ".repeat(depth + 1);
            let size = short_size(n);
            let label = if size.is_empty() { dot_escape(&n.label) } else { format!("{}\\n{}", dot_escape(&n.label), size) };
            match kids.get(n.id.as_str()) {
                Some(cs) => {
                    let _ = writeln!(s, "{pad}subgraph cluster_{} {{", idx[n.id.as_str()]);
                    let _ = writeln!(s, "{pad}  label=\"{label}\"; style=\"rounded\"; color=\"#888888\";");
                    // An anchor node so edges to the cluster have an endpoint.
                    let _ = writeln!(s, "{pad}  n{} [label=\"{}\", shape=plaintext, style=\"\"];", idx[n.id.as_str()], dot_escape(&n.label));
                    for c in cs {
                        emit(s, c, kids, idx, depth + 1);
                    }
                    let _ = writeln!(s, "{pad}}}");
                }
                None => {
                    let shape = match n.kind {
                        NodeKind::Op => ", shape=ellipse",
                        NodeKind::Token => ", shape=note",
                        _ => "",
                    };
                    let _ = writeln!(s, "{pad}n{} [label=\"{label}\", fillcolor=\"{}\"{shape}];", idx[n.id.as_str()], removable_color(n.removable));
                }
            }
        }
        for r in roots {
            emit(&mut s, r, &kids, &idx, 0);
        }
        for e in &self.edges {
            if e.kind == EdgeKind::Contains {
                continue;
            }
            let (Some(f), Some(t)) = (idx.get(e.from.as_str()), idx.get(e.to.as_str())) else { continue };
            let style = match e.kind {
                EdgeKind::Dataflow => "",
                EdgeKind::Tied => ", style=dashed, color=\"#c0392b\", dir=both",
                EdgeKind::Uses => ", style=dotted, color=\"#555555\"",
                EdgeKind::Contains => "",
            };
            let label = e.label.as_ref().map(|l| format!(", label=\"{}\"", dot_escape(l))).unwrap_or_default();
            let _ = writeln!(s, "  n{f} -> n{t} [{}{style}{label}];", if e.label.as_deref() == Some("residual") { "color=\"#999999\"" } else { "" }.trim_start_matches(", "));
        }
        s.push_str("}\n");
        s.replace("[, ", "[")
    }

    pub fn to_mermaid(&self) -> String {
        let (roots, kids) = self.children_map();
        let mut idx: HashMap<&str, usize> = HashMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            idx.insert(&n.id, i);
        }
        let mut s = String::from("flowchart TD\n");
        let mlabel = |n: &Node| -> String {
            let size = short_size(n);
            let text = if size.is_empty() { n.label.clone() } else { format!("{}<br/>{}", n.label, size) };
            text.replace('"', "#quot;")
        };
        fn emit(s: &mut String, n: &Node, kids: &HashMap<&str, Vec<&Node>>, idx: &HashMap<&str, usize>, depth: usize, mlabel: &dyn Fn(&Node) -> String) {
            let pad = "  ".repeat(depth + 1);
            match kids.get(n.id.as_str()) {
                Some(cs) => {
                    let _ = writeln!(s, "{pad}subgraph g{}[\"{}\"]", idx[n.id.as_str()], mlabel(n));
                    for c in cs {
                        emit(s, c, kids, idx, depth + 1, mlabel);
                    }
                    let _ = writeln!(s, "{pad}end");
                }
                None => {
                    let (o, c) = match n.kind {
                        NodeKind::Op => ("((", "))"),
                        NodeKind::Token => ("[/", "/]"),
                        _ => ("[", "]"),
                    };
                    let _ = writeln!(s, "{pad}n{}{o}\"{}\"{c}:::{}", idx[n.id.as_str()], mlabel(n), n.removable.as_str());
                }
            }
        }
        for r in roots {
            emit(&mut s, r, &kids, &idx, 0, &mlabel);
        }
        for e in &self.edges {
            if e.kind == EdgeKind::Contains {
                continue;
            }
            let (Some(f), Some(t)) = (idx.get(e.from.as_str()), idx.get(e.to.as_str())) else { continue };
            let pref = |i: usize| if kids.contains_key(self.nodes[i].id.as_str()) { format!("g{i}") } else { format!("n{i}") };
            let arrow = match e.kind {
                EdgeKind::Dataflow => "-->",
                EdgeKind::Tied => "<-.->",
                EdgeKind::Uses => "-.->",
                EdgeKind::Contains => "---",
            };
            match &e.label {
                Some(l) => {
                    let _ = writeln!(s, "  {} {arrow}|{}| {}", pref(*f), l.replace('"', "'"), pref(*t));
                }
                None => {
                    let _ = writeln!(s, "  {} {arrow} {}", pref(*f), pref(*t));
                }
            }
        }
        s.push_str("  classDef safe fill:#c8f0c8,stroke:#3a7d3a;\n  classDef risky fill:#fde9b6,stroke:#a77b00;\n  classDef breaks_model fill:#f7c4c0,stroke:#b42318;\n");
        s
    }
}
