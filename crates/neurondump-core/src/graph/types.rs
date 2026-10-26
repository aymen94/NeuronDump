//! Graph data types: nodes, edges, summaries, inputs, the graph itself, views and node details.

use super::*;

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
    pub(super) fine_stats: HashMap<String, f64>,
    #[serde(skip)]
    pub(super) fine_activation: HashMap<String, Value>,
    #[serde(skip)]
    pub(super) fine_importance: HashMap<String, Importance>,
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
