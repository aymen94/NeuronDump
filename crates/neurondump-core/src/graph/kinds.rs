//! Node / edge kinds, removability, view levels and grouping, with their string forms.

use super::*;

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

    pub(super) fn rank(self) -> u8 {
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
