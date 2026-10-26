//! Per-architecture forward-pass description (RoPE, activation, norms) and GGUF â†’ candle types.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum RopeKind {
    /// Adjacent pairs (llama.cpp LLAMA_ROPE_TYPE_NORM).
    Norm,
    /// Split halves (LLAMA_ROPE_TYPE_NEOX).
    Neox,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Act {
    Silu,
    Gelu,
}

/// Architecture description driving the generic forward passes.
#[derive(Debug, Clone)]
pub(super) struct ArchSpec {
    pub(super) encoder: bool,
    pub(super) rope: RopeKind,
    pub(super) act: Act,
    pub(super) embed_scale: bool,
    pub(super) layer_norm: bool,
}

pub(super) fn spec_for(arch: &str) -> Option<ArchSpec> {
    Some(match arch {
        "llama" => ArchSpec { encoder: false, rope: RopeKind::Norm, act: Act::Silu, embed_scale: false, layer_norm: false },
        "qwen2" | "qwen3" | "phi3" => ArchSpec { encoder: false, rope: RopeKind::Neox, act: Act::Silu, embed_scale: false, layer_norm: false },
        "gemma" | "gemma2" | "gemma3" => ArchSpec { encoder: false, rope: RopeKind::Neox, act: Act::Gelu, embed_scale: true, layer_norm: false },
        "nomic-bert" => ArchSpec { encoder: true, rope: RopeKind::Neox, act: Act::Silu, embed_scale: false, layer_norm: true },
        "bert" => ArchSpec { encoder: true, rope: RopeKind::None, act: Act::Gelu, embed_scale: false, layer_norm: true },
        _ => return None,
    })
}

pub(super) fn candle_dtype(t: GgmlType) -> Option<GgmlDType> {
    Some(match t.0 {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        _ => return None,
    })
}
