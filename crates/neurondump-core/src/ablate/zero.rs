//! Zeroing in place: which tensor rows / columns / slices each cut zeroes, and requantization cost.

use super::*;

/// One zeroing operation on a tensor (element ranges are half-open).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZeroOp {
    pub tensor: String,
    #[serde(default)]
    pub all: bool,
    /// Ranges of rows (along dims[1]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<(u64, u64)>,
    /// Ranges of elements inside every row (along dims[0]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cols: Vec<(u64, u64)>,
    /// Ranges along dims[2] (MoE experts).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slices: Vec<(u64, u64)>,
}

/// File-level zero operations equivalent to the zeroing part of an ablation.
pub fn zero_ops(g: &GgufFile, graph: &ModelGraph, a: &Ablation) -> Vec<ZeroOp> {
    let mut ops: BTreeMap<String, ZeroOp> = BTreeMap::new();
    fn get<'a>(ops: &'a mut BTreeMap<String, ZeroOp>, g: &GgufFile, name: &str) -> Option<&'a mut ZeroOp> {
        g.tensor(name)?;
        Some(ops.entry(name.to_string()).or_insert_with(|| ZeroOp { tensor: name.to_string(), ..Default::default() }))
    }
    macro_rules! op {
        ($name:expr) => {
            get(&mut ops, g, &$name)
        };
    }
    for t in &a.zero_tensors {
        if let Some(o) = op!(t) {
            o.all = true;
        }
    }
    for (l, b) in &a.zero_blocks {
        let bid = format!("layer.{l}.{}", match b.as_str() {
            "attention" => "attn",
            other => other,
        });
        for name in block_out_tensors(graph, &bid) {
            if let Some(o) = op!(name) {
                o.all = true;
            }
            let bias = name.trim_end_matches(".weight").to_string() + ".bias";
            if let Some(o) = op!(bias) {
                o.all = true;
            }
        }
    }
    for (l, h) in &a.zero_heads {
        let hd = graph.dims.get(l).map_or(0, |d| d.head_dim);
        if hd == 0 {
            continue;
        }
        let range = (*h as u64 * hd, (*h as u64 + 1) * hd);
        if let Some(o) = op!(format!("blk.{l}.attn_output.weight")) {
            o.cols.push(range);
        }
    }
    for (l, s, e) in &a.zero_neurons {
        let n_ff = graph.dims.get(l).map_or(0, |d| d.n_ff);
        let range = (*s as u64, *e as u64 + 1);
        let gate = format!("blk.{l}.ffn_gate.weight");
        let up = format!("blk.{l}.ffn_up.weight");
        let fused = g.tensor(&gate).is_none() && g.tensor(&up).and_then(|t| t.dims.get(1).copied()) == Some(2 * n_ff);
        if let Some(o) = op!(up) {
            o.rows.push(range);
            if fused {
                o.rows.push((range.0 + n_ff, range.1 + n_ff));
            }
        }
        if let Some(o) = op!(gate) {
            o.rows.push(range);
        }
        for b in [format!("blk.{l}.ffn_up.bias"), format!("blk.{l}.ffn_gate.bias")] {
            if let Some(o) = op!(b) {
                o.cols.push(range);
            }
        }
    }
    for (l, e) in &a.zero_experts {
        for r in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"] {
            if let Some(o) = op!(format!("blk.{l}.{r}.weight")) {
                o.slices.push((*e as u64, *e as u64 + 1));
            }
        }
    }
    ops.into_values().collect()
}

/// Bytes added by zeroing columns of quantized tensors that need requantizing to Q8_0.
pub(super) fn requant_delta(g: &GgufFile, ops: &[ZeroOp]) -> (i64, Vec<String>) {
    let mut delta = 0i64;
    let mut names = Vec::new();
    for o in ops.iter().filter(|o| !o.all && !o.cols.is_empty()) {
        let Some(t) = g.tensor(&o.tensor) else { continue };
        if !needs_requant(t.ggml_type, &o.cols) {
            continue;
        }
        if let Some(q) = GgmlType::Q8_0.bytes_for(t.n_elements) {
            delta += q as i64 - t.n_bytes as i64;
            names.push(o.tensor.clone());
        }
    }
    (delta, names)
}

/// Column zeroing works in place for float types and block-aligned ranges.
pub fn needs_requant(ty: GgmlType, cols: &[(u64, u64)]) -> bool {
    let Some(layout) = ty.layout() else { return true };
    if !layout.quantized {
        return false;
    }
    cols.iter().any(|(a, b)| a % layout.block_size != 0 || b % layout.block_size != 0)
}
