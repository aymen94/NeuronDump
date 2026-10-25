//! Streaming weight statistics. Each tensor is dequantized in bounded row chunks
//! (at most ~1M floats per thread) and reduced in two passes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use neurondump_gguf::{can_dequantize, GgmlType, GgufFile, TensorInfo};

use crate::arch::{parse_tensor_name, per_layer, Hyperparams};
use crate::util::{check_cancel, Progress, SCHEMA_VERSION};

pub const HIST_BINS: usize = 64;
const CHUNK_ELEMS: u64 = 1 << 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub min: f32,
    pub max: f32,
    pub bins: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStats {
    pub name: String,
    pub layer: Option<u32>,
    pub role: String,
    pub ggml_type: String,
    pub dims: Vec<u64>,
    pub n: u64,
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub std: f64,
    pub l2: f64,
    /// Fraction of exact zeros.
    pub sparsity: f64,
    /// Values with |x - mean| > 6 std.
    pub outliers: u64,
    pub outlier_fraction: f64,
    pub nan_or_inf: u64,
    pub histogram: Histogram,
    /// L2 norm of each row (`dims[0]`-length vectors): output units of a projection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_norms: Option<Vec<f32>>,
    /// L2 norm of each column: input units (e.g. FFN neurons of `ffn_down`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub col_norms: Option<Vec<f32>>,
    /// L2 norm of each slice along the last dim of 3D tensors (MoE experts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slice_norms: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadNorms {
    pub layer: u32,
    pub head_dim: u64,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronNorms {
    pub layer: u32,
    /// sqrt(|up_i|² + |gate_i|² + |down_:,i|²) per FFN neuron.
    pub norms: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertNorms {
    pub layer: u32,
    pub norms: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsReport {
    pub schema_version: u32,
    pub model: String,
    pub digest: String,
    pub tensors: Vec<TensorStats>,
    pub heads: Vec<HeadNorms>,
    pub ffn_neurons: Vec<NeuronNorms>,
    pub experts: Vec<ExpertNorms>,
    pub skipped: Vec<(String, String)>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatsOptions {
    /// Only these tensors (exact names); all when empty.
    #[serde(default)]
    pub tensors: Vec<String>,
    /// Only tensors of these layers; all when empty.
    #[serde(default)]
    pub layers: Vec<u32>,
}

fn wants_rows(role: &str) -> bool {
    matches!(role, "attn_q" | "attn_k" | "attn_v" | "attn_qkv" | "ffn_up" | "ffn_gate" | "ffn_gate_up" | "token_embd" | "output")
}

fn wants_cols(role: &str) -> bool {
    matches!(role, "attn_output" | "ffn_down")
}

/// Statistics for one tensor; `on_bytes` is called with the bytes processed per chunk.
pub fn tensor_stats(g: &GgufFile, t: &TensorInfo, p: &dyn Progress, on_bytes: &dyn Fn(u64)) -> Result<TensorStats> {
    let role = parse_tensor_name(&t.name);
    let rows = t.n_rows();
    let row_len = t.row_len();
    let rows_per_chunk = (CHUNK_ELEMS / row_len.max(1)).max(1);
    let n_slices = if t.dims.len() >= 3 { *t.dims.last().unwrap_or(&1) } else { 1 };
    let rows_per_slice = (rows / n_slices.max(1)).max(1);
    let keep_rows = wants_rows(&role.role) && rows <= 4_000_000;
    let keep_cols = wants_cols(&role.role) && row_len <= 4_000_000;
    let keep_slices = n_slices > 1;
    let row_bytes = t.n_bytes / rows.max(1);

    let mut row_sq = if keep_rows { vec![0f64; rows as usize] } else { vec![] };
    let mut col_sq = if keep_cols { vec![0f64; row_len as usize] } else { vec![] };
    let mut slice_sq = if keep_slices { vec![0f64; n_slices as usize] } else { vec![] };
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut sum, mut sumsq) = (0f64, 0f64);
    let (mut zeros, mut bad, mut n) = (0u64, 0u64, 0u64);
    let mut buf = Vec::new();

    let mut start = 0;
    while start < rows {
        check_cancel(p)?;
        let cnt = rows_per_chunk.min(rows - start);
        g.dequantize_rows(t, start, cnt, &mut buf)?;
        for (ri, row) in buf.chunks_exact(row_len as usize).enumerate() {
            let r = start + ri as u64;
            let mut rs = 0f64;
            for (ci, &x) in row.iter().enumerate() {
                if !x.is_finite() {
                    bad += 1;
                    continue;
                }
                min = min.min(x);
                max = max.max(x);
                let xd = x as f64;
                sum += xd;
                rs += xd * xd;
                if x == 0.0 {
                    zeros += 1;
                }
                if keep_cols {
                    col_sq[ci] += xd * xd;
                }
            }
            n += row.len() as u64;
            sumsq += rs;
            if keep_rows {
                row_sq[r as usize] = rs;
            }
            if keep_slices {
                slice_sq[((r / rows_per_slice) as usize).min(n_slices as usize - 1)] += rs;
            }
        }
        on_bytes(cnt * row_bytes / 2);
        start += cnt;
    }
    let finite = (n - bad).max(1) as f64;
    let mean = sum / finite;
    let var = (sumsq / finite - mean * mean).max(0.0);
    let std = var.sqrt();
    if !min.is_finite() {
        min = 0.0;
        max = 0.0;
    }

    // Pass 2: histogram + outliers.
    let mut bins = vec![0u64; HIST_BINS];
    let width = (max - min) as f64 / HIST_BINS as f64;
    let mut outliers = 0u64;
    let thresh = 6.0 * std;
    let mut start = 0;
    while start < rows {
        check_cancel(p)?;
        let cnt = rows_per_chunk.min(rows - start);
        g.dequantize_rows(t, start, cnt, &mut buf)?;
        for &x in &buf {
            if !x.is_finite() {
                continue;
            }
            let b = if width > 0.0 { (((x - min) as f64 / width) as usize).min(HIST_BINS - 1) } else { 0 };
            bins[b] += 1;
            if std > 0.0 && ((x as f64) - mean).abs() > thresh {
                outliers += 1;
            }
        }
        on_bytes(cnt * row_bytes - cnt * row_bytes / 2);
        start += cnt;
    }

    let sq = |v: Vec<f64>| v.into_iter().map(|x| x.sqrt() as f32).collect::<Vec<f32>>();
    Ok(TensorStats {
        name: t.name.clone(),
        layer: role.layer,
        role: role.role,
        ggml_type: t.ggml_type.name().into_owned(),
        dims: t.dims.clone(),
        n,
        min,
        max,
        mean,
        std,
        l2: sumsq.sqrt(),
        sparsity: zeros as f64 / n.max(1) as f64,
        outliers,
        outlier_fraction: outliers as f64 / n.max(1) as f64,
        nan_or_inf: bad,
        histogram: Histogram { min, max, bins },
        row_norms: keep_rows.then(|| sq(row_sq)),
        col_norms: keep_cols.then(|| sq(col_sq)),
        slice_norms: keep_slices.then(|| sq(slice_sq)),
    })
}

fn group_norms(v: &[f32], groups: u64) -> Vec<f32> {
    if groups == 0 || v.is_empty() || v.len() as u64 % groups != 0 {
        return vec![];
    }
    let per = v.len() / groups as usize;
    v.chunks_exact(per).map(|c| c.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as f32).collect()
}

pub fn compute_stats(g: &GgufFile, model: &str, digest: &str, opts: &StatsOptions, p: &dyn Progress) -> Result<StatsReport> {
    let started = Instant::now();
    let selected: Vec<&TensorInfo> = g
        .tensors
        .iter()
        .filter(|t| opts.tensors.is_empty() || opts.tensors.contains(&t.name))
        .filter(|t| opts.layers.is_empty() || parse_tensor_name(&t.name).layer.is_some_and(|l| opts.layers.contains(&l)))
        .collect();
    let mut skipped = Vec::new();
    let todo: Vec<&TensorInfo> = selected
        .into_iter()
        .filter(|t| {
            let ok = can_dequantize(t.ggml_type) && t.row_len() % t.ggml_type.layout().map_or(1, |l| l.block_size) == 0;
            if !ok {
                skipped.push((t.name.clone(), format!("dequantization of {} not supported", t.ggml_type)));
            }
            ok
        })
        .collect();
    let total: u64 = todo.iter().map(|t| t.n_bytes).sum::<u64>().max(1);
    let done = AtomicU64::new(0);
    let last_step = Mutex::new(String::new());
    let results: Vec<Result<TensorStats>> = todo
        .par_iter()
        .map(|t| {
            check_cancel(p)?;
            if let Ok(mut s) = last_step.lock() {
                *s = t.name.clone();
            }
            let on_bytes = |b: u64| {
                let d = done.fetch_add(b, Ordering::Relaxed) + b;
                let step = last_step.lock().map(|s| s.clone()).unwrap_or_default();
                p.set((d as f64 / total as f64 * 100.0) as f32, &step);
            };
            tensor_stats(g, t, p, &on_bytes)
        })
        .collect();
    let mut tensors = Vec::with_capacity(results.len());
    for r in results {
        tensors.push(r?);
    }
    tensors.sort_by(|a, b| a.layer.cmp(&b.layer).then(a.name.cmp(&b.name)));

    let hp = Hyperparams::from_gguf(g);
    let hc_key = g.arch_key("attention.head_count");
    let hkv_key = g.arch_key("attention.head_count_kv");
    let mut heads = Vec::new();
    let mut ffn_neurons = Vec::new();
    let mut experts = Vec::new();
    let layers: std::collections::BTreeSet<u32> = tensors.iter().filter_map(|t| t.layer).collect();
    for l in layers {
        let get = |role: &str| tensors.iter().find(|t| t.layer == Some(l) && t.role == role);
        let nh = per_layer(hc_key, l as usize).or(hp.head_count).unwrap_or(0);
        let nkv = per_layer(hkv_key, l as usize).or(hp.head_count_kv).unwrap_or(nh);
        let (mut q, mut k, mut v) = (vec![], vec![], vec![]);
        let mut head_dim = hp.key_length.unwrap_or(0);
        if let Some(t) = get("attn_q").and_then(|t| t.row_norms.as_ref()) {
            q = group_norms(t, nh);
            if nh > 0 {
                head_dim = t.len() as u64 / nh;
            }
        }
        if let Some(t) = get("attn_k").and_then(|t| t.row_norms.as_ref()) {
            k = group_norms(t, nkv);
        }
        if let Some(t) = get("attn_v").and_then(|t| t.row_norms.as_ref()) {
            v = group_norms(t, nkv);
        }
        if let Some(t) = get("attn_qkv").and_then(|t| t.row_norms.as_ref()) {
            if nh > 0 && (nh + 2 * nkv) > 0 && t.len() as u64 % (nh + 2 * nkv) == 0 {
                let hd = t.len() as u64 / (nh + 2 * nkv);
                head_dim = hd;
                let (qs, rest) = t.split_at((nh * hd) as usize);
                let (ks, vs) = rest.split_at((nkv * hd) as usize);
                q = group_norms(qs, nh);
                k = group_norms(ks, nkv);
                v = group_norms(vs, nkv);
            }
        }
        let o = get("attn_output").and_then(|t| t.col_norms.as_ref()).map(|c| group_norms(c, nh)).unwrap_or_default();
        if !(q.is_empty() && k.is_empty() && v.is_empty() && o.is_empty()) {
            heads.push(HeadNorms { layer: l, head_dim, q, k, v, o });
        }

        let up = get("ffn_up").and_then(|t| t.row_norms.clone());
        let gate = get("ffn_gate").and_then(|t| t.row_norms.clone());
        let down = get("ffn_down").and_then(|t| t.col_norms.clone());
        let n_ff = up.as_ref().or(down.as_ref()).map(Vec::len).unwrap_or(0);
        if n_ff > 0 {
            let norms = (0..n_ff)
                .map(|i| {
                    let a = up.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0) as f64;
                    let b = gate.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0) as f64;
                    let c = down.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0) as f64;
                    (a * a + b * b + c * c).sqrt() as f32
                })
                .collect();
            ffn_neurons.push(NeuronNorms { layer: l, norms });
        }

        let exp: Vec<&Vec<f32>> = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"]
            .iter()
            .filter_map(|r| get(r).and_then(|t| t.slice_norms.as_ref()))
            .collect();
        if let Some(first) = exp.first() {
            let norms = (0..first.len())
                .map(|e| exp.iter().map(|v| (v.get(e).copied().unwrap_or(0.0) as f64).powi(2)).sum::<f64>().sqrt() as f32)
                .collect();
            experts.push(ExpertNorms { layer: l, norms });
        }
    }
    p.set(100.0, "done");
    Ok(StatsReport {
        schema_version: SCHEMA_VERSION,
        model: model.into(),
        digest: digest.into(),
        tensors,
        heads,
        ffn_neurons,
        experts,
        skipped,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// Cosine similarity between two same-shaped tensors, streamed in chunks.
pub fn tensor_cosine(g: &GgufFile, a: &TensorInfo, b: &TensorInfo) -> Result<Option<f64>> {
    if a.dims != b.dims || !can_dequantize(a.ggml_type) || !can_dequantize(b.ggml_type) {
        return Ok(None);
    }
    let rows = a.n_rows();
    let rpc = (CHUNK_ELEMS / a.row_len().max(1)).max(1);
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    let (mut ba, mut bbuf) = (Vec::new(), Vec::new());
    let mut s = 0;
    while s < rows {
        let c = rpc.min(rows - s);
        g.dequantize_rows(a, s, c, &mut ba)?;
        g.dequantize_rows(b, s, c, &mut bbuf)?;
        for (x, y) in ba.iter().zip(&bbuf) {
            let (x, y) = (*x as f64, *y as f64);
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        s += c;
    }
    Ok(if aa > 0.0 && bb > 0.0 { Some(ab / (aa.sqrt() * bb.sqrt())) } else { None })
}

/// Whether a type is a high-precision float that has quantization headroom.
pub fn is_float_type(t: GgmlType) -> bool {
    matches!(t, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::F64)
}
