//! Static probes that read the weights only (no forward pass): the spectrum of every weight
//! matrix, what a single FFN neuron reads and writes in token space, under-trained ("glitch")
//! tokens, and the memory a context length needs.

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::gguf::{GgufFile, TensorInfo};
use crate::model::ModelHandle;
use crate::util::{check_cancel, Progress, SCHEMA_VERSION};

mod spectrum;
mod neuron;
mod glitch;
mod memory;
pub use spectrum::*;
pub use neuron::*;
pub use glitch::*;
pub use memory::*;

/// A token with a score (dot product or similarity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenScore {
    pub id: u32,
    pub text: String,
    pub score: f32,
}

/// Rows of `t` (as [n_rows, row_len]) are streamed in chunks; `f(row index, row)` sees each.
pub(crate) fn for_rows(g: &GgufFile, t: &TensorInfo, chunk: u64, p: Option<(&dyn Progress, f32, f32)>, mut f: impl FnMut(u64, &[f32])) -> Result<()> {
    let n = t.n_rows();
    let len = t.row_len() as usize;
    let mut buf = Vec::new();
    let mut start = 0;
    while start < n {
        let k = chunk.min(n - start);
        g.dequantize_rows(t, start, k, &mut buf).with_context(|| format!("reading {}", t.name))?;
        for (i, row) in buf.chunks(len).enumerate() {
            f(start + i as u64, row);
        }
        start += k;
        if let Some((p, from, to)) = p {
            check_cancel(p)?;
            p.set(from + (to - from) * start as f32 / n as f32, &format!("reading {}", t.name));
        }
    }
    Ok(())
}

/// Dot product of every row of `t` with `v`, in parallel chunks.
pub(crate) fn row_dots(g: &GgufFile, t: &TensorInfo, v: &[f32]) -> Result<Vec<f32>> {
    let n = t.n_rows();
    let len = t.row_len() as usize;
    if len != v.len() {
        bail!("{} rows have {len} values, the vector has {}", t.name, v.len());
    }
    let chunk = 1024u64;
    let starts: Vec<u64> = (0..n).step_by(chunk as usize).collect();
    let parts: Vec<Vec<f32>> = starts
        .par_iter()
        .map(|&s| {
            let k = chunk.min(n - s);
            let mut buf = Vec::new();
            g.dequantize_rows(t, s, k, &mut buf).with_context(|| format!("reading {}", t.name))?;
            Ok(buf.chunks(len).map(|r| r.iter().zip(v).map(|(a, b)| a * b).sum()).collect())
        })
        .collect::<Result<_>>()?;
    Ok(parts.concat())
}

/// Indices of the `k` largest values, largest first.
pub(crate) fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
    let k = k.min(idx.len());
    if k == 0 {
        return vec![];
    }
    idx.select_nth_unstable_by(k - 1, |a, b| v[*b].total_cmp(&v[*a]));
    idx.truncate(k);
    idx.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
    idx
}

/// Layer index of a `blk.N.` tensor.
pub(crate) fn layer_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

pub(crate) fn tensor<'a>(g: &'a GgufFile, name: &str) -> Result<&'a TensorInfo> {
    g.tensor(name).ok_or_else(|| anyhow!("this model has no {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_and_skips_nan() {
        assert_eq!(top_k(&[1.0, f32::NAN, 3.0, 2.0], 2), vec![2, 3]);
        assert_eq!(layer_of("blk.12.ffn_up.weight"), Some(12));
        assert_eq!(layer_of("output.weight"), None);
    }
}
