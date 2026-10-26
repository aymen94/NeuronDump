//! Embedding row access and nearest tokens by cosine similarity.

use super::*;

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(super) fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

pub(super) fn row(g: &GgufFile, t: &TensorInfo, id: u32) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    g.dequantize_rows(t, id as u64, 1, &mut v)?;
    Ok(v)
}

/// Top-`k` rows of `t` by cosine similarity for every query row (excluding the query id).
pub fn nearest_rows(g: &GgufFile, t: &TensorInfo, queries: &[u32], k: usize, p: &dyn Progress) -> Result<HashMap<u32, Vec<(u32, f32)>>> {
    let n = t.n_rows();
    let qs: Vec<(u32, Vec<f32>)> = queries
        .iter()
        .map(|&q| {
            let r = row(g, t, q)?;
            let nr = norm(&r).max(1e-12);
            Ok((q, r.into_iter().map(|x| x / nr).collect()))
        })
        .collect::<Result<_>>()?;
    const CHUNK: u64 = 2048;
    let chunks: Vec<u64> = (0..n.div_ceil(CHUNK)).collect();
    let done = std::sync::atomic::AtomicU64::new(0);
    let partial: Vec<Vec<Vec<(u32, f32)>>> = chunks
        .par_iter()
        .map(|&c| -> Result<Vec<Vec<(u32, f32)>>> {
            check_cancel(p)?;
            let start = c * CHUNK;
            let len = CHUNK.min(n - start);
            let mut buf = Vec::new();
            g.dequantize_rows(t, start, len, &mut buf)?;
            let dim = t.row_len() as usize;
            let mut best: Vec<Vec<(u32, f32)>> = vec![Vec::with_capacity(k + 1); qs.len()];
            for (i, r) in buf.chunks_exact(dim).enumerate() {
                let id = (start + i as u64) as u32;
                let nr = norm(r);
                if nr <= 0.0 {
                    continue;
                }
                for (qi, (qid, q)) in qs.iter().enumerate() {
                    if *qid == id {
                        continue;
                    }
                    let cos = dot(q, r) / nr;
                    let b = &mut best[qi];
                    if b.len() < k || cos > b[b.len() - 1].1 {
                        let pos = b.partition_point(|x| x.1 >= cos);
                        b.insert(pos, (id, cos));
                        b.truncate(k);
                    }
                }
            }
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            p.set(100.0 * d as f32 / chunks.len().max(1) as f32, "nearest neighbours");
            Ok(best)
        })
        .collect::<Result<_>>()?;
    let mut out = HashMap::new();
    for (qi, (qid, _)) in qs.iter().enumerate() {
        let mut all: Vec<(u32, f32)> = partial.iter().flat_map(|c| c[qi].iter().copied()).collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        out.insert(*qid, all);
    }
    Ok(out)
}
