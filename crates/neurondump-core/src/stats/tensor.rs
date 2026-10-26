//! Streaming statistics of one tensor (moments, histogram, row / column / slice norms); cosine of two tensors.

use super::*;

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

pub(super) fn group_norms(v: &[f32], groups: u64) -> Vec<f32> {
    if groups == 0 || v.is_empty() || v.len() as u64 % groups != 0 {
        return vec![];
    }
    let per = v.len() / groups as usize;
    v.chunks_exact(per).map(|c| c.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as f32).collect()
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
