//! Rewriting tensor data and metadata arrays: vocab subsets, row gathering, zeroing in place.

use super::*;

pub(super) fn subset_array(a: &Array, keep: &[usize]) -> Array {
    macro_rules! pick {
        ($v:expr, $variant:ident) => {
            Array::$variant(keep.iter().filter_map(|&i| $v.get(i).cloned()).collect())
        };
    }
    match a {
        Array::U8(v) => pick!(v, U8),
        Array::I8(v) => pick!(v, I8),
        Array::U16(v) => pick!(v, U16),
        Array::I16(v) => pick!(v, I16),
        Array::U32(v) => pick!(v, U32),
        Array::I32(v) => pick!(v, I32),
        Array::F32(v) => pick!(v, F32),
        Array::Bool(v) => pick!(v, Bool),
        Array::String(v) => pick!(v, String),
        Array::U64(v) => pick!(v, U64),
        Array::I64(v) => pick!(v, I64),
        Array::F64(v) => pick!(v, F64),
        Array::Array(v) => pick!(v, Array),
        Array::Skipped { .. } => a.clone(),
    }
}

pub(super) fn gather_rows(bytes: &[u8], row_bytes: usize, rows: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * row_bytes);
    for &r in rows {
        out.extend_from_slice(&bytes[r * row_bytes..(r + 1) * row_bytes]);
    }
    out
}

/// Zeroes part of a tensor in place. Returns the new type when the tensor had to be
/// requantized (column ranges not aligned to the quantization blocks).
pub(crate) fn apply_zero(o: &ZeroOp, dims: &[u64], ty: GgmlType, data: &mut Vec<u8>) -> Result<Option<GgmlType>> {
    if o.all {
        data.fill(0);
        return Ok(None);
    }
    let row_len = dims.first().copied().unwrap_or(1).max(1);
    let n_rows = (dims.iter().skip(1).product::<u64>()).max(1);
    let row_b = data.len() / n_rows as usize;
    for (a, b) in &o.rows {
        let (a, b) = ((*a).min(n_rows) as usize, (*b).min(n_rows) as usize);
        data[a * row_b..b * row_b].fill(0);
    }
    if let Some(&n_slices) = dims.get(2) {
        let sb = data.len() / n_slices.max(1) as usize;
        for (a, b) in &o.slices {
            let (a, b) = ((*a).min(n_slices) as usize, (*b).min(n_slices) as usize);
            data[a * sb..b * sb].fill(0);
        }
    }
    if o.cols.is_empty() {
        return Ok(None);
    }
    let layout = ty.layout().ok_or_else(|| anyhow::anyhow!("cannot zero columns of unknown type {ty}"))?;
    if !needs_requant(ty, &o.cols) {
        // Float types: element size; quantized: whole blocks.
        let (unit, unit_b) = if layout.quantized { (layout.block_size, layout.type_size) } else { (1, layout.type_size) };
        for r in 0..n_rows as usize {
            for (a, b) in &o.cols {
                let (a, b) = ((*a).min(row_len), (*b).min(row_len));
                let s = r * row_b + (a / unit * unit_b) as usize;
                let e = r * row_b + (b / unit * unit_b) as usize;
                data[s..e].fill(0);
            }
        }
        return Ok(None);
    }
    if row_len % 32 != 0 {
        bail!("cannot zero columns of a {ty} tensor with row length {row_len} (not a multiple of 32)");
    }
    let mut f = vec![0f32; (row_len * n_rows) as usize];
    dequantize_into(ty, data, &mut f)?;
    for r in 0..n_rows as usize {
        for (a, b) in &o.cols {
            let (a, b) = ((*a).min(row_len) as usize, (*b).min(row_len) as usize);
            f[r * row_len as usize + a..r * row_len as usize + b].fill(0.0);
        }
    }
    *data = quantize_q8_0(&f);
    Ok(Some(GgmlType::Q8_0))
}
