//! Streaming dequantization (ports of ggml's `dequantize_row_*` reference kernels).

use half::{bf16, f16};

use crate::error::{GgufError, Result};
use crate::file::{Endianness, GgufFile, TensorInfo};
use crate::types::GgmlType;

mod kernels;
mod stream;
pub use kernels::*;
pub use stream::*;

/// Whether [`dequantize_into`] supports this type.
pub fn can_dequantize(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::F64
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
            | GgmlType::MXFP4
    ) || ty == GgmlType(20) // IQ4_NL
        || ty == GgmlType(23) // IQ4_XS
}

/// Quantizes f32 values to Q8_0 (ggml `quantize_row_q8_0_ref`). `src.len()` must be a
/// multiple of 32.
pub fn quantize_q8_0(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / 32 * 34);
    for block in src.chunks_exact(32) {
        let amax = block.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &x in block {
            out.push((x * id).round() as i8 as u8);
        }
    }
    out
}
