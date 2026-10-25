//! Streaming dequantization (ports of ggml's `dequantize_row_*` reference kernels).

use half::{bf16, f16};

use crate::error::{GgufError, Result};
use crate::file::{Endianness, GgufFile, TensorInfo};
use crate::types::GgmlType;

const QK_K: usize = 256;

const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

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

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantizes whole blocks of `src` into `dst`. `dst.len()` must equal the element
/// count those blocks encode.
pub fn dequantize_into(ty: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<()> {
    let layout = ty
        .layout()
        .ok_or_else(|| GgufError::Malformed { offset: 0, msg: format!("cannot dequantize unknown type {ty}") })?;
    let (bs, ts) = (layout.block_size as usize, layout.type_size as usize);
    if !can_dequantize(ty) {
        return Err(GgufError::Malformed { offset: 0, msg: format!("dequantization of {ty} is not implemented") });
    }
    if src.len() % ts != 0 || src.len() / ts * bs != dst.len() {
        return Err(GgufError::Malformed {
            offset: 0,
            msg: format!("dequantize {ty}: {} bytes do not match {} elements", src.len(), dst.len()),
        });
    }
    let blocks = src.chunks_exact(ts).zip(dst.chunks_exact_mut(bs));
    match ty {
        GgmlType::F32 => {
            for (b, y) in blocks {
                y[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        GgmlType::F16 => {
            for (b, y) in blocks {
                y[0] = f16_at(b, 0);
            }
        }
        GgmlType::BF16 => {
            for (b, y) in blocks {
                y[0] = bf16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        GgmlType::F64 => {
            for (b, y) in blocks {
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                y[0] = f64::from_le_bytes(a) as f32;
            }
        }
        GgmlType::I8 => {
            for (b, y) in blocks {
                y[0] = b[0] as i8 as f32;
            }
        }
        GgmlType::I16 => {
            for (b, y) in blocks {
                y[0] = i16::from_le_bytes([b[0], b[1]]) as f32;
            }
        }
        GgmlType::I32 => {
            for (b, y) in blocks {
                y[0] = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32;
            }
        }
        GgmlType::Q4_0 => {
            for (b, y) in blocks {
                let d = f16_at(b, 0);
                let qs = &b[2..18];
                for j in 0..16 {
                    y[j] = ((qs[j] & 0xF) as i32 - 8) as f32 * d;
                    y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_1 => {
            for (b, y) in blocks {
                let (d, m) = (f16_at(b, 0), f16_at(b, 2));
                let qs = &b[4..20];
                for j in 0..16 {
                    y[j] = (qs[j] & 0xF) as f32 * d + m;
                    y[j + 16] = (qs[j] >> 4) as f32 * d + m;
                }
            }
        }
        GgmlType::Q5_0 => {
            for (b, y) in blocks {
                let d = f16_at(b, 0);
                let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                let qs = &b[6..22];
                for j in 0..16 {
                    let xh0 = (((qh >> j) << 4) & 0x10) as u8;
                    let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = (((qs[j] & 0xF) | xh0) as i32 - 16) as f32 * d;
                    y[j + 16] = (((qs[j] >> 4) | xh1) as i32 - 16) as f32 * d;
                }
            }
        }
        GgmlType::Q5_1 => {
            for (b, y) in blocks {
                let (d, m) = (f16_at(b, 0), f16_at(b, 2));
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let qs = &b[8..24];
                for j in 0..16 {
                    let xh0 = (((qh >> j) << 4) & 0x10) as u8;
                    let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = ((qs[j] & 0xF) | xh0) as f32 * d + m;
                    y[j + 16] = ((qs[j] >> 4) | xh1) as f32 * d + m;
                }
            }
        }
        GgmlType::Q8_0 => {
            for (b, y) in blocks {
                let d = f16_at(b, 0);
                for (yj, &q) in y.iter_mut().zip(&b[2..34]) {
                    *yj = q as i8 as f32 * d;
                }
            }
        }
        GgmlType::Q8_1 => {
            for (b, y) in blocks {
                let d = f16_at(b, 0);
                for (yj, &q) in y.iter_mut().zip(&b[4..36]) {
                    *yj = q as i8 as f32 * d;
                }
            }
        }
        GgmlType::Q2_K => {
            for (b, y) in blocks {
                let scales = &b[0..16];
                let qs = &b[16..80];
                let d = f16_at(b, 80);
                let dmin = f16_at(b, 82);
                let mut is = 0;
                let mut yi = 0;
                for n in 0..2 {
                    let q = &qs[n * 32..n * 32 + 32];
                    let mut shift = 0;
                    for _ in 0..4 {
                        for half in 0..2 {
                            let sc = scales[is];
                            is += 1;
                            let dl = d * (sc & 0xF) as f32;
                            let ml = dmin * (sc >> 4) as f32;
                            for l in 0..16 {
                                y[yi] = dl * ((q[l + half * 16] >> shift) & 3) as f32 - ml;
                                yi += 1;
                            }
                        }
                        shift += 2;
                    }
                }
            }
        }
        GgmlType::Q3_K => {
            const KMASK1: u32 = 0x0303_0303;
            const KMASK2: u32 = 0x0f0f_0f0f;
            for (b, y) in blocks {
                let hm = &b[0..32];
                let qs = &b[32..96];
                let sc = &b[96..108];
                let d_all = f16_at(b, 108);
                let mut aux = [0u32; 4];
                for (i, a) in aux.iter_mut().take(3).enumerate() {
                    *a = u32::from_le_bytes([sc[4 * i], sc[4 * i + 1], sc[4 * i + 2], sc[4 * i + 3]]);
                }
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
                aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
                aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
                aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
                let mut scales = [0i8; 16];
                for (i, a) in aux.iter().enumerate() {
                    for (k, byte) in a.to_le_bytes().iter().enumerate() {
                        scales[i * 4 + k] = *byte as i8;
                    }
                }
                let mut is = 0;
                let mut yi = 0;
                let mut m = 1u8;
                for n in 0..2 {
                    let q = &qs[n * 32..n * 32 + 32];
                    let mut shift = 0;
                    for _ in 0..4 {
                        for half in 0..2 {
                            let dl = d_all * (scales[is] as i32 - 32) as f32;
                            is += 1;
                            for l in 0..16 {
                                let idx = l + half * 16;
                                let hbit = if hm[idx] & m != 0 { 0 } else { 4 };
                                y[yi] = dl * (((q[idx] >> shift) & 3) as i32 - hbit) as f32;
                                yi += 1;
                            }
                        }
                        shift += 2;
                        m <<= 1;
                    }
                }
            }
        }
        GgmlType::Q4_K => {
            for (b, y) in blocks {
                let d = f16_at(b, 0);
                let dmin = f16_at(b, 2);
                let scales = &b[4..16];
                let qs = &b[16..144];
                for j in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(2 * j, scales);
                    let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
                    let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
                    let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
                    let q = &qs[j * 32..j * 32 + 32];
                    let out = &mut y[j * 64..j * 64 + 64];
                    for l in 0..32 {
                        out[l] = d1 * (q[l] & 0xF) as f32 - m1;
                        out[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
                    }
                }
            }
        }
        GgmlType::Q5_K => {
            for (b, y) in blocks {
                let d = f16_at(b, 0);
                let dmin = f16_at(b, 2);
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                let (mut u1, mut u2) = (1u8, 2u8);
                for j in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(2 * j, scales);
                    let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
                    let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
                    let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
                    let ql = &qs[j * 32..j * 32 + 32];
                    let out = &mut y[j * 64..j * 64 + 64];
                    for l in 0..32 {
                        let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                        let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                        out[l] = d1 * ((ql[l] & 0xF) + h1) as f32 - m1;
                        out[l + 32] = d2 * ((ql[l] >> 4) + h2) as f32 - m2;
                    }
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlType::Q6_K => {
            for (b, y) in blocks {
                let ql_all = &b[0..128];
                let qh_all = &b[128..192];
                let sc_all = &b[192..208];
                let d = f16_at(b, 208);
                for n in 0..2 {
                    let ql = &ql_all[n * 64..n * 64 + 64];
                    let qh = &qh_all[n * 32..n * 32 + 32];
                    let sc = &sc_all[n * 8..n * 8 + 8];
                    let out = &mut y[n * 128..n * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        out[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        out[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        out[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        out[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        GgmlType::Q8_K => {
            for (b, y) in blocks {
                let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                for (yj, &q) in y.iter_mut().zip(&b[4..4 + QK_K]) {
                    *yj = q as i8 as f32 * d;
                }
            }
        }
        GgmlType::MXFP4 => {
            for (b, y) in blocks {
                let d = 2f32.powi(b[0] as i32 - 128);
                let qs = &b[1..17];
                for j in 0..16 {
                    y[j] = KVALUES_MXFP4[(qs[j] & 0xF) as usize] as f32 * d;
                    y[j + 16] = KVALUES_MXFP4[(qs[j] >> 4) as usize] as f32 * d;
                }
            }
        }
        GgmlType(20) => {
            for (b, y) in blocks {
                let d = f16_at(b, 0);
                let qs = &b[2..18];
                for j in 0..16 {
                    y[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                    y[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
            }
        }
        GgmlType(23) => {
            for (b, y) in blocks {
                let d = f16_at(b, 0);
                let scales_h = u16::from_le_bytes([b[2], b[3]]);
                let scales_l = &b[4..8];
                let qs_all = &b[8..136];
                for ib in 0..8 {
                    let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                        | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                    let dl = d * (ls - 32) as f32;
                    let qs = &qs_all[ib * 16..ib * 16 + 16];
                    let out = &mut y[ib * 32..ib * 32 + 32];
                    for j in 0..16 {
                        out[j] = dl * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                        out[j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                    }
                }
            }
        }
        _ => unreachable!("guarded by can_dequantize"),
    }
    Ok(())
}

/// Streaming f32 iterator over a tensor; dequantizes a bounded chunk at a time.
pub struct Dequantizer<'a> {
    ty: GgmlType,
    src: &'a [u8],
    block_size: usize,
    type_size: usize,
    blocks_per_chunk: usize,
    buf: Vec<f32>,
    pos: usize,
}

impl Iterator for Dequantizer<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.pos == self.buf.len() {
            if self.src.is_empty() {
                return None;
            }
            let n_blocks = (self.src.len() / self.type_size).min(self.blocks_per_chunk);
            if n_blocks == 0 {
                return None;
            }
            let (chunk, rest) = self.src.split_at(n_blocks * self.type_size);
            self.src = rest;
            self.buf.resize(n_blocks * self.block_size, 0.0);
            if dequantize_into(self.ty, chunk, &mut self.buf).is_err() {
                return None;
            }
            self.pos = 0;
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        Some(v)
    }
}

impl GgufFile {
    fn check_dequant(&self, t: &TensorInfo) -> Result<()> {
        if self.endianness == Endianness::Big {
            return Err(GgufError::Malformed { offset: 0, msg: "big-endian tensor data is not supported".into() });
        }
        if !can_dequantize(t.ggml_type) {
            return Err(GgufError::Malformed {
                offset: 0,
                msg: format!("dequantization of {} (tensor {}) is not implemented", t.ggml_type, t.name),
            });
        }
        Ok(())
    }

    /// Streams the tensor's values as f32 without materializing the whole tensor.
    pub fn dequantize(&self, name: &str) -> Result<Dequantizer<'_>> {
        let t = self.tensor(name).ok_or_else(|| GgufError::NoSuchTensor(name.to_string()))?;
        self.dequantize_info(t)
    }

    pub fn dequantize_info(&self, t: &TensorInfo) -> Result<Dequantizer<'_>> {
        self.check_dequant(t)?;
        let l = t.ggml_type.layout().expect("checked");
        let blocks_per_chunk = (65536 / l.block_size as usize).max(1);
        Ok(Dequantizer {
            ty: t.ggml_type,
            src: self.info_bytes(t)?,
            block_size: l.block_size as usize,
            type_size: l.type_size as usize,
            blocks_per_chunk,
            buf: Vec::new(),
            pos: 0,
        })
    }

    /// Dequantizes rows `[start, start + n)` of the tensor viewed as `[n_rows, row_len]`.
    pub fn dequantize_rows(&self, t: &TensorInfo, start: u64, n: u64, out: &mut Vec<f32>) -> Result<()> {
        self.check_dequant(t)?;
        let l = t.ggml_type.layout().expect("checked");
        let row_len = t.row_len();
        if row_len % l.block_size != 0 || start + n > t.n_rows() {
            return Err(GgufError::Malformed { offset: 0, msg: format!("bad row range for {}", t.name) });
        }
        let row_bytes = row_len / l.block_size * l.type_size;
        let bytes = self.info_bytes(t)?;
        let src = &bytes[(start * row_bytes) as usize..((start + n) * row_bytes) as usize];
        out.resize((n * row_len) as usize, 0.0);
        dequantize_into(t.ggml_type, src, out)
    }
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
