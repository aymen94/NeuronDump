//! Bounded-memory access: the streaming `Dequantizer` and `GgufFile::dequantize*` row helpers.

use super::*;

/// Streaming f32 iterator over a tensor; dequantizes a bounded chunk at a time.
pub struct Dequantizer<'a> {
    pub(super) ty: GgmlType,
    pub(super) src: &'a [u8],
    pub(super) block_size: usize,
    pub(super) type_size: usize,
    pub(super) blocks_per_chunk: usize,
    pub(super) buf: Vec<f32>,
    pub(super) pos: usize,
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
    pub(super) fn check_dequant(&self, t: &TensorInfo) -> Result<()> {
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
