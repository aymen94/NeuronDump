//! GGUF v3 writer (little-endian) used by slimming.

use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use indexmap::IndexMap;

use crate::error::Result;
use crate::file::{DEFAULT_ALIGNMENT, GGUF_MAGIC};
use crate::types::GgmlType;
use crate::value::{Array, Value, ValueType};

pub struct WriterTensor<'a> {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub data: Cow<'a, [u8]>,
}

#[derive(Default)]
pub struct GgufWriter<'a> {
    pub metadata: IndexMap<String, Value>,
    pub tensors: Vec<WriterTensor<'a>>,
}

fn type_id(t: ValueType) -> u32 {
    use ValueType::*;
    match t {
        U8 => 0,
        I8 => 1,
        U16 => 2,
        I16 => 3,
        U32 => 4,
        I32 => 5,
        F32 => 6,
        Bool => 7,
        String => 8,
        Array => 9,
        U64 => 10,
        I64 => 11,
        F64 => 12,
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(*x as u8),
        Value::String(s) => put_str(out, s),
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Array(a) => put_array(out, a),
    }
}

fn put_array(out: &mut Vec<u8>, a: &Array) {
    out.extend_from_slice(&type_id(a.elem_type()).to_le_bytes());
    out.extend_from_slice(&a.len().to_le_bytes());
    macro_rules! each {
        ($v:expr) => {
            for x in $v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
    }
    match a {
        Array::U8(v) => out.extend_from_slice(v),
        Array::I8(v) => out.extend(v.iter().map(|&x| x as u8)),
        Array::U16(v) => each!(v),
        Array::I16(v) => each!(v),
        Array::U32(v) => each!(v),
        Array::I32(v) => each!(v),
        Array::F32(v) => each!(v),
        Array::U64(v) => each!(v),
        Array::I64(v) => each!(v),
        Array::F64(v) => each!(v),
        Array::Bool(v) => out.extend(v.iter().map(|&b| b as u8)),
        Array::String(v) => v.iter().for_each(|s| put_str(out, s)),
        Array::Array(v) => v.iter().for_each(|a| put_array(out, a)),
        // Callers must load arrays before writing; an empty array keeps the file valid.
        Array::Skipped { .. } => {}
    }
}

impl<'a> GgufWriter<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alignment(&self) -> u64 {
        self.metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .filter(|a| *a > 0 && a.is_power_of_two())
            .unwrap_or(DEFAULT_ALIGNMENT)
    }

    /// Serializes header + tensor infos; returns the bytes and each tensor's data offset.
    fn header(&self) -> (Vec<u8>, Vec<u64>) {
        let align = self.alignment();
        let mut h = Vec::new();
        h.extend_from_slice(GGUF_MAGIC);
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        h.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        for (k, v) in &self.metadata {
            put_str(&mut h, k);
            h.extend_from_slice(&type_id(v.value_type()).to_le_bytes());
            put_value(&mut h, v);
        }
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut off = 0u64;
        for t in &self.tensors {
            put_str(&mut h, &t.name);
            h.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                h.extend_from_slice(&d.to_le_bytes());
            }
            h.extend_from_slice(&t.ggml_type.0.to_le_bytes());
            h.extend_from_slice(&off.to_le_bytes());
            offsets.push(off);
            off = (off + t.data.len() as u64).div_ceil(align) * align;
        }
        (h, offsets)
    }

    /// Writes the file; `progress` is called with (bytes written, total tensor bytes).
    pub fn write_to(&self, path: impl AsRef<Path>, mut progress: impl FnMut(u64, u64)) -> Result<u64> {
        let align = self.alignment();
        let (header, _) = self.header();
        let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
        w.write_all(&header)?;
        let mut pos = header.len() as u64;
        let pad = |w: &mut BufWriter<File>, pos: &mut u64| -> std::io::Result<()> {
            let target = pos.div_ceil(align) * align;
            w.write_all(&vec![0u8; (target - *pos) as usize])?;
            *pos = target;
            Ok(())
        };
        pad(&mut w, &mut pos)?;
        let total: u64 = self.tensors.iter().map(|t| t.data.len() as u64).sum();
        let mut done = 0u64;
        for t in &self.tensors {
            w.write_all(&t.data)?;
            pos += t.data.len() as u64;
            done += t.data.len() as u64;
            pad(&mut w, &mut pos)?;
            progress(done, total);
        }
        w.flush()?;
        Ok(pos)
    }
}
