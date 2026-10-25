//! Bounds-checked little/big-endian cursor over the raw file bytes.

use crate::error::{GgufError, Result};
use crate::value::{Array, Value, ValueType};

const MAX_ARRAY_DEPTH: u32 = 16;

pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pub pos: usize,
    pub big_endian: bool,
    /// GGUF v2+: lengths and counts are u64 (v1 used u32).
    pub wide: bool,
}

macro_rules! read_num {
    ($name:ident, $ty:ty) => {
        pub fn $name(&mut self) -> Result<$ty> {
            let b = self.take_array::<{ std::mem::size_of::<$ty>() }>()?;
            Ok(if self.big_endian { <$ty>::from_be_bytes(b) } else { <$ty>::from_le_bytes(b) })
        }
    };
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, big_endian: false, wide: true }
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn truncated(&self, needed: u64) -> GgufError {
        GgufError::Truncated { offset: self.pos as u64, needed, len: self.data.len() as u64 }
    }

    pub fn malformed(&self, msg: impl Into<String>) -> GgufError {
        GgufError::Malformed { offset: self.pos as u64, msg: msg.into() }
    }

    pub fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        if n > self.remaining() as u64 {
            return Err(self.truncated(n));
        }
        let n = n as usize;
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let s = self.take(N as u64)?;
        let mut a = [0u8; N];
        a.copy_from_slice(s);
        Ok(a)
    }

    read_num!(u8, u8);
    read_num!(i8, i8);
    read_num!(u16, u16);
    read_num!(i16, i16);
    read_num!(u32, u32);
    read_num!(i32, i32);
    read_num!(u64, u64);
    read_num!(i64, i64);
    read_num!(f32, f32);
    read_num!(f64, f64);

    /// A count or length field (u32 in v1, u64 afterwards).
    pub fn len(&mut self) -> Result<u64> {
        if self.wide {
            self.u64()
        } else {
            self.u32().map(u64::from)
        }
    }

    fn len_size(&self) -> u64 {
        if self.wide {
            8
        } else {
            4
        }
    }

    /// Reads a string; invalid UTF-8 is decoded lossily and reported via `lossy`.
    pub fn string(&mut self, lossy: &mut bool) -> Result<String> {
        let n = self.len()?;
        let bytes = self.take(n)?;
        match std::str::from_utf8(bytes) {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => {
                *lossy = true;
                Ok(String::from_utf8_lossy(bytes).into_owned())
            }
        }
    }

    fn skip_string(&mut self) -> Result<()> {
        let n = self.len()?;
        self.take(n).map(|_| ())
    }

    pub fn value_type(&mut self) -> Result<ValueType> {
        let id = self.u32()?;
        ValueType::from_id(id).ok_or_else(|| self.malformed(format!("unknown metadata value type {id}")))
    }

    pub fn value(&mut self, ty: ValueType, load_arrays: bool, lossy: &mut bool) -> Result<Value> {
        Ok(match ty {
            ValueType::U8 => Value::U8(self.u8()?),
            ValueType::I8 => Value::I8(self.i8()?),
            ValueType::U16 => Value::U16(self.u16()?),
            ValueType::I16 => Value::I16(self.i16()?),
            ValueType::U32 => Value::U32(self.u32()?),
            ValueType::I32 => Value::I32(self.i32()?),
            ValueType::F32 => Value::F32(self.f32()?),
            ValueType::Bool => Value::Bool(self.u8()? != 0),
            ValueType::String => Value::String(self.string(lossy)?),
            ValueType::U64 => Value::U64(self.u64()?),
            ValueType::I64 => Value::I64(self.i64()?),
            ValueType::F64 => Value::F64(self.f64()?),
            ValueType::Array => Value::Array(self.array(load_arrays, lossy, 0)?),
        })
    }

    fn min_elem_size(&self, ty: ValueType) -> u64 {
        match ty {
            ValueType::String => self.len_size(),
            ValueType::Array => 4 + self.len_size(),
            t => t.fixed_size().unwrap_or(1) as u64,
        }
    }

    pub fn array(&mut self, load: bool, lossy: &mut bool, depth: u32) -> Result<Array> {
        if depth > MAX_ARRAY_DEPTH {
            return Err(self.malformed("arrays nested too deeply"));
        }
        let elem = self.value_type()?;
        let len = self.len()?;
        // Bound the length by what the remaining bytes could possibly hold before allocating.
        let min = self.min_elem_size(elem);
        if len.saturating_mul(min) > self.remaining() as u64 {
            return Err(self.truncated(len.saturating_mul(min)));
        }
        let n = len as usize;
        if !load {
            match elem.fixed_size() {
                Some(sz) => {
                    self.take(len * sz as u64)?;
                }
                None if elem == ValueType::String => {
                    for _ in 0..n {
                        self.skip_string()?;
                    }
                }
                None => {
                    for _ in 0..n {
                        self.array(false, lossy, depth + 1)?;
                    }
                }
            }
            return Ok(Array::Skipped { elem_type: elem, len });
        }
        macro_rules! collect {
            ($variant:ident, $read:ident) => {{
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.$read()?);
                }
                Array::$variant(v)
            }};
        }
        Ok(match elem {
            ValueType::U8 => Array::U8(self.take(len)?.to_vec()),
            ValueType::I8 => collect!(I8, i8),
            ValueType::U16 => collect!(U16, u16),
            ValueType::I16 => collect!(I16, i16),
            ValueType::U32 => collect!(U32, u32),
            ValueType::I32 => collect!(I32, i32),
            ValueType::F32 => collect!(F32, f32),
            ValueType::U64 => collect!(U64, u64),
            ValueType::I64 => collect!(I64, i64),
            ValueType::F64 => collect!(F64, f64),
            ValueType::Bool => Array::Bool(self.take(len)?.iter().map(|&b| b != 0).collect()),
            ValueType::String => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.string(lossy)?);
                }
                Array::String(v)
            }
            ValueType::Array => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.array(true, lossy, depth + 1)?);
                }
                Array::Array(v)
            }
        })
    }
}
