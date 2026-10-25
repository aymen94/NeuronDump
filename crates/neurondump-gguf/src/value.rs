//! GGUF metadata values.

use std::fmt;

use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};

/// GGUF metadata value type ids (`enum gguf_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    pub fn from_id(id: u32) -> Option<Self> {
        use ValueType::*;
        Some(match id {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => return None,
        })
    }

    /// Encoded size for fixed-size scalar types.
    pub fn fixed_size(self) -> Option<usize> {
        use ValueType::*;
        match self {
            U8 | I8 | Bool => Some(1),
            U16 | I16 => Some(2),
            U32 | I32 | F32 => Some(4),
            U64 | I64 | F64 => Some(8),
            String | Array => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        use ValueType::*;
        match self {
            U8 => "u8",
            I8 => "i8",
            U16 => "u16",
            I16 => "i16",
            U32 => "u32",
            I32 => "i32",
            F32 => "f32",
            Bool => "bool",
            String => "string",
            Array => "array",
            U64 => "u64",
            I64 => "i64",
            F64 => "f64",
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Array),
}

/// A homogeneous GGUF array. Arrays are stored typed so a 150k-entry vocabulary
/// costs one allocation per string and nothing more.
#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Array(Vec<Array>),
    /// Not decoded (header-only parse); only the element type and length are known.
    Skipped { elem_type: ValueType, len: u64 },
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::U8(_) => ValueType::U8,
            Value::I8(_) => ValueType::I8,
            Value::U16(_) => ValueType::U16,
            Value::I16(_) => ValueType::I16,
            Value::U32(_) => ValueType::U32,
            Value::I32(_) => ValueType::I32,
            Value::F32(_) => ValueType::F32,
            Value::Bool(_) => ValueType::Bool,
            Value::String(_) => ValueType::String,
            Value::U64(_) => ValueType::U64,
            Value::I64(_) => ValueType::I64,
            Value::F64(_) => ValueType::F64,
            Value::Array(_) => ValueType::Array,
        }
    }

    /// Type description such as `u32` or `array<string>[151936]`.
    pub fn type_description(&self) -> String {
        match self {
            Value::Array(a) => a.type_description(),
            v => v.value_type().to_string(),
        }
    }

    /// Any non-negative integer value (bools excluded).
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => i64::try_from(v).ok(),
            Value::I8(v) => Some(v.into()),
            Value::I16(v) => Some(v.into()),
            Value::I32(v) => Some(v.into()),
            Value::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => self.as_i64().map(|v| v as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl Array {
    pub fn elem_type(&self) -> ValueType {
        match self {
            Array::U8(_) => ValueType::U8,
            Array::I8(_) => ValueType::I8,
            Array::U16(_) => ValueType::U16,
            Array::I16(_) => ValueType::I16,
            Array::U32(_) => ValueType::U32,
            Array::I32(_) => ValueType::I32,
            Array::F32(_) => ValueType::F32,
            Array::Bool(_) => ValueType::Bool,
            Array::String(_) => ValueType::String,
            Array::U64(_) => ValueType::U64,
            Array::I64(_) => ValueType::I64,
            Array::F64(_) => ValueType::F64,
            Array::Array(_) => ValueType::Array,
            Array::Skipped { elem_type, .. } => *elem_type,
        }
    }

    pub fn len(&self) -> u64 {
        match self {
            Array::U8(v) => v.len() as u64,
            Array::I8(v) => v.len() as u64,
            Array::U16(v) => v.len() as u64,
            Array::I16(v) => v.len() as u64,
            Array::U32(v) => v.len() as u64,
            Array::I32(v) => v.len() as u64,
            Array::F32(v) => v.len() as u64,
            Array::Bool(v) => v.len() as u64,
            Array::String(v) => v.len() as u64,
            Array::U64(v) => v.len() as u64,
            Array::I64(v) => v.len() as u64,
            Array::F64(v) => v.len() as u64,
            Array::Array(v) => v.len() as u64,
            Array::Skipped { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_skipped(&self) -> bool {
        matches!(self, Array::Skipped { .. })
    }

    pub fn type_description(&self) -> String {
        match self {
            Array::Array(inner) => {
                let inner_ty = inner.first().map_or("?".to_string(), |a| a.type_description());
                format!("array<{inner_ty}>[{}]", inner.len())
            }
            a => format!("array<{}>[{}]", a.elem_type(), a.len()),
        }
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Array::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32s(&self) -> Option<&[f32]> {
        match self {
            Array::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Element `i` as an integer, for any integer element type.
    pub fn get_i64(&self, i: usize) -> Option<i64> {
        match self {
            Array::U8(v) => v.get(i).map(|&x| x.into()),
            Array::I8(v) => v.get(i).map(|&x| x.into()),
            Array::U16(v) => v.get(i).map(|&x| x.into()),
            Array::I16(v) => v.get(i).map(|&x| x.into()),
            Array::U32(v) => v.get(i).map(|&x| x.into()),
            Array::I32(v) => v.get(i).map(|&x| x.into()),
            Array::U64(v) => v.get(i).and_then(|&x| i64::try_from(x).ok()),
            Array::I64(v) => v.get(i).copied(),
            _ => None,
        }
    }

    /// Serializes only the first `n` elements (used for summaries of huge arrays).
    pub fn head(&self, n: usize) -> ArrayHead<'_> {
        ArrayHead { array: self, n }
    }
}

/// Serialization view over the first `n` elements of an [`Array`].
pub struct ArrayHead<'a> {
    array: &'a Array,
    n: usize,
}

fn ser_slice<S: Serializer, T: Serialize>(s: S, v: &[T], n: usize) -> Result<S::Ok, S::Error> {
    let v = &v[..v.len().min(n)];
    let mut seq = s.serialize_seq(Some(v.len()))?;
    for x in v {
        seq.serialize_element(x)?;
    }
    seq.end()
}

impl Serialize for ArrayHead<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let n = self.n;
        match self.array {
            Array::U8(v) => ser_slice(s, v, n),
            Array::I8(v) => ser_slice(s, v, n),
            Array::U16(v) => ser_slice(s, v, n),
            Array::I16(v) => ser_slice(s, v, n),
            Array::U32(v) => ser_slice(s, v, n),
            Array::I32(v) => ser_slice(s, v, n),
            Array::F32(v) => ser_slice(s, v, n),
            Array::Bool(v) => ser_slice(s, v, n),
            Array::String(v) => ser_slice(s, v, n),
            Array::U64(v) => ser_slice(s, v, n),
            Array::I64(v) => ser_slice(s, v, n),
            Array::F64(v) => ser_slice(s, v, n),
            Array::Array(v) => ser_slice(s, v, n),
            Array::Skipped { .. } => s.serialize_none(),
        }
    }
}

impl Serialize for Array {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.head(usize::MAX).serialize(s)
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::U8(v) => s.serialize_u8(*v),
            Value::I8(v) => s.serialize_i8(*v),
            Value::U16(v) => s.serialize_u16(*v),
            Value::I16(v) => s.serialize_i16(*v),
            Value::U32(v) => s.serialize_u32(*v),
            Value::I32(v) => s.serialize_i32(*v),
            Value::F32(v) => s.serialize_f32(*v),
            Value::Bool(v) => s.serialize_bool(*v),
            Value::String(v) => s.serialize_str(v),
            Value::U64(v) => s.serialize_u64(*v),
            Value::I64(v) => s.serialize_i64(*v),
            Value::F64(v) => s.serialize_f64(*v),
            Value::Array(a) => a.serialize(s),
        }
    }
}

impl fmt::Display for Value {
    /// Short human form; arrays show their type and length only.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(v) => write!(f, "{v:?}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Array(a) => f.write_str(&a.type_description()),
        }
    }
}
