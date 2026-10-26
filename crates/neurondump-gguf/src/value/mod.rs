//! GGUF metadata values.

use std::fmt;

use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};

mod array;
mod serialize;
pub use array::*;

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
