//! Typed metadata arrays: element type, length, typed views, element access, previews.

use super::*;

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
    pub(super) array: &'a Array,
    pub(super) n: usize,
}
