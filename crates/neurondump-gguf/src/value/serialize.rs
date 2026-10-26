//! JSON serialization and display of values, arrays and array previews.

use super::*;

pub(super) fn ser_slice<S: Serializer, T: Serialize>(s: S, v: &[T], n: usize) -> Result<S::Ok, S::Error> {
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
