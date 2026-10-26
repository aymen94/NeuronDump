//! Accessors: metadata lookups, tensors by name, totals, tensor byte ranges.

use super::*;

impl GgufFile {
    /// Raw bytes of the whole file (of the resident window for lazily read files).
    pub fn bytes(&self) -> &[u8] {
        self.storage.bytes()
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Value::as_f64)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// `<arch>.<suffix>` lookup.
    pub fn arch_key(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.{suffix}"))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor_index(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    pub fn total_params(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_elements).sum()
    }

    pub fn tensor_data_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_bytes).sum()
    }

    pub fn info_bytes(&self, t: &TensorInfo) -> Result<&[u8]> {
        let start = self.data_offset.saturating_add(t.offset);
        let end = start.saturating_add(t.n_bytes);
        if end > self.file_size {
            return Err(GgufError::TensorOutOfBounds {
                name: t.name.clone(),
                start,
                end,
                len: self.file_size,
            });
        }
        let base = self.storage.base();
        let have = self.bytes();
        if start < base || end - base > have.len() as u64 {
            return Err(GgufError::Malformed { offset: start, msg: format!("tensor {} data is not loaded", t.name) });
        }
        Ok(&have[(start - base) as usize..(end - base) as usize])
    }

    /// Zero-copy view of a tensor's raw (possibly quantized) bytes.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensor(name).ok_or_else(|| GgufError::NoSuchTensor(name.to_string()))?;
        self.info_bytes(t)
    }

    /// Element type of a metadata array, if `key` is an array.
    pub fn array_elem_type(&self, key: &str) -> Option<ValueType> {
        self.get(key).and_then(Value::as_array).map(|a| a.elem_type())
    }
}
