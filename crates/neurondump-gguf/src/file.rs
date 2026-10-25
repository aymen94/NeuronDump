//! GGUF file: header parsing and zero-copy tensor access.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use indexmap::IndexMap;
use memmap2::Mmap;
use serde::Serialize;

use crate::error::{GgufError, Result};
use crate::reader::Reader;
use crate::types::GgmlType;
use crate::value::{Value, ValueType};

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_DIMS: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Endianness {
    Little,
    Big,
}

#[derive(Debug, Clone, Copy)]
pub struct ParseOptions {
    /// Decode metadata arrays (vocabularies, merges, scores...). When false only their
    /// type and length are recorded, which makes header reads of big vocabularies cheaper.
    pub load_arrays: bool,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self { load_arrays: true }
    }
}

impl ParseOptions {
    pub fn header_only() -> Self {
        Self { load_arrays: false }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
    pub n_elements: u64,
    pub n_bytes: u64,
    /// True when `n_bytes` was derived from neighbouring offsets (unknown type or a
    /// shape that is not a whole number of blocks).
    pub size_estimated: bool,
}

impl TensorInfo {
    /// Rows of the tensor seen as a 2D matrix `[n_rows, dims[0]]`.
    pub fn n_rows(&self) -> u64 {
        self.dims.iter().skip(1).product::<u64>().max(1)
    }

    pub fn row_len(&self) -> u64 {
        self.dims.first().copied().unwrap_or(1)
    }
}

enum Storage {
    Mmap(Mmap),
    Bytes(Vec<u8>),
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            Storage::Mmap(m) => m,
            Storage::Bytes(b) => b,
        }
    }
}

pub struct GgufFile {
    pub version: u32,
    pub endianness: Endianness,
    pub alignment: u64,
    /// Absolute offset of the tensor data section.
    pub data_offset: u64,
    /// Size of the header (everything before the padded data section).
    pub header_size: u64,
    pub file_size: u64,
    pub metadata: IndexMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Non-fatal oddities found while parsing.
    pub warnings: Vec<String>,
    index: HashMap<String, usize>,
    storage: Storage,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("n_kv", &self.metadata.len())
            .field("n_tensors", &self.tensors.len())
            .finish()
    }
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, ParseOptions::default())
    }

    pub fn open_with(path: impl AsRef<Path>, opts: ParseOptions) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let len = file.metadata()?.len();
        if len < 8 {
            return Err(GgufError::Truncated { offset: 0, needed: 8, len });
        }
        // SAFETY: the map is read-only; concurrent truncation by another process is the
        // usual mmap caveat and is accepted here (Ollama never rewrites blobs in place).
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(Storage::Mmap(mmap), opts)
    }

    pub fn from_bytes(bytes: Vec<u8>, opts: ParseOptions) -> Result<Self> {
        Self::parse(Storage::Bytes(bytes), opts)
    }

    fn parse(storage: Storage, opts: ParseOptions) -> Result<Self> {
        let data = storage.bytes();
        let mut warnings = Vec::new();
        let mut r = Reader::new(data);

        let magic: [u8; 4] = {
            let s = r.take(4)?;
            [s[0], s[1], s[2], s[3]]
        };
        if &magic != GGUF_MAGIC {
            return Err(match &magic {
                b"lmgg" | b"ggml" => GgufError::LegacyFormat("ggml"),
                b"fmgg" | b"ggmf" => GgufError::LegacyFormat("ggmf"),
                b"tjgg" | b"ggjt" => GgufError::LegacyFormat("ggjt"),
                _ => GgufError::BadMagic(magic),
            });
        }

        let raw_version = r.u32()?;
        let (version, endianness) = if (1..=3).contains(&raw_version) {
            (raw_version, Endianness::Little)
        } else if (1..=3).contains(&raw_version.swap_bytes()) {
            r.big_endian = true;
            warnings.push("big-endian GGUF file: tensor data is not usable on little-endian hosts".into());
            (raw_version.swap_bytes(), Endianness::Big)
        } else if raw_version > 3 && raw_version < 0x1_0000 {
            warnings.push(format!("unknown GGUF version {raw_version}; parsing as v3"));
            (raw_version, Endianness::Little)
        } else {
            return Err(GgufError::UnsupportedVersion(raw_version));
        };
        r.wide = version >= 2;

        let n_tensors = r.len()?;
        let n_kv = r.len()?;

        // Each KV needs at least a key length + type id; each tensor info at least
        // name length + n_dims + type + offset.
        let min_kv = if r.wide { 12 } else { 8 };
        if n_kv.saturating_mul(min_kv) > r.remaining() as u64 {
            return Err(r.malformed(format!("metadata count {n_kv} exceeds file size")));
        }

        let mut metadata = IndexMap::with_capacity(n_kv as usize);
        let mut lossy_keys = Vec::new();
        for _ in 0..n_kv {
            let mut lossy = false;
            let key = r.string(&mut lossy)?;
            let ty = r.value_type()?;
            let mut value_lossy = false;
            let value = r.value(ty, opts.load_arrays, &mut value_lossy)?;
            if lossy {
                warnings.push(format!("metadata key {key:?} is not valid UTF-8 (decoded lossily)"));
            }
            if value_lossy {
                lossy_keys.push(key.clone());
            }
            if metadata.contains_key(&key) {
                warnings.push(format!("duplicate metadata key {key:?}; keeping the first value"));
                continue;
            }
            metadata.insert(key, value);
        }
        for key in lossy_keys {
            warnings.push(format!("metadata {key:?} contains invalid UTF-8 strings (decoded lossily)"));
        }

        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(v) => match v.as_u64() {
                Some(a) if a > 0 && a.is_power_of_two() => a,
                _ => {
                    warnings.push(format!("invalid general.alignment {v}; using {DEFAULT_ALIGNMENT}"));
                    DEFAULT_ALIGNMENT
                }
            },
        };

        let min_ti = if r.wide { 8 + 4 + 4 + 8 } else { 4 + 4 + 4 + 8 };
        if n_tensors.saturating_mul(min_ti) > r.remaining() as u64 {
            return Err(r.malformed(format!("tensor count {n_tensors} exceeds file size")));
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut index = HashMap::with_capacity(n_tensors as usize);
        for i in 0..n_tensors as usize {
            let mut lossy = false;
            let name = r.string(&mut lossy)?;
            if lossy {
                warnings.push(format!("tensor name {name:?} is not valid UTF-8"));
            }
            let n_dims = r.u32()?;
            if n_dims > MAX_DIMS {
                return Err(r.malformed(format!("tensor {name:?} has {n_dims} dimensions")));
            }
            if n_dims > 4 {
                warnings.push(format!("tensor {name:?} has {n_dims} dimensions (ggml supports 4)"));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.len()?);
            }
            let ggml_type = GgmlType(r.u32()?);
            let offset = r.u64()?;
            let n_elements = dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| r.malformed(format!("tensor {name:?} element count overflows")))?;
            let (n_bytes, size_estimated) = match ggml_type.bytes_for(n_elements) {
                Some(b) => (b, false),
                None => {
                    if ggml_type.is_known() {
                        warnings.push(format!(
                            "tensor {name:?}: {n_elements} elements is not a whole number of {ggml_type} blocks"
                        ));
                    } else {
                        warnings.push(format!(
                            "tensor {name:?} uses unknown ggml type id {}; size derived from offsets",
                            ggml_type.0
                        ));
                    }
                    (0, true)
                }
            };
            if offset % alignment != 0 {
                warnings.push(format!("tensor {name:?} offset {offset} is not {alignment}-byte aligned"));
            }
            if index.insert(name.clone(), i).is_some() {
                warnings.push(format!("duplicate tensor name {name:?}"));
            }
            tensors.push(TensorInfo { name, dims, ggml_type, offset, n_elements, n_bytes, size_estimated });
        }

        let header_size = r.pos as u64;
        let data_offset = header_size.div_ceil(alignment) * alignment;
        let file_size = data.len() as u64;
        let data_len = file_size.saturating_sub(data_offset);

        // Derive sizes we could not compute from the gap to the next tensor.
        if tensors.iter().any(|t| t.size_estimated) {
            let mut order: Vec<usize> = (0..tensors.len()).collect();
            order.sort_by_key(|&i| tensors[i].offset);
            for (pos, &i) in order.iter().enumerate() {
                if tensors[i].size_estimated {
                    let next = order.get(pos + 1).map_or(data_len, |&j| tensors[j].offset);
                    tensors[i].n_bytes = next.saturating_sub(tensors[i].offset);
                }
            }
        }

        let mut oob = 0usize;
        for t in &tensors {
            if data_offset.saturating_add(t.offset).saturating_add(t.n_bytes) > file_size {
                oob += 1;
                if oob <= 5 {
                    warnings.push(format!("tensor {:?} extends beyond the end of the file", t.name));
                }
            }
        }
        if oob > 5 {
            warnings.push(format!("{oob} tensors in total extend beyond the end of the file (truncated download?)"));
        }

        Ok(Self {
            version,
            endianness,
            alignment,
            data_offset,
            header_size,
            file_size,
            metadata,
            tensors,
            warnings,
            index,
            storage,
        })
    }

    /// Raw bytes of the whole file.
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
        Ok(&self.bytes()[start as usize..end as usize])
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
