//! GGUF file: header parsing and zero-copy tensor access.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use indexmap::IndexMap;
#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
use serde::Serialize;

use crate::error::{GgufError, Result};
use crate::reader::Reader;
use crate::types::GgmlType;
use crate::value::{Value, ValueType};

mod parse;
mod access;

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
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(Mmap),
    Bytes(Vec<u8>),
    /// Only part of the file is resident: `bytes` holds `[base, base + bytes.len())`.
    /// Used when the file is read lazily (e.g. browser `Blob.slice`).
    Window { base: u64, bytes: Vec<u8> },
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Mmap(m) => m,
            Storage::Bytes(b) => b,
            Storage::Window { bytes, .. } => bytes,
        }
    }

    fn base(&self) -> u64 {
        match self {
            Storage::Window { base, .. } => *base,
            _ => 0,
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, ParseOptions::default())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_with(path: impl AsRef<Path>, opts: ParseOptions) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let len = file.metadata()?.len();
        if len < 8 {
            return Err(GgufError::Truncated { offset: 0, needed: 8, len });
        }
        // SAFETY: the map is read-only; concurrent truncation by another process is the
        // usual mmap caveat and is accepted here (Ollama never rewrites blobs in place).
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(Storage::Mmap(mmap), None, opts)
    }

    pub fn from_bytes(bytes: Vec<u8>, opts: ParseOptions) -> Result<Self> {
        Self::parse(Storage::Bytes(bytes), None, opts)
    }

    /// Parses the header from the first bytes of a `file_size`-byte file. Fails with
    /// [`GgufError::Truncated`] when `prefix` ends inside the header (read more and retry).
    /// Tensor data is unavailable until [`GgufFile::set_window`] loads a range.
    pub fn from_prefix(prefix: Vec<u8>, file_size: u64, opts: ParseOptions) -> Result<Self> {
        Self::parse(Storage::Window { base: 0, bytes: prefix }, Some(file_size), opts)
    }

    /// Makes `[offset, offset + bytes.len())` of the file the resident data (lazy files only;
    /// replaces the previous window, so the header prefix can be dropped after parsing).
    pub fn set_window(&mut self, offset: u64, bytes: Vec<u8>) {
        self.storage = Storage::Window { base: offset, bytes };
    }

    /// Absolute byte range of a tensor's data in the file.
    pub fn data_range(&self, t: &TensorInfo) -> (u64, u64) {
        let start = self.data_offset.saturating_add(t.offset);
        (start, start.saturating_add(t.n_bytes))
    }

}
