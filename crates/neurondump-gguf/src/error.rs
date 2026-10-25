use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("not a GGUF file (magic {0:02x?})")]
    BadMagic([u8; 4]),

    #[error("legacy pre-GGUF ggml file ({0}); convert it with llama.cpp's convert scripts")]
    LegacyFormat(&'static str),

    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),

    #[error("file truncated: needed {needed} bytes at offset {offset}, file is {len} bytes")]
    Truncated { offset: u64, needed: u64, len: u64 },

    #[error("malformed GGUF at offset {offset}: {msg}")]
    Malformed { offset: u64, msg: String },

    #[error("no tensor named {0:?}")]
    NoSuchTensor(String),

    #[error("tensor {name:?} lies outside the file (bytes {start}..{end}, file is {len} bytes)")]
    TensorOutOfBounds { name: String, start: u64, end: u64, len: u64 },
}

pub type Result<T, E = GgufError> = std::result::Result<T, E>;
