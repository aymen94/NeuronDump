//! Zero-copy GGUF parsing, dequantization and writing.
//!
//! The only I/O this crate performs is memory-mapping the input file (and writing an
//! output file in [`GgufWriter`]). Malformed input yields errors or warnings, never panics.

mod dequant;
mod error;
mod file;
mod reader;
mod types;
mod value;
mod writer;

pub use dequant::{can_dequantize, dequantize_into, quantize_q8_0, Dequantizer};
pub use error::{GgufError, Result};
pub use file::{Endianness, GgufFile, ParseOptions, TensorInfo, DEFAULT_ALIGNMENT};
pub use types::{file_type_name, GgmlType, TypeLayout};
pub use value::{Array, ArrayHead, Value, ValueType};
pub use writer::{GgufWriter, WriterTensor};
