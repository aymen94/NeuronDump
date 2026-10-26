//! Runtime tracing with candle: a generic llama-family decoder forward pass and a
//! BERT-family encoder pass, instrumented per layer. Weights are built straight from
//! NeuronDump's memory-mapped GGUF (no second parse).
//!
//! Supported architectures are listed explicitly in [`SUPPORTED`]; adding one means
//! describing its [`ArchSpec`] (or implementing [`ModelTracer`] for a new family).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::{Device, Module, Tensor, D};

use neurondump_core::model::ModelHandle;
use neurondump_core::tokenizer::{Encoder, TokenType, Vocab};
use neurondump_core::trace::{
    Ablation, CorpusEval, Decision, GenStep, GenerateReport, GenerateRequest, HeadAttribution, JourneyLayer, JourneyReport, JourneyRequest, JourneyToken,
    LayerAttribution, LayerTrace, LensDiff, LensToken, NeuronAttribution, Sampling, SimPrompt, SimReport, SimRequest, SimSide, SweepEntry, SweepReport,
    SweepRequest, TokenContribution, TraceBackend, TraceReport, TraceRequest,
};
use neurondump_core::util::{check_cancel, Progress, SCHEMA_VERSION};
use neurondump_gguf::{GgmlType, GgufFile, TensorInfo};

mod spec;
mod weights;
mod model;
mod masks;
mod forward;
mod decode;
mod math;
mod backend;
mod simulate;
mod journey;
mod generate;
mod sweep;
use spec::*;
use weights::*;
pub use model::*;
use masks::*;
use math::*;
pub use backend::*;

/// Architectures with a known-good forward pass.
pub const SUPPORTED: &[&str] = &["llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3", "bert", "nomic-bert"];

/// Extension point for architectures that do not fit the generic passes.
pub trait ModelTracer {
    fn trace_tokens(&self, ids: &[u32], p: &dyn Progress) -> Result<Sample>;
}
