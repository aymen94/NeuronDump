//! Live edits: an in-memory overlay on top of a model. The original GGUF is never
//! modified. An edit stack holds the edits (each can be toggled), with undo / redo /
//! restore and named snapshots; [`compile`] turns the enabled edits into an
//! [`Intervention`] for the forward pass, and [`bake`] writes them into a **new** GGUF.
//!
//! Ablations are edits too: `zero`, `skip` and `ablate` compile to the same masks as
//! `neurondump ablate`.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use neurondump_gguf::{dequantize_into, quantize_q8_0, GgmlType, GgufWriter, Value, WriterTensor};

use crate::ablate::{ablation_for, block_out_tensors, resolve_cut, zero_ops, ZeroOp};
use crate::graph::{ModelGraph, NodeKind};
use crate::model::ModelHandle;
use crate::slim::{apply_zero, validate, Validation};
use crate::tokenizer::Vocab;
use crate::trace::{EmbedEdit, Intervention, Patch, Sampling, Steer};
use crate::util::{check_cancel, now_unix, Progress, SCHEMA_VERSION};

mod ops;
mod stack;
mod compile;
mod bake;
mod snapshots;
pub use ops::*;
pub use stack::*;
pub use compile::*;
pub use bake::*;
pub use snapshots::*;
