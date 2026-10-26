//! Structural graph of a model: Model → Sections → Layers → Blocks → Tensors → (lazy)
//! heads / neuron buckets / experts, plus dataflow, tied and uses edges.
//!
//! Built only from architecture metadata and tensor roles (same rules as `arch`), so it
//! works for any model. Dataflow edges are stored at the finest structural level and
//! lifted to the nearest visible ancestor when a coarser view is requested.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::arch::{analyze_architecture, parse_tensor_name, Component, LayerInfo, ModelKind};
use crate::model::ModelHandle;
use crate::stats::StatsReport;
use crate::tokenizer::Vocab;
use crate::trace::TraceReport;
use crate::util::{human_bytes, human_count, SCHEMA_VERSION};

mod kinds;
mod types;
mod helpers;
mod build;
mod measure;
mod fine;
mod query;
mod view;
mod export;
pub use kinds::*;
pub use types::*;
use helpers::*;
