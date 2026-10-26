//! Human-readable renderings of core results. Every function returns a String so the
//! REPL can page it.

use std::fmt::Write;

use comfy_table::presets::UTF8_FULL_CONDENSED;
use comfy_table::{Cell, ContentArrangement, Table};
use serde_json::Value;

use neurondump_core::analyze::PrunePlan;
use neurondump_core::compare::CompareReport;
use neurondump_core::diff::DiffReport;
use neurondump_core::inspect::{InspectReport, TensorRow};
use neurondump_core::jobs::JobInfo;
use neurondump_core::slim::SlimReport;
use neurondump_core::stats::StatsReport;
use neurondump_core::template::{TemplateNode, TemplateReport};
use neurondump_core::tokenizer::VocabPage;
use neurondump_core::trace::TraceReport;
use neurondump_core::util::{human_bytes, human_count};
use neurondump_core::ModelSummary;

mod model;
mod analysis;
mod graph;
mod tokens;
mod edits;
mod diff;
pub use model::*;
pub use analysis::*;
pub use graph::*;
pub use tokens::*;
pub use edits::*;
pub use diff::*;

fn table(headers: &[&str]) -> Table {
    let mut t = Table::new();
    t.load_preset(UTF8_FULL_CONDENSED).set_content_arrangement(ContentArrangement::Dynamic).set_header(headers.to_vec());
    t
}

fn opt<T: ToString>(v: Option<T>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "-".into())
}

pub fn esc(s: &str) -> String {
    let d = format!("{s:?}");
    d[1..d.len() - 1].to_string()
}

fn enum_str<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default()
}
