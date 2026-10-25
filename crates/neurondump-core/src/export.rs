//! JSON / NDJSON / CSV renderings shared by the CLI `export` and the web download.

use anyhow::{bail, Result};
use serde::Serialize;

use crate::inspect::{InspectReport, TensorRow};
use crate::tokenizer::Token;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Json,
    Ndjson,
    Csv,
}

impl Format {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "json" => Format::Json,
            "ndjson" | "jsonl" => Format::Ndjson,
            "csv" => Format::Csv,
            other => bail!("unknown format {other:?} (json, ndjson, csv)"),
        })
    }

    pub fn extension(self) -> &'static str {
        match self {
            Format::Json => "json",
            Format::Ndjson => "ndjson",
            Format::Csv => "csv",
        }
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Format::Json => "application/json",
            Format::Ndjson => "application/x-ndjson",
            Format::Csv => "text/csv; charset=utf-8",
        }
    }
}

/// Which table to render for NDJSON/CSV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Table {
    Tensors,
    Vocab,
    Metadata,
    Layers,
}

impl Table {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "tensors" => Table::Tensors,
            "vocab" | "tokens" => Table::Vocab,
            "metadata" | "meta" => Table::Metadata,
            "layers" => Table::Layers,
            other => bail!("unknown table {other:?} (tensors, vocab, metadata, layers)"),
        })
    }
}

pub fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn ndjson<T: Serialize>(rows: &[T]) -> Result<String> {
    let mut out = String::new();
    for r in rows {
        out.push_str(&serde_json::to_string(r)?);
        out.push('\n');
    }
    Ok(out)
}

pub fn tensors_csv(rows: &[TensorRow]) -> String {
    let mut s = String::from("name,layer,role,component,dims,type,n_elements,n_bytes,offset\n");
    for r in rows {
        let dims = r.dims.iter().map(u64::to_string).collect::<Vec<_>>().join("x");
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            csv_field(&r.name),
            r.layer.map(|l| l.to_string()).unwrap_or_default(),
            csv_field(&r.role),
            r.component,
            dims,
            r.ggml_type,
            r.n_elements,
            r.n_bytes,
            r.offset
        ));
    }
    s
}

pub fn vocab_csv(tokens: &[Token]) -> String {
    let mut s = String::from("id,raw,text,score,type,class,script,code_langs\n");
    for t in tokens {
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            t.id,
            csv_field(&t.raw),
            csv_field(&t.text),
            t.score.map(|v| v.to_string()).unwrap_or_default(),
            serde_json::to_value(t.token_type).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default(),
            t.class.as_str(),
            t.script.unwrap_or(""),
            t.code_langs.join("|")
        ));
    }
    s
}

pub fn render(report: &InspectReport, tokens: Option<&[Token]>, format: Format, table: Table) -> Result<String> {
    if format == Format::Json {
        return Ok(serde_json::to_string_pretty(report)?);
    }
    match (table, format) {
        (Table::Tensors, Format::Ndjson) => ndjson(&report.tensors),
        (Table::Tensors, _) => Ok(tensors_csv(&report.tensors)),
        (Table::Vocab, f) => {
            let Some(t) = tokens else { bail!("model has no vocabulary") };
            if f == Format::Ndjson {
                ndjson(t)
            } else {
                Ok(vocab_csv(t))
            }
        }
        (Table::Metadata, Format::Ndjson) => ndjson(&report.metadata),
        (Table::Metadata, _) => {
            let mut s = String::from("key,type,value\n");
            for m in &report.metadata {
                s.push_str(&format!("{},{},{}\n", csv_field(&m.key), csv_field(&m.value_type), csv_field(&m.value.to_string())));
            }
            Ok(s)
        }
        (Table::Layers, Format::Ndjson) => ndjson(&report.architecture.layers),
        (Table::Layers, _) => {
            let mut s = String::from("index,mixer,ffn,params,bytes,n_tensors,head_count,head_count_kv,ffn_length\n");
            for l in &report.architecture.layers {
                let j = serde_json::to_value(l)?;
                s.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{}\n",
                    l.index,
                    j["mixer"].as_str().unwrap_or(""),
                    j["ffn"].as_str().unwrap_or(""),
                    l.params,
                    l.bytes,
                    l.n_tensors,
                    l.head_count.map(|v| v.to_string()).unwrap_or_default(),
                    l.head_count_kv.map(|v| v.to_string()).unwrap_or_default(),
                    l.ffn_length.map(|v| v.to_string()).unwrap_or_default()
                ));
            }
            Ok(s)
        }
    }
}
