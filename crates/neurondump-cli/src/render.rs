//! Human-readable renderings of core results. Every function returns a String so the
//! REPL can page it.

use std::fmt::Write;

use comfy_table::presets::UTF8_FULL_CONDENSED;
use comfy_table::{Cell, ContentArrangement, Table};
use serde_json::Value;

use neurondump_core::analyze::PrunePlan;
use neurondump_core::compare::CompareReport;
use neurondump_core::inspect::{InspectReport, TensorRow};
use neurondump_core::jobs::JobInfo;
use neurondump_core::slim::SlimReport;
use neurondump_core::stats::StatsReport;
use neurondump_core::template::{TemplateNode, TemplateReport};
use neurondump_core::tokenizer::VocabPage;
use neurondump_core::trace::TraceReport;
use neurondump_core::util::{human_bytes, human_count};
use neurondump_core::ModelSummary;

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

pub fn list(rows: &[ModelSummary], warnings: &[String]) -> String {
    let mut t = table(&["NAME", "SOURCE", "ARCH", "PARAMS", "QUANT", "CTX", "SIZE", "ID"]);
    for r in rows {
        let arch = match (&r.architecture, &r.error) {
            (Some(a), _) => if r.has_projector { format!("{a} +vision") } else { a.clone() },
            (None, Some(e)) => format!("error: {}", e.chars().take(40).collect::<String>()),
            _ => "-".into(),
        };
        t.add_row(vec![
            Cell::new(&r.name),
            Cell::new(r.source.as_str()),
            Cell::new(arch),
            Cell::new(r.params.map(human_count).unwrap_or_else(|| "-".into())),
            Cell::new(opt(r.quantization.as_ref())),
            Cell::new(r.context_length.map(human_count).unwrap_or_else(|| "-".into())),
            Cell::new(human_bytes(r.size)),
            Cell::new(&r.id),
        ]);
    }
    let mut s = format!("{t}\n{} model(s)\n", rows.len());
    for w in warnings {
        let _ = writeln!(s, "warning: {w}");
    }
    s
}

pub fn inspect(r: &InspectReport) -> String {
    let mut s = String::new();
    let a = &r.architecture;
    let hp = &a.hyperparams;
    let _ = writeln!(s, "Model        {}  ({}, id {})", r.model.name, r.model.source.as_str(), r.model.id);
    let _ = writeln!(s, "File         {}", r.model.path.display());
    let _ = writeln!(
        s,
        "Format       GGUF v{} {}-endian, alignment {}, header {} / data {}",
        r.file.version,
        enum_str(&r.file.endianness),
        r.file.alignment,
        human_bytes(r.file.header_size),
        human_bytes(r.file.tensor_data_bytes)
    );
    let _ = writeln!(s, "Size         {}  ({} tensors, {} metadata keys)", human_bytes(r.file.file_size), r.file.n_tensors, r.file.n_kv);
    let _ = writeln!(
        s,
        "Architecture {}  [{}]  {} params, quantization {}",
        a.name,
        enum_str(&a.kind),
        human_count(a.total_params),
        opt(r.model.quantization.as_ref())
    );
    if !r.model.shares_weights_with.is_empty() {
        let _ = writeln!(s, "Shares blob  {}", r.model.shares_weights_with.join(", "));
    }
    let _ = writeln!(s);
    let mut t = table(&["HYPERPARAMETER", "VALUE"]);
    let hpv = serde_json::to_value(hp).unwrap_or_default();
    if let Value::Object(m) = hpv {
        for (k, v) in m {
            if !v.is_null() {
                t.add_row(vec![k, v.to_string()]);
            }
        }
    }
    let _ = writeln!(s, "{t}");

    let mut t = table(&["COMPONENT", "PARAMS", "BYTES"]);
    for (k, v) in &a.params_by_component {
        t.add_row(vec![k.clone(), human_count(*v), human_bytes(*a.bytes_by_component.get(k).unwrap_or(&0))]);
    }
    let _ = writeln!(s, "{t}");
    let mut t = table(&["TYPE", "TENSORS", "PARAMS", "BYTES", "BITS/W"]);
    for u in &a.types {
        t.add_row(vec![
            u.ggml_type.to_string(),
            u.tensors.to_string(),
            human_count(u.params),
            human_bytes(u.bytes),
            u.bits_per_weight.map(|b| format!("{b:.2}")).unwrap_or_else(|| "?".into()),
        ]);
    }
    let _ = writeln!(s, "{t}");
    let _ = writeln!(s, "Layers: {}", a.layers.len());
    for (k, v) in &a.layer_kinds {
        let _ = writeln!(s, "  {k:<36} {:>3} layers  {}", v.len(), compress_ranges(v));
    }
    if a.tied_embeddings {
        let _ = writeln!(s, "  embeddings are tied (no output.weight)");
    }

    if let Some(tk) = &r.tokenizer {
        let _ = writeln!(s, "\nTokenizer    {} (pre: {}), {} tokens, {} merges", tk.model, opt(tk.pre.as_ref()), tk.size, tk.merges);
        let classes: Vec<String> = tk.class_counts.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let _ = writeln!(s, "  classes    {}", classes.join("  "));
        let scripts: Vec<String> = tk.script_counts.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let _ = writeln!(s, "  scripts    {}", scripts.join("  "));
        let langs: Vec<String> = tk.code_lang_counts.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let _ = writeln!(s, "  code       {}", langs.join("  "));
        let sp: Vec<String> = tk.specials.iter().map(|t| format!("{}={} {}", t.role, t.id, esc(t.text.as_deref().unwrap_or("")))).collect();
        let _ = writeln!(s, "  specials   {}", sp.join("  "));
        let _ = writeln!(s, "  control    {} control/user-defined tokens", tk.control_tokens.len());
    }
    for tt in &r.templates.templates {
        let _ = writeln!(
            s,
            "\nTemplate     {} ({}): {} decisions, {} variables, emits {}",
            tt.name,
            enum_str(&tt.syntax),
            tt.decisions,
            tt.variables.len(),
            tt.special_tokens.iter().take(8).cloned().collect::<Vec<_>>().join(" ")
        );
    }
    if let Some(o) = &r.ollama {
        if let Some(sys) = &o.system {
            let _ = writeln!(s, "System       {}", esc(&sys.chars().take(100).collect::<String>()));
        }
        if let Some(p) = &o.params {
            let _ = writeln!(s, "Params       {p}");
        }
        for l in &o.licenses {
            let _ = writeln!(s, "License      {}", l.chars().take(80).collect::<String>());
        }
    }
    if let Some(p) = &r.projector {
        let _ = writeln!(
            s,
            "\nProjector    {} [{}], {} params, {} tensors, {}",
            p.architecture.name,
            enum_str(&p.architecture.kind),
            human_count(p.architecture.total_params),
            p.file.n_tensors,
            human_bytes(p.file.file_size)
        );
    }
    if let Some(st) = &r.stats {
        let _ = writeln!(s, "\n{}", stats(st, 20));
    }
    if !r.warnings.is_empty() {
        let _ = writeln!(s, "\nWarnings ({}):", r.warnings.len());
        for w in r.warnings.iter().take(30) {
            let _ = writeln!(s, "  - {w}");
        }
    }
    s
}

fn compress_ranges(v: &[u32]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let start = v[i];
        let mut end = start;
        while i + 1 < v.len() && v[i + 1] == end + 1 {
            i += 1;
            end = v[i];
        }
        out.push(if start == end { start.to_string() } else { format!("{start}-{end}") });
        i += 1;
    }
    out.join(",")
}

pub fn metadata(r: &InspectReport) -> String {
    let mut t = table(&["KEY", "TYPE", "VALUE"]);
    for m in &r.metadata {
        let v = m.value.to_string();
        let v: String = if v.chars().count() > 120 { format!("{}…", v.chars().take(120).collect::<String>()) } else { v };
        t.add_row(vec![m.key.clone(), m.value_type.clone(), v]);
    }
    t.to_string()
}

pub fn vocab(p: &VocabPage) -> String {
    let mut t = table(&["ID", "TEXT", "RAW", "TYPE", "CLASS", "SCRIPT/LANG", "SCORE"]);
    for tok in &p.tokens {
        let sl = if tok.code_langs.is_empty() { tok.script.unwrap_or("").to_string() } else { tok.code_langs.join(",") };
        t.add_row(vec![
            tok.id.to_string(),
            format!("\"{}\"", esc(&tok.text)),
            esc(&tok.raw),
            enum_str(&tok.token_type),
            tok.class.as_str().to_string(),
            sl,
            tok.score.map(|s| format!("{s:.3}")).unwrap_or_default(),
        ]);
    }
    format!("{t}\n{} match(es), showing {}-{}\n", p.total, p.offset, p.offset + p.tokens.len())
}

pub fn tensors(rows: &[&TensorRow]) -> String {
    let mut t = table(&["NAME", "LAYER", "ROLE", "SHAPE", "TYPE", "PARAMS", "BYTES", "OFFSET"]);
    for r in rows {
        t.add_row(vec![
            r.name.clone(),
            opt(r.layer),
            r.role.clone(),
            r.dims.iter().map(u64::to_string).collect::<Vec<_>>().join("x"),
            r.ggml_type.clone(),
            human_count(r.n_elements),
            human_bytes(r.n_bytes),
            format!("{:#x}", r.offset),
        ]);
    }
    format!("{t}\n{} tensor(s)\n", rows.len())
}

pub fn layers(r: &InspectReport) -> String {
    let mut t = table(&["LAYER", "MIXER", "FFN", "HEADS", "KV", "FFN LEN", "PARAMS", "BYTES", "TENSORS"]);
    for l in &r.architecture.layers {
        t.add_row(vec![
            l.index.to_string(),
            enum_str(&l.mixer),
            enum_str(&l.ffn),
            opt(l.head_count),
            opt(l.head_count_kv),
            opt(l.ffn_length),
            human_count(l.params),
            human_bytes(l.bytes),
            l.n_tensors.to_string(),
        ]);
    }
    t.to_string()
}

fn tree(n: &TemplateNode, prefix: &str, last: bool, out: &mut String, root: bool) {
    let label = match n.kind.as_str() {
        "root" => "template".to_string(),
        "branch" => format!("{} {}", n.branch.as_deref().unwrap_or("?"), n.condition.as_deref().unwrap_or("")),
        "emit" => {
            let t = n.text.as_deref().unwrap_or("");
            let t: String = t.chars().take(70).collect();
            if n.emits.is_empty() {
                format!("emit \"{}\"", esc(&t))
            } else {
                format!("emit \"{}\"  → {}", esc(&t), n.emits.join(" "))
            }
        }
        k => format!("{k} {}", n.condition.as_deref().unwrap_or("")),
    };
    let vars = if n.vars.is_empty() || n.kind == "emit" { String::new() } else { format!("   [reads {}]", n.vars.join(", ")) };
    if root {
        let _ = writeln!(out, "{label}");
    } else {
        let _ = writeln!(out, "{prefix}{}{}{vars}", if last { "└─ " } else { "├─ " }, label.trim_end());
    }
    let child_prefix = if root { String::new() } else { format!("{prefix}{}", if last { "   " } else { "│  " }) };
    for (i, c) in n.children.iter().enumerate() {
        tree(c, &child_prefix, i + 1 == n.children.len(), out, false);
    }
}

pub fn template(r: &TemplateReport, raw: bool) -> String {
    let mut s = String::new();
    if r.templates.is_empty() {
        s.push_str("no chat template in this model\n");
    }
    for t in &r.templates {
        let _ = writeln!(s, "=== {} ({}) — {} decisions ===", t.name, enum_str(&t.syntax), t.decisions);
        if raw {
            let _ = writeln!(s, "{}\n", t.source);
        }
        tree(&t.root, "", true, &mut s, true);
        let _ = writeln!(s, "variables:      {}", t.variables.iter().cloned().collect::<Vec<_>>().join(", "));
        let _ = writeln!(s, "special tokens: {}", t.special_tokens.iter().cloned().collect::<Vec<_>>().join(" "));
        for e in &t.errors {
            let _ = writeln!(s, "parse warning:  {e}");
        }
        s.push('\n');
    }
    if let Some(sys) = &r.ollama_system {
        let _ = writeln!(s, "SYSTEM: {}", esc(sys));
    }
    if let Some(p) = &r.ollama_params {
        let _ = writeln!(s, "PARAMS: {p}");
    }
    s
}

pub fn stats(r: &StatsReport, limit: usize) -> String {
    let mut s = String::new();
    let mut t = table(&["TENSOR", "TYPE", "MEAN", "STD", "MIN", "MAX", "L2", "SPARSITY", "OUTLIERS"]);
    for ts in r.tensors.iter().take(limit) {
        t.add_row(vec![
            ts.name.clone(),
            ts.ggml_type.clone(),
            format!("{:.2e}", ts.mean),
            format!("{:.3e}", ts.std),
            format!("{:.3}", ts.min),
            format!("{:.3}", ts.max),
            format!("{:.2}", ts.l2),
            format!("{:.2}%", ts.sparsity * 100.0),
            format!("{} ({:.3}%)", ts.outliers, ts.outlier_fraction * 100.0),
        ]);
    }
    let _ = writeln!(s, "{t}");
    if r.tensors.len() > limit {
        let _ = writeln!(s, "… {} more tensors (use --json for all)", r.tensors.len() - limit);
    }
    if !r.heads.is_empty() {
        let mut t = table(&["LAYER", "HEADS", "MIN |Q|", "MAX |Q|", "MIN |O|", "MAX |O|", "KV HEADS"]);
        for h in &r.heads {
            let mm = |v: &[f32]| {
                if v.is_empty() {
                    ("-".to_string(), "-".to_string())
                } else {
                    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    (format!("{mn:.2}"), format!("{mx:.2}"))
                }
            };
            let (qa, qb) = mm(&h.q);
            let (oa, ob) = mm(&h.o);
            t.add_row(vec![h.layer.to_string(), h.q.len().to_string(), qa, qb, oa, ob, h.k.len().to_string()]);
        }
        let _ = writeln!(s, "Per-head norms\n{t}");
    }
    if !r.ffn_neurons.is_empty() {
        let mut t = table(&["LAYER", "NEURONS", "MEDIAN NORM", "<10% OF MEDIAN"]);
        for n in &r.ffn_neurons {
            let mut v = n.norms.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = v.get(v.len() / 2).copied().unwrap_or(0.0);
            let low = v.iter().filter(|x| **x < 0.1 * med).count();
            t.add_row(vec![n.layer.to_string(), v.len().to_string(), format!("{med:.3}"), low.to_string()]);
        }
        let _ = writeln!(s, "FFN neurons\n{t}");
    }
    if !r.experts.is_empty() {
        let _ = writeln!(s, "MoE experts: {} layers, {} experts each", r.experts.len(), r.experts[0].norms.len());
    }
    for (n, why) in r.skipped.iter().take(10) {
        let _ = writeln!(s, "skipped {n}: {why}");
    }
    let _ = writeln!(s, "computed in {:.1}s", r.elapsed_ms as f64 / 1000.0);
    s
}

pub fn plan(p: &PrunePlan) -> String {
    let mut s = format!("Prune plan for {} ({}), original size {}\n", p.model, p.architecture, human_bytes(p.original_size));
    let mut t = table(&["ID", "RISK", "SAVES", "APPLY", "SEL", "TITLE"]);
    for c in &p.candidates {
        t.add_row(vec![
            c.id.clone(),
            enum_str(&c.risk),
            human_bytes(c.bytes_saved),
            if c.applicable { "yes".into() } else { "no".into() },
            if c.selected { "x".into() } else { String::new() },
            c.title.clone(),
        ]);
    }
    let _ = writeln!(s, "{t}");
    for c in &p.candidates {
        let _ = writeln!(s, "- {}: {}", c.id, c.detail);
        if let Some(r) = &c.reason {
            let _ = writeln!(s, "    not applicable: {r}");
        }
    }
    for n in &p.notes {
        let _ = writeln!(s, "note: {n}");
    }
    let _ = writeln!(s, "selected candidates save {}", human_bytes(p.total_selected_bytes()));
    s
}

pub fn slim(r: &SlimReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Wrote     {}", r.output.display());
    let _ = writeln!(s, "Modelfile {}", r.modelfile.display());
    let pct = if r.original_size > 0 { 100.0 * (1.0 - r.new_size as f64 / r.original_size as f64) } else { 0.0 };
    if r.new_size == r.original_size {
        let _ = writeln!(s, "Size      {} (unchanged)", human_bytes(r.new_size));
    } else {
        let _ = writeln!(s, "Size      {} -> {} ({pct:.1}% smaller)", human_bytes(r.original_size), human_bytes(r.new_size));
    }
    let _ = writeln!(s, "Applied   {}", if r.applied.is_empty() { "-".into() } else { r.applied.join(", ") });
    for (id, why) in &r.refused {
        let _ = writeln!(s, "Refused   {id}: {why}");
    }
    let _ = writeln!(s, "Removed   {} tokens, layers {:?}, requantized {} tensors", r.removed_tokens, r.removed_layers, r.requantized.len());
    if !r.zeroed.is_empty() {
        let _ = writeln!(s, "Zeroed    {} tensor(s) in place: {}", r.zeroed.len(), r.zeroed.join(", "));
    }
    let v = &r.validation;
    let _ = writeln!(
        s,
        "Validate  {} ({} tensors, vocab {}, block_count {})",
        if v.ok { "OK" } else { "PROBLEMS" },
        v.n_tensors,
        opt(v.n_vocab),
        opt(v.block_count)
    );
    for pr in &v.problems {
        let _ = writeln!(s, "  - {pr}");
    }
    if let Some(p) = &r.perplexity {
        let _ = writeln!(s, "Perplexity original {} -> slim {}", opt(p.original), opt(p.slim));
        let _ = writeln!(s, "{}", p.log);
    }
    s
}

pub fn trace(r: &TraceReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Trace of {} ({}, backend {}), {} sample(s), {} ms", r.model, r.architecture, r.backend, r.samples, r.elapsed_ms);
    let toks: Vec<String> = r.tokens.iter().take(40).map(|(i, t)| format!("{i}:\"{}\"", esc(t))).collect();
    let _ = writeln!(s, "tokens: {}", toks.join(" "));
    let mut t = table(&["LAYER", "|h|", "COS(in,out)", "ATTN ENTROPY", "TOP NEURONS", "LOGIT LENS (top-5)"]);
    for l in &r.layers {
        let ent = if l.head_entropy.is_empty() { "-".into() } else { format!("{:.2}", l.head_entropy.iter().sum::<f32>() / l.head_entropy.len() as f32) };
        let neurons: Vec<String> = l.top_neurons.iter().take(3).map(|(i, v)| format!("{i}:{v:.2}")).collect();
        let lens: Vec<String> = l.logit_lens.iter().take(5).map(|t| format!("\"{}\" {:.2}", esc(&t.text), t.prob)).collect();
        t.add_row(vec![
            l.index.to_string(),
            format!("{:.1}", l.hidden_norm),
            l.cos_prev.map(|c| format!("{c:.4}")).unwrap_or_else(|| "-".into()),
            ent,
            neurons.join(" "),
            lens.join("  "),
        ]);
    }
    let _ = writeln!(s, "{t}");
    if !r.prediction.is_empty() {
        let p: Vec<String> = r.prediction.iter().map(|t| format!("\"{}\" {:.3}", esc(&t.text), t.prob)).collect();
        let _ = writeln!(s, "next token: {}", p.join("  "));
    }
    if !r.embedding_contributions.is_empty() {
        let p: Vec<String> = r.embedding_contributions.iter().map(|t| format!("\"{}\" {:.3}", esc(&t.text), t.contribution)).collect();
        let _ = writeln!(s, "token contributions to pooled embedding: {}", p.join("  "));
    }
    for w in &r.warnings {
        let _ = writeln!(s, "warning: {w}");
    }
    s
}

pub fn compare(r: &CompareReport) -> String {
    let mut headers = vec!["PROPERTY".to_string()];
    headers.extend(r.models.iter().map(|m| m.name.clone()));
    let mut t = Table::new();
    t.load_preset(UTF8_FULL_CONDENSED).set_content_arrangement(ContentArrangement::Dynamic).set_header(headers);
    let rows: Vec<(&str, Box<dyn Fn(&neurondump_core::compare::ModelBrief) -> String>)> = vec![
        ("architecture", Box::new(|m| m.architecture.clone())),
        ("kind", Box::new(|m| enum_str(&m.kind))),
        ("params", Box::new(|m| human_count(m.params))),
        ("quantization", Box::new(|m| opt(m.quantization.as_ref()))),
        ("file size", Box::new(|m| human_bytes(m.file_size))),
        ("layers", Box::new(|m| m.layers.to_string())),
        ("vocab", Box::new(|m| opt(m.vocab_size))),
        ("tokenizer", Box::new(|m| opt(m.tokenizer.as_ref()))),
        ("context", Box::new(|m| opt(m.context_length))),
        ("embedding", Box::new(|m| opt(m.embedding_length))),
        ("heads", Box::new(|m| opt(m.head_count))),
        ("kv heads", Box::new(|m| opt(m.head_count_kv))),
        ("ffn", Box::new(|m| opt(m.feed_forward_length))),
        ("experts", Box::new(|m| opt(m.expert_count))),
        ("layer kinds", Box::new(|m| m.layer_kinds.join("; "))),
    ];
    for (name, f) in rows {
        let mut row = vec![name.to_string()];
        row.extend(r.models.iter().map(|m| f(m)));
        t.add_row(row);
    }
    let mut s = format!("{t}\n");
    if r.models.len() > 1 {
        let _ = writeln!(s, "vocabulary overlap (Jaccard):");
        for (i, row) in r.vocab_overlap.iter().enumerate() {
            let cells: Vec<String> = row.iter().map(|v| format!("{v:.3}")).collect();
            let _ = writeln!(s, "  {:<30} {}", r.models[i].name, cells.join("  "));
        }
    }
    let _ = writeln!(s, "{} shared tensor names, {} differ in shape/type", r.shared_tensor_names, r.tensor_diffs.len());
    let _ = writeln!(s, "{} differing metadata keys:", r.metadata_diff.len());
    for d in r.metadata_diff.iter().take(40) {
        let vals: Vec<String> = d.values.iter().map(|v| v.clone().unwrap_or_else(|| "∅".into())).collect();
        let _ = writeln!(s, "  {}: {}", d.key, vals.join(" | "));
    }
    s
}

pub fn search(r: &neurondump_core::search::SearchResults) -> String {
    let mut t = table(&["KIND", "MATCH", "DETAIL", "MODEL"]);
    for h in &r.hits {
        t.add_row(vec![enum_str(&h.kind), h.label.clone(), h.detail.clone(), h.model_name.clone()]);
    }
    let [m, tk, te, me] = r.totals;
    format!("{t}
{m} model(s), {tk} token(s), {te} tensor(s), {me} metadata key(s) match {:?}
", r.query)
}

pub fn jobs(jobs: &[JobInfo]) -> String {
    let mut t = table(&["ID", "KIND", "MODEL", "STATUS", "PROGRESS", "STEP", "ERROR"]);
    for j in jobs {
        t.add_row(vec![
            j.id.clone(),
            j.kind.as_str().into(),
            j.model.clone(),
            enum_str(&j.status),
            format!("{:.0}%", j.progress),
            j.step.clone(),
            j.error.clone().unwrap_or_default(),
        ]);
    }
    t.to_string()
}

// --- graph / ablation -------------------------------------------------------------------

fn size_cell(params: u64, bytes: u64) -> String {
    format!("{} · {}", human_count(params), human_bytes(bytes))
}

pub fn node(d: &neurondump_core::graph::NodeDetail) -> String {
    let n = &d.node;
    let mut s = String::new();
    let path: Vec<String> = d.ancestors.iter().rev().map(|a| a.id.clone()).collect();
    let _ = writeln!(s, "{}  ({})  #{}", n.label, n.kind.as_str(), n.id);
    if !path.is_empty() {
        let _ = writeln!(s, "path        {} > {}", path.join(" > "), n.id);
    }
    let _ = writeln!(s, "size        {} params · {}", human_count(n.params), human_bytes(n.bytes));
    if !n.quant.is_empty() {
        let _ = writeln!(s, "quant       {}", n.quant.join(", "));
    }
    if let Some(k) = &n.layer_kind {
        let _ = writeln!(s, "layer kind  {k}");
    }
    let _ = writeln!(s, "removable   {} — {}", n.removable.as_str(), n.removable_reason);
    let _ = writeln!(s, "importance  {:.2} ({})", n.importance.score, n.importance.source);
    if let Some(st) = &n.stats {
        let mut parts = vec![format!("l2 {:.3}", st.l2)];
        if let Some(m) = st.mean {
            parts.push(format!("mean {m:.4}"));
        }
        if let Some(v) = st.std {
            parts.push(format!("std {v:.4}"));
        }
        if let Some(v) = st.sparsity {
            parts.push(format!("zeros {:.2}%", v * 100.0));
        }
        if let Some(v) = st.outlier_fraction {
            parts.push(format!("outliers {:.3}%", v * 100.0));
        }
        let _ = writeln!(s, "weights     {}", parts.join(", "));
    }
    if let Some(a) = &n.activation {
        let _ = writeln!(s, "activation  {a}");
    }
    if !n.info.is_null() {
        let _ = writeln!(s, "info        {}", n.info);
    }
    if !d.children.is_empty() {
        let mut t = table(&["CHILD", "KIND", "PARAMS · BYTES", "REMOVABLE", "IMPORTANCE"]);
        for c in d.children.iter().take(200) {
            t.add_row(vec![
                format!("{}  #{}", c.label, c.id),
                c.kind.as_str().into(),
                size_cell(c.params, c.bytes),
                c.removable.as_str().into(),
                format!("{:.2}", c.importance.score),
            ]);
        }
        let _ = writeln!(s, "{t}");
        if d.children.len() > 200 {
            let _ = writeln!(s, "... {} more", d.children.len() - 200);
        }
    }
    for e in &d.incoming {
        let _ = writeln!(s, "in   <- {} ({}{})", e.from, enum_str(&e.kind), e.label.as_ref().map(|l| format!(": {l}")).unwrap_or_default());
    }
    for e in &d.outgoing {
        let _ = writeln!(s, "out  -> {} ({}{})", e.to, enum_str(&e.kind), e.label.as_ref().map(|l| format!(": {l}")).unwrap_or_default());
    }
    s
}

pub fn impact(i: &neurondump_core::ablate::StaticImpact) -> String {
    let mut s = String::new();
    let mut t = table(&["CUT", "KIND", "MODE", "PARAMS · BYTES", "REMOVABLE"]);
    for c in &i.cut {
        t.add_row(vec![format!("{}  #{}", c.label, c.id), c.kind.as_str().into(), enum_str(&c.mode), size_cell(c.params, c.bytes), c.removable.as_str().into()]);
    }
    let _ = writeln!(s, "Static impact on {} ({})", i.model, i.architecture);
    let _ = writeln!(s, "{t}");
    for u in &i.unknown {
        let _ = writeln!(s, "unknown id: {u} (see `graph` / `node`)");
    }
    for r in &i.redundant {
        let _ = writeln!(s, "ignored {r}: an ancestor is already cut");
    }
    let _ = writeln!(
        s,
        "Removed     {} params ({}), zeroed {} params — {:.2}% of the model",
        human_count(i.params_removed),
        human_bytes(i.bytes_removed),
        human_count(i.params_zeroed),
        i.percent_of_model
    );
    let delta = if i.requant_bytes_delta != 0 {
        format!(" (requantization {}{})", if i.requant_bytes_delta > 0 { "+" } else { "-" }, human_bytes(i.requant_bytes_delta.unsigned_abs()))
    } else {
        String::new()
    };
    let _ = writeln!(s, "File size   {} -> {}{delta}", human_bytes(i.original_size), human_bytes(i.new_size));
    if i.other_files_removed > 0 {
        let _ = writeln!(s, "Other files {} no longer needed (projector)", human_bytes(i.other_files_removed));
    }
    let _ = writeln!(s, "Verdict     {}{}", i.verdict.as_str().to_uppercase(), if i.refused { " — refused without --force" } else { "" });
    for r in &i.reasons {
        let _ = writeln!(s, "  - {r}");
    }
    for d in &i.dependencies {
        let _ = writeln!(s, "Depends     {d}");
    }
    if !i.disconnected.is_empty() {
        let shown: Vec<&str> = i.disconnected.iter().take(12).map(|x| x.as_str()).collect();
        let more = if i.disconnected.len() > 12 { format!(" … +{}", i.disconnected.len() - 12) } else { String::new() };
        let _ = writeln!(s, "Cut off     {}{more}", shown.join(", "));
    }
    let _ = writeln!(s, "llama.cpp   {}", if i.llama_cpp_loads { "loads the slimmed file" } else { "would NOT load the slimmed file" });
    for l in &i.legality {
        let _ = writeln!(s, "  - {l}");
    }
    if !i.layer_renumbering.is_empty() {
        let r: Vec<String> = i.layer_renumbering.iter().take(8).map(|(a, b)| format!("{a}->{b}")).collect();
        let _ = writeln!(s, "Renumber    {}{}", r.join(" "), if i.layer_renumbering.len() > 8 { " …" } else { "" });
    }
    for f in &i.file_ops {
        let _ = writeln!(s, "slim would  {f}");
    }
    let _ = writeln!(s, "Simulation  {}", if i.simulation_supported { "available (run without --static)" } else { "not available" });
    for n in &i.simulation_notes {
        let _ = writeln!(s, "  - {n}");
    }
    s
}

fn toks(v: &[neurondump_core::trace::LensToken], n: usize) -> String {
    v.iter().take(n).map(|t| format!("\"{}\" {:.3}", esc(&t.text), t.prob)).collect::<Vec<_>>().join("  ")
}

pub fn simulation(r: &neurondump_core::trace::SimReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Simulated ablation of {} on {} ({}, {} ms) — no file written", r.cut.join(", "), r.model, r.backend, r.elapsed_ms);
    for p in &r.prompts {
        let _ = writeln!(s, "\nprompt: \"{}\" ({} tokens)", esc(&p.prompt), p.n_tokens);
        let mut t = table(&["", "BASELINE", "ABLATED"]);
        if !p.baseline.top_k.is_empty() || !p.ablated.top_k.is_empty() {
            t.add_row(vec!["top-k next token".to_string(), toks(&p.baseline.top_k, 5), toks(&p.ablated.top_k, 5)]);
        }
        if !p.baseline.generated.is_empty() || !p.ablated.generated.is_empty() {
            t.add_row(vec![
                format!("greedy text ({} tok)", p.baseline.generated_tokens.max(p.ablated.generated_tokens)),
                format!("\"{}\"", esc(&p.baseline.generated)),
                format!("\"{}\"", esc(&p.ablated.generated)),
            ]);
        }
        t.add_row(vec!["tokens/s".to_string(), format!("{:.2}", p.baseline.tokens_per_s), format!("{:.2}", p.ablated.tokens_per_s)]);
        if let (Some(a), Some(b)) = (p.baseline.embedding_norm, p.ablated.embedding_norm) {
            t.add_row(vec!["embedding |v|".to_string(), format!("{a:.3}"), format!("{b:.3}")]);
        }
        let _ = writeln!(s, "{t}");
        if let Some(k) = p.kl {
            let _ = writeln!(s, "KL(baseline || ablated) = {k:.4} nats; top-1 {}", if p.top1_changed { "CHANGED" } else { "unchanged" });
        }
        if let Some(c) = p.embedding_cosine {
            let _ = writeln!(s, "embedding cosine(baseline, ablated) = {c:.4}");
        }
        if !p.lens.is_empty() {
            let target = p.baseline.top_k.first().map(|t| esc(&t.text)).unwrap_or_default();
            let head = format!("P(\"{target}\") base → ablated");
            let mut lt = table(&["LAYER", "BASELINE LENS", "ABLATED LENS", head.as_str()]);
            for l in &p.lens {
                let one = |x: &Option<neurondump_core::trace::LensToken>| x.as_ref().map(|t| format!("\"{}\" {:.2}", esc(&t.text), t.prob)).unwrap_or_default();
                lt.add_row(vec![
                    format!("{}{}", l.layer, if l.skipped { " (cut)" } else { "" }),
                    one(&l.baseline),
                    one(&l.ablated),
                    format!("{:.3} → {:.3}", l.baseline_target_prob, l.ablated_target_prob),
                ]);
            }
            let _ = writeln!(s, "{lt}");
        }
    }
    if let Some(c) = &r.corpus {
        let _ = writeln!(
            s,
            "corpus: {} sample(s), {} positions: perplexity {} -> {}, mean KL {}, top-1 agreement {:.1}%",
            c.samples,
            c.tokens,
            c.baseline_ppl.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into()),
            c.ablated_ppl.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into()),
            c.mean_kl.map(|v| format!("{v:.4}")).unwrap_or_else(|| "-".into()),
            c.top1_agreement * 100.0
        );
    }
    let _ = writeln!(s, "weights in use: {} -> {}", human_bytes(r.weights_bytes_baseline), human_bytes(r.weights_bytes_ablated));
    for w in &r.warnings {
        let _ = writeln!(s, "note: {w}");
    }
    s
}

pub fn sweep(r: &neurondump_core::trace::SweepReport) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Sweep of {} by {}{} — {} sample(s), {} positions, baseline perplexity {} ({} ms)",
        r.model,
        r.level,
        r.layer.map(|l| format!(" (layer {l})")).unwrap_or_default(),
        r.samples,
        r.tokens,
        r.baseline_ppl.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into()),
        r.elapsed_ms
    );
    let mut t = table(&["RANK", "REMOVED", "MEAN KL", "PERPLEXITY", "PPL +%", "TOP-1 SAME", "DAMAGE"]);
    let max = r.entries.iter().map(|e| e.mean_kl).fold(0f64, f64::max);
    let mut rows: Vec<_> = r.entries.iter().collect();
    rows.sort_by_key(|e| e.rank);
    for e in rows {
        let bar = if max > 0.0 { "█".repeat(((e.mean_kl / max) * 20.0).round() as usize) } else { String::new() };
        t.add_row(vec![
            e.rank.to_string(),
            format!("{}  #{}", e.label, e.id),
            format!("{:.4}", e.mean_kl),
            e.ppl.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into()),
            e.ppl_increase.map(|v| format!("{:+.1}%", v * 100.0)).unwrap_or_else(|| "-".into()),
            format!("{:.1}%", e.top1_agreement * 100.0),
            bar,
        ]);
    }
    let _ = writeln!(s, "{t}");
    for w in &r.warnings {
        let _ = writeln!(s, "note: {w}");
    }
    s
}

// --- token inspector / edits -------------------------------------------------------------

pub fn tokens(r: &neurondump_core::tokens::TokenizeReport) -> String {
    let mut s = String::new();
    if let Some(c) = &r.chat {
        let _ = writeln!(s, "chat template: {} (system: {})", c.source, c.system.as_deref().map(esc).unwrap_or_else(|| "none".into()));
        let _ = writeln!(s, "templated prompt:\n  \"{}\"", esc(&r.full_text));
    }
    let _ = writeln!(
        s,
        "{} tokens ({} tokenizer){}",
        r.tokens.len(),
        r.tokenizer,
        if r.tied_embeddings { ", embeddings tied to the output head" } else { "" }
    );
    let mut t = table(&["#", "ID", "TEXT", "CLASS", "ADDED", "|EMB|", "|OUT|", "PREDICTABLE", "NEAREST (cosine)"]);
    for k in &r.tokens {
        t.add_row(vec![
            k.index.to_string(),
            k.id.to_string(),
            format!("\"{}\"", esc(&k.text)),
            format!("{}{}", enum_str(&k.class), if k.special { " (special)" } else { "" }),
            if k.added { "template".into() } else { String::new() },
            opt(k.embedding_norm.map(|v| format!("{v:.3}"))),
            opt(k.output_norm.map(|v| format!("{v:.3}"))),
            if k.banned { "banned".into() } else if k.in_output { "yes".into() } else { "no".into() },
            k.neighbors.iter().take(5).map(|n| format!("\"{}\" {:.2}", esc(&n.text), n.cosine)).collect::<Vec<_>>().join("  "),
        ]);
    }
    let _ = writeln!(s, "{t}");
    for w in &r.warnings {
        let _ = writeln!(s, "note: {w}");
    }
    s
}

pub fn decision(d: &neurondump_core::trace::Decision) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "decision at position {}: \"{}\" {:.3} vs runner-up \"{}\" {:.3} (logit diff {:.3}, attributed {:.3})",
        d.position,
        esc(&d.chosen.text),
        d.chosen.prob,
        esc(&d.runner_up.text),
        d.runner_up.prob,
        d.logit_diff,
        d.attributed
    );
    let _ = writeln!(s, "candidates: {}", toks(&d.candidates, 8));
    let mut t = table(&["LAYER", "ATTENTION", "FFN", "PUSH"]);
    let max = d.layers.iter().map(|l| (l.attn + l.ffn).abs()).fold(1e-6f32, f32::max);
    let _ = writeln!(s, "embedding: {:+.3}", d.embedding);
    for l in &d.layers {
        let v = l.attn + l.ffn;
        let n = ((v.abs() / max) * 20.0).round() as usize;
        t.add_row(vec![l.layer.to_string(), format!("{:+.3}", l.attn), format!("{:+.3}", l.ffn), format!("{}{}", if v < 0.0 { "-" } else { "+" }, "#".repeat(n))]);
    }
    let _ = writeln!(s, "{t}");
    if !d.top_heads.is_empty() {
        let _ = writeln!(s, "top heads: {}", d.top_heads.iter().take(8).map(|h| format!("L{}H{} {:+.3}", h.layer, h.head, h.value)).collect::<Vec<_>>().join("  "));
    }
    if !d.top_neurons.is_empty() {
        let _ = writeln!(
            s,
            "top neurons: {}",
            d.top_neurons.iter().take(8).map(|n| format!("L{}N{} {:+.3} (act {:.2})", n.layer, n.neuron, n.value, n.activation)).collect::<Vec<_>>().join("  ")
        );
    }
    s
}

pub fn journey(r: &neurondump_core::trace::JourneyReport) -> String {
    let mut s = String::new();
    let tok = &r.tokens[r.position];
    let _ = writeln!(
        s,
        "journey of token {} \"{}\" (id {}) on {}{} ({} ms)",
        r.position,
        esc(&tok.text),
        tok.id,
        r.model,
        if r.edited { " [edited]" } else { "" },
        r.elapsed_ms
    );
    let _ = writeln!(
        s,
        "sequence: {}",
        r.tokens.iter().enumerate().map(|(i, t)| if i == r.position { format!("[{}]", esc(&t.text)) } else { esc(&t.text) }).collect::<Vec<_>>().join("|")
    );
    if let Some(n) = &r.next {
        let _ = writeln!(s, "next token in the sequence: \"{}\"", esc(&n.text));
    }
    if !r.prediction.is_empty() {
        let _ = writeln!(s, "final prediction: {}", toks(&r.prediction, 5));
    }
    let _ = writeln!(
        s,
        "target first top-1 at layer {}, first in top-5 at layer {}",
        opt(r.first_top1),
        opt(r.first_top5)
    );
    let mut t = table(&["LAYER", "|h|", "LOGIT LENS", "P(target)", "RANK", "ATTENDS TO", "TOP FFN BUCKET"]);
    for l in &r.layers {
        t.add_row(vec![
            format!("{}{}", l.index, if l.skipped { " (skip)" } else { "" }),
            format!("{:.1}", l.residual_norm),
            toks(&l.lens, 3),
            format!("{:.3}", l.target_prob),
            l.target_rank.to_string(),
            l.attended.iter().take(3).map(|(k, w)| format!("{}:\"{}\" {:.2}", k, esc(&r.tokens.get(*k).map(|t| t.text.clone()).unwrap_or_default()), w)).collect::<Vec<_>>().join(" "),
            l.ffn_buckets.first().map(|(id, v)| format!("{} {:.3}", id.rsplit('.').next().unwrap_or(id), v)).unwrap_or_default(),
        ]);
    }
    let _ = writeln!(s, "{t}");
    if let Some(d) = &r.decision {
        s.push_str(&decision(d));
    }
    if !r.highlight.is_empty() {
        let _ = writeln!(s, "contributing nodes: {}", r.highlight.join(", "));
    }
    for w in &r.warnings {
        let _ = writeln!(s, "note: {w}");
    }
    s
}

pub fn generate(r: &neurondump_core::trace::GenerateReport, explain: bool) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "generated {} token(s) on {}{} ({:.2} tokens/s, {} ms)",
        r.steps.len(),
        r.model,
        if r.edited { " [edited]" } else { "" },
        r.tokens_per_s,
        r.elapsed_ms
    );
    let _ = writeln!(s, "text: \"{}\"", esc(&r.generated));
    let mut t = table(&["STEP", "TOKEN", "P", "ENTROPY", "TOP ALTERNATIVE", "TOP LAYER PUSH"]);
    for st in &r.steps {
        let top = st.decision.as_ref().and_then(|d| {
            d.layers.iter().max_by(|a, b| (a.attn + a.ffn).abs().partial_cmp(&(b.attn + b.ffn).abs()).unwrap_or(std::cmp::Ordering::Equal)).map(|l| format!("L{} {:+.2}", l.layer, l.attn + l.ffn))
        });
        t.add_row(vec![
            format!("{}{}", st.index, if st.forced { " (forced)" } else { "" }),
            format!("\"{}\"", esc(&st.token.text)),
            format!("{:.3}", st.token.prob),
            format!("{:.2}", st.entropy),
            st.alternatives.first().map(|a| format!("\"{}\" {:.3}", esc(&a.text), a.prob)).unwrap_or_default(),
            top.unwrap_or_default(),
        ]);
    }
    let _ = writeln!(s, "{t}");
    if explain {
        for st in &r.steps {
            if let Some(d) = &st.decision {
                let _ = writeln!(s, "\n-- step {} --", st.index);
                s.push_str(&decision(d));
            }
        }
    }
    for w in &r.warnings {
        let _ = writeln!(s, "note: {w}");
    }
    s
}

pub fn edits_list(edits: &[neurondump_core::edits::Edit], c: Option<&neurondump_core::edits::Compiled>) -> String {
    let mut s = String::new();
    if edits.is_empty() {
        let _ = writeln!(s, "edit stack is empty (the original model)");
    }
    for e in edits {
        let _ = writeln!(s, "  #{:<3} [{}] {}", e.id, if e.enabled { "on " } else { "off" }, e.label);
    }
    if let Some(c) = c {
        if c.active > 0 {
            let _ = writeln!(s, "{} active edit(s); changed nodes: {}", c.active, if c.touched.is_empty() { "-".into() } else { c.touched.join(", ") });
        }
        for n in &c.notes {
            let _ = writeln!(s, "note: {n}");
        }
    }
    s
}

pub fn edit_view(v: &neurondump_core::app::EditView) -> String {
    let mut s = edits_list(&v.edits, v.compiled.as_ref());
    let _ = writeln!(s, "undo: {}  redo: {}  snapshots: {}", if v.can_undo { "yes" } else { "no" }, if v.can_redo { "yes" } else { "no" }, if v.snapshots.is_empty() { "-".into() } else { v.snapshots.join(", ") });
    if let Some(e) = &v.error {
        let _ = writeln!(s, "error: {e}");
    }
    s
}

pub fn compare_decisions(r: &neurondump_core::trace::SimReport) -> String {
    let mut s = String::new();
    for p in &r.prompts {
        if let Some(e) = &p.edited_prompt {
            let _ = writeln!(s, "\nedited prompt text: \"{}\"", esc(e));
        }
        if let (Some(b), Some(a)) = (&p.decision_baseline, &p.decision_edited) {
            let _ = writeln!(s, "\ndecision breakdown, baseline → edited (\"{}\")", esc(&p.prompt));
            let mut t = table(&["LAYER", "BASELINE PUSH", "EDITED PUSH", "CHANGE"]);
            for (lb, la) in b.layers.iter().zip(&a.layers) {
                let (x, y) = (lb.attn + lb.ffn, la.attn + la.ffn);
                if (y - x).abs() < 1e-3 {
                    continue;
                }
                t.add_row(vec![lb.layer.to_string(), format!("{x:+.3}"), format!("{y:+.3}"), format!("{:+.3}", y - x)]);
            }
            let _ = writeln!(
                s,
                "baseline \"{}\" vs \"{}\" (diff {:.3}); edited \"{}\" vs \"{}\" (diff {:.3})",
                esc(&b.chosen.text),
                esc(&b.runner_up.text),
                b.logit_diff,
                esc(&a.chosen.text),
                esc(&a.runner_up.text),
                a.logit_diff
            );
            let _ = writeln!(s, "{t}");
        }
    }
    s
}

pub fn bake(r: &neurondump_core::edits::BakeReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "baked {} -> {}", r.model, r.output.display());
    let _ = writeln!(s, "size {} -> {}", human_bytes(r.original_size), human_bytes(r.new_size));
    let _ = writeln!(s, "applied: {}", if r.applied.is_empty() { "-".into() } else { r.applied.join("; ") });
    for (e, why) in &r.skipped {
        let _ = writeln!(s, "skipped: {e} ({why})");
    }
    let _ = writeln!(s, "tensors changed: {}{}", r.tensors_changed.len(), if r.requantized.is_empty() { String::new() } else { format!(" ({} re-encoded as Q8_0)", r.requantized.len()) });
    let _ = writeln!(
        s,
        "validate: {}",
        if r.validation.ok { format!("OK ({} tensors)", r.validation.n_tensors) } else { r.validation.problems.join("; ") }
    );
    let _ = writeln!(s, "Modelfile: {} (ollama create <name> -f {})", r.modelfile.display(), r.modelfile.display());
    for n in &r.notes {
        let _ = writeln!(s, "note: {n}");
    }
    s
}
