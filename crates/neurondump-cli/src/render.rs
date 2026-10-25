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
    let _ = writeln!(s, "Size      {} -> {} ({pct:.1}% smaller)", human_bytes(r.original_size), human_bytes(r.new_size));
    let _ = writeln!(s, "Applied   {}", if r.applied.is_empty() { "-".into() } else { r.applied.join(", ") });
    for (id, why) in &r.refused {
        let _ = writeln!(s, "Refused   {id}: {why}");
    }
    let _ = writeln!(s, "Removed   {} tokens, layers {:?}, requantized {} tensors", r.removed_tokens, r.removed_layers, r.requantized.len());
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
