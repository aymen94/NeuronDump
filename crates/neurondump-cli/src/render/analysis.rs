//! Weight stats, prune plans, slim and trace reports, model comparison, search results, jobs.

use super::*;

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
