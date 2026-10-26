//! Graph node details, static impact, simulated ablation and sweep rankings.

use super::*;

pub(super) fn size_cell(params: u64, bytes: u64) -> String {
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

pub(super) fn toks(v: &[neurondump_core::trace::LensToken], n: usize) -> String {
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
