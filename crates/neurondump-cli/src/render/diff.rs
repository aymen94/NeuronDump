//! Model diff report: weights per layer / block / expert / tensor, metadata, vocab, template.

use super::*;

pub(super) fn small(x: f64) -> String {
    if x == 0.0 {
        "0".into()
    } else if x.abs() < 1e-3 {
        format!("{x:.2e}")
    } else {
        format!("{x:.4}")
    }
}

pub(super) fn list_names(s: &mut String, title: &str, names: &[String]) {
    if names.is_empty() {
        return;
    }
    let shown: Vec<&str> = names.iter().take(10).map(String::as_str).collect();
    let more = if names.len() > 10 { format!(" … +{}", names.len() - 10) } else { String::new() };
    let _ = writeln!(s, "{title} ({}): {}{more}", names.len(), shown.join(", "));
}

pub fn diff(r: &DiffReport, top: usize) -> String {
    let mut s = String::new();
    for (tag, m) in [("A", &r.a), ("B", &r.b)] {
        let removed = if m.removed_layers.is_empty() { String::new() } else { format!(", slimmed: layers {:?} removed", m.removed_layers) };
        let _ = writeln!(s, "{tag}  {}  ({}, {}, {}, {} tensors{removed})", m.name, m.architecture, opt(m.quantization.as_ref()), human_bytes(m.file_size), m.tensors);
    }
    if !r.a.removed_layers.is_empty() || !r.b.removed_layers.is_empty() {
        let _ = writeln!(s, "layers realigned to the original numbering before matching");
    }
    match &r.total {
        Some(t) => {
            let _ = writeln!(
                s,
                "\nweights: {} tensors compared ({} identical): relative L2 {}, cosine {:.6}, max |diff| {}",
                r.compared,
                r.identical,
                small(t.rel_l2),
                t.cosine,
                small(t.max_abs as f64)
            );
        }
        None => {
            let _ = writeln!(s, "\nweights: not compared");
        }
    }
    list_names(&mut s, "only in A", &r.only_in_a);
    list_names(&mut s, "only in B", &r.only_in_b);
    let mism: Vec<String> = r.shape_mismatch.iter().map(|m| format!("{} {:?}→{:?}", m.name, m.dims_a, m.dims_b)).collect();
    list_names(&mut s, "shape mismatch", &mism);
    let skipped: Vec<String> = r.skipped.iter().map(|(n, why)| format!("{n} ({why})")).collect();
    list_names(&mut s, "skipped", &skipped);

    // Metadata, vocabulary, template.
    if r.metadata.is_empty() {
        let _ = writeln!(s, "\nmetadata: identical");
    } else {
        let mut t = table(&["METADATA KEY", "A", "B"]);
        for d in r.metadata.iter().take(40) {
            t.add_row(vec![d.key.clone(), opt(d.a.as_ref()), opt(d.b.as_ref())]);
        }
        let more = if r.metadata.len() > 40 { format!("… +{} keys (see --json)\n", r.metadata.len() - 40) } else { String::new() };
        let _ = write!(s, "\n{t}\n{more}");
    }
    let v = &r.vocab;
    let _ = writeln!(s, "\nvocab: {} → {} tokens, {} changed", v.size_a, v.size_b, v.changed);
    for (id, a, b) in v.examples.iter().take(10) {
        let f = |x: &Option<String>| x.as_deref().map(|t| format!("\"{}\"", esc(t))).unwrap_or_else(|| "∅".into());
        let _ = writeln!(s, "  {id:>7}  {} → {}", f(a), f(b));
    }
    let c = &r.chat_template;
    if c.changed {
        let _ = writeln!(s, "chat template: changed ({} → {} chars)", c.chars_a, c.chars_b);
        if let Some((line, a, b)) = &c.first_difference {
            let _ = writeln!(s, "  first difference, line {line}:\n  A: {}\n  B: {}", esc(a), esc(b));
        }
    } else {
        let _ = writeln!(s, "chat template: identical");
    }

    // Per layer, with the block breakdown.
    let layers: Vec<_> = r.groups.iter().filter(|g| g.kind == "layer").collect();
    if !layers.is_empty() {
        let block = |l: Option<u32>, names: &[&str]| {
            r.groups
                .iter()
                .find(|g| g.kind == "block" && g.layer == l && g.block.as_deref().is_some_and(|b| names.contains(&b)))
                .map(|g| small(g.metrics.rel_l2))
                .unwrap_or_else(|| "-".into())
        };
        let mut t = table(&["LAYER", "REL L2", "COSINE", "MAX |DIFF|", "ATTN/SSM", "FFN", "MOE"]);
        for g in &layers {
            t.add_row(vec![
                g.label.clone(),
                small(g.metrics.rel_l2),
                format!("{:.6}", g.metrics.cosine),
                small(g.metrics.max_abs as f64),
                block(g.layer, &["attention", "ssm"]),
                block(g.layer, &["ffn"]),
                block(g.layer, &["moe"]),
            ]);
        }
        let _ = writeln!(s, "\nper layer (relative L2 per block):\n{t}");
    }
    let mut experts: Vec<_> = r.groups.iter().filter(|g| g.kind == "expert").collect();
    if !experts.is_empty() {
        experts.sort_by(|a, b| b.metrics.rel_l2.total_cmp(&a.metrics.rel_l2));
        let mut t = table(&["EXPERT", "REL L2", "COSINE", "MAX |DIFF|"]);
        for g in experts.iter().take(top.min(20)) {
            t.add_row(vec![g.id.clone(), small(g.metrics.rel_l2), format!("{:.6}", g.metrics.cosine), small(g.metrics.max_abs as f64)]);
        }
        let _ = writeln!(s, "\nmost changed experts ({} total):\n{t}", experts.len());
    }
    if !r.tensors.is_empty() {
        let mut t = table(&["TENSOR", "TYPES", "REL L2", "COSINE", "MAX |DIFF|"]);
        for d in r.tensors.iter().take(top) {
            let types = if d.type_a == d.type_b { d.type_a.clone() } else { format!("{} → {}", d.type_a, d.type_b) };
            t.add_row(vec![d.name.clone(), types, small(d.metrics.rel_l2), format!("{:.6}", d.metrics.cosine), small(d.metrics.max_abs as f64)]);
        }
        let _ = writeln!(s, "\ntop {} most changed tensors:\n{t}", top.min(r.tensors.len()));
    }
    let _ = writeln!(s, "({:.1} s)", r.elapsed_ms as f64 / 1000.0);
    s
}
