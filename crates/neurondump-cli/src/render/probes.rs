//! Renderings of the probes: causal tracing, head roles, layer similarity, weight spectrum,
//! neuron view, glitch tokens and the memory estimate.

use super::*;
use neurondump_core::probe::{GlitchReport, MemoryEstimate, NeuronReport, SpectrumReport, TokenScore};
use neurondump_core::trace::{CausalReport, HeadRolesReport, SimilarityReport};

/// A one-character shade for 0..1.
fn shade(v: f32) -> char {
    const S: [char; 5] = [' ', '░', '▒', '▓', '█'];
    S[((v.clamp(0.0, 1.0) * 4.0).round()) as usize]
}

fn warnings(s: &mut String, w: &[String]) {
    for w in w {
        let _ = writeln!(s, "warning: {w}");
    }
}

pub fn causal(r: &CausalReport) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Causal tracing of {} ({}, {} ms): target {:?} — clean {:.3}, corrupted {:.3}",
        r.model,
        r.component.as_str(),
        r.elapsed_ms,
        r.target.text,
        r.clean_prob,
        r.corrupt_prob
    );
    let _ = writeln!(s, "corrupted top: {}", r.corrupt_top.iter().map(|t| format!("{:?} {:.3}", t.text, t.prob)).collect::<Vec<_>>().join("  "));
    let _ = writeln!(s, "\nshare of the clean answer restored per layer (rows) and corrupted token (columns):");
    let w = r.corrupt_tokens.len();
    let _ = writeln!(s, "        {}", (0..w).map(|i| format!("{:>2}", i % 100)).collect::<String>());
    for (l, row) in r.effect.iter().enumerate() {
        let cells: String = row.iter().map(|e| e.map_or("  ".to_string(), |e| format!(" {}", shade(e)))).collect();
        let _ = writeln!(s, "  L{l:<4}{cells}");
    }
    let _ = writeln!(s, "\ntokens: {}", r.corrupt_tokens.iter().enumerate().map(|(i, t)| format!("{i}:{:?}", t.text)).collect::<Vec<_>>().join(" "));
    let mut t = table(&["LAYER", "POSITION", "TOKEN", "EFFECT"]);
    for p in &r.peaks {
        t.add_row(vec![p.layer.to_string(), p.position.to_string(), r.corrupt_tokens.get(p.position).map(|t| format!("{:?}", t.text)).unwrap_or_default(), format!("{:.3}", p.effect)]);
    }
    let _ = writeln!(s, "\nstrongest cells:\n{t}");
    warnings(&mut s, &r.warnings);
    s
}

pub fn head_roles(r: &HeadRolesReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Head roles of {} — {} random sequence(s) of {} tokens repeated twice ({} ms)", r.model, r.samples, r.length, r.elapsed_ms);
    let _ = writeln!(s, "{}", r.counts.iter().map(|(k, n)| format!("{k}: {n}")).collect::<Vec<_>>().join(" · "));
    let mut t = table(&["LAYER", "HEAD", "ROLE", "INDUCTION", "PREV TOKEN", "DUPLICATE", "SINK", "SELF"]);
    let mut rows: Vec<_> = r.heads.iter().filter(|h| h.role != "mixed" && h.role != "sink").collect();
    rows.sort_by(|a, b| b.induction.max(b.prev_token).max(b.duplicate).total_cmp(&a.induction.max(a.prev_token).max(a.duplicate)));
    for h in rows.iter().take(40) {
        t.add_row(vec![
            h.layer.to_string(),
            h.head.to_string(),
            h.role.clone(),
            format!("{:.2}", h.induction),
            format!("{:.2}", h.prev_token),
            format!("{:.2}", h.duplicate),
            format!("{:.2}", h.sink),
            format!("{:.2}", h.self_attn),
        ]);
    }
    let _ = writeln!(s, "{t}");
    warnings(&mut s, &r.warnings);
    s
}

pub fn similarity(r: &SimilarityReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Layer similarity of {} — {} sample(s), {} positions, linear CKA ({} ms)", r.model, r.samples, r.positions, r.elapsed_ms);
    let _ = writeln!(s, "matrix (embeddings, then each layer; darker = more similar):");
    for (i, row) in r.cka.iter().enumerate() {
        let _ = writeln!(s, "  {:>4} {}", if i == 0 { "emb".to_string() } else { format!("L{}", i - 1) }, row.iter().map(|v| shade(*v)).collect::<String>());
    }
    let mut t = table(&["LAYER", "CKA IN→OUT", "COSINE IN→OUT", "UPDATE / INPUT"]);
    for c in &r.layers {
        t.add_row(vec![c.layer.to_string(), format!("{:.4}", c.cka_in_out), format!("{:.4}", c.cos_in_out), format!("{:.3}", c.rel_update)]);
    }
    let _ = writeln!(s, "{t}");
    if r.redundant.is_empty() {
        let _ = writeln!(s, "no layer leaves the representation almost unchanged (CKA ≥ 0.99)");
    } else {
        let _ = writeln!(s, "most redundant (CKA in→out ≥ 0.99): {}", r.redundant.iter().map(|l| format!("layer {l}")).collect::<Vec<_>>().join(", "));
    }
    warnings(&mut s, &r.warnings);
    s
}

pub fn spectrum(r: &SpectrumReport, limit: usize) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Weight spectrum of {} — {} matrices ({} ms)", r.model, r.tensors.len(), r.elapsed_ms);
    let mut t = table(&["TENSOR", "SHAPE", "QUANT", "σ MAX", "STABLE RANK", "RANK %", "OUTLIER", "KURTOSIS"]);
    let mut rows: Vec<_> = r.tensors.iter().collect();
    rows.sort_by(|a, b| a.rank_ratio.total_cmp(&b.rank_ratio));
    for x in rows.iter().take(limit) {
        t.add_row(vec![
            x.name.clone(),
            format!("{}×{}", x.rows, x.cols),
            x.quant.clone(),
            format!("{:.3}", x.sigma_max),
            format!("{:.1}", x.stable_rank),
            format!("{:.2}%", x.rank_ratio * 100.0),
            format!("{:.1}", x.outlier_ratio),
            format!("{:.1}", x.kurtosis),
        ]);
    }
    let _ = writeln!(s, "lowest stable rank first (the most compressible):\n{t}");
    let _ = writeln!(s, "most outliers: {}", r.most_outliers.join(", "));
    if !r.skipped.is_empty() {
        let _ = writeln!(s, "skipped: {}", r.skipped.len());
    }
    s
}

fn scores(v: &[TokenScore]) -> String {
    v.iter().map(|t| format!("{:?} {:+.3}", t.text, t.score)).collect::<Vec<_>>().join("  ")
}

pub fn neuron(r: &NeuronReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Neuron {} of layer {} ({} neurons) in {}", r.neuron, r.layer, r.n_ff, r.model);
    let _ = writeln!(s, "input row ({}) norm {:.3} · output column norm {:.3}", r.read_from, r.read_norm, r.write_norm);
    let _ = writeln!(s, "\nreads (embeddings along its input row):\n  {}", scores(&r.reads));
    let _ = writeln!(s, "\npromotes (through the unembedding):\n  {}", scores(&r.promotes));
    let _ = writeln!(s, "\nsuppresses:\n  {}", scores(&r.suppresses));
    warnings(&mut s, &r.warnings);
    s
}

pub fn glitch(r: &GlitchReport, limit: usize) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Under-trained token candidates in {} — {} of {} tokens checked, {} flagged", r.model, r.checked, r.vocab_size, r.flagged);
    let _ = writeln!(s, "token_embd row norms: mean {:.3}, std {:.3}, median {:.3}", r.embed.mean, r.embed.std, r.embed.median);
    if let Some(u) = &r.unembed {
        let _ = writeln!(s, "output row norms: mean {:.3}, std {:.3}, median {:.3}", u.mean, u.std, u.median);
    }
    let mut t = table(&["ID", "TEXT", "TYPE", "EMBED NORM", "EMBED Z", "OUTPUT Z", "REACHABLE"]);
    for c in r.candidates.iter().take(limit) {
        t.add_row(vec![
            c.id.to_string(),
            format!("{:?}", c.text),
            c.token_type.clone(),
            format!("{:.3}", c.embed_norm),
            format!("{:.2}", c.embed_z),
            c.unembed_z.map(|z| format!("{z:.2}")).unwrap_or_else(|| "-".into()),
            if c.reachable { "yes".into() } else { "no".into() },
        ]);
    }
    let _ = writeln!(s, "{t}");
    warnings(&mut s, &r.warnings);
    s
}

pub fn memory(r: &MemoryEstimate) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Memory of {} — weights {}, {} of {} layers cache K/V, {} values per token",
        r.model,
        human_bytes(r.weights_bytes),
        r.kv_layers,
        r.layers,
        human_count(r.values_per_token)
    );
    let mut headers = vec!["CONTEXT".to_string()];
    headers.extend(r.kv_types.iter().map(|k| format!("KV {k}")));
    headers.extend(r.kv_types.iter().map(|k| format!("TOTAL ({k})")));
    let hs: Vec<&str> = headers.iter().map(String::as_str).collect();
    let mut t = table(&hs);
    for row in &r.rows {
        let mut cells = vec![format!("{}{}", row.context, if Some(row.context) == r.train_context { " (trained)" } else { "" })];
        cells.extend(row.kv_bytes.iter().map(|b| human_bytes(*b)));
        cells.extend(row.total_bytes.iter().map(|b| human_bytes(*b)));
        t.add_row(cells);
    }
    let _ = writeln!(s, "{t}");
    for n in &r.notes {
        let _ = writeln!(s, "note: {n}");
    }
    s
}
