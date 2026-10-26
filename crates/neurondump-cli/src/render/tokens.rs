//! Token inspector: tokenization, decision breakdown, token journey, step-by-step generation.

use super::*;

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
