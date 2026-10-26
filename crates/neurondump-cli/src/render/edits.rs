//! Live edits: the edit stack, compiled view, baseline vs edited decisions, bake result.

use super::*;

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
