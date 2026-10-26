//! Model listing and static inspection: list, inspect, metadata, vocab, tensors, layers, template tree.

use super::*;

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

pub(super) fn compress_ranges(v: &[u32]) -> String {
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

pub(super) fn tree(n: &TemplateNode, prefix: &str, last: bool, out: &mut String, root: bool) {
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
