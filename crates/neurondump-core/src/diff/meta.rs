//! Non-weight differences: model sides, metadata keys, vocabulary tokens, chat template.

use super::*;

pub(super) fn short_value(v: &neurondump_gguf::Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 120 {
        format!("{}… ({} chars)", s.chars().take(100).collect::<String>(), s.chars().count())
    } else {
        s
    }
}

pub(super) fn side(m: &ModelHandle) -> DiffSide {
    DiffSide {
        name: m.entry.name.clone(),
        path: m.entry.path.to_string_lossy().to_string(),
        architecture: m.architecture().to_string(),
        quantization: quantization_label(&m.gguf),
        tensors: m.gguf.tensors.len(),
        file_size: m.gguf.file_size,
        removed_layers: removed_layers(&m.gguf),
    }
}

pub(super) fn metadata_diff(a: &GgufFile, b: &GgufFile) -> Vec<ValueDiff> {
    let skip = |k: &str| {
        k == "tokenizer.chat_template"
            || matches!(k, "tokenizer.ggml.tokens" | "tokenizer.ggml.scores" | "tokenizer.ggml.token_type" | "tokenizer.ggml.merges")
    };
    let keys: BTreeSet<&String> = a.metadata.keys().chain(b.metadata.keys()).collect();
    keys.into_iter()
        .filter(|k| !skip(k))
        .filter_map(|k| {
            let (va, vb) = (a.get(k), b.get(k));
            if va == vb {
                return None;
            }
            Some(ValueDiff { key: k.clone(), a: va.map(short_value), b: vb.map(short_value) })
        })
        .collect()
}

pub(super) fn vocab_diff(a: &GgufFile, b: &GgufFile) -> VocabDiff {
    let toks = |g: &GgufFile| -> Vec<String> {
        g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).and_then(|a| a.as_strings()).map(<[String]>::to_vec).unwrap_or_default()
    };
    let (ta, tb) = (toks(a), toks(b));
    let mut d = VocabDiff { size_a: ta.len(), size_b: tb.len(), ..Default::default() };
    for i in 0..ta.len().max(tb.len()) {
        let (x, y) = (ta.get(i), tb.get(i));
        if x == y {
            continue;
        }
        if x.is_some() && y.is_some() {
            d.changed += 1;
        }
        if d.examples.len() < MAX_EXAMPLES {
            d.examples.push((i, x.cloned(), y.cloned()));
        }
    }
    d
}

pub(super) fn template_of(m: &ModelHandle) -> Option<String> {
    m.gguf.get_str("tokenizer.chat_template").map(String::from).or_else(|| m.ollama.template.clone())
}

pub(super) fn template_diff(a: &ModelHandle, b: &ModelHandle) -> TemplateDiff {
    let (ta, tb) = (template_of(a).unwrap_or_default(), template_of(b).unwrap_or_default());
    let mut d = TemplateDiff { changed: ta != tb, chars_a: ta.chars().count(), chars_b: tb.chars().count(), first_difference: None };
    if d.changed {
        let (la, lb): (Vec<&str>, Vec<&str>) = (ta.lines().collect(), tb.lines().collect());
        let i = (0..la.len().max(lb.len())).find(|&i| la.get(i) != lb.get(i)).unwrap_or(0);
        d.first_difference = Some((i + 1, la.get(i).unwrap_or(&"").to_string(), lb.get(i).unwrap_or(&"").to_string()));
    }
    d
}
