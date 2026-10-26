//! `bake`: write the edits into a new GGUF (scaled / zeroed / dropped tensors) plus an Ollama Modelfile.

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BakeReport {
    pub schema_version: u32,
    pub model: String,
    pub output: PathBuf,
    pub modelfile: PathBuf,
    pub original_size: u64,
    pub new_size: u64,
    /// Edits written into the weights or the Modelfile.
    pub applied: Vec<String>,
    /// (edit, reason) for edits that cannot be baked.
    pub skipped: Vec<(String, String)>,
    pub tensors_changed: Vec<String>,
    /// Tensors re-encoded as Q8_0 (quantized types without an exact in-place scale).
    pub requantized: Vec<String>,
    pub notes: Vec<String>,
    pub validation: Validation,
}

/// Weight change on a dequantized tensor (element ranges are half-open).
#[derive(Debug, Clone)]
pub(super) enum WOp {
    Zero(ZeroOp),
    /// Scale everything.
    All(f32),
    /// Columns (inside every row) [a, b).
    Cols(u64, u64, f32),
    /// Slices along dims[2] [a, b).
    Slices(u64, u64, f32),
    /// Row r := row `from` of the original tensor, then times the factor.
    Row { row: u64, from: Option<u64>, factor: f32 },
}

pub(super) fn encode_f32(ty: GgmlType, f: &[f32], row_len: u64) -> (GgmlType, Vec<u8>) {
    match ty {
        GgmlType::F32 => (ty, f.iter().flat_map(|x| x.to_le_bytes()).collect()),
        GgmlType::F16 => (ty, f.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()),
        GgmlType::BF16 => (ty, f.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect()),
        _ if row_len % 32 == 0 => (GgmlType::Q8_0, quantize_q8_0(f)),
        _ => (GgmlType::F16, f.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect()),
    }
}

/// Writes the enabled edits into a new GGUF (and a Modelfile next to it).
pub fn bake(m: &ModelHandle, graph: &ModelGraph, vocab: Option<&Vocab>, edits: &[Edit], out: &Path, p: &dyn Progress) -> Result<BakeReport> {
    if out.canonicalize().ok().is_some_and(|o| m.entry.path.canonicalize().ok() == Some(o)) {
        bail!("refusing to bake into the original model file; bake always writes a new GGUF");
    }
    let g = &m.gguf;
    let c = compile(graph, vocab, edits)?;
    let a = &c.intervention;
    let mut applied = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut notes = c.notes.clone();
    for e in edits.iter().filter(|e| e.enabled) {
        let why = match &e.op {
            EditOp::Ban { .. } | EditOp::Boost { .. } => Some("logit edits are applied at sampling time; a GGUF has no logit bias"),
            EditOp::Patch { .. } => Some("activation patching depends on a second prompt at run time"),
            EditOp::Steer { .. } => Some("steering adds a run-time vector (no bias tensor to hold it in most architectures)"),
            EditOp::Force { .. } => Some("forced tokens only exist during one generation"),
            _ => None,
        };
        match why {
            Some(w) => skipped.push((e.label.clone(), w.into())),
            None => applied.push(e.label.clone()),
        }
    }

    // --- weight operations per tensor ------------------------------------------------------
    let mut ops: BTreeMap<String, Vec<WOp>> = BTreeMap::new();
    let mut add = |name: String, op: WOp| {
        if g.tensor(&name).is_some() {
            ops.entry(name).or_default().push(op);
        }
    };
    // Skipped layers = identity: zero every block output of the layer.
    let mut zero_part = a.clone();
    for l in &a.skip_layers {
        for b in ["attention", "ssm", "ffn", "moe"] {
            zero_part.zero_blocks.push((*l, b.to_string()));
        }
    }
    zero_part.skip_layers.clear();
    for o in zero_ops(g, graph, &zero_part) {
        add(o.tensor.clone(), WOp::Zero(o));
    }
    let block_short = |b: &str| match b {
        "attention" => "attn",
        other => other,
    }
    .to_string();
    let post_norm = |l: u32, b: &str| -> Option<String> {
        let n = match b {
            "attention" | "ssm" => format!("blk.{l}.post_attention_norm.weight"),
            _ => format!("blk.{l}.post_ffw_norm.weight"),
        };
        g.tensor(&n).map(|_| n)
    };
    let scale_block = |l: u32, b: &str, f: f32, add: &mut dyn FnMut(String, WOp)| {
        // With a post-norm the norm's output is what reaches the residual stream.
        if let Some(n) = post_norm(l, b) {
            add(n, WOp::All(f));
            return;
        }
        for name in block_out_tensors(graph, &format!("layer.{l}.{}", block_short(b))) {
            add(name.trim_end_matches(".weight").to_string() + ".bias", WOp::All(f));
            add(name, WOp::All(f));
        }
    };
    for (l, f) in &a.scale_layers {
        for b in ["attention", "ssm", "ffn", "moe"] {
            if graph.nodes.contains_key(&format!("layer.{l}.{}", block_short(b))) {
                scale_block(*l, b, *f, &mut add);
            }
        }
    }
    for (l, b, f) in &a.scale_blocks {
        scale_block(*l, b, *f, &mut add);
    }
    for (l, h, f) in &a.scale_heads {
        let hd = graph.dims.get(l).map_or(0, |d| d.head_dim);
        if post_norm(*l, "attention").is_some() {
            notes.push(format!("layer {l} has a post-attention norm: a baked head scale is renormalized, so it is weaker than at run time"));
        }
        add(format!("blk.{l}.attn_output.weight"), WOp::Cols(*h as u64 * hd, (*h as u64 + 1) * hd, *f));
    }
    for (l, s, e, f) in &a.scale_neurons {
        add(format!("blk.{l}.ffn_down.weight"), WOp::Cols(*s as u64, *e as u64 + 1, *f));
    }
    for (l, e, f) in &a.scale_experts {
        add(format!("blk.{l}.ffn_down_exps.weight"), WOp::Slices(*e as u64, *e as u64 + 1, *f));
    }
    for ed in &a.embed_edits {
        add("token_embd.weight".into(), WOp::Row { row: ed.token as u64, from: ed.replace_with.map(|r| r as u64), factor: ed.scale.unwrap_or(1.0) });
    }
    if !a.embed_edits.is_empty() && g.tensor("output.weight").is_none() {
        notes.push("embeddings are tied: the edited token_embd rows also change the output head (at run time only the input side changes)".into());
    }
    if !a.mask_tokens.is_empty() {
        notes.push("banned tokens are not baked: use `slim` with a vocabulary cut to remove tokens".into());
    }

    // --- write ------------------------------------------------------------------------------
    let mut w = GgufWriter::new();
    w.metadata = g.metadata.clone();
    w.metadata.insert("general.edited_by".into(), Value::String(format!("neurondump {}", env!("CARGO_PKG_VERSION"))));
    w.metadata.insert("neurondump.edits".into(), Value::String(serde_json::to_string(&edits.iter().filter(|e| e.enabled).collect::<Vec<_>>())?));
    if let Some(t) = &c.template {
        // Jinja templates in the file would disagree with the new (Go) template.
        w.metadata.shift_remove("tokenizer.chat_template");
        notes.push("tokenizer.chat_template removed: the Modelfile carries the edited template".into());
        let _ = t;
    }
    let mut changed = Vec::new();
    let mut requantized = Vec::new();
    let total = g.tensors.len().max(1);
    for (i, t) in g.tensors.iter().enumerate() {
        check_cancel(p)?;
        p.set(5.0 + 45.0 * i as f32 / total as f32, &format!("preparing {}", t.name));
        let src = g.info_bytes(t)?;
        let Some(list) = ops.get(&t.name) else {
            w.tensors.push(WriterTensor { name: t.name.clone(), dims: t.dims.clone(), ggml_type: t.ggml_type, data: Cow::Borrowed(src) });
            continue;
        };
        changed.push(t.name.clone());
        let mut ty = t.ggml_type;
        let data: Vec<u8> = if list.iter().all(|o| matches!(o, WOp::Zero(_))) {
            let mut buf = src.to_vec();
            for o in list {
                if let WOp::Zero(z) = o {
                    if let Some(nt) = apply_zero(z, &t.dims, ty, &mut buf)? {
                        ty = nt;
                        requantized.push(t.name.clone());
                    }
                }
            }
            buf
        } else {
            let row_len = t.dims.first().copied().unwrap_or(1).max(1);
            let n_rows = t.dims.iter().skip(1).product::<u64>().max(1);
            let mut f = vec![0f32; (row_len * n_rows) as usize];
            dequantize_into(t.ggml_type, src, &mut f).with_context(|| format!("dequantizing {}", t.name))?;
            let orig = if list.iter().any(|o| matches!(o, WOp::Row { from: Some(_), .. })) { Some(f.clone()) } else { None };
            let rl = row_len as usize;
            for o in list {
                match o {
                    WOp::Zero(z) => {
                        if z.all {
                            f.fill(0.0);
                        }
                        for (s, e) in &z.rows {
                            let (s, e) = ((*s).min(n_rows) as usize, (*e).min(n_rows) as usize);
                            f[s * rl..e * rl].fill(0.0);
                        }
                        for (s, e) in &z.cols {
                            for r in 0..n_rows as usize {
                                f[r * rl + (*s).min(row_len) as usize..r * rl + (*e).min(row_len) as usize].fill(0.0);
                            }
                        }
                        if let Some(&ns) = t.dims.get(2) {
                            let per = f.len() / ns.max(1) as usize;
                            for (s, e) in &z.slices {
                                f[(*s).min(ns) as usize * per..(*e).min(ns) as usize * per].fill(0.0);
                            }
                        }
                    }
                    WOp::All(k) => f.iter_mut().for_each(|x| *x *= k),
                    WOp::Cols(s, e, k) => {
                        for r in 0..n_rows as usize {
                            for x in &mut f[r * rl + (*s).min(row_len) as usize..r * rl + (*e).min(row_len) as usize] {
                                *x *= k;
                            }
                        }
                    }
                    WOp::Slices(s, e, k) => {
                        if let Some(&ns) = t.dims.get(2) {
                            let per = f.len() / ns.max(1) as usize;
                            for x in &mut f[(*s).min(ns) as usize * per..(*e).min(ns) as usize * per] {
                                *x *= k;
                            }
                        }
                    }
                    WOp::Row { row, from, factor } => {
                        if *row >= n_rows {
                            continue;
                        }
                        let r = *row as usize;
                        if let (Some(fr), Some(o)) = (from, &orig) {
                            let fr = (*fr).min(n_rows - 1) as usize;
                            f[r * rl..(r + 1) * rl].copy_from_slice(&o[fr * rl..(fr + 1) * rl]);
                        }
                        for x in &mut f[r * rl..(r + 1) * rl] {
                            *x *= factor;
                        }
                    }
                }
            }
            let (nt, bytes) = encode_f32(t.ggml_type, &f, row_len);
            if nt != t.ggml_type {
                requantized.push(t.name.clone());
            }
            ty = nt;
            bytes
        };
        w.tensors.push(WriterTensor { name: t.name.clone(), dims: t.dims.clone(), ggml_type: ty, data: Cow::Owned(data) });
    }
    if let Some(d) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
    }
    p.set(50.0, "writing");
    let new_size = w
        .write_to(out, |done, total| p.set(50.0 + 45.0 * done as f32 / total.max(1) as f32, "writing"))
        .with_context(|| format!("writing {}", out.display()))?;
    drop(w);
    p.set(96.0, "validating");
    let validation = validate(out);
    let name = out.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "baked.gguf".into());
    let modelfile = out.with_file_name(format!("{}.Modelfile", name.trim_end_matches(".gguf")));
    std::fs::write(&modelfile, modelfile_text(m, &name, &c)).with_context(|| format!("writing {}", modelfile.display()))?;
    if c.sampling.is_some() || c.system.is_some() || c.template.is_some() {
        notes.push("sampling, system prompt and template edits are written to the Modelfile".into());
    }
    p.set(100.0, "done");
    Ok(BakeReport {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        output: out.to_path_buf(),
        modelfile,
        original_size: g.file_size,
        new_size,
        applied,
        skipped,
        tensors_changed: changed,
        requantized,
        notes,
        validation,
    })
}

/// Modelfile for a baked model, with the edited template / system prompt / sampling.
pub(super) fn modelfile_text(m: &ModelHandle, gguf_name: &str, c: &Compiled) -> String {
    let mut s = format!("# Generated by NeuronDump (bake) from {}\nFROM ./{gguf_name}\n", m.entry.name);
    if let Some(p) = &m.entry.projector {
        s.push_str(&format!("FROM {}\n", p.display()));
    }
    if let Some(t) = c.template.as_ref().or(m.ollama.template.as_ref()) {
        s.push_str(&format!("TEMPLATE \"\"\"{t}\"\"\"\n"));
    }
    match &c.system {
        Some(sys) if !sys.is_empty() => s.push_str(&format!("SYSTEM \"\"\"{sys}\"\"\"\n")),
        Some(_) => s.push_str("SYSTEM \"\"\"\"\"\"\n"),
        None => {
            if let Some(sys) = &m.ollama.system {
                s.push_str(&format!("SYSTEM \"\"\"{sys}\"\"\"\n"));
            }
        }
    }
    let mut overridden: Vec<&str> = Vec::new();
    if let Some(sp) = &c.sampling {
        s.push_str(&format!("PARAMETER temperature {}\n", sp.temperature));
        overridden.push("temperature");
        if let Some(k) = sp.top_k {
            s.push_str(&format!("PARAMETER top_k {k}\n"));
            overridden.push("top_k");
        }
        if let Some(pp) = sp.top_p {
            s.push_str(&format!("PARAMETER top_p {pp}\n"));
            overridden.push("top_p");
        }
        if let Some(seed) = sp.seed {
            s.push_str(&format!("PARAMETER seed {seed}\n"));
            overridden.push("seed");
        }
    }
    if let Some(serde_json::Value::Object(params)) = &m.ollama.params {
        for (k, v) in params.iter().filter(|(k, _)| !overridden.contains(&k.as_str())) {
            let vals: Vec<&serde_json::Value> = match v {
                serde_json::Value::Array(a) => a.iter().collect(),
                v => vec![v],
            };
            for v in vals {
                match v {
                    serde_json::Value::String(x) => s.push_str(&format!("PARAMETER {k} {}\n", serde_json::to_string(x).unwrap_or_default())),
                    other => s.push_str(&format!("PARAMETER {k} {other}\n")),
                }
            }
        }
    }
    for lic in &m.ollama.licenses {
        s.push_str(&format!("LICENSE \"\"\"{lic}\"\"\"\n"));
    }
    s
}
