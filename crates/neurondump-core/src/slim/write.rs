//! `slim`: apply the chosen plan items (vocab, layers, tensors, zeroing) and write the new GGUF.

use super::*;

pub fn slim(m: &ModelHandle, opts: &SlimOptions, output_dir: &Path, p: &dyn Progress) -> Result<SlimReport> {
    let name = safe_file_name(&opts.out)?;
    let name = if name.to_lowercase().ends_with(".gguf") { name.to_string() } else { format!("{name}.gguf") };
    if opts.plan.digest != m.entry.digest {
        bail!("plan was made for {} ({}), not {} ({})", opts.plan.model, opts.plan.digest, m.entry.name, m.entry.digest);
    }
    let dir = output_dir.join("slim");
    std::fs::create_dir_all(&dir)?;
    let out_path = dir.join(&name);
    let g = &m.gguf;
    let arch = g.architecture().unwrap_or("unknown").to_string();

    let mut applied = Vec::new();
    let mut refused = Vec::new();
    let mut remove_tokens: BTreeSet<u32> = BTreeSet::new();
    let mut keep_only: Option<BTreeSet<u32>> = None;
    let mut remove_layers: BTreeSet<u32> = BTreeSet::new();
    let mut drop_tensors: HashSet<String> = HashSet::new();
    let mut requant: HashSet<String> = HashSet::new();
    let mut strip_keys: HashSet<String> = HashSet::new();
    let mut include_projector = true;
    let mut include_template = true;
    let mut zero: BTreeMap<String, Vec<ZeroOp>> = BTreeMap::new();

    for c in chosen(&opts.plan, &opts.only) {
        if !c.applicable {
            refused.push((c.id.clone(), c.reason.clone().unwrap_or_else(|| "not applicable".into())));
            continue;
        }
        let ids = |key: &str| -> Vec<u32> {
            c.data[key].as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect()).unwrap_or_default()
        };
        let strs = |key: &str| -> Vec<String> {
            c.data[key].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default()
        };
        match c.kind {
            CandidateKind::VocabTrim => {
                let k: BTreeSet<u32> = ids("keep_ids").into_iter().collect();
                keep_only = Some(match keep_only {
                    Some(prev) => prev.intersection(&k).copied().collect(),
                    None => k,
                });
            }
            CandidateKind::ReservedTokens => remove_tokens.extend(ids("remove_ids")),
            CandidateKind::LayerRedundancy => {
                if let Some(l) = c.data["layer"].as_u64() {
                    remove_layers.insert(l as u32);
                }
            }
            CandidateKind::TiedEmbeddings => {
                drop_tensors.insert("output.weight".into());
            }
            CandidateKind::QuantHeadroom => requant.extend(strs("tensors")),
            CandidateKind::UnusedProjector => include_projector = false,
            CandidateKind::LargeMetadata => strip_keys.extend(strs("keys")),
            CandidateKind::ZeroWeights => {
                let ops: Vec<ZeroOp> = serde_json::from_value(c.data["ops"].clone()).context("zero_weights: invalid ops")?;
                for o in ops {
                    if g.tensor(&o.tensor).is_none() {
                        bail!("zero_weights: no tensor {:?} in this model", o.tensor);
                    }
                    zero.entry(o.tensor.clone()).or_default().push(o);
                }
            }
            CandidateKind::DropTensors => drop_tensors.extend(strs("tensors")),
            CandidateKind::StripTemplate => {
                include_template = false;
                strip_keys.insert("tokenizer.chat_template".into());
            }
            CandidateKind::LowNormHeads | CandidateKind::LowNormNeurons | CandidateKind::LowNormExperts => {
                refused.push((c.id.clone(), "structured pruning is not supported for any architecture yet".into()));
                continue;
            }
        }
        applied.push(c.id.clone());
    }

    // --- Vocabulary remap ------------------------------------------------------------------
    p.set(5.0, "planning");
    let mut metadata = g.metadata.clone();
    let vocab = Vocab::from_gguf(g).ok();
    let n_vocab = vocab.as_ref().map_or(0, |v| v.tokens.len());
    let mut keep_ids: Option<Vec<usize>> = None;
    if keep_only.is_some() || !remove_tokens.is_empty() {
        let Some(vocab) = &vocab else { bail!("vocabulary trimming requested but the model has no vocabulary") };
        let protected = protected_ids(g, vocab);
        let mut keep: BTreeSet<u32> = keep_only.unwrap_or_else(|| (0..n_vocab as u32).collect());
        for r in &remove_tokens {
            keep.remove(r);
        }
        // Special ids referenced by metadata are never dropped.
        for k in crate::tokenizer::special_token_keys(g) {
            if let Some(id) = g.get_u64(&k) {
                keep.insert(id as u32);
            }
        }
        if keep_only_was_corpus(&applied) {
            keep.extend(protected.iter().copied());
        }
        let keep_v: Vec<usize> = keep.iter().map(|&i| i as usize).filter(|&i| i < n_vocab).collect();
        let mut remap = vec![u32::MAX; n_vocab];
        for (new, &old) in keep_v.iter().enumerate() {
            remap[old] = new as u32;
        }
        for (k, v) in metadata.iter_mut() {
            if k == "tokenizer.ggml.tokens" || k == "tokenizer.ggml.scores" || k == "tokenizer.ggml.token_type" {
                if let Value::Array(a) = v {
                    if a.len() as usize == n_vocab {
                        *a = subset_array(a, &keep_v);
                    }
                }
            } else if k.starts_with("tokenizer.ggml.") && k.ends_with("_token_id") {
                if let Some(old) = v.as_u64() {
                    if (old as usize) < n_vocab {
                        *v = Value::U32(remap[old as usize]);
                    }
                }
            }
        }
        // BPE merges survive only if both sides and the result are still in the vocabulary.
        if let Some(Value::Array(Array::String(merges))) = metadata.get_mut("tokenizer.ggml.merges") {
            let kept: HashSet<&str> = keep_v.iter().map(|&i| vocab.tokens[i].raw.as_str()).collect();
            merges.retain(|mg| match mg.split_once(' ') {
                Some((a, b)) => kept.contains(a) && kept.contains(b) && kept.contains(format!("{a}{b}").as_str()),
                None => false,
            });
        }
        let vs_key = format!("{arch}.vocab_size");
        if metadata.contains_key(&vs_key) {
            metadata.insert(vs_key, Value::U32(keep_v.len() as u32));
        }
        keep_ids = Some(keep_v);
    }

    // --- Layer removal ----------------------------------------------------------------------
    let block_count = g.arch_key("block_count").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut layer_map: BTreeMap<u32, u32> = BTreeMap::new();
    if !remove_layers.is_empty() {
        if block_count == 0 {
            bail!("layer removal requested but {arch}.block_count is missing");
        }
        let mut next = 0;
        for l in 0..block_count as u32 {
            if !remove_layers.contains(&l) {
                layer_map.insert(l, next);
                next += 1;
            }
        }
        let keep_layers: Vec<usize> = layer_map.keys().map(|&l| l as usize).collect();
        let prefix = format!("{arch}.");
        for (k, v) in metadata.iter_mut() {
            if !k.starts_with(&prefix) {
                continue;
            }
            if let Value::Array(a) = v {
                if a.len() == block_count {
                    *a = subset_array(a, &keep_layers);
                }
            }
        }
        metadata.insert(format!("{arch}.block_count"), Value::U32(next));
        // Next-n (speculative) layers sit at the end; keep their count in sync.
        let nextn_key = format!("{arch}.nextn_predict_layers");
        if let Some(n) = g.get_u64(&nextn_key) {
            let first = block_count.saturating_sub(n) as u32;
            let gone = remove_layers.iter().filter(|&&l| l >= first).count() as u64;
            if gone > 0 {
                metadata.insert(nextn_key, Value::U32(n.saturating_sub(gone) as u32));
            }
        }
    }
    for k in &strip_keys {
        metadata.shift_remove(k);
    }
    metadata.insert("general.slimmed_by".into(), Value::String(format!("neurondump {}", env!("CARGO_PKG_VERSION"))));
    if !remove_layers.is_empty() {
        // Original layer numbers (composed with earlier slims), so `diff` can realign layers.
        let prior = crate::diff::removed_layers(g);
        let mut original = (0u32..).filter(|l| !prior.contains(l));
        let mut all: BTreeSet<u32> = prior.iter().copied().collect();
        let mut last = 0;
        for &l in &remove_layers {
            all.insert(original.nth((l - last) as usize).expect("unbounded"));
            last = l + 1;
        }
        metadata.insert(
            crate::diff::REMOVED_LAYERS_KEY.into(),
            Value::Array(neurondump_gguf::Array::U32(all.into_iter().collect())),
        );
    }

    // --- Tensors -----------------------------------------------------------------------------
    let vt: HashSet<String> = vocab_tensors(g, n_vocab as u64).into_iter().map(|t| t.name.clone()).collect();
    let mut w = GgufWriter::new();
    w.metadata = metadata;
    let mut requantized = Vec::new();
    let mut zeroed = Vec::new();
    let total = g.tensors.len().max(1);
    for (i, t) in g.tensors.iter().enumerate() {
        check_cancel(p)?;
        p.set(10.0 + 40.0 * i as f32 / total as f32, &format!("preparing {}", t.name));
        if drop_tensors.contains(&t.name) {
            continue;
        }
        let role = parse_tensor_name(&t.name);
        let mut name = t.name.clone();
        if let Some(l) = role.layer {
            if !layer_map.is_empty() {
                match layer_map.get(&l) {
                    Some(&nl) => name = format!("blk.{nl}.{}", &t.name[format!("blk.{l}.").len()..]),
                    None => continue,
                }
            }
        }
        let mut dims = t.dims.clone();
        let mut ty = t.ggml_type;
        let src = g.info_bytes(t)?;
        let mut data: Cow<[u8]> = Cow::Borrowed(src);
        if let (Some(keep), true) = (&keep_ids, vt.contains(&t.name)) {
            data = Cow::Owned(gather_rows(src, row_bytes(t) as usize, keep));
            dims[1] = keep.len() as u64;
        }
        if requant.contains(&t.name) && t.ggml_type != GgmlType::Q8_0 {
            let mut f = Vec::new();
            if let Some(keep) = keep_ids.as_ref().filter(|_| vt.contains(&t.name)) {
                let mut row = Vec::new();
                for &r in keep {
                    g.dequantize_rows(t, r as u64, 1, &mut row)?;
                    f.extend_from_slice(&row);
                }
            } else {
                g.dequantize_rows(t, 0, t.n_rows(), &mut f)?;
            }
            data = Cow::Owned(quantize_q8_0(&f));
            ty = GgmlType::Q8_0;
            requantized.push(t.name.clone());
        }
        if let Some(ops) = zero.get(&t.name) {
            let mut buf = data.into_owned();
            for o in ops {
                if let Some(new_ty) = apply_zero(o, &dims, ty, &mut buf)? {
                    ty = new_ty;
                    requantized.push(t.name.clone());
                }
            }
            data = Cow::Owned(buf);
            zeroed.push(t.name.clone());
        }
        w.tensors.push(WriterTensor { name, dims, ggml_type: ty, data });
    }

    p.set(50.0, "writing");
    let new_size = w
        .write_to(&out_path, |done, total| {
            p.set(50.0 + 40.0 * done as f32 / total.max(1) as f32, "writing");
        })
        .with_context(|| format!("writing {}", out_path.display()))?;
    drop(w);

    p.set(92.0, "validating");
    let validation = validate(&out_path);
    let modelfile = dir.join(format!("{}.Modelfile", name.trim_end_matches(".gguf")));
    std::fs::write(&modelfile, modelfile_text_with(m, &name, include_projector, include_template))?;

    let perplexity = match (&opts.perplexity_corpus, find_tool("llama-perplexity")) {
        (Some(corpus), Some(tool)) => {
            // llama-perplexity needs at least 2 x ctx tokens: size the context to the corpus.
            let n_tokens = match (&vocab, std::fs::read_to_string(corpus)) {
                (Some(v), Ok(text)) => crate::tokenizer::Encoder::new(v).map(|e| e.encode(&text, true, None).ids.len()).unwrap_or(0),
                _ => 0,
            };
            let ctx = (n_tokens / 2).clamp(16, 512);
            p.set(95.0, "perplexity (original)");
            let (a, la) = run_perplexity(&tool, &m.entry.path, corpus, ctx);
            p.set(97.0, "perplexity (slim)");
            let (b, lb) = run_perplexity(&tool, &out_path, corpus, ctx);
            Some(Perplexity {
                original: a,
                slim: b,
                log: format!("context {ctx} ({n_tokens} corpus tokens)\n--- original\n{la}\n--- slim\n{lb}"),
            })
        }
        (Some(_), None) => Some(Perplexity { original: None, slim: None, log: "llama-perplexity not found in PATH".into() }),
        _ => None,
    };

    p.set(100.0, "done");
    Ok(SlimReport {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        output: out_path,
        modelfile,
        original_size: g.file_size,
        new_size,
        applied,
        refused,
        removed_tokens: keep_ids.map_or(0, |k| n_vocab - k.len()),
        removed_layers: remove_layers.into_iter().collect(),
        requantized,
        zeroed,
        validation,
        perplexity,
    })
}
