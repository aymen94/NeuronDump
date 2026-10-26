//! `ModelGraph::build`: sections, layers, blocks, tensors and dataflow edges from the architecture.

use super::*;

impl ModelGraph {
    pub fn build(m: &ModelHandle, vocab: Option<&Vocab>, inputs: &GraphInputs) -> ModelGraph {
        let g = &m.gguf;
        let arch = analyze_architecture(g);
        let mut nodes: IndexMap<String, Node> = IndexMap::new();
        let mut edges: Vec<Edge> = Vec::new();
        let mut notes = Vec::new();
        let encoder = arch.kind == ModelKind::Encoder;
        let pattern_period = g.arch_key("full_attention_interval").and_then(|v| v.as_u64()).filter(|&p| p > 1);
        let hybrid = arch.kind == ModelKind::Hybrid || pattern_period.is_some();
        let n_layers = arch.layers.len() as u32;
        let nextn_layers = g.arch_key("nextn_predict_layers").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let main_layers = n_layers.saturating_sub(nextn_layers);

        let mut root = new_node("model", NodeKind::Model, &m.entry.name, None);
        root.removable = Removable::BreaksModel;
        root.removable_reason = "the whole model".into();
        nodes.insert("model".into(), root);

        let section = |nodes: &mut IndexMap<String, Node>, id: &str, label: &str, rem: Removable, why: &str| {
            let mut n = new_node(id, NodeKind::Section, label, Some("model"));
            n.removable = rem;
            n.removable_reason = why.into();
            nodes.insert(id.into(), n);
        };
        section(&mut nodes, "input", "Input", Removable::BreaksModel, "token embeddings are the model's only input");
        section(&mut nodes, "body", "Body", Removable::BreaksModel, "removing every layer leaves no model");
        section(
            &mut nodes,
            "output",
            if encoder { "Output (pooling)" } else { "Output" },
            Removable::BreaksModel,
            "the output norm and head produce the predictions",
        );

        // --- top-level tensors ---------------------------------------------------------
        let roles: Vec<_> = g.tensors.iter().map(|t| parse_tensor_name(&t.name)).collect();
        let mut has_mtp_section = false;
        let mut has_vision_section = false;
        let mut has_extra = false;
        for (t, r) in g.tensors.iter().zip(&roles) {
            if r.layer.is_some() {
                continue;
            }
            let role = r.role.strip_prefix("unknown:").unwrap_or(&r.role);
            let vision_prefix = role.starts_with("v.") || role.starts_with("mm.") || role.starts_with("a.");
            let sec = match r.component {
                _ if vision_prefix => "vision",
                Component::Embedding => "input",
                Component::Output => "output",
                Component::Mtp => "mtp",
                Component::Vision => "vision",
                Component::Norm if role.starts_with("output") => "output",
                Component::Norm if role.starts_with("token_embd") => "input",
                _ if role.starts_with("rope") => "input",
                _ if role.starts_with("output") || role.starts_with("cls") => "output",
                _ => "extra",
            };
            match sec {
                "mtp" if !has_mtp_section => {
                    has_mtp_section = true;
                    section(&mut nodes, "mtp", "Multi-token prediction", Removable::Safe, "speculative-decoding head; normal generation does not use it");
                }
                "vision" if !has_vision_section => {
                    has_vision_section = true;
                    section(&mut nodes, "vision", "Vision encoder (in file)", Removable::Risky, "image input stops working; llama.cpp may expect these tensors");
                }
                "extra" if !has_extra => {
                    has_extra = true;
                    section(&mut nodes, "extra", "Other tensors", Removable::Risky, "unrecognized top-level tensors");
                }
                _ => {}
            }
            let id = tensor_id(&t.name);
            let mut n = new_node(&id, NodeKind::Tensor, &t.name, Some(sec));
            n.params = t.n_elements;
            n.bytes = t.n_bytes;
            n.quant = vec![t.ggml_type.to_string()];
            n.info = json!({ "name": t.name, "dims": t.dims, "role": r.role, "type": t.ggml_type.to_string() });
            let (rem, why) = match (sec, role) {
                ("input", "token_embd") => (Removable::BreaksModel, "maps token ids to vectors; nothing runs without it"),
                ("input", r) if r.starts_with("rope") => (Removable::BreaksModel, "rotary position frequencies used by every attention layer"),
                ("input", "pos_embd") => (Removable::BreaksModel, "position embeddings: the model loses word order"),
                ("input", _) => (Removable::BreaksModel, "part of the input embedding path"),
                ("output", "output_norm") => (Removable::BreaksModel, "final normalization before the output head"),
                ("output", _) => (Removable::BreaksModel, "produces the predictions"),
                ("mtp", _) => (Removable::Safe, "speculative-decoding head only"),
                _ => (Removable::Risky, "top-level tensor"),
            };
            n.removable = rem;
            n.removable_reason = why.into();
            nodes.insert(id, n);
        }

        // Tied output head: a virtual node pointing at the embedding matrix.
        let tied = arch.tied_embeddings;
        if tied {
            let mut n = new_node("output.head", NodeKind::Tensor, "output head (tied to token_embd)", Some("output"));
            n.removable = Removable::BreaksModel;
            n.removable_reason = "shares its weights with token_embd".into();
            n.info = json!({ "virtual": true, "tied_to": "token_embd.weight" });
            nodes.insert("output.head".into(), n);
            edges.push(Edge { from: "output.head".into(), to: tensor_id("token_embd.weight"), kind: EdgeKind::Tied, label: Some("shared weights".into()) });
        }
        let pooling = g.arch_key("pooling_type").and_then(|v| v.as_u64()).unwrap_or(0);
        let sink_id = if encoder || pooling > 0 { "output.embedding" } else { "output.logits" };
        {
            let label = if sink_id == "output.embedding" { "pooled embedding" } else { "next-token logits" };
            let mut n = new_node(sink_id, NodeKind::Op, label, Some("output"));
            n.removable = Removable::BreaksModel;
            n.removable_reason = "the model's result".into();
            nodes.insert(sink_id.into(), n);
        }

        // --- layers --------------------------------------------------------------------
        let mut dims: BTreeMap<u32, LayerDims> = BTreeMap::new();
        let emb = arch.hyperparams.embedding_length.unwrap_or(0);
        let mut layer_tensors: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, r) in roles.iter().enumerate() {
            if let Some(l) = r.layer {
                layer_tensors.entry(l).or_default().push(i);
            }
        }
        for l in &arch.layers {
            let lid = format!("layer.{}", l.index);
            let mut n = new_node(&lid, NodeKind::Layer, &format!("layer {}", l.index), Some("body"));
            n.layer = Some(l.index);
            n.layer_kind = Some(layer_kind_label(l));
            n.params = l.params;
            n.bytes = l.bytes;
            let is_nextn = l.index >= main_layers && nextn_layers > 0;
            let (rem, why) = if is_nextn {
                (Removable::Risky, "next-n prediction layer: only used for speculative decoding, but its metadata must be updated too".to_string())
            } else if hybrid {
                (
                    Removable::BreaksModel,
                    format!(
                        "hybrid model: layer kinds follow a fixed pattern{}; removing one layer misaligns every later layer",
                        pattern_period.map(|p| format!(" (every {p}th layer is full attention)")).unwrap_or_default()
                    ),
                )
            } else if n_layers <= 1 {
                (Removable::BreaksModel, "the only layer".into())
            } else if l.index == 0 || l.index + 1 == main_layers {
                (Removable::Risky, "edge layer: llama.cpp loads the file after renumbering, but first/last layers usually matter most".into())
            } else {
                (Removable::Risky, "llama.cpp loads it after renumbering the layers; quality loss must be measured".into())
            };
            n.removable = rem;
            n.removable_reason = why;
            nodes.insert(lid.clone(), n);

            let idx = layer_tensors.get(&l.index).cloned().unwrap_or_default();
            let comp_has = |c: Component| idx.iter().any(|&i| roles[i].component == c);
            let has_ssm = comp_has(Component::LinearAttention);
            let has_moe = comp_has(Component::Moe);
            let has_ffn = idx.iter().any(|&i| {
                let r = roles[i].role.as_str();
                matches!(r, "ffn_up" | "ffn_down" | "ffn_gate" | "ffn_gate_up")
            });

            // Dimensions for the fine level.
            let n_head = l.head_count.unwrap_or(0);
            let n_kv = l.head_count_kv.unwrap_or(n_head);
            let head_dim = arch.hyperparams.key_length.unwrap_or(if n_head > 0 { emb / n_head } else { 0 });
            let n_ff = g
                .tensor(&format!("blk.{}.ffn_down.weight", l.index))
                .map(|t| t.row_len())
                .or(l.ffn_length)
                .unwrap_or(0);
            let n_expert = g
                .tensor(&format!("blk.{}.ffn_down_exps.weight", l.index))
                .or_else(|| g.tensor(&format!("blk.{}.ffn_gate_exps.weight", l.index)))
                .and_then(|t| t.dims.get(2).copied())
                .unwrap_or(0);
            let mut bucket = 64u64;
            while n_ff / bucket > 256 {
                bucket *= 2;
            }
            dims.insert(l.index, LayerDims { n_head, n_kv, head_dim, n_ff, n_expert, bucket });

            // Blocks.
            let mut blocks: IndexMap<&'static str, Vec<usize>> = IndexMap::new();
            for &i in &idx {
                let b = block_of(&roles[i].role, roles[i].component, has_ssm, has_ffn, has_moe);
                blocks.entry(b).or_default().push(i);
            }
            let order = ["attention", "ssm", "ffn", "moe", "mtp", "other"];
            blocks.sort_by_key(|k, _| order.iter().position(|o| o == k).unwrap_or(9));
            for (b, ts) in &blocks {
                let bid = format!("{lid}.{}", block_short(b));
                let mut bn = new_node(&bid, NodeKind::Block, block_label(b), Some(&lid));
                bn.layer = Some(l.index);
                bn.block = Some(b.to_string());
                bn.layer_kind = Some(layer_kind_label(l));
                let (rem, why) = match *b {
                    "mtp" => (Removable::Safe, "speculative-decoding head only"),
                    "other" => (Removable::Risky, "unrecognized tensors"),
                    _ if hybrid && *b == "ssm" => (Removable::Risky, "zeroed in place: shapes stay, the layer stops mixing tokens"),
                    _ => (Removable::Risky, "zeroed in place (output projection set to 0): shapes stay, so llama.cpp still loads it"),
                };
                bn.removable = rem;
                bn.removable_reason = why.into();
                bn.lazy = match *b {
                    "attention" => n_head > 0,
                    "ffn" => n_ff > 0,
                    "moe" => n_expert > 0,
                    _ => false,
                };
                bn.info = match *b {
                    "attention" => json!({ "heads": n_head, "kv_heads": n_kv, "head_dim": head_dim, "gqa_group": if n_kv > 0 { n_head / n_kv } else { 0 } }),
                    "ffn" => json!({ "neurons": n_ff, "bucket": bucket }),
                    "moe" => json!({ "experts": n_expert }),
                    _ => Value::Null,
                };
                let mut quants = BTreeSet::new();
                for &i in ts {
                    let t = &g.tensors[i];
                    bn.params += t.n_elements;
                    bn.bytes += t.n_bytes;
                    quants.insert(t.ggml_type.to_string());
                    let tid = tensor_id(&t.name);
                    let short = t.name.strip_prefix(&format!("blk.{}.", l.index)).unwrap_or(&t.name);
                    let mut tn = new_node(&tid, NodeKind::Tensor, short, Some(&bid));
                    tn.layer = Some(l.index);
                    tn.block = Some(b.to_string());
                    tn.layer_kind = Some(layer_kind_label(l));
                    tn.params = t.n_elements;
                    tn.bytes = t.n_bytes;
                    tn.quant = vec![t.ggml_type.to_string()];
                    tn.info = json!({ "name": t.name, "dims": t.dims, "role": roles[i].role, "type": t.ggml_type.to_string() });
                    tn.removable = if *b == "mtp" { Removable::Safe } else { Removable::Risky };
                    tn.removable_reason = "zeroed in place: llama.cpp still loads it, its contribution becomes 0".into();
                    nodes.insert(tid, tn);
                }
                bn.quant = quants.into_iter().collect();
                nodes.insert(bid, bn);
            }
            if let Some(n) = nodes.get_mut(&lid) {
                let mut q: BTreeSet<String> = BTreeSet::new();
                for &i in &idx {
                    q.insert(g.tensors[i].ggml_type.to_string());
                }
                n.quant = q.into_iter().collect();
            }
            for (op, label) in [("add_mix", "residual add (mixer)"), ("add_ffn", "residual add (FFN)")] {
                let oid = format!("{lid}.{op}");
                let mut on = new_node(&oid, NodeKind::Op, label, Some(&lid));
                on.layer = Some(l.index);
                on.removable = Removable::BreaksModel;
                on.removable_reason = "the residual stream itself".into();
                nodes.insert(oid, on);
            }
        }

        // --- dataflow ------------------------------------------------------------------
        let flow = |edges: &mut Vec<Edge>, a: &str, b: &str, label: Option<&str>| {
            edges.push(Edge { from: a.into(), to: b.into(), kind: EdgeKind::Dataflow, label: label.map(String::from) });
        };
        let mut cur = tensor_id("token_embd.weight");
        if !nodes.contains_key(&cur) {
            cur = "input".into();
        }
        for extra in ["pos_embd.weight", "token_types.weight"] {
            let id = tensor_id(extra);
            if nodes.contains_key(&id) {
                flow(&mut edges, &cur, &id, Some("add"));
                cur = id;
            }
        }
        let en = tensor_id("token_embd_norm.weight");
        if nodes.contains_key(&en) {
            flow(&mut edges, &cur, &en, None);
            cur = en;
        }
        let mut last_main = cur.clone();
        for l in &arch.layers {
            let lid = format!("layer.{}", l.index);
            let is_nextn = l.index >= main_layers && nextn_layers > 0;
            let start = if is_nextn { last_main.clone() } else { cur.clone() };
            let mut residual = start.clone();
            let blocks: Vec<(String, String)> = nodes
                .values()
                .filter(|n| n.kind == NodeKind::Block && n.parent.as_deref() == Some(lid.as_str()))
                .map(|n| (n.id.clone(), n.block.clone().unwrap_or_default()))
                .collect();
            let mixer = blocks.iter().find(|(_, b)| b == "attention" || b == "ssm").cloned();
            let ffnb = blocks.iter().find(|(_, b)| b == "ffn" || b == "moe").cloned();
            let extra_mtp = blocks.iter().find(|(_, b)| b == "mtp").cloned();
            for (stage_block, add) in [(mixer, "add_mix"), (ffnb, "add_ffn")] {
                let Some((bid, _)) = stage_block else { continue };
                let add_id = format!("{lid}.{add}");
                let ts: Vec<(String, Stage)> = nodes
                    .values()
                    .filter(|n| n.parent.as_deref() == Some(bid.as_str()))
                    .map(|n| (n.id.clone(), stage_of(n.info["role"].as_str().unwrap_or(""))))
                    .collect();
                let of = |s: Stage| -> Vec<String> { ts.iter().filter(|(_, st)| *st == s).map(|(i, _)| i.clone()).collect() };
                let (pre, inn, out, post, post_add) = (of(Stage::Pre), of(Stage::In), of(Stage::Out), of(Stage::Post), of(Stage::PostAdd));
                let mut frontier = vec![residual.clone()];
                for stage in [&pre, &inn, &out, &post] {
                    if stage.is_empty() {
                        continue;
                    }
                    for f in &frontier {
                        for s in stage {
                            flow(&mut edges, f, s, None);
                        }
                    }
                    frontier = stage.clone();
                }
                for f in &frontier {
                    flow(&mut edges, f, &add_id, None);
                }
                flow(&mut edges, &residual, &add_id, Some("residual"));
                residual = add_id;
                for pa in &post_add {
                    flow(&mut edges, &residual, pa, None);
                    residual = pa.clone();
                }
            }
            if let Some((bid, _)) = extra_mtp {
                flow(&mut edges, &residual, &bid, Some("speculative"));
            }
            if !is_nextn {
                cur = residual;
                last_main = cur.clone();
            }
        }
        let on = tensor_id("output_norm.weight");
        if nodes.contains_key(&on) {
            flow(&mut edges, &cur, &on, None);
            cur = on;
        }
        let head = if nodes.contains_key(&tensor_id("output.weight")) {
            Some(tensor_id("output.weight"))
        } else if nodes.contains_key("output.head") {
            Some("output.head".to_string())
        } else {
            None
        };
        if let Some(h) = head.filter(|_| sink_id == "output.logits") {
            flow(&mut edges, &cur, &h, None);
            cur = h;
        }
        flow(&mut edges, &cur, sink_id, None);
        if has_mtp_section {
            flow(&mut edges, &last_main, "mtp", Some("speculative"));
        }
        if has_vision_section {
            edges.push(Edge { from: "vision".into(), to: "input".into(), kind: EdgeKind::Uses, label: Some("image embeddings".into()) });
        }

        // --- tokenizer, template, projector ----------------------------------------------
        if let Some(v) = vocab {
            section(&mut nodes, "tokenizer", &format!("Tokenizer ({})", v.model_name), Removable::BreaksModel, "turns text into token ids");
            if let Some(n) = nodes.get_mut("tokenizer") {
                n.params = v.tokens.len() as u64;
                n.bytes = v.tokens.iter().map(|t| t.raw.len() as u64 + 12).sum::<u64>() + v.merges.iter().map(|x| x.len() as u64 + 8).sum::<u64>();
                n.info = json!({ "tokens": v.tokens.len(), "merges": v.merges.len(), "model": v.model_name, "classes": v.summary.class_counts });
            }
            if nodes.contains_key(&tensor_id("token_embd.weight")) {
                edges.push(Edge { from: "tokenizer".into(), to: tensor_id("token_embd.weight"), kind: EdgeKind::Uses, label: Some("token ids".into()) });
            }
            let report = crate::template::template_report(m);
            let mut used: BTreeSet<String> = BTreeSet::new();
            for t in &report.templates {
                used.extend(t.special_tokens.iter().cloned());
            }
            let mut ids: BTreeMap<u32, Vec<String>> = BTreeMap::new();
            for s in &v.summary.specials {
                ids.entry(s.id).or_default().push(s.role.clone());
            }
            for s in &used {
                if let Some(&id) = v.by_raw.get(s) {
                    ids.entry(id).or_default().push("template".into());
                }
            }
            for (id, roles_) in &ids {
                let Some(tok) = v.tokens.get(*id as usize) else { continue };
                let nid = format!("token.{id}");
                let mut n = new_node(&nid, NodeKind::Token, &format!("{:?}", tok.raw), Some("tokenizer"));
                n.params = 1;
                n.info = json!({ "id": id, "text": tok.raw, "roles": roles_ });
                let structural = roles_.iter().any(|r| r == "bos" || r == "eos");
                n.removable = if structural { Removable::BreaksModel } else { Removable::Risky };
                n.removable_reason = if structural {
                    "referenced by the tokenizer metadata (bos/eos)".into()
                } else {
                    "used by the chat template".into()
                };
                nodes.insert(nid, n);
            }
            if !report.templates.is_empty() {
                section(&mut nodes, "template", "Chat template", Removable::Risky, "raw-text inference still works, chat formatting is lost");
                if let Some(n) = nodes.get_mut("template") {
                    n.info = json!({
                        "templates": report.templates.iter().map(|t| json!({ "name": t.name, "decisions": t.decisions, "variables": t.variables.len() })).collect::<Vec<_>>(),
                        "special_tokens": used,
                    });
                    n.params = report.templates.iter().map(|t| t.decisions as u64).sum();
                    n.bytes = report.templates.iter().map(|t| t.source.len() as u64).sum();
                }
                for s in &used {
                    if let Some(&id) = v.by_raw.get(s) {
                        edges.push(Edge { from: "template".into(), to: format!("token.{id}"), kind: EdgeKind::Uses, label: None });
                    }
                }
            }
        }
        if let Some(p) = &m.entry.projector {
            section(&mut nodes, "projector", "Vision projector (separate file)", Removable::Safe, "text-only use does not need it; image input stops working");
            if let Some(n) = nodes.get_mut("projector") {
                n.bytes = std::fs::metadata(p).map(|md| md.len()).unwrap_or(0);
                if let Some(Ok(pg)) = m.open_projector() {
                    n.params = pg.total_params();
                    let mut q: BTreeSet<String> = BTreeSet::new();
                    for t in &pg.tensors {
                        q.insert(t.ggml_type.to_string());
                    }
                    n.quant = q.into_iter().collect();
                    n.info = json!({ "file": p, "tensors": pg.tensors.len(), "architecture": pg.architecture() });
                }
            }
            edges.push(Edge { from: "projector".into(), to: "input".into(), kind: EdgeKind::Uses, label: Some("image embeddings".into()) });
        }

        // --- aggregate sizes for sections ------------------------------------------------
        for sec in ["input", "output", "body", "mtp", "vision", "extra"] {
            let (p, b, q) = nodes.values().filter(|n| n.kind == NodeKind::Tensor).filter(|n| ancestors_of(&nodes, &n.id).iter().any(|a| a == sec)).fold(
                (0u64, 0u64, BTreeSet::new()),
                |(p, b, mut q), n| {
                    q.extend(n.quant.iter().cloned());
                    (p + n.params, b + n.bytes, q)
                },
            );
            if let Some(n) = nodes.get_mut(sec) {
                n.params = p;
                n.bytes = b;
                n.quant = q.into_iter().collect();
            }
        }
        if let Some(n) = nodes.get_mut("model") {
            n.params = g.total_params();
            n.bytes = g.file_size;
        }

        let mut graph = ModelGraph {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            digest: m.entry.digest.clone(),
            architecture: arch.name.clone(),
            kind: arch.kind,
            total_params: g.total_params(),
            file_size: g.file_size,
            tied_embeddings: tied,
            pattern_period,
            n_layers,
            nodes,
            edges,
            dims,
            fine_stats: HashMap::new(),
            fine_activation: HashMap::new(),
            fine_importance: HashMap::new(),
            notes: vec![],
        };
        graph.apply_measurements(inputs, &mut notes);
        graph.fill_contains();
        graph.notes = notes;
        graph
    }

}
