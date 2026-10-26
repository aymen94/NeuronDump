//! `static_impact`: size, disconnected parts, dependants, loadability and verdict of a cut.

use super::*;

pub(super) fn reachable(starts: &[String], adj: &BTreeMap<String, Vec<String>>, blocked: &HashSet<String>) -> HashSet<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut q: VecDeque<String> = starts.iter().filter(|s| !blocked.contains(*s)).cloned().collect();
    while let Some(x) = q.pop_front() {
        if !seen.insert(x.clone()) {
            continue;
        }
        for y in adj.get(&x).into_iter().flatten() {
            if !blocked.contains(y) && !seen.contains(y) {
                q.push_back(y.clone());
            }
        }
    }
    seen
}

pub fn static_impact(m: &ModelHandle, graph: &ModelGraph, vocab: Option<&Vocab>, ids: &[String], force: bool) -> StaticImpact {
    let g = &m.gguf;
    let n_vocab = vocab.map_or(0, |v| v.tokens.len() as u32);
    let (targets, unknown, redundant) = resolve_cut(graph, vocab, ids);
    let ablation = ablation_for(graph, &targets, n_vocab);
    let mut reasons = Vec::new();
    let mut dependencies = Vec::new();
    let mut legality = Vec::new();
    let mut file_ops = Vec::new();
    let mut verdict = targets.iter().map(|t| t.removable).max().unwrap_or(Removable::Safe);
    let mut llama_ok = true;
    for t in &targets {
        reasons.push(format!("{}: {} ({})", t.id, t.removable.as_str(), t.reason));
    }
    if targets.is_empty() {
        reasons.push("nothing selected".into());
    }

    // --- sizes -----------------------------------------------------------------------
    let mut params_removed = 0u64;
    let mut params_zeroed = 0u64;
    let mut bytes_removed = 0u64;
    let mut other_files = 0u64;
    for t in &targets {
        match t.mode {
            CutMode::Remove if t.id == "projector" => other_files += t.bytes,
            CutMode::Remove if t.id == "template" => bytes_removed += t.bytes,
            CutMode::Remove => {
                params_removed += t.params;
                bytes_removed += t.bytes;
            }
            CutMode::Zero => params_zeroed += t.params,
            CutMode::None => {}
        }
    }
    let ops = zero_ops(g, graph, &ablation);
    let (delta, requant) = requant_delta(g, &ops);
    if !requant.is_empty() {
        file_ops.push(format!(
            "requantize {} tensor(s) to Q8_0 to zero unaligned columns ({}{}): {}",
            requant.len(),
            if delta >= 0 { "+" } else { "-" },
            human_bytes(delta.unsigned_abs()),
            requant.join(", ")
        ));
    }

    // --- layers ----------------------------------------------------------------------
    let n_layers = graph.n_layers;
    let removed: BTreeSet<u32> = ablation.skip_layers.iter().copied().collect();
    let mut renumber = Vec::new();
    if !removed.is_empty() {
        if removed.len() as u32 >= n_layers {
            verdict = Removable::BreaksModel;
            llama_ok = false;
            reasons.push("every layer is removed".into());
        }
        let mut next = 0;
        for l in 0..n_layers {
            if removed.contains(&l) {
                continue;
            }
            if l != next {
                renumber.push((l, next));
            }
            next += 1;
        }
        file_ops.push(format!("remove layer(s) {:?}; block_count {} -> {}", removed, n_layers, n_layers as usize - removed.len()));
        legality.push(format!("layers renumbered: {} later layer(s) move down", renumber.len()));
        if let Some(p) = graph.pattern_period {
            // Removing whole aligned periods keeps the hybrid pattern intact.
            let nextn = g.arch_key("nextn_predict_layers").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let main: BTreeSet<u32> = removed.iter().copied().filter(|l| *l < n_layers.saturating_sub(nextn)).collect();
            let p = p as u32;
            let whole = main.iter().all(|l| (l / p * p..l / p * p + p).all(|x| main.contains(&x)));
            if whole && !main.is_empty() {
                verdict = verdict.min(Removable::Risky).max(targets.iter().filter(|t| t.kind != NodeKind::Layer).map(|t| t.removable).max().unwrap_or(Removable::Safe));
                reasons.push(format!("removed layers form whole periods of {p} (the hybrid pattern stays aligned)"));
            } else if !main.is_empty() {
                verdict = Removable::BreaksModel;
                llama_ok = false;
                reasons.push(format!("hybrid pattern (period {p}) broken: remove whole aligned groups of {p} layers instead"));
            }
        }
    }

    // --- dependencies ------------------------------------------------------------------
    let cut_ids: HashSet<&str> = targets.iter().map(|t| t.id.as_str()).collect();
    let cut_under = |x: &str| cut_ids.contains(x) || graph.ancestors(x).iter().any(|a| cut_ids.contains(a.as_str()));
    if graph.tied_embeddings && (cut_under("t.token_embd.weight") || cut_under("output.head")) {
        dependencies.push("tied embeddings: token_embd is also the output head, so cutting one cuts both".into());
    }
    if let Some(v) = vocab {
        let tpl_specials: BTreeSet<u32> = graph
            .nodes
            .values()
            .filter(|n| n.kind == NodeKind::Token && n.info["roles"].as_array().is_some_and(|r| r.iter().any(|x| x == "template")))
            .filter_map(|n| n.info["id"].as_u64().map(|x| x as u32))
            .collect();
        let structural: BTreeSet<u32> = v.summary.specials.iter().filter(|s| s.role == "bos" || s.role == "eos").map(|s| s.id).collect();
        let hit_tpl: Vec<u32> = tpl_specials.iter().copied().filter(|id| ablation.masks_token(*id)).collect();
        let hit_struct: Vec<u32> = structural.iter().copied().filter(|id| ablation.masks_token(*id)).collect();
        if !hit_tpl.is_empty() {
            dependencies.push(format!("the chat template uses {} of the cut tokens ({:?}): chat formatting breaks", hit_tpl.len(), hit_tpl));
            verdict = verdict.max(Removable::Risky);
        }
        if !hit_struct.is_empty() {
            dependencies.push(format!("bos/eos token(s) {hit_struct:?} are cut: slim always keeps them, simulation masks them"));
            verdict = Removable::BreaksModel;
        }
        if !ablation.mask_tokens.is_empty() {
            let n: u32 = ablation.mask_tokens.iter().map(|(a, b)| b - a + 1).sum();
            file_ops.push(format!("remove {n} token(s) from the vocabulary, token_embd/output rows and merges"));
        }
    }
    if cut_ids.contains("tokenizer") {
        dependencies.push("without the tokenizer no text can be encoded".into());
    }

    // --- graph connectivity -------------------------------------------------------------
    let mut fwd: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut bwd: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for e in graph.edges.iter().filter(|e| e.kind == EdgeKind::Dataflow) {
        fwd.entry(e.from.clone()).or_default().push(e.to.clone());
        bwd.entry(e.to.clone()).or_default().push(e.from.clone());
    }
    // Skipped layers become identities: connect the input of each run of consecutive
    // removed layers to its output.
    let mut runs: Vec<Vec<u32>> = Vec::new();
    for &l in &removed {
        match runs.last_mut() {
            Some(r) if r.last() == Some(&(l.wrapping_sub(1))) => r.push(l),
            _ => runs.push(vec![l]),
        }
    }
    for run in &runs {
        let lids: Vec<String> = run.iter().map(|l| format!("layer.{l}")).collect();
        let inside = |x: &str| lids.iter().any(|lid| x == lid || graph.ancestors(x).contains(lid));
        let ins: Vec<String> = graph.edges.iter().filter(|e| e.kind == EdgeKind::Dataflow && !inside(&e.from) && inside(&e.to)).map(|e| e.from.clone()).collect();
        let outs: Vec<String> = graph.edges.iter().filter(|e| e.kind == EdgeKind::Dataflow && inside(&e.from) && !inside(&e.to)).map(|e| e.to.clone()).collect();
        for a in &ins {
            for b in &outs {
                fwd.entry(a.clone()).or_default().push(b.clone());
                bwd.entry(b.clone()).or_default().push(a.clone());
            }
        }
    }
    let blocked: HashSet<String> = graph
        .nodes
        .keys()
        .filter(|id| {
            let ablated_tensor = graph.nodes.get(*id).and_then(|n| n.info["name"].as_str()).is_some_and(|nm| ablation.zero_tensors.iter().any(|z| z == nm));
            cut_under(id) || ablated_tensor
        })
        .cloned()
        .collect();
    let source = if graph.nodes.contains_key("t.token_embd.weight") { "t.token_embd.weight" } else { "input" };
    let sink = if graph.nodes.contains_key("output.logits") { "output.logits" } else { "output.embedding" };
    let from_src = reachable(&[source.to_string()], &fwd, &blocked);
    let to_sink = reachable(&[sink.to_string()], &bwd, &blocked);
    // Nodes already off the main path without any cut (e.g. speculative next-n layers).
    let none = HashSet::new();
    let base_src = reachable(&[source.to_string()], &fwd, &none);
    let base_sink = reachable(&[sink.to_string()], &bwd, &none);
    let flow_nodes: BTreeSet<&String> = fwd.keys().chain(bwd.keys()).collect();
    let mut disconnected: BTreeSet<String> = BTreeSet::new();
    for n in flow_nodes {
        if blocked.contains(n) || (from_src.contains(n) && to_sink.contains(n)) || !(base_src.contains(n) && base_sink.contains(n)) {
            continue;
        }
        // Report at block / layer granularity.
        let anc = graph.ancestors(n);
        let shown = anc.iter().find(|a| graph.nodes.get(*a).is_some_and(|x| x.kind == NodeKind::Block)).cloned().unwrap_or_else(|| n.clone());
        if !shown.ends_with("mtp") && graph.nodes.get(&shown).is_none_or(|x| x.block.as_deref() != Some("mtp")) {
            disconnected.insert(shown);
        }
    }
    if !from_src.contains(sink) {
        verdict = Removable::BreaksModel;
        reasons.push("no path from the input embeddings to the output remains".into());
    }
    if !disconnected.is_empty() {
        reasons.push(format!("{} node(s) are cut off from the input or the output", disconnected.len()));
    }

    // --- file operations / legality -----------------------------------------------------
    for t in &targets {
        match (t.mode, t.id.as_str()) {
            (CutMode::Remove, "projector") => file_ops.push("drop the vision projector from the Modelfile".into()),
            (CutMode::Remove, "template") => file_ops.push("strip tokenizer.chat_template and the Modelfile TEMPLATE".into()),
            (CutMode::Remove, "mtp" | "vision" | "extra") => {
                file_ops.push(format!("drop the {} tensors of {}", graph.tensors_under(&t.id).len(), t.id));
                legality.push(format!("{}: not verified that llama.cpp loads the file without these tensors", t.id));
            }
            (CutMode::Zero, _) if t.removable == Removable::BreaksModel => {
                legality.push(format!("{}: zeroed tensors keep their shapes, so the file loads, but the model's output is garbage", t.id));
            }
            _ => {}
        }
    }
    if !ops.is_empty() {
        file_ops.push(format!("zero weights in place in {} tensor(s) (shapes unchanged)", ops.len()));
        legality.push("zeroed weights keep every tensor shape: llama.cpp loads the file unchanged".into());
    }
    if targets.iter().any(|t| t.mode == CutMode::None) {
        llama_ok = false;
    }

    // --- simulation support -----------------------------------------------------------------
    let mut sim_notes = Vec::new();
    let backend_ok = crate::trace::require_backend(m);
    let mut sim_supported = backend_ok.is_ok();
    if let Err(e) = &backend_ok {
        sim_notes.push(format!("{e:#}"));
    }
    if !ablation.zero_experts.is_empty() || ablation.zero_blocks.iter().any(|(_, b)| b == "moe" || b == "ssm") {
        sim_supported = false;
        sim_notes.push("the runtime forward pass has no MoE / SSM support: experts and SSM blocks can only be cut statically".into());
    }
    for t in &targets {
        if matches!(t.id.as_str(), "projector" | "template") {
            sim_notes.push(format!("{} has no effect on a raw-text forward pass", t.id));
        }
        if t.id == "tokenizer" {
            sim_supported = false;
            sim_notes.push("the tokenizer cannot be simulated away".into());
        }
    }
    if !ablation.mask_tokens.is_empty() {
        sim_notes.push("simulation masks the tokens in the output distribution; slim removes them from the vocabulary".into());
    }

    let refused = verdict == Removable::BreaksModel && !force;
    let total = graph.total_params.max(1);
    let new_size = (graph.file_size as i64 - bytes_removed as i64 + delta).max(0) as u64;
    StaticImpact {
        schema_version: SCHEMA_VERSION,
        model: graph.model.clone(),
        architecture: graph.architecture.clone(),
        cut: targets,
        unknown,
        redundant,
        total_params: graph.total_params,
        params_removed,
        params_zeroed,
        percent_of_model: 100.0 * (params_removed + params_zeroed) as f64 / total as f64,
        original_size: graph.file_size,
        bytes_removed,
        requant_bytes_delta: delta,
        new_size,
        other_files_removed: other_files,
        verdict,
        reasons,
        disconnected: {
            let mut d: Vec<String> = disconnected.into_iter().collect();
            d.sort_by_key(|x| (x.strip_prefix("layer.").and_then(|r| r.split('.').next()).and_then(|n| n.parse::<u32>().ok()).unwrap_or(u32::MAX), x.clone()));
            d
        },
        dependencies,
        llama_cpp_loads: llama_ok,
        legality,
        layer_renumbering: renumber,
        file_ops,
        refused,
        ablation,
        simulation_supported: sim_supported,
        simulation_notes: sim_notes,
    }
}
