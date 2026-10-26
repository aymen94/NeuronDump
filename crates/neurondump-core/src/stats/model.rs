//! `compute_stats`: every tensor in parallel, plus per-head, per-neuron and per-expert norms.

use super::*;

pub fn compute_stats(g: &GgufFile, model: &str, digest: &str, opts: &StatsOptions, p: &dyn Progress) -> Result<StatsReport> {
    let started = Instant::now();
    let selected: Vec<&TensorInfo> = g
        .tensors
        .iter()
        .filter(|t| opts.tensors.is_empty() || opts.tensors.contains(&t.name))
        .filter(|t| opts.layers.is_empty() || parse_tensor_name(&t.name).layer.is_some_and(|l| opts.layers.contains(&l)))
        .collect();
    let mut skipped = Vec::new();
    let todo: Vec<&TensorInfo> = selected
        .into_iter()
        .filter(|t| {
            let ok = can_dequantize(t.ggml_type) && t.row_len() % t.ggml_type.layout().map_or(1, |l| l.block_size) == 0;
            if !ok {
                skipped.push((t.name.clone(), format!("dequantization of {} not supported", t.ggml_type)));
            }
            ok
        })
        .collect();
    let total: u64 = todo.iter().map(|t| t.n_bytes).sum::<u64>().max(1);
    let done = AtomicU64::new(0);
    let last_step = Mutex::new(String::new());
    let results: Vec<Result<TensorStats>> = todo
        .par_iter()
        .map(|t| {
            check_cancel(p)?;
            if let Ok(mut s) = last_step.lock() {
                *s = t.name.clone();
            }
            let on_bytes = |b: u64| {
                let d = done.fetch_add(b, Ordering::Relaxed) + b;
                let step = last_step.lock().map(|s| s.clone()).unwrap_or_default();
                p.set((d as f64 / total as f64 * 100.0) as f32, &step);
            };
            tensor_stats(g, t, p, &on_bytes)
        })
        .collect();
    let mut tensors = Vec::with_capacity(results.len());
    for r in results {
        tensors.push(r?);
    }
    tensors.sort_by(|a, b| a.layer.cmp(&b.layer).then(a.name.cmp(&b.name)));

    let hp = Hyperparams::from_gguf(g);
    let hc_key = g.arch_key("attention.head_count");
    let hkv_key = g.arch_key("attention.head_count_kv");
    let mut heads = Vec::new();
    let mut ffn_neurons = Vec::new();
    let mut experts = Vec::new();
    let layers: std::collections::BTreeSet<u32> = tensors.iter().filter_map(|t| t.layer).collect();
    for l in layers {
        let get = |role: &str| tensors.iter().find(|t| t.layer == Some(l) && t.role == role);
        let nh = per_layer(hc_key, l as usize).or(hp.head_count).unwrap_or(0);
        let nkv = per_layer(hkv_key, l as usize).or(hp.head_count_kv).unwrap_or(nh);
        let (mut q, mut k, mut v) = (vec![], vec![], vec![]);
        let mut head_dim = hp.key_length.unwrap_or(0);
        if let Some(t) = get("attn_q").and_then(|t| t.row_norms.as_ref()) {
            q = group_norms(t, nh);
            if nh > 0 {
                head_dim = t.len() as u64 / nh;
            }
        }
        if let Some(t) = get("attn_k").and_then(|t| t.row_norms.as_ref()) {
            k = group_norms(t, nkv);
        }
        if let Some(t) = get("attn_v").and_then(|t| t.row_norms.as_ref()) {
            v = group_norms(t, nkv);
        }
        if let Some(t) = get("attn_qkv").and_then(|t| t.row_norms.as_ref()) {
            if nh > 0 && (nh + 2 * nkv) > 0 && t.len() as u64 % (nh + 2 * nkv) == 0 {
                let hd = t.len() as u64 / (nh + 2 * nkv);
                head_dim = hd;
                let (qs, rest) = t.split_at((nh * hd) as usize);
                let (ks, vs) = rest.split_at((nkv * hd) as usize);
                q = group_norms(qs, nh);
                k = group_norms(ks, nkv);
                v = group_norms(vs, nkv);
            }
        }
        let o = get("attn_output").and_then(|t| t.col_norms.as_ref()).map(|c| group_norms(c, nh)).unwrap_or_default();
        if !(q.is_empty() && k.is_empty() && v.is_empty() && o.is_empty()) {
            heads.push(HeadNorms { layer: l, head_dim, q, k, v, o });
        }

        let up = get("ffn_up").and_then(|t| t.row_norms.clone());
        let gate = get("ffn_gate").and_then(|t| t.row_norms.clone());
        let down = get("ffn_down").and_then(|t| t.col_norms.clone());
        let n_ff = up.as_ref().or(down.as_ref()).map(Vec::len).unwrap_or(0);
        if n_ff > 0 {
            let norms = (0..n_ff)
                .map(|i| {
                    let a = up.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0) as f64;
                    let b = gate.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0) as f64;
                    let c = down.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(0.0) as f64;
                    (a * a + b * b + c * c).sqrt() as f32
                })
                .collect();
            ffn_neurons.push(NeuronNorms { layer: l, norms });
        }

        let exp: Vec<&Vec<f32>> = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"]
            .iter()
            .filter_map(|r| get(r).and_then(|t| t.slice_norms.as_ref()))
            .collect();
        if let Some(first) = exp.first() {
            let norms = (0..first.len())
                .map(|e| exp.iter().map(|v| (v.get(e).copied().unwrap_or(0.0) as f64).powi(2)).sum::<f64>().sqrt() as f32)
                .collect();
            experts.push(ExpertNorms { layer: l, norms });
        }
    }
    p.set(100.0, "done");
    Ok(StatsReport {
        schema_version: SCHEMA_VERSION,
        model: model.into(),
        digest: digest.into(),
        tensors,
        heads,
        ffn_neurons,
        experts,
        skipped,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}
