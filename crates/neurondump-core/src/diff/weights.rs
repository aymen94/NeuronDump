//! `diff`: streamed per-tensor sums from both files, aggregated per graph node and MoE expert.

use super::*;

/// Running sums from which every metric (and any aggregate of them) follows exactly.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Sums {
    pub(super) aa: f64,
    pub(super) bb: f64,
    pub(super) ab: f64,
    pub(super) dd: f64,
    pub(super) max_abs: f32,
    pub(super) n: u64,
}

impl Sums {
    pub(super) fn add(&mut self, a: &[f32], b: &[f32]) {
        for (&x, &y) in a.iter().zip(b) {
            let (xd, yd) = (x as f64, y as f64);
            self.aa += xd * xd;
            self.bb += yd * yd;
            self.ab += xd * yd;
            self.dd += (xd - yd) * (xd - yd);
            self.max_abs = self.max_abs.max((x - y).abs());
        }
        self.n += a.len() as u64;
    }

    pub(super) fn merge(&mut self, o: &Sums) {
        self.aa += o.aa;
        self.bb += o.bb;
        self.ab += o.ab;
        self.dd += o.dd;
        self.max_abs = self.max_abs.max(o.max_abs);
        self.n += o.n;
    }

    pub(super) fn metrics(&self) -> Metrics {
        let rel_l2 = if self.aa > 0.0 { (self.dd / self.aa).sqrt() } else if self.dd > 0.0 { 1.0 } else { 0.0 };
        let cosine = if self.aa > 0.0 && self.bb > 0.0 {
            self.ab / (self.aa.sqrt() * self.bb.sqrt())
        } else if self.aa == self.bb {
            1.0
        } else {
            0.0
        };
        Metrics { rel_l2, cosine, max_abs: self.max_abs, n: self.n }
    }
}

/// Per-tensor sums, plus per-expert sums for 3D MoE expert tensors.
pub(super) fn diff_tensor(
    ga: &GgufFile,
    ta: &TensorInfo,
    gb: &GgufFile,
    tb: &TensorInfo,
    p: &dyn Progress,
    on_rows: &dyn Fn(u64),
) -> Result<(Sums, Vec<Sums>)> {
    let rows = ta.n_rows();
    let row_len = ta.row_len() as usize;
    let rpc = (CHUNK_ELEMS / ta.row_len().max(1)).max(1);
    // ggml dims [n_embd, n_ff, n_expert]: each expert is a contiguous run of dims[1] rows.
    let per_expert = (ta.dims.len() == 3 && ta.name.contains("_exps")).then(|| ta.dims[1].max(1));
    let mut experts = vec![Sums::default(); per_expert.map_or(0, |r| (rows / r) as usize)];
    let mut total = Sums::default();
    let (mut ba, mut bb) = (Vec::new(), Vec::new());
    let mut s = 0;
    while s < rows {
        check_cancel(p)?;
        let c = rpc.min(rows - s);
        ga.dequantize_rows(ta, s, c, &mut ba)?;
        gb.dequantize_rows(tb, s, c, &mut bb)?;
        match per_expert {
            Some(r) => {
                for i in 0..c as usize {
                    let e = ((s + i as u64) / r) as usize;
                    let span = i * row_len..(i + 1) * row_len;
                    let mut row = Sums::default();
                    row.add(&ba[span.clone()], &bb[span]);
                    experts[e].merge(&row);
                    total.merge(&row);
                }
            }
            None => total.add(&ba, &bb),
        }
        on_rows(c);
        s += c;
    }
    Ok((total, experts))
}

/// Diffs model `b` against model `a`. `graph_a` (model A's graph) enables the per-node
/// aggregates used for coloring; without it only layer aggregates are produced.
pub fn diff(a: &ModelHandle, b: &ModelHandle, graph_a: Option<&ModelGraph>, o: &DiffOptions, p: &dyn Progress) -> Result<DiffReport> {
    let started = Instant::now();
    let (ga, gb) = (&*a.gguf, &*b.gguf);
    let (ra, rb) = (removed_layers(ga), removed_layers(gb));
    p.set(0.0, "matching tensors");

    let by_a: HashMap<String, &TensorInfo> = ga.tensors.iter().map(|t| (canonical_name(&t.name, &ra), t)).collect();
    let by_b: HashMap<String, &TensorInfo> = gb.tensors.iter().map(|t| (canonical_name(&t.name, &rb), t)).collect();
    let mut only_in_a = Vec::new();
    let mut shape_mismatch = Vec::new();
    let mut skipped = Vec::new();
    let mut pairs: Vec<(&TensorInfo, &TensorInfo)> = Vec::new();
    for t in &ga.tensors {
        match by_b.get(&canonical_name(&t.name, &ra)) {
            None => only_in_a.push(t.name.clone()),
            Some(u) if u.dims != t.dims => {
                shape_mismatch.push(ShapeMismatch { name: t.name.clone(), dims_a: t.dims.clone(), dims_b: u.dims.clone() })
            }
            Some(u) => {
                let bad = [t, *u].into_iter().find(|x| !can_dequantize(x.ggml_type) || x.row_len() % x.ggml_type.layout().map_or(1, |l| l.block_size) != 0);
                match bad {
                    Some(x) => skipped.push((t.name.clone(), format!("cannot dequantize {} ({})", x.ggml_type, if std::ptr::eq(x, t) { "A" } else { "B" }))),
                    None => pairs.push((t, *u)),
                }
            }
        }
    }
    let only_in_b: Vec<String> =
        gb.tensors.iter().filter(|t| !by_a.contains_key(&canonical_name(&t.name, &rb))).map(|t| t.name.clone()).collect();

    // --- weights ---------------------------------------------------------------------------
    let total_rows: u64 = pairs.iter().map(|(t, _)| t.n_rows()).sum::<u64>().max(1);
    let done = AtomicU64::new(0);
    let last = Mutex::new(0f32);
    let on_rows = |n: u64| {
        let d = done.fetch_add(n, Ordering::Relaxed) + n;
        let pct = (d as f64 / total_rows as f64 * 100.0) as f32;
        let mut l = last.lock().unwrap();
        if pct - *l >= 0.5 {
            *l = pct;
            p.set(pct.min(99.0), "comparing weights");
        }
    };
    let results: Vec<(usize, Sums, Vec<Sums>)> = if o.metadata_only {
        Vec::new()
    } else {
        pairs
            .par_iter()
            .enumerate()
            .map(|(i, (ta, tb))| diff_tensor(ga, ta, gb, tb, p, &on_rows).map(|(s, e)| (i, s, e)))
            .collect::<Result<_>>()?
    };

    let mut total: Option<Sums> = None;
    let mut by_node: HashMap<String, (Sums, usize)> = HashMap::new();
    let mut tensors = Vec::with_capacity(results.len());
    let mut identical = 0;
    for (i, s, experts) in &results {
        let (ta, tb) = pairs[*i];
        if s.dd == 0.0 {
            identical += 1;
        }
        total.get_or_insert_with(Sums::default).merge(s);
        let id = format!("t.{}", ta.name);
        let layer = parse_tensor_name(&ta.name).layer;
        let owners: Vec<String> = match graph_a {
            Some(g) => g.ancestors(&id),
            None => layer.map(|l| vec![format!("layer.{l}")]).unwrap_or_default(),
        };
        if let (Some(l), false) = (layer, experts.is_empty()) {
            for (e, es) in experts.iter().enumerate() {
                let eid = format!("layer.{l}.moe.expert.{e}");
                let ent = by_node.entry(eid).or_default();
                ent.0.merge(es);
                ent.1 += 1;
            }
        }
        for o in owners {
            let ent = by_node.entry(o).or_default();
            ent.0.merge(s);
            ent.1 += 1;
        }
        tensors.push(TensorDiff {
            name: ta.name.clone(),
            id,
            layer,
            dims: ta.dims.clone(),
            type_a: ta.ggml_type.to_string(),
            type_b: tb.ggml_type.to_string(),
            metrics: s.metrics(),
        });
    }
    tensors.sort_by(|x, y| y.metrics.rel_l2.total_cmp(&x.metrics.rel_l2));
    if let Some(n) = o.top {
        tensors.truncate(n);
    }

    let group = |id: &str, kind: &str, label: String, layer: Option<u32>, block: Option<String>, v: &(Sums, usize)| GroupDiff {
        id: id.to_string(),
        kind: kind.to_string(),
        label,
        layer,
        block,
        tensors: v.1,
        metrics: v.0.metrics(),
    };
    let mut groups = Vec::new();
    match graph_a {
        Some(g) => {
            for n in g.nodes.values().filter(|n| n.kind != NodeKind::Tensor) {
                if let Some(v) = by_node.get(&n.id) {
                    groups.push(group(&n.id, n.kind.as_str(), n.label.clone(), n.layer, n.block.clone(), v));
                }
            }
        }
        None => {
            let mut layers: Vec<(&String, &(Sums, usize))> = by_node.iter().filter(|(k, _)| !k.contains(".moe.")).collect();
            layers.sort_by_key(|(k, _)| k[6..].parse::<u32>().unwrap_or(u32::MAX));
            for (k, v) in layers {
                let l = k[6..].parse().ok();
                groups.push(group(k, "layer", format!("layer {}", l.unwrap_or(0)), l, None, v));
            }
        }
    }
    let mut experts: Vec<(&String, &(Sums, usize))> = by_node.iter().filter(|(k, _)| k.contains(".moe.expert.")).collect();
    experts.sort_by_key(|(k, _)| {
        let mut it = k.split('.');
        (it.nth(1).and_then(|x| x.parse::<u32>().ok()), it.nth(2).and_then(|x| x.parse::<u32>().ok()))
    });
    for (k, v) in experts {
        let mut it = k.split('.');
        let l = it.nth(1).and_then(|x| x.parse().ok());
        let label = format!("expert {}", k.rsplit('.').next().unwrap_or(""));
        groups.push(group(k, "expert", label, l, Some("moe".into()), v));
    }

    p.set(100.0, "done");
    Ok(DiffReport {
        schema_version: SCHEMA_VERSION,
        a: side(a),
        b: side(b),
        metadata: metadata_diff(ga, gb),
        vocab: vocab_diff(ga, gb),
        chat_template: template_diff(a, b),
        only_in_a,
        only_in_b,
        shape_mismatch,
        skipped,
        compared: results.len(),
        identical,
        total: total.map(|s| s.metrics()),
        groups,
        tensors,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}
