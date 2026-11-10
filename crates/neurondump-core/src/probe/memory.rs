//! Memory a model needs at a context length: the weights plus the KV cache, which grows with
//! the context, in the cache types llama.cpp / Ollama offer (f16, q8_0, q4_0).

use super::*;

/// Bytes per cached value: f16, q8_0 (34 bytes / 32 values), q4_0 (18 bytes / 32 values).
pub const KV_TYPES: [(&str, f64); 3] = [("f16", 2.0), ("q8_0", 34.0 / 32.0), ("q4_0", 18.0 / 32.0)];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRow {
    pub context: u64,
    /// KV cache bytes per cache type, in `KV_TYPES` order.
    pub kv_bytes: Vec<u64>,
    /// Weights + KV cache, per cache type.
    pub total_bytes: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub model: String,
    pub weights_bytes: u64,
    pub layers: u64,
    /// Layers that keep a KV cache (attention layers).
    pub kv_layers: u64,
    /// Key + value values cached per token, over all layers.
    pub values_per_token: u64,
    pub kv_types: Vec<String>,
    /// KV bytes per token per cache type.
    pub kv_bytes_per_token: Vec<f64>,
    /// Context length the model was trained for.
    pub train_context: Option<u64>,
    pub sliding_window: Option<u64>,
    /// Layers that only cache the last `sliding_window` tokens.
    pub swa_layers: u64,
    pub rows: Vec<MemoryRow>,
    pub notes: Vec<String>,
}

/// The estimate at the usual context lengths (and the trained one), plus `extra` lengths.
pub fn memory_estimate(m: &ModelHandle, extra: &[u64]) -> Result<MemoryEstimate> {
    let g = &m.gguf;
    let k = |s: &str| g.arch_key(s);
    let layers = k("block_count").and_then(|v| v.as_u64()).context("block_count missing")?;
    let n_embd = k("embedding_length").and_then(|v| v.as_u64()).unwrap_or(0);
    let hc = k("attention.head_count");
    let hkv = k("attention.head_count_kv");
    let mut notes = Vec::new();
    // Sliding-window layers as llama.cpp lays them out: Gemma 2 alternates local / global,
    // Gemma 3 has five local layers per global one, unless the file gives the pattern.
    let sliding_window = k("attention.sliding_window").and_then(|v| v.as_u64()).filter(|w| *w > 0);
    let pattern = k("attention.sliding_window_pattern").and_then(|v| v.as_u64()).or(match m.architecture() {
        "gemma2" => Some(2),
        "gemma3" => Some(6),
        _ => None,
    });
    let is_swa = |i: usize| sliding_window.is_some() && pattern.is_some_and(|p| p > 0 && (i as u64) % p < p - 1);
    let mut values_per_token = 0u64;
    let mut values_swa = 0u64;
    let mut kv_layers = 0u64;
    let mut swa_layers = 0u64;
    for i in 0..layers as usize {
        let n_head = crate::arch::per_layer(hc, i).unwrap_or(0);
        let n_kv = crate::arch::per_layer(hkv, i).unwrap_or(n_head);
        if n_head == 0 || n_kv == 0 {
            continue;
        }
        let head_dim = n_embd.checked_div(n_head).unwrap_or(0);
        let kd = k("attention.key_length").and_then(|v| v.as_u64()).unwrap_or(head_dim);
        let vd = k("attention.value_length").and_then(|v| v.as_u64()).unwrap_or(head_dim);
        values_per_token += n_kv * (kd + vd);
        kv_layers += 1;
        if is_swa(i) {
            values_swa += n_kv * (kd + vd);
            swa_layers += 1;
        }
    }
    if kv_layers == 0 {
        notes.push("no attention heads found: recurrent / state-space layers keep a fixed-size state instead of a KV cache".into());
    } else if kv_layers < layers {
        notes.push(format!("{} of {layers} layers have no KV cache (recurrent or state-space layers)", layers - kv_layers));
    }
    match sliding_window {
        Some(w) if swa_layers > 0 => notes.push(format!(
            "{swa_layers} of {kv_layers} attention layers use a {w}-token sliding window and cache at most {w} tokens (counted); llama.cpp's --swa-full caches the whole context there"
        )),
        Some(w) => notes.push(format!("the file declares a {w}-token sliding window but not which layers use it: every layer is counted at full context")),
        None => {}
    }
    let train_context = k("context_length").and_then(|v| v.as_u64());
    let weights_bytes: u64 = g.tensors.iter().map(|t| t.n_bytes).sum();
    let kv_bytes_per_token: Vec<f64> = KV_TYPES.iter().map(|(_, b)| values_per_token as f64 * b).collect();
    let mut ctxs: Vec<u64> = vec![2048, 4096, 8192, 16384, 32768, 131072];
    ctxs.extend(train_context);
    ctxs.extend(extra.iter().copied().filter(|c| *c > 0));
    ctxs.sort_unstable();
    ctxs.dedup();
    let rows = ctxs
        .into_iter()
        .map(|c| {
            // Global layers cache every token; sliding-window layers at most the window.
            let window = sliding_window.unwrap_or(c).min(c);
            let kv: Vec<u64> = KV_TYPES
                .iter()
                .map(|(_, b)| (b * ((values_per_token - values_swa) as f64 * c as f64 + values_swa as f64 * window as f64)) as u64)
                .collect();
            MemoryRow { context: c, total_bytes: kv.iter().map(|x| x + weights_bytes).collect(), kv_bytes: kv }
        })
        .collect();
    notes.push("runtimes add compute buffers (often a few hundred MiB) on top".into());
    Ok(MemoryEstimate {
        model: m.entry.name.clone(),
        weights_bytes,
        layers,
        kv_layers,
        values_per_token,
        kv_types: KV_TYPES.iter().map(|(n, _)| n.to_string()).collect(),
        kv_bytes_per_token,
        train_context,
        sliding_window,
        swa_layers,
        rows,
        notes,
    })
}
