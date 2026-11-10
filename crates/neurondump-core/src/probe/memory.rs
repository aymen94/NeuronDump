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
    let mut values_per_token = 0u64;
    let mut kv_layers = 0u64;
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
    }
    if kv_layers == 0 {
        notes.push("no attention heads found: recurrent / state-space layers keep a fixed-size state instead of a KV cache".into());
    } else if kv_layers < layers {
        notes.push(format!("{} of {layers} layers have no KV cache (recurrent or state-space layers)", layers - kv_layers));
    }
    let sliding_window = k("attention.sliding_window").and_then(|v| v.as_u64());
    if let Some(w) = sliding_window {
        notes.push(format!("sliding-window layers ({w} tokens) cache less than shown; the figures assume every attention layer caches the whole context"));
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
            let kv: Vec<u64> = kv_bytes_per_token.iter().map(|b| (b * c as f64) as u64).collect();
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
        rows,
        notes,
    })
}
