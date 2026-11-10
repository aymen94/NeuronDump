//! One FFN neuron in token space, from the weights alone: the tokens whose embedding points
//! along its input row (what it reads on the direct path), and the tokens its output column
//! pushes up or down through the unembedding (what it writes). Both ignore everything
//! between the embedding and the neuron, and between the neuron and the output, so they are
//! a first look, not a measurement on real text (use the journey for that).

use super::*;
use crate::tokenizer::Vocab;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronOptions {
    pub layer: u32,
    pub neuron: u32,
    #[serde(default = "default_top")]
    pub top_k: usize,
}

fn default_top() -> usize {
    15
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronReport {
    pub schema_version: u32,
    pub model: String,
    pub layer: u32,
    pub neuron: u32,
    /// Neurons in the layer.
    pub n_ff: u64,
    /// Tensor of the input row (`ffn_gate`, or `ffn_up` without a gate).
    pub read_from: String,
    pub read_norm: f32,
    pub write_norm: f32,
    /// Tokens whose embedding (through the FFN norm) aligns best with the input row.
    pub reads: Vec<TokenScore>,
    /// Tokens the output column raises / lowers most through the unembedding.
    pub promotes: Vec<TokenScore>,
    pub suppresses: Vec<TokenScore>,
    pub warnings: Vec<String>,
}

pub fn neuron(m: &ModelHandle, vocab: &Vocab, o: &NeuronOptions, p: &dyn Progress) -> Result<NeuronReport> {
    let g = &m.gguf;
    let b = |s: &str| format!("blk.{}.{s}", o.layer);
    let down = g.tensor(&b("ffn_down.weight")).ok_or_else(|| {
        if g.tensor(&b("ffn_down_exps.weight")).is_some() {
            anyhow!("layer {} is a mixture of experts: expert neurons are not supported yet", o.layer)
        } else {
            anyhow!("layer {} has no ffn_down.weight", o.layer)
        }
    })?;
    let n_ff = down.row_len();
    if o.neuron as u64 >= n_ff {
        bail!("neuron {} is out of range (layer {} has {n_ff})", o.neuron, o.layer);
    }
    let mut warnings = Vec::new();
    let norm_gain = |name: &str, dim: usize| -> Result<Option<Vec<f32>>> {
        match g.tensor(name) {
            Some(t) if t.n_elements as usize == dim => {
                let mut v = Vec::new();
                g.dequantize_rows(t, 0, t.n_rows(), &mut v)?;
                Ok(Some(v))
            }
            _ => Ok(None),
        }
    };

    // What it writes: column `neuron` of ffn_down ([n_embd rows] × [n_ff]).
    p.set(5.0, "output column");
    let mut col = vec![0f32; down.n_rows() as usize];
    for_rows(g, down, 256, Some((p, 5.0, 35.0)), |r, row| col[r as usize] = row[o.neuron as usize])?;
    let write_norm = col.iter().map(|x| x * x).sum::<f32>().sqrt();
    let dim = col.len();
    if let Some(gn) = norm_gain("output_norm.weight", dim)? {
        col.iter_mut().zip(&gn).for_each(|(x, g)| *x *= g);
    }
    let unembed = g.tensor("output.weight").or_else(|| g.tensor("token_embd.weight")).ok_or_else(|| anyhow!("no output head"))?;
    p.set(40.0, "unembedding");
    let logits = row_dots(g, unembed, &col)?;
    let neg: Vec<f32> = logits.iter().map(|x| -x).collect();

    // What it reads: row `neuron` of ffn_gate (or ffn_up; the first half when fused).
    p.set(70.0, "input row");
    let (read_t, read_from) = match g.tensor(&b("ffn_gate.weight")) {
        Some(t) => (t, "ffn_gate"),
        None => (tensor(g, &b("ffn_up.weight"))?, "ffn_up"),
    };
    let mut row = Vec::new();
    g.dequantize_rows(read_t, o.neuron as u64, 1, &mut row)?;
    let read_norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    if let Some(gn) = norm_gain(&b("ffn_norm.weight"), row.len())? {
        row.iter_mut().zip(&gn).for_each(|(x, g)| *x *= g);
    }
    let embd = tensor(g, "token_embd.weight")?;
    let reads = if embd.row_len() as usize == row.len() {
        row_dots(g, embd, &row)?
    } else {
        warnings.push("token_embd has another width than the FFN input: reads skipped".into());
        vec![]
    };
    if o.layer > 0 {
        warnings.push("reads follow the direct path from the embeddings only; earlier layers also feed this neuron".into());
    }
    let tok = |scores: &[f32], i: usize, sign: f32| TokenScore {
        id: i as u32,
        text: vocab.tokens.get(i).map(|t| t.text.clone()).unwrap_or_default(),
        score: scores[i] * sign,
    };
    let k = o.top_k.clamp(1, 100);
    p.set(100.0, "done");
    Ok(NeuronReport {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        layer: o.layer,
        neuron: o.neuron,
        n_ff,
        read_from: read_from.into(),
        read_norm,
        write_norm,
        reads: top_k(&reads, k).into_iter().map(|i| tok(&reads, i, 1.0)).collect(),
        promotes: top_k(&logits, k).into_iter().map(|i| tok(&logits, i, 1.0)).collect(),
        suppresses: top_k(&neg, k).into_iter().map(|i| tok(&neg, i, -1.0)).collect(),
        warnings,
    })
}
