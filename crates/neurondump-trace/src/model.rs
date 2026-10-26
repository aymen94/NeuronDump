//! Loaded model, forward options and records; loading, RoPE, logit lens, embedding.

use super::*;

pub struct Sample {
    pub tokens: Vec<u32>,
    pub layers: Vec<LayerTrace>,
    pub prediction: Vec<LensToken>,
    pub neuron_means: Vec<Vec<f32>>,
    pub contributions: Vec<TokenContribution>,
}

pub(super) struct Model<'a> {
    pub(super) w: Weights<'a>,
    pub(super) spec: ArchSpec,
    pub(super) layers: Vec<Layer>,
    pub(super) out_norm: Option<Norm>,
    pub(super) lm_head: Option<Linear>,
    pub(super) embd_norm: Option<Norm>,
    pub(super) n_embd: usize,
    pub(super) rope_dim: usize,
    pub(super) rope_base: f32,
    pub(super) rope_factors: Option<Vec<f32>>,
    pub(super) pooling: u64,
    /// Produces a pooled embedding (encoders, or decoders with a pooling type).
    pub(super) pooled: bool,
    pub(super) vocab: &'a Vocab,
    pub(super) top_k: usize,
}

#[derive(Clone, Copy, Default)]
pub(super) struct FwdOpt {
    /// Per-layer measurements (entropy, neurons, logit lens).
    pub(super) measure: bool,
    /// Keep the full logit-lens distribution of every layer.
    pub(super) lens_probs: bool,
    /// Logits for every position (perplexity, KL), as [seq, vocab].
    pub(super) all_logits: bool,
    /// Record everything about this position (journey, decision breakdown).
    pub(super) record: Option<usize>,
    /// Keep every layer's residual rows and attention outputs (steering / patching sources).
    pub(super) keep: bool,
}

/// One layer at the recorded position.
#[derive(Default, Clone)]
pub(super) struct LayerRec {
    /// [head][key 0..=pos].
    pub(super) attn: Vec<Vec<f32>>,
    /// Attention output before the projection (n_head * head_dim), head factors applied.
    pub(super) z: Vec<f32>,
    /// What the mixer added to the residual stream.
    pub(super) attn_out: Vec<f32>,
    /// RMS of the projection output before a post-attention norm.
    pub(super) attn_pre_rms: Option<f32>,
    pub(super) attn_factor: f32,
    /// FFN activations (after the gate), neuron factors applied.
    pub(super) inner: Vec<f32>,
    pub(super) ffn_out: Vec<f32>,
    pub(super) ffn_pre_rms: Option<f32>,
    pub(super) ffn_factor: f32,
    /// Residual stream after the layer.
    pub(super) h: Vec<f32>,
    pub(super) skipped: bool,
}

#[derive(Default, Clone)]
pub(super) struct Record {
    pub(super) pos: usize,
    /// Residual stream before layer 0.
    pub(super) emb: Vec<f32>,
    pub(super) layers: Vec<LayerRec>,
}

pub(super) struct FwdOut {
    pub(super) sample: Sample,
    /// Next-token logits at the last position (token mask and bias applied).
    pub(super) last_logits: Vec<f32>,
    pub(super) all_logits: Option<Tensor>,
    pub(super) lens_probs: Vec<Vec<f32>>,
    pub(super) pooled: Option<Vec<f32>>,
    pub(super) record: Option<Record>,
    /// With `keep`: residual rows after every layer, and attention outputs before the projection.
    pub(super) hidden: Vec<Vec<Vec<f32>>>,
    pub(super) zs: Vec<Vec<Vec<f32>>>,
}

impl<'a> Model<'a> {
    pub(super) fn load(m: &'a ModelHandle, vocab: &'a Vocab, top_k: usize, p: &dyn Progress) -> Result<Self> {
        let g = &m.gguf;
        let arch = m.architecture().to_string();
        let spec = spec_for(&arch).ok_or_else(|| anyhow!("no forward pass for {arch}"))?;
        let w = Weights { g, dev: Device::Cpu };
        let k = |s: &str| g.arch_key(s);
        let n_embd = k("embedding_length").and_then(|v| v.as_u64()).context("embedding_length")? as usize;
        let n_layer = k("block_count").and_then(|v| v.as_u64()).context("block_count")? as usize;
        let hc = k("attention.head_count");
        let hkv = k("attention.head_count_kv");
        let eps = k("attention.layer_norm_rms_epsilon").or(k("attention.layer_norm_epsilon")).and_then(|v| v.as_f64()).unwrap_or(1e-6);
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            check_cancel(p)?;
            p.set(5.0 + 40.0 * i as f32 / n_layer as f32, &format!("loading layer {i}"));
            let b = |s: &str| format!("blk.{i}.{s}");
            let n_head = neurondump_core::arch::per_layer(hc, i).context("head_count")? as usize;
            let n_kv = neurondump_core::arch::per_layer(hkv, i).unwrap_or(n_head as u64) as usize;
            let head_dim = k("attention.key_length").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(n_embd / n_head);
            let ln = spec.layer_norm;
            let up_rows = g.tensor(&b("ffn_up.weight")).and_then(|t| t.dims.get(1).copied()).unwrap_or(0);
            let down_in = g.tensor(&b("ffn_down.weight")).map(|t| t.row_len()).unwrap_or(0);
            let fused_up = g.tensor(&b("ffn_gate.weight")).is_none() && up_rows == 2 * down_in;
            layers.push(Layer {
                attn_norm: norm(&w, &b("attn_norm"), eps, ln)?,
                q: linear(&w, &b("attn_q"))?,
                k: linear(&w, &b("attn_k"))?,
                v: linear(&w, &b("attn_v"))?,
                qkv: linear(&w, &b("attn_qkv"))?,
                q_norm: norm(&w, &b("attn_q_norm"), eps, false)?,
                k_norm: norm(&w, &b("attn_k_norm"), eps, false)?,
                o: req(linear(&w, &b("attn_output"))?, "attn_output")?,
                attn_post_norm: norm(&w, &b("post_attention_norm"), eps, false)?,
                ffn_norm: norm(&w, &b("ffn_norm"), eps, ln)?,
                gate: linear(&w, &b("ffn_gate"))?,
                up: req(linear(&w, &b("ffn_up"))?, "ffn_up")?,
                down: req(linear(&w, &b("ffn_down"))?, "ffn_down")?,
                ffn_post_norm: norm(&w, &b("post_ffw_norm"), eps, false)?,
                attn_out_norm: if spec.encoder { norm(&w, &b("attn_output_norm"), eps, true)? } else { None },
                out_norm: if spec.encoder { norm(&w, &b("layer_output_norm"), eps, true)? } else { None },
                fused_up,
                n_head,
                n_kv,
                head_dim,
            });
        }
        let out_norm = norm(&w, "output_norm", eps, spec.layer_norm)?;
        let lm_head = if spec.encoder {
            None
        } else if w.info("output.weight").is_some() {
            Some(Linear { w: w.matmul("output.weight")?, b: None, name: "output.weight".into() })
        } else {
            Some(Linear { w: w.matmul("token_embd.weight")?, b: None, name: "token_embd.weight".into() })
        };
        let embd_norm = norm(&w, "token_embd_norm", eps, true)?;
        let head_dim0 = layers.first().map_or(64, |l| l.head_dim);
        let rope_dim = k("rope.dimension_count").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(head_dim0);
        let rope_base = k("rope.freq_base").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;
        let rope_factors = match g.tensor("rope_freqs.weight") {
            Some(t) => {
                let mut v = Vec::new();
                g.dequantize_rows(t, 0, t.n_rows(), &mut v)?;
                Some(v)
            }
            None => None,
        };
        let pooling_key = k("pooling_type").and_then(|v| v.as_u64());
        let pooled = spec.encoder || pooling_key.is_some_and(|p| p > 0);
        let pooling = pooling_key.unwrap_or(1);
        Ok(Self { w, spec, layers, out_norm, lm_head, embd_norm, n_embd, rope_dim, rope_base, rope_factors, pooling, pooled, vocab, top_k })
    }

    /// cos/sin tables [seq, rope_dim/2].
    pub(super) fn rope_tables(&self, seq: usize) -> Result<(Tensor, Tensor)> {
        let half = self.rope_dim / 2;
        let mut inv: Vec<f32> = (0..half).map(|i| self.rope_base.powf(-(2.0 * i as f32) / self.rope_dim as f32)).collect();
        if let Some(f) = &self.rope_factors {
            for (x, fac) in inv.iter_mut().zip(f) {
                *x /= fac;
            }
        }
        let mut cos = Vec::with_capacity(seq * half);
        let mut sin = Vec::with_capacity(seq * half);
        for p in 0..seq {
            for f in &inv {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok((Tensor::from_vec(cos, (seq, half), &self.w.dev)?, Tensor::from_vec(sin, (seq, half), &self.w.dev)?))
    }

    /// x: [seq, heads, head_dim]
    pub(super) fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        if self.spec.rope == RopeKind::None {
            return Ok(x.clone());
        }
        let (seq, h, hd) = x.dims3()?;
        let rd = self.rope_dim.min(hd);
        let half = rd / 2;
        let cos = cos.reshape((seq, 1, half))?;
        let sin = sin.reshape((seq, 1, half))?;
        let rot = x.narrow(2, 0, rd)?;
        let rotated = match self.spec.rope {
            RopeKind::Neox => {
                let x1 = rot.narrow(2, 0, half)?;
                let x2 = rot.narrow(2, half, half)?;
                let a = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
                let b = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
                Tensor::cat(&[a, b], 2)?
            }
            _ => {
                let pairs = rot.reshape((seq, h, half, 2))?;
                let x0 = pairs.narrow(3, 0, 1)?.squeeze(3)?;
                let x1 = pairs.narrow(3, 1, 1)?.squeeze(3)?;
                let a = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
                let b = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
                Tensor::stack(&[a, b], 3)?.reshape((seq, h, rd))?
            }
        };
        if rd < hd {
            Ok(Tensor::cat(&[rotated, x.narrow(2, rd, hd - rd)?], 2)?)
        } else {
            Ok(rotated)
        }
    }

    /// Logits of the output head for hidden states [n, dim] (output norm applied).
    pub(super) fn head_logits(&self, h: &Tensor) -> Result<Option<Tensor>> {
        let Some(head) = &self.lm_head else { return Ok(None) };
        let x = match &self.out_norm {
            Some(n) => n.forward(h)?,
            None => h.clone(),
        };
        Ok(Some(head.forward(&x)?))
    }

    pub(super) fn lens(&self, h_last: &Tensor) -> Result<Vec<LensToken>> {
        Ok(self.lens_full(h_last, &Masks::default())?.0)
    }

    /// Top-k of the logit lens plus the full distribution (token mask applied).
    pub(super) fn lens_full(&self, h_last: &Tensor, masks: &Masks) -> Result<(Vec<LensToken>, Vec<f32>)> {
        let Some(logits) = self.head_logits(h_last)? else { return Ok((vec![], vec![])) };
        let mut l = logits.squeeze(0)?.to_vec1::<f32>()?;
        masks.apply_logits(&mut l);
        let lp = log_softmax(&l);
        let probs: Vec<f32> = lp.iter().map(|x| x.exp() as f32).collect();
        Ok((self.top_tokens(&probs), probs))
    }

    pub(super) fn top_tokens(&self, probs: &[f32]) -> Vec<LensToken> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        let k = self.top_k.min(idx.len());
        idx.select_nth_unstable_by(k.saturating_sub(1), |a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut top: Vec<usize> = idx[..k].to_vec();
        top.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
        top.into_iter()
            .map(|i| LensToken { id: i as u32, text: self.vocab.tokens.get(i).map(|t| t.text.clone()).unwrap_or_default(), prob: probs[i] })
            .collect()
    }

    pub(super) fn forward(&self, ids: &[u32], p: &dyn Progress) -> Result<Sample> {
        Ok(self.run(ids, &Masks::default(), FwdOpt { measure: true, ..Default::default() }, p)?.sample)
    }

    /// Input embeddings for `ids`, with embedding edits applied.
    pub(super) fn embed(&self, ids: &[u32], masks: &Masks) -> Result<Tensor> {
        let h = self.w.rows("token_embd.weight", ids)?;
        if masks.embed.is_empty() || !ids.iter().any(|i| masks.embed.contains_key(i)) {
            return Ok(h);
        }
        let mut v = h.to_vec2::<f32>()?;
        for (i, id) in ids.iter().enumerate() {
            let Some((scale, repl)) = masks.embed.get(id) else { continue };
            if let Some(r) = repl {
                v[i] = row_of(&self.w.rows("token_embd.weight", &[*r])?, 0)?;
            }
            for x in &mut v[i] {
                *x *= scale;
            }
        }
        Ok(Tensor::from_vec(v.concat(), (ids.len(), self.n_embd_rows()), &self.w.dev)?)
    }

    pub(super) fn n_embd_rows(&self) -> usize {
        self.w.info("token_embd.weight").map_or(self.n_embd, |t| t.row_len() as usize)
    }

}

impl ModelTracer for Model<'_> {
    fn trace_tokens(&self, ids: &[u32], p: &dyn Progress) -> Result<Sample> {
        self.forward(ids, p)
    }
}
