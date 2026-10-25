//! Runtime tracing with candle: a generic llama-family decoder forward pass and a
//! BERT-family encoder pass, instrumented per layer. Weights are built straight from
//! NeuronDump's memory-mapped GGUF (no second parse).
//!
//! Supported architectures are listed explicitly in [`SUPPORTED`]; adding one means
//! describing its [`ArchSpec`] (or implementing [`ModelTracer`] for a new family).

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::{Device, Module, Tensor, D};

use neurondump_core::model::ModelHandle;
use neurondump_core::tokenizer::{Encoder, Vocab};
use neurondump_core::trace::{LayerTrace, LensToken, TokenContribution, TraceBackend, TraceReport, TraceRequest};
use neurondump_core::util::{check_cancel, Progress, SCHEMA_VERSION};
use neurondump_gguf::{GgmlType, GgufFile, TensorInfo};

/// Architectures with a known-good forward pass.
pub const SUPPORTED: &[&str] = &["llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3", "bert", "nomic-bert"];

#[derive(Debug, Clone, Copy, PartialEq)]
enum RopeKind {
    /// Adjacent pairs (llama.cpp LLAMA_ROPE_TYPE_NORM).
    Norm,
    /// Split halves (LLAMA_ROPE_TYPE_NEOX).
    Neox,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Act {
    Silu,
    Gelu,
}

/// Architecture description driving the generic forward passes.
#[derive(Debug, Clone)]
struct ArchSpec {
    encoder: bool,
    rope: RopeKind,
    act: Act,
    embed_scale: bool,
    layer_norm: bool,
}

fn spec_for(arch: &str) -> Option<ArchSpec> {
    Some(match arch {
        "llama" => ArchSpec { encoder: false, rope: RopeKind::Norm, act: Act::Silu, embed_scale: false, layer_norm: false },
        "qwen2" | "qwen3" | "phi3" => ArchSpec { encoder: false, rope: RopeKind::Neox, act: Act::Silu, embed_scale: false, layer_norm: false },
        "gemma" | "gemma2" | "gemma3" => ArchSpec { encoder: false, rope: RopeKind::Neox, act: Act::Gelu, embed_scale: true, layer_norm: false },
        "nomic-bert" => ArchSpec { encoder: true, rope: RopeKind::Neox, act: Act::Silu, embed_scale: false, layer_norm: true },
        "bert" => ArchSpec { encoder: true, rope: RopeKind::None, act: Act::Gelu, embed_scale: false, layer_norm: true },
        _ => return None,
    })
}

/// Extension point for architectures that do not fit the generic passes.
pub trait ModelTracer {
    fn trace_tokens(&self, ids: &[u32], p: &dyn Progress) -> Result<Sample>;
}

fn candle_dtype(t: GgmlType) -> Option<GgmlDType> {
    Some(match t.0 {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        _ => return None,
    })
}

struct Weights<'a> {
    g: &'a GgufFile,
    dev: Device,
}

impl<'a> Weights<'a> {
    fn info(&self, name: &str) -> Option<&'a TensorInfo> {
        self.g.tensor(name)
    }

    /// Dense f32 tensor in candle order (dims reversed).
    fn dense(&self, name: &str) -> Result<Tensor> {
        let t = self.info(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let mut v = Vec::new();
        self.g.dequantize_rows(t, 0, t.n_rows(), &mut v)?;
        let dims: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        Ok(Tensor::from_vec(v, dims, &self.dev)?)
    }

    fn opt_dense(&self, name: &str) -> Result<Option<Tensor>> {
        if self.info(name).is_some() {
            self.dense(name).map(Some)
        } else {
            Ok(None)
        }
    }

    fn matmul(&self, name: &str) -> Result<QMatMul> {
        let t = self.info(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let dims: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        match candle_dtype(t.ggml_type) {
            Some(dt) => {
                let q = qtensor_from_ggml(dt, self.g.info_bytes(t)?, dims, &self.dev)?;
                Ok(QMatMul::from_qtensor(q)?)
            }
            None => Ok(QMatMul::Tensor(self.dense(name)?)),
        }
    }

    fn opt_matmul(&self, name: &str) -> Result<Option<QMatMul>> {
        if self.info(name).is_some() {
            self.matmul(name).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Rows of an embedding matrix for the given ids, as [n, dim].
    fn rows(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        let t = self.info(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let dim = t.row_len() as usize;
        let mut out = Vec::with_capacity(ids.len() * dim);
        let mut row = Vec::new();
        for &id in ids {
            self.g.dequantize_rows(t, id as u64, 1, &mut row)?;
            out.extend_from_slice(&row);
        }
        Ok(Tensor::from_vec(out, (ids.len(), dim), &self.dev)?)
    }
}

struct Linear {
    w: QMatMul,
    b: Option<Tensor>,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.w.forward(x)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

struct Norm {
    w: Tensor,
    b: Option<Tensor>,
    eps: f64,
    layer: bool,
}

impl Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.layer {
            let mean = x.mean_keepdim(D::Minus1)?;
            let xc = x.broadcast_sub(&mean)?;
            let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
            xc.broadcast_div(&(var + self.eps)?.sqrt()?)?
        } else {
            let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
            x.broadcast_div(&(ms + self.eps)?.sqrt()?)?
        };
        let x = x.broadcast_mul(&self.w)?;
        Ok(match &self.b {
            Some(b) => x.broadcast_add(b)?,
            None => x,
        })
    }
}

struct Layer {
    attn_norm: Option<Norm>,
    q: Option<Linear>,
    k: Option<Linear>,
    v: Option<Linear>,
    qkv: Option<Linear>,
    q_norm: Option<Norm>,
    k_norm: Option<Norm>,
    o: Linear,
    attn_post_norm: Option<Norm>,
    ffn_norm: Option<Norm>,
    gate: Option<Linear>,
    up: Linear,
    down: Linear,
    ffn_post_norm: Option<Norm>,
    /// Encoder post-norms (BERT style).
    attn_out_norm: Option<Norm>,
    out_norm: Option<Norm>,
    /// `ffn_up` holds gate and up stacked (phi3): first half is the gate.
    fused_up: bool,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
}

pub struct Sample {
    pub tokens: Vec<u32>,
    pub layers: Vec<LayerTrace>,
    pub prediction: Vec<LensToken>,
    pub neuron_means: Vec<Vec<f32>>,
    pub contributions: Vec<TokenContribution>,
}

struct Model<'a> {
    w: Weights<'a>,
    spec: ArchSpec,
    layers: Vec<Layer>,
    out_norm: Option<Norm>,
    lm_head: Option<QMatMul>,
    embd_norm: Option<Norm>,
    n_embd: usize,
    rope_dim: usize,
    rope_base: f32,
    rope_factors: Option<Vec<f32>>,
    pooling: u64,
    vocab: &'a Vocab,
    top_k: usize,
}

fn norm(w: &Weights, base: &str, eps: f64, layer: bool) -> Result<Option<Norm>> {
    let Some(wt) = w.opt_dense(&format!("{base}.weight"))? else { return Ok(None) };
    Ok(Some(Norm { w: wt, b: w.opt_dense(&format!("{base}.bias"))?, eps, layer }))
}

fn linear(w: &Weights, base: &str) -> Result<Option<Linear>> {
    let Some(m) = w.opt_matmul(&format!("{base}.weight"))? else { return Ok(None) };
    Ok(Some(Linear { w: m, b: w.opt_dense(&format!("{base}.bias"))? }))
}

fn req<T>(v: Option<T>, what: &str) -> Result<T> {
    v.ok_or_else(|| anyhow!("missing tensor {what}"))
}

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    Ok(e.broadcast_div(&s)?)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    if aa > 0.0 && bb > 0.0 {
        (ab / (aa.sqrt() * bb.sqrt())) as f32
    } else {
        0.0
    }
}

impl<'a> Model<'a> {
    fn load(m: &'a ModelHandle, vocab: &'a Vocab, top_k: usize, p: &dyn Progress) -> Result<Self> {
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
            Some(w.matmul("output.weight")?)
        } else {
            Some(w.matmul("token_embd.weight")?)
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
        let pooling = k("pooling_type").and_then(|v| v.as_u64()).unwrap_or(1);
        Ok(Self { w, spec, layers, out_norm, lm_head, embd_norm, n_embd, rope_dim, rope_base, rope_factors, pooling, vocab, top_k })
    }

    /// cos/sin tables [seq, rope_dim/2].
    fn rope_tables(&self, seq: usize) -> Result<(Tensor, Tensor)> {
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
    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
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

    fn lens(&self, h_last: &Tensor) -> Result<Vec<LensToken>> {
        let Some(head) = &self.lm_head else { return Ok(vec![]) };
        let x = match &self.out_norm {
            Some(n) => n.forward(h_last)?,
            None => h_last.clone(),
        };
        let logits = head.forward(&x)?;
        let probs = softmax_last(&logits)?.squeeze(0)?.to_vec1::<f32>()?;
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        let k = self.top_k.min(idx.len());
        idx.select_nth_unstable_by(k.saturating_sub(1), |a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut top: Vec<usize> = idx[..k].to_vec();
        top.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
        Ok(top
            .into_iter()
            .map(|i| LensToken { id: i as u32, text: self.vocab.tokens.get(i).map(|t| t.text.clone()).unwrap_or_default(), prob: probs[i] })
            .collect())
    }

    fn forward(&self, ids: &[u32], p: &dyn Progress) -> Result<Sample> {
        let seq = ids.len();
        let mut h = self.w.rows("token_embd.weight", ids)?;
        if self.spec.embed_scale {
            h = (h * (self.n_embd as f64).sqrt())?;
        }
        if self.w.info("pos_embd.weight").is_some() {
            let pos: Vec<u32> = (0..seq as u32).collect();
            h = (h + self.w.rows("pos_embd.weight", &pos)?)?;
        }
        if self.w.info("token_types.weight").is_some() {
            h = h.broadcast_add(&self.w.rows("token_types.weight", &[0])?)?;
        }
        if let Some(n) = &self.embd_norm {
            h = n.forward(&h)?;
        }
        let (cos, sin) = self.rope_tables(seq)?;
        let causal = !self.spec.encoder;
        let mask = if causal && seq > 1 {
            let m: Vec<f32> = (0..seq * seq).map(|i| if i % seq > i / seq { f32::NEG_INFINITY } else { 0.0 }).collect();
            Some(Tensor::from_vec(m, (seq, seq), &self.w.dev)?)
        } else {
            None
        };

        let mut traces = Vec::new();
        let mut neuron_means = Vec::new();
        let n_layers = self.layers.len();
        for (li, l) in self.layers.iter().enumerate() {
            check_cancel(p)?;
            p.set(50.0 + 45.0 * li as f32 / n_layers as f32, &format!("layer {li}"));
            let h_in = h.clone();

            // --- attention ---
            let x = match &l.attn_norm {
                Some(n) if !self.spec.encoder => n.forward(&h)?,
                _ => h.clone(),
            };
            let (q, k, v) = if let Some(qkv) = &l.qkv {
                let y = qkv.forward(&x)?;
                let qd = l.n_head * l.head_dim;
                let kd = l.n_kv * l.head_dim;
                (y.narrow(1, 0, qd)?, y.narrow(1, qd, kd)?, y.narrow(1, qd + kd, kd)?)
            } else {
                (req(l.q.as_ref(), "attn_q")?.forward(&x)?, req(l.k.as_ref(), "attn_k")?.forward(&x)?, req(l.v.as_ref(), "attn_v")?.forward(&x)?)
            };
            let mut q = q.reshape((seq, l.n_head, l.head_dim))?;
            let mut k = k.reshape((seq, l.n_kv, l.head_dim))?;
            let v = v.reshape((seq, l.n_kv, l.head_dim))?;
            if let Some(n) = &l.q_norm {
                q = n.forward(&q)?;
            }
            if let Some(n) = &l.k_norm {
                k = n.forward(&k)?;
            }
            let q = self.rope(&q, &cos, &sin)?.transpose(0, 1)?.contiguous()?;
            let k = self.rope(&k, &cos, &sin)?.transpose(0, 1)?.contiguous()?;
            let v = v.transpose(0, 1)?.contiguous()?;
            let group = l.n_head / l.n_kv.max(1);
            let (k, v) = if group > 1 {
                let rep = |t: &Tensor| -> Result<Tensor> {
                    Ok(t.unsqueeze(1)?.broadcast_as((l.n_kv, group, seq, l.head_dim))?.contiguous()?.reshape((l.n_head, seq, l.head_dim))?)
                };
                (rep(&k)?, rep(&v)?)
            } else {
                (k, v)
            };
            let mut scores = (q.matmul(&k.t()?.contiguous()?)? / (l.head_dim as f64).sqrt())?;
            if let Some(m) = &mask {
                scores = scores.broadcast_add(m)?;
            }
            let probs = softmax_last(&scores)?;
            // Mean attention entropy per head over query positions.
            let pv = probs.to_vec3::<f32>()?;
            let head_entropy: Vec<f32> = pv
                .iter()
                .map(|rows| {
                    let e: f32 = rows.iter().map(|r| -r.iter().filter(|x| **x > 0.0).map(|x| x * x.ln()).sum::<f32>()).sum();
                    e / rows.len().max(1) as f32
                })
                .collect();
            let att = probs.matmul(&v)?.transpose(0, 1)?.contiguous()?.reshape((seq, l.n_head * l.head_dim))?;
            let mut att = l.o.forward(&att)?;
            if let Some(n) = &l.attn_post_norm {
                att = n.forward(&att)?;
            }
            h = (h + att)?;
            if let Some(n) = &l.attn_out_norm {
                h = n.forward(&h)?;
            }

            // --- feed-forward ---
            let x = match &l.ffn_norm {
                Some(n) => n.forward(&h)?,
                None => h.clone(),
            };
            let up = l.up.forward(&x)?;
            let act = |t: &Tensor| -> Result<Tensor> {
                Ok(match self.spec.act {
                    Act::Silu => t.silu()?,
                    Act::Gelu => t.gelu()?,
                })
            };
            let inner = match &l.gate {
                Some(g) => (act(&g.forward(&x)?)? * up)?,
                None if l.fused_up => {
                    let n = up.dim(1)? / 2;
                    (act(&up.narrow(1, 0, n)?)? * up.narrow(1, n, n)?)?
                }
                None => act(&up)?,
            };
            let means: Vec<f32> = inner.abs()?.mean(0)?.to_vec1::<f32>()?;
            let mut ffn = l.down.forward(&inner)?;
            if let Some(n) = &l.ffn_post_norm {
                ffn = n.forward(&ffn)?;
            }
            h = (h + ffn)?;
            if let Some(n) = &l.out_norm {
                h = n.forward(&h)?;
            }

            // --- measurements ---
            let hv = h.to_vec2::<f32>()?;
            let iv = h_in.to_vec2::<f32>()?;
            let hidden_norm = hv.iter().map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt()).sum::<f32>() / seq as f32;
            let cos_prev = iv.iter().zip(&hv).map(|(a, b)| cosine(a, b)).sum::<f32>() / seq as f32;
            let mut order: Vec<usize> = (0..means.len()).collect();
            order.sort_by(|a, b| means[*b].partial_cmp(&means[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let top_neurons = order.iter().take(self.top_k).map(|&i| (i as u32, means[i])).collect();
            let logit_lens = self.lens(&h.narrow(0, seq - 1, 1)?)?;
            traces.push(LayerTrace { index: li as u32, hidden_norm, cos_prev: Some(cos_prev), head_entropy, top_neurons, router: vec![], logit_lens });
            neuron_means.push(means);
        }

        let mut contributions = Vec::new();
        let prediction = if self.spec.encoder {
            let hv = h.to_vec2::<f32>()?;
            let pooled: Vec<f32> = if self.pooling == 2 {
                hv[0].clone()
            } else {
                (0..self.n_embd).map(|j| hv.iter().map(|r| r[j]).sum::<f32>() / seq as f32).collect()
            };
            for (i, r) in hv.iter().enumerate() {
                contributions.push(TokenContribution {
                    id: ids[i],
                    text: self.vocab.tokens.get(ids[i] as usize).map(|t| t.text.clone()).unwrap_or_default(),
                    contribution: cosine(r, &pooled),
                });
            }
            vec![]
        } else {
            self.lens(&h.narrow(0, seq - 1, 1)?)?
        };
        Ok(Sample { tokens: ids.to_vec(), layers: traces, prediction, neuron_means, contributions })
    }
}

impl ModelTracer for Model<'_> {
    fn trace_tokens(&self, ids: &[u32], p: &dyn Progress) -> Result<Sample> {
        self.forward(ids, p)
    }
}

pub struct CandleBackend;

impl CandleBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceBackend for CandleBackend {
    fn name(&self) -> &str {
        "candle-cpu"
    }

    fn supported_architectures(&self) -> Vec<String> {
        SUPPORTED.iter().map(|s| s.to_string()).collect()
    }

    fn trace(&self, m: &ModelHandle, req: &TraceRequest, p: &dyn Progress) -> Result<TraceReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let top_k = req.top_k.unwrap_or(5).max(1);
        let max_tokens = req.max_tokens.unwrap_or(128).max(1);
        let mut texts: Vec<String> = req.prompt.iter().cloned().collect();
        texts.extend(req.corpus.iter().cloned());
        if texts.is_empty() {
            bail!("nothing to trace");
        }
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, top_k, p)?;
        let mut warnings: Vec<String> = enc.warnings.clone();
        if m.architecture() == "gemma2" {
            warnings.push("gemma2 logit/attention soft-capping is not applied".into());
        }

        let mut samples = Vec::new();
        for (i, t) in texts.iter().enumerate() {
            check_cancel(p)?;
            let mut ids = enc.encode(t, true, None).ids;
            if ids.is_empty() {
                continue;
            }
            if ids.len() > max_tokens {
                warnings.push(format!("sample {i}: truncated from {} to {max_tokens} tokens", ids.len()));
                ids.truncate(max_tokens);
            }
            samples.push(model.trace_tokens(&ids, p)?);
        }
        let Some(last) = samples.last() else { bail!("inputs produced no tokens") };
        let n = samples.len();
        // Aggregate: mean of layer metrics over samples; lens/prediction from the last one.
        let mut layers = last.layers.clone();
        if n > 1 {
            for (li, l) in layers.iter_mut().enumerate() {
                l.hidden_norm = samples.iter().map(|s| s.layers[li].hidden_norm).sum::<f32>() / n as f32;
                l.cos_prev = Some(samples.iter().filter_map(|s| s.layers[li].cos_prev).sum::<f32>() / n as f32);
                for (h, e) in l.head_entropy.iter_mut().enumerate() {
                    *e = samples.iter().map(|s| s.layers[li].head_entropy.get(h).copied().unwrap_or(0.0)).sum::<f32>() / n as f32;
                }
                let dim = last.neuron_means[li].len();
                let agg: Vec<f32> = (0..dim).map(|j| samples.iter().map(|s| s.neuron_means[li][j]).sum::<f32>() / n as f32).collect();
                let mut order: Vec<usize> = (0..dim).collect();
                order.sort_by(|a, b| agg[*b].partial_cmp(&agg[*a]).unwrap_or(std::cmp::Ordering::Equal));
                l.top_neurons = order.iter().take(top_k).map(|&i| (i as u32, agg[i])).collect();
            }
        }
        let mut hm: HashMap<u32, String> = HashMap::new();
        for &id in &last.tokens {
            hm.entry(id).or_insert_with(|| vocab.tokens.get(id as usize).map(|t| t.text.clone()).unwrap_or_default());
        }
        p.set(100.0, "done");
        Ok(TraceReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            backend: self.name().into(),
            supported: true,
            tokens: last.tokens.iter().map(|id| (*id, hm[id].clone())).collect(),
            layers,
            prediction: last.prediction.clone(),
            embedding_contributions: last.contributions.clone(),
            samples: n,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}
