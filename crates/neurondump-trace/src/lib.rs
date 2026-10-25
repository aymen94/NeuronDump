//! Runtime tracing with candle: a generic llama-family decoder forward pass and a
//! BERT-family encoder pass, instrumented per layer. Weights are built straight from
//! NeuronDump's memory-mapped GGUF (no second parse).
//!
//! Supported architectures are listed explicitly in [`SUPPORTED`]; adding one means
//! describing its [`ArchSpec`] (or implementing [`ModelTracer`] for a new family).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::{Device, Module, Tensor, D};

use neurondump_core::model::ModelHandle;
use neurondump_core::tokenizer::{Encoder, Vocab};
use neurondump_core::trace::{
    Ablation, CorpusEval, LayerTrace, LensDiff, LensToken, SimPrompt, SimReport, SimRequest, SimSide, SweepEntry, SweepReport, SweepRequest,
    TokenContribution, TraceBackend, TraceReport, TraceRequest,
};
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

thread_local! {
    /// GGUF names of tensors treated as all-zero during the current forward pass.
    static ZEROED: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

fn is_zeroed(name: &str) -> bool {
    ZEROED.with(|z| !z.borrow().is_empty() && z.borrow().contains(name))
}

/// Sets the zeroed-tensor set for the lifetime of the guard.
struct ZeroGuard;

impl ZeroGuard {
    fn new(names: &[String]) -> Self {
        ZEROED.with(|z| *z.borrow_mut() = names.iter().cloned().collect());
        ZeroGuard
    }
}

impl Drop for ZeroGuard {
    fn drop(&mut self) {
        ZEROED.with(|z| z.borrow_mut().clear());
    }
}

struct Linear {
    w: QMatMul,
    b: Option<Tensor>,
    name: String,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.w.forward(x)?;
        if is_zeroed(&self.name) {
            y = y.zeros_like()?;
        }
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
    name: String,
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
        let mut x = x.broadcast_mul(&self.w)?;
        if is_zeroed(&self.name) {
            x = x.zeros_like()?;
        }
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
    lm_head: Option<Linear>,
    embd_norm: Option<Norm>,
    n_embd: usize,
    rope_dim: usize,
    rope_base: f32,
    rope_factors: Option<Vec<f32>>,
    pooling: u64,
    /// Produces a pooled embedding (encoders, or decoders with a pooling type).
    pooled: bool,
    vocab: &'a Vocab,
    top_k: usize,
}

/// Per-pass ablation masks (built from a core [`Ablation`]).
#[derive(Default)]
struct Masks {
    skip: HashSet<usize>,
    zero_attn: HashSet<usize>,
    zero_ffn: HashSet<usize>,
    heads: HashMap<usize, Vec<usize>>,
    neurons: HashMap<usize, Vec<(usize, usize)>>,
    zero_tensors: Vec<String>,
    tokens: Vec<(u32, u32)>,
}

impl Masks {
    fn from(a: &Ablation) -> Self {
        let mut m = Masks { zero_tensors: a.zero_tensors.clone(), tokens: a.mask_tokens.clone(), ..Default::default() };
        m.skip.extend(a.skip_layers.iter().map(|&l| l as usize));
        for (l, b) in &a.zero_blocks {
            match b.as_str() {
                "attention" | "ssm" => m.zero_attn.insert(*l as usize),
                _ => m.zero_ffn.insert(*l as usize),
            };
        }
        for (l, h) in &a.zero_heads {
            m.heads.entry(*l as usize).or_default().push(*h as usize);
        }
        for (l, s, e) in &a.zero_neurons {
            m.neurons.entry(*l as usize).or_default().push((*s as usize, *e as usize));
        }
        m
    }
}

#[derive(Clone, Copy, Default)]
struct FwdOpt {
    /// Per-layer measurements (entropy, neurons, logit lens).
    measure: bool,
    /// Keep the full logit-lens distribution of every layer.
    lens_probs: bool,
    /// Logits for every position (perplexity, KL), as [seq, vocab].
    all_logits: bool,
}

struct FwdOut {
    sample: Sample,
    /// Next-token logits at the last position (token mask applied).
    last_logits: Vec<f32>,
    all_logits: Option<Tensor>,
    lens_probs: Vec<Vec<f32>>,
    pooled: Option<Vec<f32>>,
}

fn mask_logits(logits: &mut [f32], tokens: &[(u32, u32)]) {
    for (a, b) in tokens {
        let (a, b) = (*a as usize, (*b as usize + 1).min(logits.len()));
        if a < b {
            logits[a..b].fill(f32::NEG_INFINITY);
        }
    }
}

/// log-softmax in f64 (stable).
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| ((x as f64) - max).exp()).sum();
    let lse = max + sum.ln();
    logits.iter().map(|&x| x as f64 - lse).collect()
}

const LOG_FLOOR: f64 = -23.025_850_929_940_457; // ln(1e-10)

/// KL(p || q) from log-probabilities; q is floored at 1e-10 so masked tokens stay finite.
fn kl(lp: &[f64], lq: &[f64]) -> f64 {
    lp.iter().zip(lq).filter(|(p, _)| p.is_finite()).map(|(p, q)| p.exp() * (p - q.max(LOG_FLOOR))).sum::<f64>().max(0.0)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

fn norm(w: &Weights, base: &str, eps: f64, layer: bool) -> Result<Option<Norm>> {
    let Some(wt) = w.opt_dense(&format!("{base}.weight"))? else { return Ok(None) };
    Ok(Some(Norm { w: wt, b: w.opt_dense(&format!("{base}.bias"))?, eps, layer, name: format!("{base}.weight") }))
}

fn linear(w: &Weights, base: &str) -> Result<Option<Linear>> {
    let Some(m) = w.opt_matmul(&format!("{base}.weight"))? else { return Ok(None) };
    Ok(Some(Linear { w: m, b: w.opt_dense(&format!("{base}.bias"))?, name: format!("{base}.weight") }))
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

    /// Logits of the output head for hidden states [n, dim] (output norm applied).
    fn head_logits(&self, h: &Tensor) -> Result<Option<Tensor>> {
        let Some(head) = &self.lm_head else { return Ok(None) };
        let x = match &self.out_norm {
            Some(n) => n.forward(h)?,
            None => h.clone(),
        };
        Ok(Some(head.forward(&x)?))
    }

    fn lens(&self, h_last: &Tensor) -> Result<Vec<LensToken>> {
        Ok(self.lens_full(h_last, &[])?.0)
    }

    /// Top-k of the logit lens plus the full distribution (token mask applied).
    fn lens_full(&self, h_last: &Tensor, mask: &[(u32, u32)]) -> Result<(Vec<LensToken>, Vec<f32>)> {
        let Some(logits) = self.head_logits(h_last)? else { return Ok((vec![], vec![])) };
        let mut l = logits.squeeze(0)?.to_vec1::<f32>()?;
        mask_logits(&mut l, mask);
        let lp = log_softmax(&l);
        let probs: Vec<f32> = lp.iter().map(|x| x.exp() as f32).collect();
        Ok((self.top_tokens(&probs), probs))
    }

    fn top_tokens(&self, probs: &[f32]) -> Vec<LensToken> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        let k = self.top_k.min(idx.len());
        idx.select_nth_unstable_by(k.saturating_sub(1), |a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut top: Vec<usize> = idx[..k].to_vec();
        top.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
        top.into_iter()
            .map(|i| LensToken { id: i as u32, text: self.vocab.tokens.get(i).map(|t| t.text.clone()).unwrap_or_default(), prob: probs[i] })
            .collect()
    }

    fn forward(&self, ids: &[u32], p: &dyn Progress) -> Result<Sample> {
        Ok(self.run(ids, &Masks::default(), FwdOpt { measure: true, ..Default::default() }, p)?.sample)
    }

    fn run(&self, ids: &[u32], masks: &Masks, opt: FwdOpt, p: &dyn Progress) -> Result<FwdOut> {
        let _zero = ZeroGuard::new(&masks.zero_tensors);
        let seq = ids.len();
        let mut h = self.w.rows("token_embd.weight", ids)?;
        if is_zeroed("token_embd.weight") {
            h = h.zeros_like()?;
        }
        if self.spec.embed_scale {
            h = (h * (self.n_embd as f64).sqrt())?;
        }
        if self.w.info("pos_embd.weight").is_some() && !is_zeroed("pos_embd.weight") {
            let pos: Vec<u32> = (0..seq as u32).collect();
            h = (h + self.w.rows("pos_embd.weight", &pos)?)?;
        }
        if self.w.info("token_types.weight").is_some() && !is_zeroed("token_types.weight") {
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
        let mut lens_probs = Vec::new();
        let n_layers = self.layers.len();
        let measure_lens = |h: &Tensor, lens_probs: &mut Vec<Vec<f32>>| -> Result<Vec<LensToken>> {
            if opt.lens_probs {
                let (top, probs) = self.lens_full(&h.narrow(0, seq - 1, 1)?, &masks.tokens)?;
                lens_probs.push(probs);
                Ok(top)
            } else {
                self.lens(&h.narrow(0, seq - 1, 1)?)
            }
        };
        for (li, l) in self.layers.iter().enumerate() {
            check_cancel(p)?;
            if opt.measure {
                p.set(50.0 + 45.0 * li as f32 / n_layers as f32, &format!("layer {li}"));
            }
            if masks.skip.contains(&li) {
                // Identity: the residual stream passes through unchanged.
                if opt.measure {
                    let hv = h.to_vec2::<f32>()?;
                    let hidden_norm = hv.iter().map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt()).sum::<f32>() / seq as f32;
                    let logit_lens = measure_lens(&h, &mut lens_probs)?;
                    traces.push(LayerTrace { index: li as u32, hidden_norm, cos_prev: Some(1.0), head_entropy: vec![], top_neurons: vec![], router: vec![], logit_lens });
                    neuron_means.push(vec![]);
                }
                continue;
            }
            let h_in = h.clone();
            let zero_attn = masks.zero_attn.contains(&li);
            let zero_ffn = masks.zero_ffn.contains(&li);

            // --- attention ---
            let mut head_entropy: Vec<f32> = vec![];
            if !zero_attn {
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
            if opt.measure {
                // Mean attention entropy per head over query positions.
                let pv = probs.to_vec3::<f32>()?;
                head_entropy = pv
                    .iter()
                    .map(|rows| {
                        let e: f32 = rows.iter().map(|r| -r.iter().filter(|x| **x > 0.0).map(|x| x * x.ln()).sum::<f32>()).sum();
                        e / rows.len().max(1) as f32
                    })
                    .collect();
            }
            let mut att = probs.matmul(&v)?.transpose(0, 1)?.contiguous()?.reshape((seq, l.n_head * l.head_dim))?;
            if let Some(hs) = masks.heads.get(&li) {
                let mut m = vec![1f32; l.n_head * l.head_dim];
                for &hh in hs.iter().filter(|&&hh| hh < l.n_head) {
                    m[hh * l.head_dim..(hh + 1) * l.head_dim].fill(0.0);
                }
                att = att.broadcast_mul(&Tensor::from_vec(m, (1, l.n_head * l.head_dim), &self.w.dev)?)?;
            }
            let mut att = l.o.forward(&att)?;
            if let Some(n) = &l.attn_post_norm {
                att = n.forward(&att)?;
            }
            h = (h + att)?;
            }
            if let Some(n) = &l.attn_out_norm {
                h = n.forward(&h)?;
            }

            // --- feed-forward ---
            let mut means: Vec<f32> = vec![];
            if !zero_ffn {
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
            let mut inner = inner;
            if let Some(rs) = masks.neurons.get(&li) {
                let n_ff = inner.dim(1)?;
                let mut m = vec![1f32; n_ff];
                for &(s0, e) in rs {
                    if s0 < n_ff {
                        m[s0..(e + 1).min(n_ff)].fill(0.0);
                    }
                }
                inner = inner.broadcast_mul(&Tensor::from_vec(m, (1, n_ff), &self.w.dev)?)?;
            }
            if opt.measure {
                means = inner.abs()?.mean(0)?.to_vec1::<f32>()?;
            }
            let mut ffn = l.down.forward(&inner)?;
            if let Some(n) = &l.ffn_post_norm {
                ffn = n.forward(&ffn)?;
            }
            h = (h + ffn)?;
            }
            if let Some(n) = &l.out_norm {
                h = n.forward(&h)?;
            }

            // --- measurements ---
            if !opt.measure {
                continue;
            }
            let hv = h.to_vec2::<f32>()?;
            let iv = h_in.to_vec2::<f32>()?;
            let hidden_norm = hv.iter().map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt()).sum::<f32>() / seq as f32;
            let cos_prev = iv.iter().zip(&hv).map(|(a, b)| cosine(a, b)).sum::<f32>() / seq as f32;
            let mut order: Vec<usize> = (0..means.len()).collect();
            order.sort_by(|a, b| means[*b].partial_cmp(&means[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let top_neurons = order.iter().take(self.top_k).map(|&i| (i as u32, means[i])).collect();
            let logit_lens = measure_lens(&h, &mut lens_probs)?;
            traces.push(LayerTrace { index: li as u32, hidden_norm, cos_prev: Some(cos_prev), head_entropy, top_neurons, router: vec![], logit_lens });
            neuron_means.push(means);
        }

        let mut contributions = Vec::new();
        let pooled = if self.pooled {
            let hv = h.to_vec2::<f32>()?;
            Some(match self.pooling {
                2 => hv[0].clone(),
                3 => hv[seq - 1].clone(),
                _ => (0..self.n_embd).map(|j| hv.iter().map(|r| r[j]).sum::<f32>() / seq as f32).collect(),
            })
        } else {
            None
        };
        let prediction = if self.spec.encoder {
            let hv = h.to_vec2::<f32>()?;
            let pooled = pooled.clone().unwrap_or_default();
            for (i, r) in hv.iter().enumerate() {
                contributions.push(TokenContribution {
                    id: ids[i],
                    text: self.vocab.tokens.get(ids[i] as usize).map(|t| t.text.clone()).unwrap_or_default(),
                    contribution: cosine(r, &pooled),
                });
            }
            vec![]
        } else if opt.measure {
            self.lens(&h.narrow(0, seq - 1, 1)?)?
        } else {
            vec![]
        };
        let mut last_logits = Vec::new();
        let mut all_logits = None;
        if self.lm_head.is_some() {
            if opt.all_logits {
                let lg = self.head_logits(&h)?.ok_or_else(|| anyhow!("no output head"))?;
                let mut last = lg.narrow(0, seq - 1, 1)?.squeeze(0)?.to_vec1::<f32>()?;
                mask_logits(&mut last, &masks.tokens);
                last_logits = last;
                all_logits = Some(lg);
            } else if let Some(lg) = self.head_logits(&h.narrow(0, seq - 1, 1)?)? {
                last_logits = lg.squeeze(0)?.to_vec1::<f32>()?;
                mask_logits(&mut last_logits, &masks.tokens);
            }
        }
        Ok(FwdOut {
            sample: Sample { tokens: ids.to_vec(), layers: traces, prediction, neuron_means, contributions },
            last_logits,
            all_logits,
            lens_probs,
            pooled,
        })
    }

    /// Greedy continuation; returns (ids, text, tokens/s).
    fn generate(&self, ids: &[u32], masks: &Masks, n: usize, eos: Option<u32>, p: &dyn Progress) -> Result<(Vec<u32>, String, f64)> {
        let mut cur = ids.to_vec();
        let mut out = Vec::new();
        let t = Instant::now();
        for _ in 0..n {
            check_cancel(p)?;
            let f = self.run(&cur, masks, FwdOpt::default(), p)?;
            if f.last_logits.is_empty() {
                break;
            }
            let next = argmax(&f.last_logits) as u32;
            out.push(next);
            cur.push(next);
            if Some(next) == eos {
                break;
            }
        }
        let secs = t.elapsed().as_secs_f64().max(1e-9);
        let text: String = out.iter().map(|i| self.vocab.tokens.get(*i as usize).map(|t| t.text.as_str()).unwrap_or("")).collect();
        let tps = if out.is_empty() { 0.0 } else { out.len() as f64 / secs };
        Ok((out, text, tps))
    }

    /// Per-position log-probabilities [seq-1][vocab] predicting the next token.
    fn position_logprobs(&self, f: &FwdOut, masks: &Masks) -> Result<Vec<Vec<f64>>> {
        let Some(lg) = &f.all_logits else { return Ok(vec![]) };
        let rows = lg.to_vec2::<f32>()?;
        let n = rows.len().saturating_sub(1);
        Ok(rows
            .into_iter()
            .take(n)
            .map(|mut r| {
                mask_logits(&mut r, &masks.tokens);
                log_softmax(&r)
            })
            .collect())
    }
}

/// Accumulated corpus metrics for one configuration.
#[derive(Default, Clone)]
struct EvalAcc {
    nll: f64,
    kl: f64,
    agree: usize,
    positions: usize,
}

impl EvalAcc {
    fn add(&mut self, ids: &[u32], base: &[Vec<f64>], abl: &[Vec<f64>]) {
        for (t, (lp, lq)) in base.iter().zip(abl).enumerate() {
            let target = ids[t + 1] as usize;
            self.nll -= lq.get(target).copied().unwrap_or(LOG_FLOOR).max(LOG_FLOOR);
            self.kl += kl(lp, lq);
            let am = |v: &[f64]| v.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
            if am(lp) == am(lq) {
                self.agree += 1;
            }
            self.positions += 1;
        }
    }

    fn ppl(&self) -> Option<f64> {
        (self.positions > 0).then(|| (self.nll / self.positions as f64).exp()).filter(|v| v.is_finite())
    }

    fn mean_kl(&self) -> f64 {
        if self.positions == 0 {
            0.0
        } else {
            self.kl / self.positions as f64
        }
    }

    fn agreement(&self) -> f64 {
        if self.positions == 0 {
            1.0
        } else {
            self.agree as f64 / self.positions as f64
        }
    }
}

fn cosine64(a: &[f32], b: &[f32]) -> f32 {
    cosine(a, b)
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

    fn simulate(&self, m: &ModelHandle, req: &SimRequest, p: &dyn Progress) -> Result<SimReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let top_k = if req.top_k == 0 { 5 } else { req.top_k };
        let max_tokens = if req.max_tokens == 0 { 128 } else { req.max_tokens };
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, top_k, p)?;
        let base = Masks::default();
        let abl = Masks::from(&req.ablation);
        let mut warnings: Vec<String> = enc.warnings.clone();
        if !req.ablation.zero_experts.is_empty() {
            warnings.push("expert ablation is not supported by this forward pass; ignored".into());
        }
        let eos = m.gguf.get_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        let encode = |t: &str, warnings: &mut Vec<String>| -> Vec<u32> {
            let mut ids = enc.encode(t, true, None).ids;
            if ids.len() > max_tokens {
                warnings.push(format!("input truncated from {} to {max_tokens} tokens", ids.len()));
                ids.truncate(max_tokens);
            }
            ids
        };
        let n_steps = (req.prompts.len() * 2 + req.corpus.len()).max(1) as f32;
        let mut step = 0f32;
        let mut prompts = Vec::new();
        for text in &req.prompts {
            check_cancel(p)?;
            let ids = encode(text, &mut warnings);
            if ids.is_empty() {
                continue;
            }
            p.set(10.0 + 85.0 * step / n_steps, "baseline");
            step += 1.0;
            let opt = FwdOpt { measure: true, lens_probs: true, all_logits: false };
            let b = model.run(&ids, &base, opt, p)?;
            p.set(10.0 + 85.0 * step / n_steps, "ablated");
            step += 1.0;
            let a = model.run(&ids, &abl, opt, p)?;
            let (kl_v, top1_changed, btop, atop) = if b.last_logits.is_empty() {
                (None, false, vec![], vec![])
            } else {
                let lb = log_softmax(&b.last_logits);
                let la = log_softmax(&a.last_logits);
                let pb: Vec<f32> = lb.iter().map(|x| x.exp() as f32).collect();
                let pa: Vec<f32> = la.iter().map(|x| x.exp() as f32).collect();
                (Some(kl(&lb, &la)), argmax(&b.last_logits) != argmax(&a.last_logits), model.top_tokens(&pb), model.top_tokens(&pa))
            };
            let target = btop.first().map(|t| t.id as usize);
            let lens: Vec<LensDiff> = (0..b.sample.layers.len())
                .map(|li| {
                    let tp = |v: &Vec<Vec<f32>>| target.and_then(|t| v.get(li).and_then(|r| r.get(t).copied())).unwrap_or(0.0);
                    LensDiff {
                        layer: li as u32,
                        baseline: b.sample.layers[li].logit_lens.first().cloned(),
                        ablated: a.sample.layers.get(li).and_then(|l| l.logit_lens.first().cloned()),
                        baseline_target_prob: tp(&b.lens_probs),
                        ablated_target_prob: tp(&a.lens_probs),
                        skipped: abl.skip.contains(&li),
                    }
                })
                .collect();
            let (bg, ag) = if req.gen_tokens > 0 && !b.last_logits.is_empty() {
                p.set(10.0 + 85.0 * step / n_steps, "generating");
                (model.generate(&ids, &base, req.gen_tokens, eos, p)?, model.generate(&ids, &abl, req.gen_tokens, eos, p)?)
            } else {
                let t0 = Instant::now();
                model.run(&ids, &base, FwdOpt::default(), p)?;
                let tb = ids.len() as f64 / t0.elapsed().as_secs_f64().max(1e-9);
                let t1 = Instant::now();
                model.run(&ids, &abl, FwdOpt::default(), p)?;
                let ta = ids.len() as f64 / t1.elapsed().as_secs_f64().max(1e-9);
                ((vec![], String::new(), tb), (vec![], String::new(), ta))
            };
            let norm = |v: &Option<Vec<f32>>| v.as_ref().map(|x| x.iter().map(|y| y * y).sum::<f32>().sqrt());
            prompts.push(SimPrompt {
                prompt: text.clone(),
                n_tokens: ids.len(),
                baseline: SimSide { top_k: btop, generated: bg.1, generated_tokens: bg.0.len(), tokens_per_s: bg.2, embedding_norm: norm(&b.pooled) },
                ablated: SimSide { top_k: atop, generated: ag.1, generated_tokens: ag.0.len(), tokens_per_s: ag.2, embedding_norm: norm(&a.pooled) },
                kl: kl_v,
                top1_changed,
                embedding_cosine: match (&b.pooled, &a.pooled) {
                    (Some(x), Some(y)) => Some(cosine64(x, y)),
                    _ => None,
                },
                lens,
            });
        }
        let mut corpus = None;
        if !req.corpus.is_empty() && model.lm_head.is_some() {
            let mut base_acc = EvalAcc::default();
            let mut acc = EvalAcc::default();
            let mut n = 0;
            for text in &req.corpus {
                check_cancel(p)?;
                p.set(10.0 + 85.0 * step / n_steps, "corpus");
                step += 1.0;
                let ids = encode(text, &mut warnings);
                if ids.len() < 2 {
                    continue;
                }
                let opt = FwdOpt { all_logits: true, ..Default::default() };
                let b = model.run(&ids, &base, opt, p)?;
                let bl = model.position_logprobs(&b, &base)?;
                let a = model.run(&ids, &abl, opt, p)?;
                let al = model.position_logprobs(&a, &abl)?;
                base_acc.add(&ids, &bl, &bl);
                acc.add(&ids, &bl, &al);
                n += 1;
            }
            corpus = Some(CorpusEval {
                samples: n,
                tokens: acc.positions,
                baseline_ppl: base_acc.ppl(),
                ablated_ppl: acc.ppl(),
                mean_kl: (acc.positions > 0).then(|| acc.mean_kl()),
                top1_agreement: acc.agreement(),
            });
        }
        let total: u64 = m.gguf.tensors.iter().map(|t| t.n_bytes).sum();
        let skipped: u64 = m
            .gguf
            .tensors
            .iter()
            .filter(|t| neurondump_core::arch::parse_tensor_name(&t.name).layer.is_some_and(|l| abl.skip.contains(&(l as usize))))
            .map(|t| t.n_bytes)
            .sum();
        p.set(100.0, "done");
        warnings.push("in-memory masks: process memory is the same for both runs; weight bytes are what a slimmed file would hold".into());
        if req.gen_tokens > 0 {
            warnings.push("generation recomputes the whole sequence for every token (no KV cache, CPU): tokens/s is only for comparing baseline and ablated".into());
        }
        Ok(SimReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            architecture: m.architecture().to_string(),
            backend: self.name().into(),
            cut: req.cut.clone(),
            ablation: req.ablation.clone(),
            prompts,
            corpus,
            weights_bytes_baseline: total,
            weights_bytes_ablated: total - skipped,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn sweep(&self, m: &ModelHandle, req: &SweepRequest, p: &dyn Progress) -> Result<SweepReport> {
        let started = Instant::now();
        let vocab = Vocab::from_gguf(&m.gguf)?;
        let enc = Encoder::new(&vocab)?;
        let max_tokens = req.max_tokens.unwrap_or(64).max(2);
        p.set(1.0, "loading weights");
        let model = Model::load(m, &vocab, 5, p)?;
        if model.lm_head.is_none() {
            bail!("sweeps measure next-token predictions; this model has no output head (embedding model)");
        }
        let mut warnings: Vec<String> = enc.warnings.clone();
        let masks: Vec<Masks> = req.targets.iter().map(|t| Masks::from(&t.ablation)).collect();
        let base = Masks::default();
        let mut base_acc = EvalAcc::default();
        let mut accs = vec![EvalAcc::default(); req.targets.len()];
        let texts: Vec<Vec<u32>> = req
            .corpus
            .iter()
            .map(|t| {
                let mut ids = enc.encode(t, true, None).ids;
                if ids.len() > max_tokens {
                    warnings.push(format!("sample truncated from {} to {max_tokens} tokens", ids.len()));
                    ids.truncate(max_tokens);
                }
                ids
            })
            .filter(|ids| ids.len() >= 2)
            .collect();
        if texts.is_empty() {
            bail!("the corpus produced no samples of at least 2 tokens");
        }
        let total = (texts.len() * (req.targets.len() + 1)) as f32;
        let mut done = 0f32;
        let opt = FwdOpt { all_logits: true, ..Default::default() };
        for ids in &texts {
            check_cancel(p)?;
            p.set(5.0 + 94.0 * done / total, "baseline");
            let b = model.run(ids, &base, opt, p)?;
            let bl = model.position_logprobs(&b, &base)?;
            base_acc.add(ids, &bl, &bl);
            done += 1.0;
            for (i, mk) in masks.iter().enumerate() {
                check_cancel(p)?;
                p.set(5.0 + 94.0 * done / total, &format!("{} ({}/{})", req.targets[i].label, i + 1, masks.len()));
                let a = model.run(ids, mk, opt, p)?;
                let al = model.position_logprobs(&a, mk)?;
                accs[i].add(ids, &bl, &al);
                done += 1.0;
            }
        }
        let base_ppl = base_acc.ppl();
        let mut entries: Vec<SweepEntry> = req
            .targets
            .iter()
            .zip(&accs)
            .map(|(t, a)| SweepEntry {
                id: t.id.clone(),
                label: t.label.clone(),
                mean_kl: a.mean_kl(),
                ppl: a.ppl(),
                ppl_increase: match (a.ppl(), base_ppl) {
                    (Some(x), Some(b)) if b > 0.0 => Some(x / b - 1.0),
                    _ => None,
                },
                top1_agreement: a.agreement(),
                rank: 0,
            })
            .collect();
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|a, b| entries[*b].mean_kl.partial_cmp(&entries[*a].mean_kl).unwrap_or(std::cmp::Ordering::Equal));
        for (r, &i) in order.iter().enumerate() {
            entries[i].rank = r + 1;
        }
        p.set(100.0, "done");
        warnings.dedup();
        Ok(SweepReport {
            schema_version: SCHEMA_VERSION,
            model: m.entry.name.clone(),
            digest: m.entry.digest.clone(),
            architecture: m.architecture().to_string(),
            level: req.level.clone(),
            layer: req.layer,
            samples: texts.len(),
            tokens: base_acc.positions,
            baseline_ppl: base_ppl,
            ranking: order.iter().map(|&i| entries[i].id.clone()).collect(),
            entries,
            warnings,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}
