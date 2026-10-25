//! Token inspector, static part (no inference): tokenization with the chat template,
//! per-token embedding row statistics and nearest neighbours, and the 3D embedding map
//! (token_embd rows projected with PCA).

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use neurondump_gguf::{GgufFile, TensorInfo};

use crate::chat::{render_chat, ChatRender, ChatRequest};
use crate::model::ModelHandle;
use crate::tokenizer::{Encoder, TokenClass, TokenType, Vocab};
use crate::util::{check_cancel, Progress, SCHEMA_VERSION};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
    /// Wrap the text in the model's chat template first.
    #[serde(default)]
    pub chat: bool,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    /// Nearest neighbours per token (0 = skip the embedding scan).
    #[serde(default = "default_neighbors")]
    pub neighbors: usize,
}

fn default_neighbors() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbor {
    pub id: u32,
    pub text: String,
    pub cosine: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenRow {
    pub index: usize,
    pub id: u32,
    pub raw: String,
    pub text: String,
    pub class: TokenClass,
    pub token_type: TokenType,
    /// Control / user-defined token (chat markers, BOS, ...).
    pub special: bool,
    /// Added by the chat template (outside the user's text) or by the tokenizer (BOS/EOS).
    pub added: bool,
    /// Byte range in `full_text` when it could be aligned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<(usize, usize)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_norm: Option<f32>,
    /// Norm of the token's row in the output head (the tied embedding row when tied).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_norm: Option<f32>,
    /// The model can predict this token (an output row exists and it is not banned).
    pub in_output: bool,
    pub banned: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub neighbors: Vec<Neighbor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizeReport {
    pub schema_version: u32,
    pub model: String,
    pub tokenizer: String,
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat: Option<ChatRender>,
    pub full_text: String,
    pub tokens: Vec<TokenRow>,
    pub tied_embeddings: bool,
    pub has_output_head: bool,
    pub embedding_tensor: Option<String>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

fn row(g: &GgufFile, t: &TensorInfo, id: u32) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    g.dequantize_rows(t, id as u64, 1, &mut v)?;
    Ok(v)
}

/// Top-`k` rows of `t` by cosine similarity for every query row (excluding the query id).
pub fn nearest_rows(g: &GgufFile, t: &TensorInfo, queries: &[u32], k: usize, p: &dyn Progress) -> Result<HashMap<u32, Vec<(u32, f32)>>> {
    let n = t.n_rows();
    let qs: Vec<(u32, Vec<f32>)> = queries
        .iter()
        .map(|&q| {
            let r = row(g, t, q)?;
            let nr = norm(&r).max(1e-12);
            Ok((q, r.into_iter().map(|x| x / nr).collect()))
        })
        .collect::<Result<_>>()?;
    const CHUNK: u64 = 2048;
    let chunks: Vec<u64> = (0..n.div_ceil(CHUNK)).collect();
    let done = std::sync::atomic::AtomicU64::new(0);
    let partial: Vec<Vec<Vec<(u32, f32)>>> = chunks
        .par_iter()
        .map(|&c| -> Result<Vec<Vec<(u32, f32)>>> {
            check_cancel(p)?;
            let start = c * CHUNK;
            let len = CHUNK.min(n - start);
            let mut buf = Vec::new();
            g.dequantize_rows(t, start, len, &mut buf)?;
            let dim = t.row_len() as usize;
            let mut best: Vec<Vec<(u32, f32)>> = vec![Vec::with_capacity(k + 1); qs.len()];
            for (i, r) in buf.chunks_exact(dim).enumerate() {
                let id = (start + i as u64) as u32;
                let nr = norm(r);
                if nr <= 0.0 {
                    continue;
                }
                for (qi, (qid, q)) in qs.iter().enumerate() {
                    if *qid == id {
                        continue;
                    }
                    let cos = dot(q, r) / nr;
                    let b = &mut best[qi];
                    if b.len() < k || cos > b[b.len() - 1].1 {
                        let pos = b.partition_point(|x| x.1 >= cos);
                        b.insert(pos, (id, cos));
                        b.truncate(k);
                    }
                }
            }
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            p.set(100.0 * d as f32 / chunks.len().max(1) as f32, "nearest neighbours");
            Ok(best)
        })
        .collect::<Result<_>>()?;
    let mut out = HashMap::new();
    for (qi, (qid, _)) in qs.iter().enumerate() {
        let mut all: Vec<(u32, f32)> = partial.iter().flat_map(|c| c[qi].iter().copied()).collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        out.insert(*qid, all);
    }
    Ok(out)
}

/// Aligns decoded token texts to `text`; tokens that do not match get no span.
fn spans(vocab: &Vocab, ids: &[u32], text: &str) -> Vec<Option<(usize, usize)>> {
    let mut out = Vec::with_capacity(ids.len());
    let mut pos = 0;
    let mut lost = false;
    for (i, &id) in ids.iter().enumerate() {
        let t = vocab.tokens.get(id as usize).map(|t| t.text.as_str()).unwrap_or("");
        if lost || t.is_empty() {
            out.push(None);
            continue;
        }
        let rest = &text[pos..];
        if rest.starts_with(t) {
            out.push(Some((pos, pos + t.len())));
            pos += t.len();
        } else if let Some(stripped) = t.strip_prefix(' ').filter(|s| rest.starts_with(*s) && !s.is_empty()) {
            // SentencePiece space prefix that is not in the text.
            out.push(Some((pos, pos + stripped.len())));
            pos += stripped.len();
        } else if i == 0 {
            // Added BOS / CLS.
            out.push(None);
        } else {
            lost = true;
            out.push(None);
        }
    }
    out
}

pub fn tokenize(m: &ModelHandle, vocab: &Vocab, req: &TokenizeRequest, banned: &[(u32, u32)], p: &dyn Progress) -> Result<TokenizeReport> {
    let started = Instant::now();
    let g = &m.gguf;
    let enc = Encoder::new(vocab)?;
    let mut warnings = enc.warnings.clone();
    let chat = if req.chat {
        Some(render_chat(m, Some(vocab), &ChatRequest { prompt: req.text.clone(), system: req.system.clone(), template: req.template.clone() })?)
    } else {
        None
    };
    if chat.as_ref().is_some_and(|c| c.source == "none") {
        warnings.push("this model has no recognizable chat template: the text is used as is".into());
    }
    let full_text = chat.as_ref().map(|c| c.text.clone()).unwrap_or_else(|| req.text.clone());
    let e = enc.encode(&full_text, true, None);
    let sp = spans(vocab, &e.ids, &full_text);
    let user = chat.as_ref().and_then(|c| c.prompt_span).unwrap_or((0, full_text.len()));
    let emb = g.tensor("token_embd.weight");
    let out_t = g.tensor("output.weight");
    let tied = out_t.is_none() && emb.is_some();
    let has_head = out_t.is_some() || (tied && !matches!(m.architecture(), "bert" | "nomic-bert" | "jina-bert-v2"));
    let uniq: Vec<u32> = e.ids.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let mut nn = HashMap::new();
    if let (Some(t), true) = (emb, req.neighbors > 0) {
        let q: Vec<u32> = uniq.iter().copied().take(64).collect();
        if uniq.len() > 64 {
            warnings.push(format!("nearest neighbours computed for the first 64 of {} distinct tokens", uniq.len()));
        }
        nn = nearest_rows(g, t, &q, req.neighbors.min(32), p)?;
    }
    let mut emb_norm: HashMap<u32, f32> = HashMap::new();
    let mut out_norm: HashMap<u32, f32> = HashMap::new();
    for &id in &uniq {
        if let Some(t) = emb.filter(|t| (id as u64) < t.n_rows()) {
            emb_norm.insert(id, norm(&row(g, t, id)?));
        }
        let ot = if tied { emb } else { out_t };
        if let Some(t) = ot.filter(|t| (id as u64) < t.n_rows() && has_head) {
            out_norm.insert(id, if tied { emb_norm.get(&id).copied().unwrap_or(0.0) } else { norm(&row(g, t, id)?) });
        }
    }
    let text_of = |id: u32| vocab.tokens.get(id as usize).map(|t| t.text.clone()).unwrap_or_default();
    let tokens = e
        .ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let tok = &vocab.tokens[id as usize];
            let special = matches!(tok.token_type, TokenType::Control | TokenType::UserDefined);
            let span = sp[i];
            let added = match span {
                None => true,
                Some((a, b)) => chat.is_some() && (b <= user.0 || a >= user.1),
            };
            let is_banned = banned.iter().any(|(a, b)| id >= *a && id <= *b);
            TokenRow {
                index: i,
                id,
                raw: tok.raw.clone(),
                text: tok.text.clone(),
                class: tok.class,
                token_type: tok.token_type,
                special,
                added,
                span,
                embedding_norm: emb_norm.get(&id).copied(),
                output_norm: out_norm.get(&id).copied(),
                in_output: has_head && out_norm.get(&id).is_some_and(|n| *n > 0.0) && !is_banned,
                banned: is_banned,
                neighbors: nn.get(&id).map(|v| v.iter().map(|(n, c)| Neighbor { id: *n, text: text_of(*n), cosine: *c }).collect()).unwrap_or_default(),
            }
        })
        .collect();
    Ok(TokenizeReport {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        tokenizer: vocab.model_name.clone(),
        input: req.text.clone(),
        chat,
        full_text,
        tokens,
        tied_embeddings: tied,
        has_output_head: has_head,
        embedding_tensor: emb.map(|t| t.name.clone()),
        warnings,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

// --- embedding map ------------------------------------------------------------------------

/// The vocabulary as a 3D point cloud: token_embd rows projected on their first three
/// principal components (computed on a sample of rows, then applied to every row).
#[derive(Debug, Clone)]
pub struct EmbeddingMap {
    pub n: usize,
    pub dim: usize,
    pub sample: usize,
    pub tensor: String,
    /// Fraction of the variance on each axis.
    pub explained: [f32; 3],
    /// n × 3, scaled to roughly [-1, 1].
    pub coords: Vec<f32>,
    /// Index into [`TokenClass::ALL`] per token.
    pub classes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingMapInfo {
    pub count: usize,
    pub dim: usize,
    pub sample: usize,
    pub tensor: String,
    pub explained: [f32; 3],
    /// The binary layout: little-endian f32, `stride` values per token (x, y, z, class).
    pub stride: usize,
    pub classes: Vec<&'static str>,
}

impl EmbeddingMap {
    pub fn info(&self) -> EmbeddingMapInfo {
        EmbeddingMapInfo {
            count: self.n,
            dim: self.dim,
            sample: self.sample,
            tensor: self.tensor.clone(),
            explained: self.explained,
            stride: 4,
            classes: TokenClass::ALL.iter().map(|c| c.as_str()).collect(),
        }
    }

    /// Float32Array payload: x, y, z, class per token.
    pub fn to_f32_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.n * 16);
        for i in 0..self.n {
            for k in 0..3 {
                out.extend_from_slice(&self.coords[i * 3 + k].to_le_bytes());
            }
            out.extend_from_slice(&(self.classes.get(i).copied().unwrap_or(0) as f32).to_le_bytes());
        }
        out
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut b = Vec::with_capacity(64 + self.n * 13);
        b.extend_from_slice(b"NDEM1");
        for v in [self.n as u64, self.dim as u64, self.sample as u64, self.tensor.len() as u64] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(self.tensor.as_bytes());
        for e in self.explained {
            b.extend_from_slice(&e.to_le_bytes());
        }
        for c in &self.coords {
            b.extend_from_slice(&c.to_le_bytes());
        }
        b.extend_from_slice(&self.classes);
        std::fs::write(path, b)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let b = std::fs::read(path)?;
        let bad = || anyhow!("corrupt embedding map cache {}", path.display());
        if !b.starts_with(b"NDEM1") {
            return Err(bad());
        }
        let mut pos = 5;
        let mut u64_at = |b: &[u8]| -> Result<u64> {
            let v = u64::from_le_bytes(b.get(pos..pos + 8).ok_or_else(bad)?.try_into().map_err(|_| bad())?);
            pos += 8;
            Ok(v)
        };
        let (n, dim, sample, tl) = (u64_at(&b)? as usize, u64_at(&b)? as usize, u64_at(&b)? as usize, u64_at(&b)? as usize);
        let tensor = String::from_utf8_lossy(b.get(pos..pos + tl).ok_or_else(bad)?).to_string();
        pos += tl;
        let f32s = |b: &[u8], at: usize, k: usize| -> Result<Vec<f32>> {
            let s = b.get(at..at + k * 4).ok_or_else(bad)?;
            Ok(s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let ex = f32s(&b, pos, 3)?;
        pos += 12;
        let coords = f32s(&b, pos, n * 3)?;
        pos += n * 12;
        let classes = b.get(pos..pos + n).ok_or_else(bad)?.to_vec();
        Ok(Self { n, dim, sample, tensor, explained: [ex[0], ex[1], ex[2]], coords, classes })
    }
}

pub fn embedding_map(m: &ModelHandle, vocab: Option<&Vocab>, p: &dyn Progress) -> Result<EmbeddingMap> {
    let g = &m.gguf;
    let t = g.tensor("token_embd.weight").ok_or_else(|| anyhow!("this model has no token_embd.weight"))?;
    let n = t.n_rows() as usize;
    let dim = t.row_len() as usize;
    if n < 4 || dim < 3 {
        bail!("token_embd is too small for a 3D map");
    }
    // 1. Sample rows evenly.
    let s = n.min(16384);
    p.set(2.0, "sampling rows");
    let picks: Vec<usize> = (0..s).map(|i| i * n / s).collect();
    let rows: Vec<Vec<f32>> = picks.par_iter().map(|&r| row(g, t, r as u32)).collect::<Result<_>>().context("reading token_embd")?;
    let mut mean = vec![0f64; dim];
    for r in &rows {
        for (m, x) in mean.iter_mut().zip(r) {
            *m += *x as f64;
        }
    }
    let mean: Vec<f32> = mean.iter().map(|x| (*x / s as f64) as f32).collect();
    let x: Vec<Vec<f32>> = rows.into_par_iter().map(|r| r.iter().zip(&mean).map(|(a, b)| a - b).collect()).collect();
    let total_var: f64 = x.par_iter().map(|r| r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sum::<f64>() / s as f64;

    // 2. Block power iteration for the top three components.
    let mut v: Vec<[f32; 3]> = (0..dim).map(|i| [((i * 7 + 1) as f32).sin(), ((i * 13 + 2) as f32).cos(), ((i * 29 + 3) as f32).sin()]).collect();
    let ortho = |v: &mut Vec<[f32; 3]>| {
        for k in 0..3 {
            for j in 0..k {
                let d: f64 = v.iter().map(|r| r[k] as f64 * r[j] as f64).sum();
                for r in v.iter_mut() {
                    r[k] -= (d as f32) * r[j];
                }
            }
            let nk = v.iter().map(|r| (r[k] as f64).powi(2)).sum::<f64>().sqrt().max(1e-12) as f32;
            for r in v.iter_mut() {
                r[k] /= nk;
            }
        }
    };
    ortho(&mut v);
    let iters = 30;
    for it in 0..iters {
        check_cancel(p)?;
        p.set(5.0 + 60.0 * it as f32 / iters as f32, "principal components");
        let y: Vec<[f32; 3]> = x.par_iter().map(|r| {
            let mut o = [0f32; 3];
            for (a, vv) in r.iter().zip(&v) {
                o[0] += a * vv[0];
                o[1] += a * vv[1];
                o[2] += a * vv[2];
            }
            o
        }).collect();
        let z: Vec<[f64; 3]> = x
            .par_iter()
            .zip(&y)
            .fold(|| vec![[0f64; 3]; dim], |mut acc, (r, yy)| {
                for (a, o) in r.iter().zip(acc.iter_mut()) {
                    o[0] += (*a * yy[0]) as f64;
                    o[1] += (*a * yy[1]) as f64;
                    o[2] += (*a * yy[2]) as f64;
                }
                acc
            })
            .reduce(|| vec![[0f64; 3]; dim], |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    x[0] += y[0];
                    x[1] += y[1];
                    x[2] += y[2];
                }
                a
            });
        v = z.into_iter().map(|r| [r[0] as f32, r[1] as f32, r[2] as f32]).collect();
        ortho(&mut v);
    }
    let mut explained = [0f32; 3];
    let lam: Vec<f64> = (0..3)
        .map(|k| x.par_iter().map(|r| r.iter().zip(&v).map(|(a, vv)| (*a * vv[k]) as f64).sum::<f64>().powi(2)).sum::<f64>() / s as f64)
        .collect();
    for k in 0..3 {
        explained[k] = (lam[k] / total_var.max(1e-12)) as f32;
    }
    drop(x);

    // 3. Project every row.
    const CHUNK: usize = 4096;
    let n_chunks = n.div_ceil(CHUNK);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let parts: Vec<Vec<f32>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| -> Result<Vec<f32>> {
            check_cancel(p)?;
            let start = c * CHUNK;
            let len = CHUNK.min(n - start);
            let mut buf = Vec::new();
            g.dequantize_rows(t, start as u64, len as u64, &mut buf)?;
            let mut out = Vec::with_capacity(len * 3);
            for r in buf.chunks_exact(dim) {
                let mut o = [0f32; 3];
                for ((a, mu), vv) in r.iter().zip(&mean).zip(&v) {
                    let a = a - mu;
                    o[0] += a * vv[0];
                    o[1] += a * vv[1];
                    o[2] += a * vv[2];
                }
                out.extend_from_slice(&o);
            }
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            p.set(65.0 + 33.0 * d as f32 / n_chunks as f32, "projecting");
            Ok(out)
        })
        .collect::<Result<_>>()?;
    let mut coords: Vec<f32> = parts.concat();
    // Scale by the 99.5th percentile of |x| on the first axis, so a few outliers do not
    // squash the cloud; keeps the axes' relative spread.
    let mut a0: Vec<f32> = coords.iter().step_by(3).map(|x| x.abs()).collect();
    let q = (a0.len() as f32 * 0.995) as usize;
    let q = q.min(a0.len().saturating_sub(1));
    a0.select_nth_unstable_by(q, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let scale = 1.0 / a0[q].max(1e-12);
    for c in &mut coords {
        *c *= scale;
    }
    let classes = (0..n)
        .map(|i| {
            vocab
                .and_then(|v| v.tokens.get(i))
                .and_then(|t| TokenClass::ALL.iter().position(|c| *c == t.class))
                .unwrap_or(TokenClass::ALL.len() - 1) as u8
        })
        .collect();
    p.set(100.0, "done");
    Ok(EmbeddingMap { n, dim, sample: s, tensor: t.name.clone(), explained, coords, classes })
}
