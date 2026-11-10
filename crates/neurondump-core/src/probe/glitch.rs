//! Under-trained ("glitch") token candidates: tokens whose embedding or unembedding row is
//! unusually small, which happens when a token was (almost) never seen in training. Prompts
//! that contain them can make a model behave erratically. Each candidate is also checked for
//! reachability: whether encoding its text gives that token back.

use super::*;
use crate::tokenizer::{Encoder, TokenType, Vocab};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlitchOptions {
    /// Candidates returned.
    #[serde(default = "default_top")]
    pub top: usize,
    /// z-score below which a norm counts as abnormal.
    #[serde(default = "default_z")]
    pub z: f32,
}

fn default_top() -> usize {
    100
}

fn default_z() -> f32 {
    -3.0
}

impl Default for GlitchOptions {
    fn default() -> Self {
        Self { top: default_top(), z: default_z() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormStats {
    pub tensor: String,
    pub mean: f32,
    pub std: f32,
    pub median: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlitchToken {
    pub id: u32,
    pub text: String,
    pub token_type: String,
    pub embed_norm: f32,
    pub embed_z: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unembed_norm: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unembed_z: Option<f32>,
    /// Encoding the token's text gives back exactly this token.
    pub reachable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlitchReport {
    pub schema_version: u32,
    pub model: String,
    pub vocab_size: usize,
    /// Tokens checked (normal and user-defined ones; control, byte and unused tokens are skipped).
    pub checked: usize,
    pub embed: NormStats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unembed: Option<NormStats>,
    /// Tokens below the z threshold.
    pub flagged: usize,
    /// Lowest z first.
    pub candidates: Vec<GlitchToken>,
    pub warnings: Vec<String>,
}

fn row_norms(g: &GgufFile, t: &TensorInfo, p: &dyn Progress, from: f32, to: f32) -> Result<Vec<f32>> {
    let mut out = vec![0f32; t.n_rows() as usize];
    for_rows(g, t, 2048, Some((p, from, to)), |i, r| out[i as usize] = r.iter().map(|x| x * x).sum::<f32>().sqrt())?;
    Ok(out)
}

fn stats(name: &str, v: &[f32], keep: &[usize]) -> NormStats {
    let mut xs: Vec<f32> = keep.iter().map(|&i| v[i]).collect();
    xs.sort_by(f32::total_cmp);
    let n = xs.len().max(1) as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let std = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
    NormStats {
        tensor: name.into(),
        mean,
        std,
        median: xs.get(xs.len() / 2).copied().unwrap_or(0.0),
        min: xs.first().copied().unwrap_or(0.0),
        max: xs.last().copied().unwrap_or(0.0),
    }
}

pub fn glitch_tokens(m: &ModelHandle, vocab: &Vocab, o: &GlitchOptions, p: &dyn Progress) -> Result<GlitchReport> {
    let g = &m.gguf;
    let embd = tensor(g, "token_embd.weight")?;
    let mut warnings = Vec::new();
    p.set(1.0, "embedding norms");
    let en = row_norms(g, embd, p, 1.0, 45.0)?;
    let un = match g.tensor("output.weight") {
        Some(t) if t.n_rows() == embd.n_rows() => Some(row_norms(g, t, p, 45.0, 90.0)?),
        Some(_) => {
            warnings.push("output.weight has another row count than token_embd: unembedding skipped".into());
            None
        }
        None => None,
    };
    let keep: Vec<usize> = (0..en.len())
        .filter(|&i| vocab.tokens.get(i).is_some_and(|t| matches!(t.token_type, TokenType::Normal | TokenType::UserDefined)))
        .collect();
    if keep.is_empty() {
        bail!("no normal tokens to check");
    }
    let es = stats("token_embd.weight", &en, &keep);
    let us = un.as_ref().map(|u| stats("output.weight", u, &keep));
    let z = |x: f32, s: &NormStats| if s.std > 0.0 { (x - s.mean) / s.std } else { 0.0 };
    // A token's score is its lowest z: either side can show it was not trained.
    let score = |i: usize| -> f32 {
        let a = z(en[i], &es);
        match (&un, &us) {
            (Some(u), Some(s)) => a.min(z(u[i], s)),
            _ => a,
        }
    };
    let mut order: Vec<usize> = keep.iter().copied().filter(|&i| score(i) <= o.z).collect();
    let flagged = order.len();
    order.sort_by(|a, b| score(*a).total_cmp(&score(*b)));
    order.truncate(o.top.clamp(1, 2000));
    p.set(92.0, "reachability");
    let enc = Encoder::new(vocab)?;
    let candidates = order
        .into_iter()
        .map(|i| {
            let t = &vocab.tokens[i];
            GlitchToken {
                id: i as u32,
                text: t.text.clone(),
                token_type: format!("{:?}", t.token_type).to_lowercase(),
                embed_norm: en[i],
                embed_z: z(en[i], &es),
                unembed_norm: un.as_ref().map(|u| u[i]),
                unembed_z: match (&un, &us) {
                    (Some(u), Some(s)) => Some(z(u[i], s)),
                    _ => None,
                },
                reachable: enc.encode(&t.text, false, None).ids == [i as u32],
            }
        })
        .collect();
    p.set(100.0, "done");
    Ok(GlitchReport {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        vocab_size: vocab.tokens.len(),
        checked: keep.len(),
        embed: es,
        unembed: us,
        flagged,
        candidates,
        warnings,
    })
}
