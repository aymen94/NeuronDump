//! Probe types: causal tracing, attention head roles and layer similarity.

use super::*;

/// Which part of a layer causal tracing restores from the clean run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CausalComponent {
    /// The residual stream after the layer.
    #[default]
    Residual,
    /// What the attention (or mixer) block added at the layer.
    Attn,
    /// What the feed-forward block added at the layer.
    Ffn,
}

impl CausalComponent {
    pub fn as_str(self) -> &'static str {
        match self {
            CausalComponent::Residual => "residual",
            CausalComponent::Attn => "attn",
            CausalComponent::Ffn => "ffn",
        }
    }
}

/// Causal tracing: run a clean and a corrupted prompt, then restore the clean state at one
/// layer and position inside the corrupted run and measure how much of the clean answer
/// comes back.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalRequest {
    /// Templated clean prompt.
    pub clean: String,
    /// Templated corrupted prompt (e.g. another subject). Positions are aligned from the end.
    pub corrupt: String,
    /// Token whose probability is measured (default: the clean run's top-1).
    #[serde(default)]
    pub target: Option<u32>,
    #[serde(default)]
    pub component: CausalComponent,
    #[serde(default)]
    pub max_tokens: usize,
    #[serde(default)]
    pub ablation: Ablation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPeak {
    pub layer: u32,
    /// Position in the corrupted prompt.
    pub position: usize,
    pub effect: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub component: CausalComponent,
    pub clean_tokens: Vec<JourneyToken>,
    pub corrupt_tokens: Vec<JourneyToken>,
    pub target: LensToken,
    /// Probability of the target in the clean and in the corrupted run.
    pub clean_prob: f32,
    pub corrupt_prob: f32,
    /// Top predictions of the corrupted run.
    pub corrupt_top: Vec<LensToken>,
    /// [layer][corrupted position]: probability of the target with that state restored
    /// (null where the position has no clean counterpart).
    pub probs: Vec<Vec<Option<f32>>>,
    /// Same cells as the share of the clean-corrupt gap restored: 0 = corrupted, 1 = clean.
    pub effect: Vec<Vec<Option<f32>>>,
    /// The strongest cells, largest effect first.
    pub peaks: Vec<CausalPeak>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}

/// Head roles from synthetic prompts: random tokens repeated twice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadRolesRequest {
    /// Number of random sequences averaged.
    pub samples: usize,
    /// Length of the random part (the sequence is that part twice).
    pub length: usize,
    pub seed: u64,
    #[serde(default)]
    pub ablation: Ablation,
}

impl Default for HeadRolesRequest {
    fn default() -> Self {
        Self { samples: 4, length: 24, seed: 7, ablation: Ablation::default() }
    }
}

/// Mean attention of one head on the synthetic sequences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeadScore {
    pub layer: u32,
    pub head: u32,
    /// Attention to the previous token.
    pub prev_token: f32,
    /// In the repeat: attention to the token after the earlier copy of the current one.
    pub induction: f32,
    /// In the repeat: attention to the earlier copy of the current token.
    pub duplicate: f32,
    /// Attention to the first position (attention sink).
    pub sink: f32,
    /// Attention to the current position.
    pub self_attn: f32,
    /// `induction`, `previous-token`, `duplicate-token`, `self`, `sink` or `mixed`.
    pub role: String,
}

impl HeadScore {
    /// The strongest pattern above its threshold; specific patterns win over the sink.
    pub fn classify(&mut self) {
        self.role = if self.induction >= 0.3 {
            "induction"
        } else if self.prev_token >= 0.4 {
            "previous-token"
        } else if self.duplicate >= 0.3 {
            "duplicate-token"
        } else if self.self_attn >= 0.4 {
            "self"
        } else if self.sink >= 0.5 {
            "sink"
        } else {
            "mixed"
        }
        .into();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadRolesReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub samples: usize,
    pub length: usize,
    pub heads: Vec<HeadScore>,
    /// role -> number of heads.
    pub counts: Vec<(String, usize)>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}

/// Representation similarity between layers on a corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityRequest {
    pub corpus: Vec<String>,
    /// Tokens per sample.
    pub max_tokens: usize,
    /// Positions used in total (the kernel matrices are positions × positions).
    pub max_positions: usize,
    #[serde(default)]
    pub ablation: Ablation,
}

impl Default for SimilarityRequest {
    fn default() -> Self {
        Self { corpus: vec![], max_tokens: 64, max_positions: 256, ablation: Ablation::default() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerChange {
    pub layer: u32,
    /// CKA between the residual stream before and after the layer (1 = same geometry).
    pub cka_in_out: f32,
    /// Mean cosine between the residual vectors before and after the layer.
    pub cos_in_out: f32,
    /// Mean |update| / |input|: how much the layer writes relative to what it reads.
    pub rel_update: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityReport {
    pub schema_version: u32,
    pub model: String,
    pub architecture: String,
    pub samples: usize,
    pub positions: usize,
    /// `embeddings`, then `layer 0` … : the residual stream at each point.
    pub labels: Vec<String>,
    /// Linear CKA between every pair of points in `labels`.
    pub cka: Vec<Vec<f32>>,
    pub layers: Vec<LayerChange>,
    /// Layers that change the residual stream least (CKA in/out ≥ 0.99), most redundant first.
    pub redundant: Vec<u32>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u64,
}

/// Linear CKA from two centered Gram matrices (n × n, row-major).
pub fn cka_from_grams(k: &[f64], l: &[f64]) -> f64 {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let d = (dot(k, k) * dot(l, l)).sqrt();
    if d > 0.0 {
        dot(k, l) / d
    } else {
        0.0
    }
}

/// Centered Gram matrix H X Xᵀ H of row vectors `x` (n rows), in f64.
pub fn centered_gram(x: &[Vec<f32>]) -> Vec<f64> {
    use rayon::prelude::*;
    let n = x.len();
    let mut k = vec![0f64; n * n];
    k.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for (j, v) in row.iter_mut().enumerate() {
            *v = x[i].iter().zip(&x[j]).map(|(a, b)| *a as f64 * *b as f64).sum();
        }
    });
    let row_mean: Vec<f64> = (0..n).map(|i| k[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64).collect();
    let mean = row_mean.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] += mean - row_mean[i] - row_mean[j];
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cka_is_one_for_scaled_rotations_and_lower_otherwise() {
        let x: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32, (i * i) as f32 % 5.0, 1.0 - i as f32 * 0.3]).collect();
        // Rotation in the first two coordinates plus scaling keeps CKA at 1.
        let (c, s) = (0.6f32, 0.8f32);
        let y: Vec<Vec<f32>> = x.iter().map(|r| vec![3.0 * (c * r[0] - s * r[1]), 3.0 * (s * r[0] + c * r[1]), 3.0 * r[2]]).collect();
        let z: Vec<Vec<f32>> = (0..6).map(|i| vec![((i * 7) % 4) as f32, 0.5, (i % 2) as f32]).collect();
        let (kx, ky, kz) = (centered_gram(&x), centered_gram(&y), centered_gram(&z));
        assert!((cka_from_grams(&kx, &ky) - 1.0).abs() < 1e-9);
        assert!(cka_from_grams(&kx, &kz) < 0.9);
    }

    #[test]
    fn head_roles_prefer_specific_patterns() {
        let mut h = HeadScore { induction: 0.5, sink: 0.9, ..Default::default() };
        h.classify();
        assert_eq!(h.role, "induction");
        let mut h = HeadScore { sink: 0.8, ..Default::default() };
        h.classify();
        assert_eq!(h.role, "sink");
    }
}
