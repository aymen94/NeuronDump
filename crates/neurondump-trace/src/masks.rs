//! Intervention masks: ablations, zeroed tensors, steering, patching and edits resolved per layer / head.

use super::*;

/// Per-pass intervention masks (built from a core [`Ablation`]; steering and patching
/// vectors are computed by [`Model::masks`]).
#[derive(Default)]
pub(super) struct Masks {
    pub(super) skip: HashSet<usize>,
    pub(super) zero_attn: HashSet<usize>,
    pub(super) zero_ffn: HashSet<usize>,
    /// Factor of the mixer / FFN contribution (layer and block scales multiplied).
    pub(super) attn_scale: HashMap<usize, f32>,
    pub(super) ffn_scale: HashMap<usize, f32>,
    /// (head, factor); 0 zeroes the head.
    pub(super) heads: HashMap<usize, Vec<(usize, f32)>>,
    /// (first, last, factor) inclusive neuron ranges.
    pub(super) neurons: HashMap<usize, Vec<(usize, usize, f32)>>,
    pub(super) zero_tensors: Vec<String>,
    pub(super) tokens: Vec<(u32, u32)>,
    pub(super) bias: Vec<(u32, f32)>,
    /// token -> (scale, replacement token).
    pub(super) embed: HashMap<u32, (f32, Option<u32>)>,
    /// Vector added to the residual stream after the layer, [1, n_embd].
    pub(super) steer: HashMap<usize, Tensor>,
    /// Source residual rows after the layer, and whether every position is patched.
    pub(super) patch_h: HashMap<usize, (Vec<Vec<f32>>, bool)>,
    /// (head, source attention output rows before the projection, all positions).
    pub(super) patch_z: HashMap<usize, Vec<(usize, Vec<Vec<f32>>, bool)>>,
}

impl Masks {
    pub(super) fn from(a: &Ablation) -> Self {
        let mut m = Masks { zero_tensors: a.zero_tensors.clone(), tokens: a.mask_tokens.clone(), bias: a.logit_bias.clone(), ..Default::default() };
        m.skip.extend(a.skip_layers.iter().map(|&l| l as usize));
        let is_mixer = |b: &str| matches!(b, "attention" | "ssm");
        for (l, b) in &a.zero_blocks {
            if is_mixer(b) {
                m.zero_attn.insert(*l as usize);
            } else {
                m.zero_ffn.insert(*l as usize);
            }
        }
        for (l, f) in &a.scale_layers {
            *m.attn_scale.entry(*l as usize).or_insert(1.0) *= f;
            *m.ffn_scale.entry(*l as usize).or_insert(1.0) *= f;
        }
        for (l, b, f) in &a.scale_blocks {
            let map = if is_mixer(b) { &mut m.attn_scale } else { &mut m.ffn_scale };
            *map.entry(*l as usize).or_insert(1.0) *= f;
        }
        for (l, f) in m.attn_scale.clone() {
            if f == 0.0 {
                m.zero_attn.insert(l);
            }
        }
        for (l, f) in m.ffn_scale.clone() {
            if f == 0.0 {
                m.zero_ffn.insert(l);
            }
        }
        for (l, h) in &a.zero_heads {
            m.heads.entry(*l as usize).or_default().push((*h as usize, 0.0));
        }
        for (l, h, f) in &a.scale_heads {
            m.heads.entry(*l as usize).or_default().push((*h as usize, *f));
        }
        for (l, s, e) in &a.zero_neurons {
            m.neurons.entry(*l as usize).or_default().push((*s as usize, *e as usize, 0.0));
        }
        for (l, s, e, f) in &a.scale_neurons {
            m.neurons.entry(*l as usize).or_default().push((*s as usize, *e as usize, *f));
        }
        for e in &a.embed_edits {
            let entry = m.embed.entry(e.token).or_insert((1.0, None));
            entry.0 *= e.scale.unwrap_or(1.0);
            if e.replace_with.is_some() {
                entry.1 = e.replace_with;
            }
        }
        m
    }

    pub(super) fn factor(map: &HashMap<usize, f32>, l: usize) -> f32 {
        map.get(&l).copied().unwrap_or(1.0)
    }

    pub(super) fn apply_logits(&self, logits: &mut [f32]) {
        for (t, b) in &self.bias {
            if let Some(x) = logits.get_mut(*t as usize) {
                *x += b;
            }
        }
        mask_logits(logits, &self.tokens);
    }
}
