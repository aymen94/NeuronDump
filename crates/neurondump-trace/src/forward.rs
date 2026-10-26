//! The instrumented forward pass (`Model::run`): every layer, with ablations, steering and patching.

use super::*;

impl<'a> Model<'a> {
    pub(super) fn run(&self, ids: &[u32], masks: &Masks, opt: FwdOpt, p: &dyn Progress) -> Result<FwdOut> {
        let _zero = ZeroGuard::new(&masks.zero_tensors);
        let seq = ids.len();
        let mut h = self.embed(ids, masks)?;
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
        let rpos = opt.record.map(|r| r.min(seq - 1));
        let mut record = rpos.map(|pos| Record { pos, emb: Vec::new(), layers: Vec::new() });
        if let (Some(r), Some(pos)) = (record.as_mut(), rpos) {
            r.emb = row_of(&h, pos)?;
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
        let mut hidden = Vec::new();
        let mut zs = Vec::new();
        let n_layers = self.layers.len();
        let measure_lens = |h: &Tensor, lens_probs: &mut Vec<Vec<f32>>| -> Result<Vec<LensToken>> {
            if opt.lens_probs {
                let (top, probs) = self.lens_full(&h.narrow(0, seq - 1, 1)?, masks)?;
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
            let mut rec = LayerRec { attn_factor: Masks::factor(&masks.attn_scale, li), ffn_factor: Masks::factor(&masks.ffn_scale, li), ..Default::default() };
            if masks.skip.contains(&li) {
                // Identity: the residual stream passes through unchanged.
                rec.skipped = true;
                if opt.keep {
                    zs.push(vec![]);
                }
                if opt.measure {
                    let hv = h.to_vec2::<f32>()?;
                    let hidden_norm = hv.iter().map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt()).sum::<f32>() / seq as f32;
                    let logit_lens = measure_lens(&h, &mut lens_probs)?;
                    traces.push(LayerTrace { index: li as u32, hidden_norm, cos_prev: Some(1.0), head_entropy: vec![], top_neurons: vec![], router: vec![], logit_lens });
                    neuron_means.push(vec![]);
                }
            } else {
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
                    if let Some(pos) = rpos {
                        let pr = probs.narrow(1, pos, 1)?.squeeze(1)?.to_vec2::<f32>()?;
                        rec.attn = pr.into_iter().map(|mut r| {
                            r.truncate(if causal { pos + 1 } else { seq });
                            r
                        }).collect();
                    }
                    let width = l.n_head * l.head_dim;
                    let mut att = probs.matmul(&v)?.transpose(0, 1)?.contiguous()?.reshape((seq, width))?;
                    if let Some(hs) = masks.heads.get(&li) {
                        let mut m = vec![1f32; width];
                        for &(hh, f) in hs.iter().filter(|(hh, _)| *hh < l.n_head) {
                            for x in &mut m[hh * l.head_dim..(hh + 1) * l.head_dim] {
                                *x *= f;
                            }
                        }
                        att = att.broadcast_mul(&Tensor::from_vec(m, (1, width), &self.w.dev)?)?;
                    }
                    if let Some(ps) = masks.patch_z.get(&li) {
                        for (hh, src, all) in ps.iter().filter(|(hh, _, _)| *hh < l.n_head) {
                            att = patch_rows(&att, src, *all, Some((hh * l.head_dim, (hh + 1) * l.head_dim)))?;
                        }
                    }
                    if opt.keep {
                        zs.push(att.to_vec2::<f32>()?);
                    }
                    if let Some(pos) = rpos {
                        rec.z = row_of(&att, pos)?;
                    }
                    let mut att = l.o.forward(&att)?;
                    if let Some(n) = &l.attn_post_norm {
                        if let Some(pos) = rpos {
                            rec.attn_pre_rms = Some(rms(&row_of(&att, pos)?, n.eps));
                        }
                        att = n.forward(&att)?;
                    }
                    if rec.attn_factor != 1.0 {
                        att = (att * rec.attn_factor as f64)?;
                    }
                    if let Some(pos) = rpos {
                        rec.attn_out = row_of(&att, pos)?;
                    }
                    h = (h + att)?;
                } else if opt.keep {
                    zs.push(vec![]);
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
                        for &(s0, e, f) in rs {
                            if s0 < n_ff {
                                for x in &mut m[s0..(e + 1).min(n_ff)] {
                                    *x *= f;
                                }
                            }
                        }
                        inner = inner.broadcast_mul(&Tensor::from_vec(m, (1, n_ff), &self.w.dev)?)?;
                    }
                    if opt.measure {
                        means = inner.abs()?.mean(0)?.to_vec1::<f32>()?;
                    }
                    if let Some(pos) = rpos {
                        rec.inner = row_of(&inner, pos)?;
                    }
                    let mut ffn = l.down.forward(&inner)?;
                    if let Some(n) = &l.ffn_post_norm {
                        if let Some(pos) = rpos {
                            rec.ffn_pre_rms = Some(rms(&row_of(&ffn, pos)?, n.eps));
                        }
                        ffn = n.forward(&ffn)?;
                    }
                    if rec.ffn_factor != 1.0 {
                        ffn = (ffn * rec.ffn_factor as f64)?;
                    }
                    if let Some(pos) = rpos {
                        rec.ffn_out = row_of(&ffn, pos)?;
                    }
                    h = (h + ffn)?;
                }
                if let Some(n) = &l.out_norm {
                    h = n.forward(&h)?;
                }

                // --- measurements ---
                if opt.measure {
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
            }

            // --- edits on the residual stream after the layer ---
            if let Some(v) = masks.steer.get(&li) {
                h = h.broadcast_add(v)?;
            }
            if let Some((src, all)) = masks.patch_h.get(&li) {
                h = patch_rows(&h, src, *all, None)?;
            }
            if opt.keep {
                hidden.push(h.to_vec2::<f32>()?);
            }
            if let (Some(r), Some(pos)) = (record.as_mut(), rpos) {
                rec.h = row_of(&h, pos)?;
                r.layers.push(rec);
            }
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
                masks.apply_logits(&mut last);
                last_logits = last;
                all_logits = Some(lg);
            } else if let Some(lg) = self.head_logits(&h.narrow(0, seq - 1, 1)?)? {
                last_logits = lg.squeeze(0)?.to_vec1::<f32>()?;
                masks.apply_logits(&mut last_logits);
            }
        }
        Ok(FwdOut {
            sample: Sample { tokens: ids.to_vec(), layers: traces, prediction, neuron_means, contributions },
            last_logits,
            all_logits,
            lens_probs,
            pooled,
            record,
            hidden,
            zs,
        })
    }

}
