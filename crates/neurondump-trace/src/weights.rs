//! Weights read from the memory-mapped GGUF: lookup, dequantization, linear / norm / layer blocks.

use super::*;

pub(super) struct Weights<'a> {
    pub(super) g: &'a GgufFile,
    pub(super) dev: Device,
}

impl<'a> Weights<'a> {
    pub(super) fn info(&self, name: &str) -> Option<&'a TensorInfo> {
        self.g.tensor(name)
    }

    /// Dense f32 tensor in candle order (dims reversed).
    pub(super) fn dense(&self, name: &str) -> Result<Tensor> {
        let t = self.info(name).ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let mut v = Vec::new();
        self.g.dequantize_rows(t, 0, t.n_rows(), &mut v)?;
        let dims: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        Ok(Tensor::from_vec(v, dims, &self.dev)?)
    }

    pub(super) fn opt_dense(&self, name: &str) -> Result<Option<Tensor>> {
        if self.info(name).is_some() {
            self.dense(name).map(Some)
        } else {
            Ok(None)
        }
    }

    pub(super) fn matmul(&self, name: &str) -> Result<QMatMul> {
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

    pub(super) fn opt_matmul(&self, name: &str) -> Result<Option<QMatMul>> {
        if self.info(name).is_some() {
            self.matmul(name).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Rows of an embedding matrix for the given ids, as [n, dim].
    pub(super) fn rows(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
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

pub(super) fn is_zeroed(name: &str) -> bool {
    ZEROED.with(|z| !z.borrow().is_empty() && z.borrow().contains(name))
}

/// Sets the zeroed-tensor set for the lifetime of the guard.
pub(super) struct ZeroGuard;

impl ZeroGuard {
    pub(super) fn new(names: &[String]) -> Self {
        ZEROED.with(|z| *z.borrow_mut() = names.iter().cloned().collect());
        ZeroGuard
    }
}

impl Drop for ZeroGuard {
    fn drop(&mut self) {
        ZEROED.with(|z| z.borrow_mut().clear());
    }
}

pub(super) struct Linear {
    pub(super) w: QMatMul,
    pub(super) b: Option<Tensor>,
    pub(super) name: String,
}

impl Linear {
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

pub(super) struct Norm {
    pub(super) w: Tensor,
    pub(super) b: Option<Tensor>,
    pub(super) eps: f64,
    pub(super) layer: bool,
    pub(super) name: String,
}

impl Norm {
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

pub(super) struct Layer {
    pub(super) attn_norm: Option<Norm>,
    pub(super) q: Option<Linear>,
    pub(super) k: Option<Linear>,
    pub(super) v: Option<Linear>,
    pub(super) qkv: Option<Linear>,
    pub(super) q_norm: Option<Norm>,
    pub(super) k_norm: Option<Norm>,
    pub(super) o: Linear,
    pub(super) attn_post_norm: Option<Norm>,
    pub(super) ffn_norm: Option<Norm>,
    pub(super) gate: Option<Linear>,
    pub(super) up: Linear,
    pub(super) down: Linear,
    pub(super) ffn_post_norm: Option<Norm>,
    /// Encoder post-norms (BERT style).
    pub(super) attn_out_norm: Option<Norm>,
    pub(super) out_norm: Option<Norm>,
    /// `ffn_up` holds gate and up stacked (phi3): first half is the gate.
    pub(super) fused_up: bool,
    pub(super) n_head: usize,
    pub(super) n_kv: usize,
    pub(super) head_dim: usize,
}

pub(super) fn norm(w: &Weights, base: &str, eps: f64, layer: bool) -> Result<Option<Norm>> {
    let Some(wt) = w.opt_dense(&format!("{base}.weight"))? else { return Ok(None) };
    Ok(Some(Norm { w: wt, b: w.opt_dense(&format!("{base}.bias"))?, eps, layer, name: format!("{base}.weight") }))
}

pub(super) fn linear(w: &Weights, base: &str) -> Result<Option<Linear>> {
    let Some(m) = w.opt_matmul(&format!("{base}.weight"))? else { return Ok(None) };
    Ok(Some(Linear { w: m, b: w.opt_dense(&format!("{base}.bias"))?, name: format!("{base}.weight") }))
}

pub(super) fn req<T>(v: Option<T>, what: &str) -> Result<T> {
    v.ok_or_else(|| anyhow!("missing tensor {what}"))
}
