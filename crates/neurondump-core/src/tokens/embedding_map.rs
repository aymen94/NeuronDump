//! The 3D embedding map: PCA of `token_embd` rows, packed for the web view.

use super::*;

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
