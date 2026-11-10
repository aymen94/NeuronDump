//! Spectrum of the weight matrices: largest singular value (power iteration), stable rank,
//! and outlier weights. A low stable rank means the matrix uses few directions (a candidate
//! for low-rank compression); large outlier ratios make a matrix sensitive to quantization.

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumOptions {
    /// Only these layers (default: all).
    #[serde(default)]
    pub layers: Option<Vec<u32>>,
    /// Also the (large) embedding and output matrices.
    #[serde(default)]
    pub embeddings: bool,
    /// Power iterations for the largest singular value.
    #[serde(default = "default_iters")]
    pub iters: usize,
}

fn default_iters() -> usize {
    12
}

impl Default for SpectrumOptions {
    fn default() -> Self {
        Self { layers: None, embeddings: false, iters: default_iters() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpectrum {
    pub name: String,
    pub layer: Option<u32>,
    pub rows: u64,
    pub cols: u64,
    pub quant: String,
    /// Frobenius norm.
    pub fro: f64,
    /// Largest singular value.
    pub sigma_max: f64,
    /// ‖W‖²_F / σ²_max: how many directions carry the energy (≤ min(rows, cols)).
    pub stable_rank: f64,
    /// stable_rank / min(rows, cols), 0..1.
    pub rank_ratio: f64,
    pub max_abs: f32,
    pub rms: f32,
    /// max |w| / rms.
    pub outlier_ratio: f32,
    /// Excess kurtosis of the weights (0 for a Gaussian; heavy tails are positive).
    pub kurtosis: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumReport {
    pub schema_version: u32,
    pub model: String,
    pub tensors: Vec<TensorSpectrum>,
    /// Lowest rank ratio first (the most compressible matrices).
    pub lowest_rank: Vec<String>,
    /// Highest outlier ratio first.
    pub most_outliers: Vec<String>,
    pub skipped: Vec<String>,
    pub elapsed_ms: u64,
}

pub fn spectrum(m: &ModelHandle, o: &SpectrumOptions, p: &dyn Progress) -> Result<SpectrumReport> {
    let started = std::time::Instant::now();
    let g = &m.gguf;
    let mut skipped = Vec::new();
    let picked: Vec<&TensorInfo> = g
        .tensors
        .iter()
        .filter(|t| {
            let big = matches!(t.name.as_str(), "token_embd.weight" | "output.weight");
            let layer_ok = match (&o.layers, layer_of(&t.name)) {
                (Some(ls), Some(l)) => ls.contains(&l),
                (Some(_), None) => false,
                (None, l) => l.is_some() || (big && o.embeddings),
            };
            if !layer_ok || !t.name.ends_with(".weight") {
                return false;
            }
            if t.dims.len() != 2 || t.dims[0] < 2 || t.dims[1] < 2 {
                // Norm vectors are 1-D; MoE experts are 3-D stacks.
                if t.dims.len() > 2 {
                    skipped.push(format!("{} (expert stack)", t.name));
                }
                return false;
            }
            if !t.ggml_type.is_known() {
                skipped.push(format!("{} (unknown type)", t.name));
                return false;
            }
            true
        })
        .collect();
    if picked.is_empty() {
        bail!("no weight matrices matched");
    }
    let total: u64 = picked.iter().map(|t| t.n_elements).sum();
    let mut done = 0u64;
    let mut tensors = Vec::with_capacity(picked.len());
    for t in picked {
        check_cancel(p)?;
        p.set(1.0 + 98.0 * done as f32 / total.max(1) as f32, &t.name);
        let mut w = Vec::new();
        g.dequantize_rows(t, 0, t.n_rows(), &mut w).with_context(|| format!("reading {}", t.name))?;
        tensors.push(matrix_spectrum(t, &w, o.iters.clamp(2, 64)));
        done += t.n_elements;
    }
    let mut lowest: Vec<&TensorSpectrum> = tensors.iter().collect();
    lowest.sort_by(|a, b| a.rank_ratio.total_cmp(&b.rank_ratio));
    let mut outl: Vec<&TensorSpectrum> = tensors.iter().collect();
    outl.sort_by(|a, b| b.outlier_ratio.total_cmp(&a.outlier_ratio));
    p.set(100.0, "done");
    Ok(SpectrumReport {
        schema_version: SCHEMA_VERSION,
        model: m.entry.name.clone(),
        lowest_rank: lowest.iter().take(10).map(|t| t.name.clone()).collect(),
        most_outliers: outl.iter().take(10).map(|t| t.name.clone()).collect(),
        tensors,
        skipped,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// Statistics of one matrix given as `rows` rows of `cols` values.
pub fn matrix_spectrum(t: &TensorInfo, w: &[f32], iters: usize) -> TensorSpectrum {
    let cols = t.row_len() as usize;
    let rows = t.n_rows() as usize;
    let n = w.len() as f64;
    let (sum, sq, max_abs) = w.par_chunks(cols).map(|r| {
        r.iter().fold((0f64, 0f64, 0f32), |(s, q, m), &x| (s + x as f64, q + (x as f64) * (x as f64), m.max(x.abs())))
    }).reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2.max(b.2)));
    let mean = sum / n;
    let var = (sq / n - mean * mean).max(1e-30);
    let m4: f64 = w.par_chunks(cols).map(|r| r.iter().map(|&x| (x as f64 - mean).powi(4)).sum::<f64>()).sum::<f64>() / n;
    let fro = sq.sqrt();
    let sigma = sigma_max(w, rows, cols, iters);
    let stable_rank = if sigma > 0.0 { sq / (sigma * sigma) } else { 0.0 };
    let rms = (sq / n).sqrt() as f32;
    TensorSpectrum {
        name: t.name.clone(),
        layer: layer_of(&t.name),
        rows: rows as u64,
        cols: cols as u64,
        quant: t.ggml_type.name().into_owned(),
        fro,
        sigma_max: sigma,
        stable_rank,
        rank_ratio: stable_rank / rows.min(cols).max(1) as f64,
        max_abs,
        rms,
        outlier_ratio: if rms > 0.0 { max_abs / rms } else { 0.0 },
        kurtosis: (m4 / (var * var) - 3.0) as f32,
    }
}

/// Largest singular value by power iteration on WᵀW, from a fixed start vector.
fn sigma_max(w: &[f32], rows: usize, cols: usize, iters: usize) -> f64 {
    let mut v: Vec<f64> = (0..cols).map(|i| 1.0 + ((i * 2654435761) % 1000) as f64 / 1000.0).collect();
    let norm = |x: &[f64]| x.iter().map(|a| a * a).sum::<f64>().sqrt();
    let nv = norm(&v);
    v.iter_mut().for_each(|x| *x /= nv);
    let mut sigma = 0.0;
    for _ in 0..iters {
        // u = W v
        let u: Vec<f64> = w.par_chunks(cols).map(|r| r.iter().zip(&v).map(|(a, b)| *a as f64 * b).sum()).collect();
        sigma = norm(&u);
        if sigma == 0.0 {
            return 0.0;
        }
        // v = Wᵀ u, normalized
        let chunk = (rows / rayon::current_num_threads().max(1)).max(64);
        v = w
            .par_chunks(cols * chunk)
            .enumerate()
            .map(|(ci, block)| {
                let mut acc = vec![0f64; cols];
                for (ri, r) in block.chunks(cols).enumerate() {
                    let ui = u[ci * chunk + ri];
                    for (a, x) in acc.iter_mut().zip(r) {
                        *a += ui * *x as f64;
                    }
                }
                acc
            })
            .reduce(|| vec![0f64; cols], |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            });
        let nv = norm(&v);
        if nv == 0.0 {
            return 0.0;
        }
        v.iter_mut().for_each(|x| *x /= nv);
    }
    sigma
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;

    fn info(rows: u64, cols: u64) -> TensorInfo {
        TensorInfo { name: "blk.0.test.weight".into(), dims: vec![cols, rows], ggml_type: GgmlType::F32, offset: 0, n_elements: rows * cols, n_bytes: rows * cols * 4, size_estimated: false }
    }

    #[test]
    fn rank_one_matrix_has_stable_rank_one() {
        let (rows, cols) = (8, 5);
        let a: Vec<f32> = (0..rows).map(|i| 1.0 + i as f32).collect();
        let b: Vec<f32> = (0..cols).map(|j| 0.5 - j as f32 * 0.2).collect();
        let w: Vec<f32> = a.iter().flat_map(|x| b.iter().map(move |y| x * y)).collect();
        let s = matrix_spectrum(&info(rows as u64, cols as u64), &w, 20);
        assert!((s.stable_rank - 1.0).abs() < 1e-6, "{}", s.stable_rank);
        assert_eq!(s.layer, Some(0));
    }

    #[test]
    fn identity_has_full_stable_rank() {
        let n = 6;
        let w: Vec<f32> = (0..n * n).map(|i| if i % (n + 1) == 0 { 1.0 } else { 0.0 }).collect();
        let s = matrix_spectrum(&info(n as u64, n as u64), &w, 30);
        assert!((s.sigma_max - 1.0).abs() < 1e-6);
        assert!((s.rank_ratio - 1.0).abs() < 1e-6);
    }
}
