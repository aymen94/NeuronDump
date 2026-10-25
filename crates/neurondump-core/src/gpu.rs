//! GPU support: device choice, NVIDIA device information through NVML (feature `nvml`),
//! live samples for running jobs, and a VRAM fit check that is plain arithmetic over the GGUF
//! (so it works in every build).

use std::fmt;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use neurondump_gguf::{GgmlType, GgufFile};

use crate::arch::{analyze_architecture, parse_tensor_name, Mixer};
use crate::util::{human_bytes, SCHEMA_VERSION};

/// Where inference runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceChoice {
    /// CUDA if compiled in and available, else CPU.
    #[default]
    Auto,
    Cpu,
    Cuda(usize),
}

impl DeviceChoice {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        Ok(match s.as_str() {
            "" | "auto" => DeviceChoice::Auto,
            "cpu" => DeviceChoice::Cpu,
            "cuda" | "gpu" => DeviceChoice::Cuda(0),
            _ => match s.strip_prefix("cuda:").and_then(|n| n.parse().ok()) {
                Some(n) => DeviceChoice::Cuda(n),
                None => bail!("unknown device {s:?} (auto, cpu, cuda, cuda:N)"),
            },
        })
    }
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceChoice::Auto => f.write_str("auto"),
            DeviceChoice::Cpu => f.write_str("cpu"),
            DeviceChoice::Cuda(n) => write!(f, "cuda:{n}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub index: u32,
    pub name: String,
    pub vram_total: u64,
    pub vram_free: u64,
    pub vram_used: u64,
    /// e.g. "7.5"
    pub compute_capability: Option<String>,
    /// Percent, sampled now.
    pub utilization: Option<u32>,
    pub temperature_c: Option<u32>,
    pub power_w: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub schema_version: u32,
    /// Built with NVML (feature `nvml`, implied by `cuda`).
    pub nvml_compiled: bool,
    /// Built with CUDA inference (feature `cuda`).
    pub cuda_compiled: bool,
    /// CUDA inference usable right now (compiled in and a device answers).
    pub cuda_available: bool,
    pub driver_version: Option<String>,
    /// CUDA version supported by the driver, e.g. "12.7".
    pub cuda_driver_version: Option<String>,
    pub devices: Vec<GpuDevice>,
    pub messages: Vec<String>,
}

/// One live sample, attached to running jobs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuSample {
    pub index: u32,
    pub vram_used: u64,
    pub vram_total: u64,
    pub utilization: Option<u32>,
}

impl GpuSample {
    pub fn short(&self) -> String {
        format!(
            "GPU {}{} · {}/{}",
            self.index,
            self.utilization.map(|u| format!(" {u}%")).unwrap_or_default(),
            human_bytes(self.vram_used),
            human_bytes(self.vram_total)
        )
    }
}

#[cfg(feature = "nvml")]
mod nv {
    use std::sync::OnceLock;

    use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
    use nvml_wrapper::Nvml;

    use super::{GpuDevice, GpuSample};

    fn nvml() -> Result<&'static Nvml, String> {
        static N: OnceLock<Result<Nvml, String>> = OnceLock::new();
        N.get_or_init(|| Nvml::init().map_err(|e| format!("NVML unavailable (no NVIDIA driver?): {e}"))).as_ref().map_err(Clone::clone)
    }

    pub fn versions() -> (Option<String>, Option<String>) {
        let Ok(n) = nvml() else { return (None, None) };
        let cuda = n.sys_cuda_driver_version().ok().map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10));
        (n.sys_driver_version().ok(), cuda)
    }

    pub fn devices() -> Result<Vec<GpuDevice>, String> {
        let n = nvml()?;
        let count = n.device_count().map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for i in 0..count {
            let d = n.device_by_index(i).map_err(|e| e.to_string())?;
            let mem = d.memory_info().ok();
            out.push(GpuDevice {
                index: i,
                name: d.name().unwrap_or_else(|_| "unknown".into()),
                vram_total: mem.as_ref().map_or(0, |m| m.total),
                vram_free: mem.as_ref().map_or(0, |m| m.free),
                vram_used: mem.as_ref().map_or(0, |m| m.used),
                compute_capability: d.cuda_compute_capability().ok().map(|c| format!("{}.{}", c.major, c.minor)),
                utilization: d.utilization_rates().ok().map(|u| u.gpu),
                temperature_c: d.temperature(TemperatureSensor::Gpu).ok(),
                power_w: d.power_usage().ok().map(|mw| mw as f64 / 1000.0),
            });
        }
        Ok(out)
    }

    pub fn sample(index: u32) -> Option<GpuSample> {
        let d = nvml().ok()?.device_by_index(index).ok()?;
        let mem = d.memory_info().ok()?;
        Some(GpuSample { index, vram_used: mem.used, vram_total: mem.total, utilization: d.utilization_rates().ok().map(|u| u.gpu) })
    }
}

/// Device information. Never fails: problems are reported in `messages`.
pub fn gpu_info() -> GpuInfo {
    let backend = crate::trace::backend();
    let cuda_compiled = backend.is_some_and(|b| b.cuda_compiled());
    let cuda_available = cuda_compiled && backend.is_some_and(|b| b.cuda_available());
    let mut messages = Vec::new();
    #[allow(unused_mut)]
    let mut info = GpuInfo {
        schema_version: SCHEMA_VERSION,
        nvml_compiled: cfg!(feature = "nvml"),
        cuda_compiled,
        cuda_available,
        driver_version: None,
        cuda_driver_version: None,
        devices: vec![],
        messages: vec![],
    };
    #[cfg(feature = "nvml")]
    {
        let (drv, cuda) = nv::versions();
        info.driver_version = drv;
        info.cuda_driver_version = cuda;
        match nv::devices() {
            Ok(d) if d.is_empty() => messages.push("no NVIDIA GPU found".into()),
            Ok(d) => info.devices = d,
            Err(e) => messages.push(e),
        }
    }
    #[cfg(not(feature = "nvml"))]
    messages.push("this build has no GPU information (build with --features nvml, or cuda)".into());
    if !cuda_compiled {
        messages.push(if backend.is_some() {
            "not compiled with CUDA: inference runs on the CPU (see SETUP-CUDA.md to build with --features cuda)".into()
        } else {
            "not compiled with trace: no inference at all (build with --features trace, or cuda)".into()
        });
    } else if !cuda_available {
        messages.push("compiled with CUDA, but no CUDA device answered (driver missing or too old?)".into());
    }
    info.messages = messages;
    info
}

/// Live VRAM / utilization of one GPU (None without NVML or GPU).
pub fn sample(index: u32) -> Option<GpuSample> {
    #[cfg(feature = "nvml")]
    {
        nv::sample(index)
    }
    #[cfg(not(feature = "nvml"))]
    {
        let _ = index;
        None
    }
}

// --- fit check ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitReport {
    pub schema_version: u32,
    pub model: String,
    /// Tokens processed in one forward pass (and kept in the KV cache).
    pub ctx: usize,
    /// Weights NeuronDump puts on the GPU (quantized matrices stay quantized).
    pub weights_bytes: u64,
    /// f16 KV cache for `ctx` tokens over the attention layers.
    pub kv_cache_bytes: u64,
    /// Peak working memory of one forward pass over `ctx` tokens.
    pub activation_bytes: u64,
    /// CUDA context, cuBLAS workspace and allocator slack.
    pub overhead_bytes: u64,
    pub total_bytes: u64,
    pub gpu: Option<String>,
    pub vram_total: Option<u64>,
    pub vram_free: Option<u64>,
    /// None when no GPU is known.
    pub fits: Option<bool>,
    pub verdict: String,
    pub details: Vec<String>,
}

const OVERHEAD: u64 = 500 << 20;

fn supported_quant(t: GgmlType) -> bool {
    matches!(t.0, 0 | 1 | 2 | 3 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 30)
}

/// VRAM needed to run `g` with `ctx` tokens per forward pass on a GPU.
pub fn fit(model: &str, g: &GgufFile, ctx: usize, gpu: Option<&GpuDevice>) -> FitReport {
    let arch = analyze_architecture(g);
    let hp = &arch.hyperparams;
    let tied = g.tensor("output.weight").is_none();
    let mut weights = 0u64;
    let mut largest_matrix = 0u64;
    for t in &g.tensors {
        let r = parse_tensor_name(&t.name);
        let role = r.role.as_str();
        // Embedding rows are read from the memory-mapped file per token, unless they are also
        // the output head; vision / MTP tensors are never loaded.
        if role == "token_embd" && !tied || role.starts_with("v.") || role.starts_with("mm.") || role.starts_with("mtp.") {
            continue;
        }
        let bytes = if t.dims.len() >= 2 && supported_quant(t.ggml_type) { t.n_bytes } else { t.n_elements * 4 };
        weights += bytes;
        if t.dims.len() >= 2 {
            largest_matrix = largest_matrix.max(t.n_elements * 4);
        }
    }
    let n_embd = hp.embedding_length.unwrap_or(0);
    let n_head = hp.head_count.unwrap_or(0);
    let n_kv = hp.head_count_kv.unwrap_or(n_head);
    let hd = hp.key_length.unwrap_or(if n_head > 0 { n_embd / n_head } else { 0 });
    let n_ff = hp.feed_forward_length.unwrap_or(4 * n_embd);
    let vocab = hp.vocab_size.unwrap_or(0);
    let attn_layers = arch.layers.iter().filter(|l| matches!(l.mixer, Mixer::FullAttention | Mixer::SlidingAttention)).count() as u64;
    let c = ctx as u64;
    let kv = 2 * attn_layers * c * n_kv * hd * 2;
    // One layer is live at a time; logits for every position when perplexity / KL is measured.
    let residual = 8 * c * n_embd * 4;
    let qkv = 2 * c * (n_head + 2 * n_kv) * hd * 4 + 2 * n_head * c * hd * 4;
    let scores = 2 * n_head * c * c * 4;
    let ffn = 3 * c * n_ff * 4;
    let logits = c * vocab * 4;
    // Quantized matmuls over more than a few rows dequantize the weight to f32 first.
    let dequant = if ctx > 8 { largest_matrix } else { 0 };
    let activations = residual + qkv + scores + ffn + logits + dequant;
    let total = weights + kv + activations + OVERHEAD;
    let mut details = vec![
        format!("weights {} (quantized matrices stay quantized; norms and unsupported types as f32)", human_bytes(weights)),
        format!("KV cache {} ({attn_layers} attention layers × {ctx} tokens × {n_kv} KV heads × {hd} dims × 2 × f16)", human_bytes(kv)),
        format!(
            "activations {} (attention scores {} = {n_head} heads × {ctx}²; logits {}; dequantized matrix {})",
            human_bytes(activations),
            human_bytes(scores),
            human_bytes(logits),
            human_bytes(dequant)
        ),
        format!("CUDA context and workspace ≈ {}", human_bytes(OVERHEAD)),
    ];
    if tied {
        details.push("token_embd is also the output head, so it is loaded on the GPU".into());
    }
    let (fits, verdict) = match gpu {
        Some(d) => {
            let free = d.vram_free.max(1);
            let ok = total <= free;
            let v = if ok {
                format!("fits on {}: needs {} of {} free ({} total)", d.name, human_bytes(total), human_bytes(d.vram_free), human_bytes(d.vram_total))
            } else if total <= d.vram_total {
                format!(
                    "does not fit right now on {}: needs {} but only {} of {} is free (close other GPU programs)",
                    d.name,
                    human_bytes(total),
                    human_bytes(d.vram_free),
                    human_bytes(d.vram_total)
                )
            } else {
                format!("does not fit on {}: needs {}, the GPU has {}", d.name, human_bytes(total), human_bytes(d.vram_total))
            };
            (Some(ok), v)
        }
        None => (None, format!("needs about {} of VRAM (no GPU information in this build or no NVIDIA GPU)", human_bytes(total))),
    };
    if fits == Some(false) && scores > total / 3 {
        details.push(format!("most of it is the attention scores for {ctx} tokens: a smaller --ctx (or --max-tokens) shrinks them quadratically"));
    }
    FitReport {
        schema_version: SCHEMA_VERSION,
        model: model.to_string(),
        ctx,
        weights_bytes: weights,
        kv_cache_bytes: kv,
        activation_bytes: activations,
        overhead_bytes: OVERHEAD,
        total_bytes: total,
        gpu: gpu.map(|d| d.name.clone()),
        vram_total: gpu.map(|d| d.vram_total),
        vram_free: gpu.map(|d| d.vram_free),
        fits,
        verdict,
        details,
    }
}

/// Default context for a fit check: 4096 tokens or the model's own limit.
pub fn default_ctx(g: &GgufFile) -> usize {
    let c = g.arch_key("context_length").and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
    c.min(4096)
}

/// The first NVIDIA GPU (or `index`), if NVML can see it.
pub fn device(index: u32) -> Option<GpuDevice> {
    #[cfg(feature = "nvml")]
    {
        nv::devices().ok()?.into_iter().find(|d| d.index == index)
    }
    #[cfg(not(feature = "nvml"))]
    {
        let _ = index;
        None
    }
}
