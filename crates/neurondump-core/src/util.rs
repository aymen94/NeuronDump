//! Small shared helpers.

use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA_VERSION: u32 = 1;

/// FNV-1a 64-bit; used for stable opaque ids and cache keys.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

pub fn short_id(s: &str) -> String {
    format!("{:012x}", fnv1a(s.as_bytes()) & 0xffff_ffff_ffff)
}

pub fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

pub fn human_count(n: u64) -> String {
    let v = n as f64;
    if v >= 1e9 {
        format!("{:.2}B", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.1}K", v / 1e3)
    } else {
        n.to_string()
    }
}

/// Levenshtein distance, used for "did you mean" suggestions.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Progress reporting + cooperative cancellation, implemented by the CLI (progress bars)
/// and the job system (shared state for SSE).
pub trait Progress: Sync {
    fn set(&self, percent: f32, step: &str);
    fn cancelled(&self) -> bool {
        false
    }
    /// Live GPU sample while a GPU job runs.
    fn gpu(&self, _sample: &crate::gpu::GpuSample) {}
}

pub struct NoProgress;

impl Progress for NoProgress {
    fn set(&self, _: f32, _: &str) {}
}

#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

pub fn check_cancel(p: &dyn Progress) -> anyhow::Result<()> {
    if p.cancelled() {
        Err(Cancelled.into())
    } else {
        Ok(())
    }
}

/// Converts a GGUF value to JSON, summarizing arrays longer than `max_items`.
pub fn value_json(v: &neurondump_gguf::Value, max_items: Option<usize>) -> serde_json::Value {
    use neurondump_gguf::Value;
    match (v, max_items) {
        (Value::Array(a), Some(n)) if a.len() as usize > n => serde_json::json!({
            "type": a.type_description(),
            "len": a.len(),
            "head": serde_json::to_value(a.head(n)).unwrap_or_default(),
            "truncated": true,
        }),
        (Value::Array(a), _) if a.is_skipped() => serde_json::json!({ "type": a.type_description(), "len": a.len() }),
        _ => serde_json::to_value(v).unwrap_or(serde_json::Value::Null),
    }
}
