//! Named edit snapshots on disk: save, load, list.

use super::*;

/// Snapshot files live in `<output_dir>/edits/<digest>/<name>.json`.
pub fn snapshot_dir(output_dir: &Path, m: &ModelHandle) -> PathBuf {
    output_dir.join("edits").join(&m.entry.digest)
}

pub fn snapshot_name(name: &str) -> Result<String> {
    let n = name.trim();
    if n.is_empty() || n.len() > 80 || !n.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) || n.starts_with('.') {
        bail!("snapshot names use letters, digits, '-', '_' and '.' only: {name:?}");
    }
    Ok(n.trim_end_matches(".json").to_string())
}

pub fn save_snapshot(output_dir: &Path, m: &ModelHandle, name: &str, edits: &[Edit]) -> Result<PathBuf> {
    let name = snapshot_name(name)?;
    let dir = snapshot_dir(output_dir, m);
    std::fs::create_dir_all(&dir)?;
    let p = dir.join(format!("{name}.json"));
    std::fs::write(&p, serde_json::to_vec_pretty(&EditsFile::new(m, Some(name), edits.to_vec()))?)?;
    Ok(p)
}

pub fn load_snapshot(output_dir: &Path, m: &ModelHandle, name: &str) -> Result<EditsFile> {
    let name = snapshot_name(name)?;
    let p = snapshot_dir(output_dir, m).join(format!("{name}.json"));
    let text = std::fs::read_to_string(&p).with_context(|| format!("no snapshot {name:?} ({})", p.display()))?;
    EditsFile::parse(&text)
}

pub fn list_snapshots(output_dir: &Path, m: &ModelHandle) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(snapshot_dir(output_dir, m))
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".json")).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}
