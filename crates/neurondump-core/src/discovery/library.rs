//! The model library: scanning all sources, rescans on change, resolving names / ids / paths, summaries, dedup.

use super::*;

impl Library {
    pub fn discover(options: DiscoveryOptions) -> Self {
        let mut entries = Vec::new();
        let mut warnings = Vec::new();
        if !options.no_defaults {
            for dir in ollama_model_dirs() {
                ollama::scan(&dir, &mut entries, &mut warnings);
            }
            files::scan_default_caches(&mut entries, &mut warnings);
        }
        // Env dirs are ambient configuration, so missing ones are skipped without a warning.
        let env_dirs: Vec<PathBuf> = env_model_dirs().into_iter().filter(|d| d.is_dir()).collect();
        for dir in options.extra_dirs.iter().chain(&env_dirs) {
            if dir.join("manifests").is_dir() && dir.join("blobs").is_dir() {
                ollama::scan(dir, &mut entries, &mut warnings);
            } else {
                files::scan_dir(dir, SourceKind::Directory, &mut entries, &mut warnings);
            }
        }
        dedup(&mut entries);
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        let fingerprint = Self::compute_fingerprint(&options);
        Self { entries, warnings, options, fingerprint }
    }

    pub fn rescan(&mut self) {
        let extra: Vec<ModelEntry> =
            self.entries.iter().filter(|e| e.source == SourceKind::File).cloned().collect();
        *self = Self::discover(self.options.clone());
        for e in extra {
            if !self.entries.iter().any(|x| x.path == e.path) {
                self.entries.push(e);
            }
        }
    }

    /// Cheap change detector over Ollama manifest dirs and scan dirs (count + newest mtime).
    pub(super) fn compute_fingerprint(options: &DiscoveryOptions) -> u64 {
        let mut dirs: Vec<PathBuf> = Vec::new();
        if !options.no_defaults {
            for d in ollama_model_dirs() {
                dirs.push(d.join("manifests"));
                dirs.push(d.join("manifests-v2"));
            }
        }
        dirs.extend(options.extra_dirs.iter().cloned());
        dirs.extend(env_model_dirs());
        let mut count = 0u64;
        let mut newest = 0u64;
        for d in dirs {
            walk_mtimes(&d, 0, &mut count, &mut newest);
        }
        count.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ newest
    }

    /// Rescans if anything changed on disk since the last scan. Returns true if it did.
    pub fn refresh_if_changed(&mut self) -> bool {
        if Self::compute_fingerprint(&self.options) != self.fingerprint {
            self.rescan();
            true
        } else {
            false
        }
    }

    pub fn by_id(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Resolves `llama3.2:1b`, `ns/model:tag`, `host/ns/model:tag`, an id, a file
    /// stem, or a path to a `.gguf` (which is added to the library).
    pub fn resolve(&mut self, query: &str) -> Result<ModelEntry> {
        let q = query.trim();
        if let Some(e) = self.find(q) {
            return Ok(e.clone());
        }
        let p = Path::new(q);
        if p.is_file() {
            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| q.to_string());
            let e = ModelEntry::from_file(p, SourceKind::File, name);
            if let Some(existing) = self.entries.iter().find(|x| x.path == e.path) {
                return Ok(existing.clone());
            }
            self.entries.push(e.clone());
            return Ok(e);
        }
        let mut suggestions: Vec<(usize, &str)> = self
            .entries
            .iter()
            .map(|e| (edit_distance(&e.name.to_lowercase(), &q.to_lowercase()), e.name.as_str()))
            .collect();
        suggestions.sort();
        let close: Vec<&str> = suggestions.iter().filter(|(d, _)| *d <= 4.max(q.len() / 3)).take(3).map(|s| s.1).collect();
        if close.is_empty() {
            bail!("no model named {q:?} (run `neurondump list` to see available models)");
        }
        bail!("no model named {q:?}; did you mean: {}?", close.join(", "))
    }

    pub(super) fn find(&self, q: &str) -> Option<&ModelEntry> {
        if let Some(e) = self.entries.iter().find(|e| e.id == q || e.name == q) {
            return Some(e);
        }
        let norm = ollama::normalize_name(q);
        if let Some(e) = self
            .entries
            .iter()
            .find(|e| e.ollama.as_ref().is_some_and(|o| o.full_name() == norm))
        {
            return Some(e);
        }
        let lower = q.to_lowercase();
        self.entries.iter().find(|e| {
            e.name.to_lowercase() == lower
                || e.path.file_stem().is_some_and(|s| s.to_string_lossy().to_lowercase() == lower)
        })
    }

    /// Header-only summaries of all entries, read in parallel.
    #[cfg(feature = "native")]
    pub fn summaries(&self) -> Vec<ModelSummary> {
        use rayon::prelude::*;
        self.entries.par_iter().map(summarize).collect()
    }
}

#[cfg(feature = "native")]
pub fn summarize(e: &ModelEntry) -> ModelSummary {
    use neurondump_gguf::ParseOptions;
    let mut s = ModelSummary {
        id: e.id.clone(),
        name: e.name.clone(),
        source: e.source,
        architecture: None,
        params: None,
        quantization: None,
        context_length: None,
        size: e.size,
        path: e.path.clone(),
        has_projector: e.projector.is_some(),
        error: None,
    };
    match GgufFile::open_with(&e.path, ParseOptions::header_only()) {
        Ok(g) => {
            s.architecture = g.architecture().map(str::to_string);
            s.params = Some(g.total_params());
            s.quantization = quantization_label(&g);
            s.context_length = g.arch_key("context_length").and_then(|v| v.as_u64());
        }
        Err(err) => s.error = Some(err.to_string()),
    }
    s
}

pub(super) fn walk_mtimes(dir: &Path, depth: usize, count: &mut u64, newest: &mut u64) {
    if depth > 6 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for ent in rd.flatten() {
        let Ok(md) = ent.metadata() else { continue };
        *count += 1;
        if let Ok(m) = md.modified() {
            let secs = m.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            *newest = (*newest).max(secs);
        }
        if md.is_dir() {
            walk_mtimes(&ent.path(), depth + 1, count, newest);
        }
    }
}

/// Removes duplicate discoveries of the same entry and links entries sharing weights.
pub(super) fn dedup(entries: &mut Vec<ModelEntry>) {
    let mut seen: HashMap<(String, PathBuf), usize> = HashMap::new();
    let mut keep = Vec::with_capacity(entries.len());
    for e in entries.drain(..) {
        let key = (e.name.clone(), e.path.clone());
        if seen.contains_key(&key) {
            continue;
        }
        // Plain files that are already known through another source (e.g. Ollama blob).
        if e.ollama.is_none() && keep.iter().any(|k: &ModelEntry| k.path == e.path) {
            continue;
        }
        seen.insert(key, keep.len());
        keep.push(e);
    }
    let mut by_path: HashMap<PathBuf, Vec<String>> = HashMap::new();
    for e in &keep {
        by_path.entry(e.path.clone()).or_default().push(e.name.clone());
    }
    for e in &mut keep {
        if let Some(names) = by_path.get(&e.path) {
            e.shares_weights_with = names.iter().filter(|n| **n != e.name).cloned().collect();
        }
    }
    *entries = keep;
}
