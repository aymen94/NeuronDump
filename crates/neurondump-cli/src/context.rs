//! Command context (`Ctx`) and helpers: JSON output, edits files, inspecting one or all models, stdout.

use super::*;

pub struct Ctx {
    pub app: Arc<App>,
    pub global: Global,
    pub web: Option<WebLauncher>,
}

impl Ctx {
    pub(super) fn bar(&self, label: &str) -> Bar {
        Bar::new(label, self.global.quiet)
    }

    pub(super) fn run_job(&self, kind: JobKind, model: &str, params: Value) -> Result<Value> {
        let bar = self.bar(kind.as_str());
        let r = self.app.run(kind, model, params, &bar);
        bar.finish();
        let (v, cached) = r?;
        if cached && !self.global.quiet {
            eprintln!("(cached result)");
        }
        Ok(v)
    }
}

pub(super) fn to_json<T: serde::Serialize>(v: &T) -> Result<String> {
    Ok(serde_json::to_string_pretty(v)? + "\n")
}

/// Reads an edits.json (portable patch) given on the command line.
pub(super) fn load_edits(p: Option<&PathBuf>) -> Result<Option<Vec<neurondump_core::edits::Edit>>> {
    let Some(p) = p else { return Ok(None) };
    let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(Some(neurondump_core::edits::EditsFile::parse(&text)?.edits))
}

/// Inspect with statistics coming from the shared cache (computed with a progress bar).
pub(super) fn inspect_one(ctx: &Ctx, model: &str, opts: &InspectOptions) -> Result<InspectReport> {
    let m = ctx.app.open(model)?;
    let mut r = inspect(&m, &InspectOptions { full: opts.full, stats: false }, &NoProgress)?;
    if opts.stats {
        let t = Instant::now();
        let bar = ctx.bar("stats");
        let s = ctx.app.stats(&m, true, &bar);
        bar.finish();
        r.stats = s?;
        r.timings.stats_ms = t.elapsed().as_millis() as u64;
    }
    Ok(r)
}

/// Writes to stdout, treating a closed pipe (`| head`) as a normal end of output.
pub fn write_stdout(s: &str) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    if let Err(e) = out.write_all(s.as_bytes()).and_then(|_| out.flush()) {
        if e.kind() != std::io::ErrorKind::BrokenPipe {
            eprintln!("error writing output: {e}");
        }
    }
}

pub(super) fn file_safe(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' }).collect()
}

pub(super) fn inspect_all(ctx: &Ctx, opts: &InspectOptions, out: Option<&PathBuf>) -> Result<String> {
    let entries = ctx.app.library().entries.clone();
    let mut s = String::new();
    let mut failed = 0;
    let mut results = Vec::new();
    for e in &entries {
        let t = Instant::now();
        let r = inspect_one(ctx, &e.id, opts).and_then(|r| Ok(serde_json::to_value(r)?));
        match r {
            Ok(v) => {
                let warnings = v["warnings"].as_array().map_or(0, Vec::len);
                s.push_str(&format!(
                    "ok    {:<45} {:<12} {:>4} layers {:>8} tokens  {:>3} warnings  {} ms\n",
                    e.name,
                    v["architecture"]["name"].as_str().unwrap_or("?"),
                    v["architecture"]["layers"].as_array().map_or(0, Vec::len),
                    v["tokenizer"]["size"].as_u64().unwrap_or(0),
                    warnings,
                    t.elapsed().as_millis()
                ));
                if let Some(dir) = out {
                    std::fs::create_dir_all(dir)?;
                    std::fs::write(dir.join(format!("{}.inspect.json", file_safe(&e.name))), serde_json::to_vec_pretty(&v)?)?;
                }
                results.push(json!({ "model": e.name, "ok": true, "warnings": v["warnings"] }));
            }
            Err(err) => {
                failed += 1;
                s.push_str(&format!("FAIL  {:<45} {err:#}\n", e.name));
                results.push(json!({ "model": e.name, "ok": false, "error": format!("{err:#}") }));
            }
        }
    }
    s.push_str(&format!("{} model(s), {failed} failed\n", entries.len()));
    if failed > 0 {
        let out = if ctx.global.json { to_json(&results)? } else { s };
        write_stdout(&out);
        bail!("{failed} model(s) failed to inspect");
    }
    if ctx.global.json {
        to_json(&results)
    } else {
        Ok(s)
    }
}
