//! Background jobs from the REPL (`cmd &`) and showing a finished job's result.

use super::*;

/// Commands that run as core jobs when suffixed with `&`.
pub(super) fn background(ctx: &Ctx, model: &str, cmd: &str, rest: &[String], selection: &[String]) -> Result<String> {
    let has = |f: &str| rest.iter().any(|a| a == f);
    let values = |f: &str| -> Vec<String> {
        let mut v = Vec::new();
        let mut i = 0;
        while i < rest.len() {
            if rest[i] == f && i + 1 < rest.len() {
                v.push(rest[i + 1].clone());
                i += 1;
            }
            i += 1;
        }
        v
    };
    let (kind, params) = match cmd {
        "stats" => (JobKind::Stats, json!({ "tensors": values("--tensor") })),
        "analyze" => (
            JobKind::Analyze,
            json!({ "corpus_files": values("--corpus"), "compute_stats": has("--stats"), "skip_layer_similarity": has("--fast") }),
        ),
        "trace" => {
            let prompt: Vec<&String> = rest.iter().filter(|a| !a.starts_with("--")).collect();
            (JobKind::Trace, json!({ "prompt": prompt.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ") }))
        }
        "info" | "inspect" => (JobKind::Inspect, json!({ "stats": true })),
        "simulate" => {
            if selection.is_empty() {
                bail!("nothing selected (`cut <id>` first)");
            }
            let prompt: Vec<&String> = rest.iter().take_while(|a| !a.starts_with("--")).collect();
            let prompt = (!prompt.is_empty()).then(|| prompt.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" "));
            let tokens = values("--tokens").first().and_then(|t| t.parse::<usize>().ok()).unwrap_or(16);
            (
                JobKind::Ablate,
                json!({ "cut": selection, "prompt": prompt, "corpus_file": values("--corpus").first(), "tokens": tokens, "force": has("--force") }),
            )
        }
        "sweep" => (
            JobKind::Sweep,
            json!({
                "level": values("--level").first().cloned().unwrap_or_else(|| "layer".into()),
                "layer": values("--layer").first().and_then(|l| l.parse::<u32>().ok()),
                "corpus_file": values("--corpus").first(),
            }),
        ),
        other => bail!("`{other}` cannot run in the background (stats, analyze, trace, info, simulate, sweep)"),
    };
    let id = ctx.app.submit(kind, model, params)?;
    Ok(format!("started job {id} (see `jobs`, `job {id}`, `cancel {id}`)\n"))
}

pub(super) fn show_job(ctx: &Ctx, id: &str) -> Result<String> {
    let Some(info) = ctx.app.jobs.get(id) else { bail!("no job {id}") };
    let Some(v) = ctx.app.jobs.result(id) else {
        return Ok(format!("{}\n", render::jobs(&[info])));
    };
    Ok(match info.kind {
        JobKind::Stats => render::stats(&serde_json::from_value(v)?, 40),
        JobKind::Analyze => render::plan(&serde_json::from_value(v)?),
        JobKind::Slim => render::slim(&serde_json::from_value(v)?),
        JobKind::Trace => render::trace(&serde_json::from_value(v)?),
        JobKind::Ablate => {
            let r: neurondump_core::ablate::AblateResult = serde_json::from_value(v)?;
            let mut s = render::impact(&r.impact);
            if let Some(sim) = &r.simulation {
                s.push('\n');
                s.push_str(&render::simulation(sim));
            }
            s
        }
        JobKind::Sweep => render::sweep(&serde_json::from_value(v)?),
        JobKind::Journey => render::journey(&serde_json::from_value(v)?),
        JobKind::Generate => render::generate(&serde_json::from_value(v)?, false),
        JobKind::Compare => {
            let r: neurondump_core::app::CompareResult = serde_json::from_value(v)?;
            render::simulation(&r.simulation) + &render::compare_decisions(&r.simulation)
        }
        JobKind::Bake => render::bake(&serde_json::from_value(v)?),
        JobKind::Diff => render::diff(&serde_json::from_value(v)?, 20),
        JobKind::Inspect => {
            let stats: Value = v["stats"].clone();
            if stats.is_null() {
                serde_json::to_string_pretty(&v["architecture"]["hyperparams"])?
            } else {
                render::stats(&serde_json::from_value(stats)?, 40)
            }
        }
    })
}
