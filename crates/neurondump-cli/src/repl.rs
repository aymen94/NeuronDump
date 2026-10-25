//! Interactive REPL (`neurondump cli`). Lines are translated into the same clap
//! commands as the one-shot CLI, with the current model injected.

use std::borrow::Cow;
use std::io::{BufRead, IsTerminal, Write};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use clap::Parser;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};
use serde_json::{json, Value};

use neurondump_core::tokenizer::TokenClass;
use neurondump_core::JobKind;

use crate::{render, run_command, Cli, Ctx, WebArgs};

const COMMANDS: &[(&str, &str)] = &[
    ("use", "use <model>            switch the current model"),
    ("models", "models                 list all local models"),
    ("info", "info [--metadata]      overview of the current model"),
    ("search", "search <text>          models + tokens, tensors, metadata of the current model"),
    ("vocab", "vocab [query] [--class c] [--regex] [--limit n] [--tokenize text]"),
    ("tensors", "tensors [--layer N] [--role r] [--sort size|type|layer]"),
    ("layer", "layer <N>              tensors of one layer"),
    ("layers", "layers                 per-layer kinds"),
    ("template", "template [--raw]       chat template decision tree"),
    ("stats", "stats [--tensor name]  weight statistics (append & to run in background)"),
    ("analyze", "analyze [--corpus f] [--stats] [--fast]   prune plan (& for background)"),
    ("slim", "slim --plan f --out name.gguf [--only vocab,layers]"),
    ("trace", "trace <prompt>         runtime trace (needs --features trace)"),
    ("compare", "compare <model>...     compare with other models"),
    ("export", "export --format json|ndjson|csv [--table t] [--out f]"),
    ("jobs", "jobs                   background jobs"),
    ("job", "job <id>               show a finished job's result"),
    ("cancel", "cancel <id>            cancel a job"),
    ("web", "web [--port N]         start the web UI on this session"),
    ("rescan", "rescan                 rediscover models"),
    ("help", "help                   this help"),
    ("exit", "exit                   quit"),
];

struct ReplHelper {
    models: Mutex<Vec<String>>,
    tensors: Mutex<Vec<String>>,
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> rustyline::Result<(usize, Vec<Pair>)> {
        let before = &line[..pos];
        let start = before.rfind(' ').map_or(0, |i| i + 1);
        let word = &before[start..];
        let words: Vec<&str> = before[..start].split_whitespace().collect();
        let pairs = |items: Vec<String>| -> Vec<Pair> {
            items.into_iter().filter(|c| c.starts_with(word)).map(|c| Pair { display: c.clone(), replacement: c }).collect()
        };
        let cands = match (words.first().copied(), words.last().copied()) {
            (None, _) => pairs(COMMANDS.iter().map(|c| c.0.to_string()).collect()),
            (_, Some("--class")) => pairs(TokenClass::ALL.iter().map(|c| c.as_str().to_string()).collect()),
            (_, Some("--tensor")) | (_, Some("--role")) => pairs(self.tensors.lock().unwrap().clone()),
            (_, Some("--sort")) => pairs(["size", "type", "layer", "name", "offset"].map(String::from).to_vec()),
            (_, Some("--format")) => pairs(["json", "ndjson", "csv"].map(String::from).to_vec()),
            (_, Some("--table")) => pairs(["tensors", "vocab", "metadata", "layers"].map(String::from).to_vec()),
            (Some("use" | "compare"), _) => pairs(self.models.lock().unwrap().clone()),
            _ => vec![],
        };
        Ok((start, cands))
    }
}

impl Hinter for ReplHelper {
    type Hint = String;
}

impl Highlighter for ReplHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(&'s self, prompt: &'p str, _default: bool) -> Cow<'b, str> {
        Cow::Borrowed(prompt)
    }
}

impl Validator for ReplHelper {}

impl Helper for ReplHelper {}

/// Minimal shell-like splitting with quotes.
fn split_args(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut any = false;
    for c in line.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => cur.push(c),
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                any = true;
            }
            None if c.is_whitespace() => {
                if !cur.is_empty() || any {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            None => cur.push(c),
        }
    }
    if !cur.is_empty() || any {
        out.push(cur);
    }
    out
}

fn page(text: &str) {
    let lines: Vec<&str> = text.lines().collect();
    let height = 50;
    if lines.len() <= height || !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
        return;
    }
    let stdin = std::io::stdin();
    for (i, chunk) in lines.chunks(height).enumerate() {
        for l in chunk {
            println!("{l}");
        }
        if (i + 1) * height < lines.len() {
            print!("-- more ({}/{}) Enter: next, q: quit -- ", ((i + 1) * height).min(lines.len()), lines.len());
            let _ = std::io::stdout().flush();
            let mut s = String::new();
            if stdin.lock().read_line(&mut s).is_err() || s.trim() == "q" {
                break;
            }
        }
    }
}

fn pick_model(ctx: &Ctx) -> Option<String> {
    let names: Vec<String> = ctx.app.library().entries.iter().map(|e| e.name.clone()).collect();
    if names.is_empty() {
        eprintln!("no models found; use `use <path.gguf>` or start with --scan <dir>");
        return None;
    }
    if !std::io::stdin().is_terminal() {
        return names.first().cloned();
    }
    let sel = dialoguer::FuzzySelect::new().with_prompt("Pick a model (type to filter)").items(&names).default(0).interact_opt().ok()??;
    Some(names[sel].clone())
}

/// Commands that run as core jobs when suffixed with `&`.
fn background(ctx: &Ctx, model: &str, cmd: &str, rest: &[String]) -> Result<String> {
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
        other => bail!("`{other}` cannot run in the background (stats, analyze, trace, info)"),
    };
    let id = ctx.app.submit(kind, model, params)?;
    Ok(format!("started job {id} (see `jobs`, `job {id}`, `cancel {id}`)\n"))
}

fn show_job(ctx: &Ctx, id: &str) -> Result<String> {
    let Some(info) = ctx.app.jobs.get(id) else { bail!("no job {id}") };
    let Some(v) = ctx.app.jobs.result(id) else {
        return Ok(format!("{}\n", render::jobs(&[info])));
    };
    Ok(match info.kind {
        JobKind::Stats => render::stats(&serde_json::from_value(v)?, 40),
        JobKind::Analyze => render::plan(&serde_json::from_value(v)?),
        JobKind::Slim => render::slim(&serde_json::from_value(v)?),
        JobKind::Trace => render::trace(&serde_json::from_value(v)?),
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

pub fn run(ctx: &Ctx) -> Result<()> {
    let helper = ReplHelper {
        models: Mutex::new(ctx.app.library().entries.iter().map(|e| e.name.clone()).collect()),
        tensors: Mutex::new(Vec::new()),
    };
    let mut ed: Editor<ReplHelper, FileHistory> = Editor::new()?;
    ed.set_helper(Some(helper));
    let hist = neurondump_core::config::config_path().with_file_name("history.txt");
    let _ = ed.load_history(&hist);

    let mut current = pick_model(ctx);
    let set_tensors = |ed: &mut Editor<ReplHelper, FileHistory>, model: &Option<String>| {
        if let (Some(h), Some(m)) = (ed.helper(), model) {
            if let Ok(mh) = ctx.app.open(m) {
                let mut names: Vec<String> = mh.gguf.tensors.iter().map(|t| t.name.clone()).collect();
                let mut roles: Vec<String> =
                    names.iter().map(|n| neurondump_core::arch::parse_tensor_name(n).role).collect();
                roles.sort();
                roles.dedup();
                names.extend(roles);
                *h.tensors.lock().unwrap() = names;
            }
        }
    };
    set_tensors(&mut ed, &current);
    println!("NeuronDump REPL — type `help` for commands, Tab to complete, `exit` to quit.");
    let web_running: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

    loop {
        let prompt = format!("neurondump[{}]> ", current.as_deref().unwrap_or("-"));
        let line = match ed.readline(&prompt) {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => continue,
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let _ = ed.add_history_entry(line);
        let mut args = split_args(line);
        let bg = args.last().is_some_and(|a| a == "&");
        if bg {
            args.pop();
        }
        let cmd = args.remove(0);
        let snapshot = current.clone();
        let need_model = || -> Result<String> {
            snapshot.clone().ok_or_else(|| anyhow::anyhow!("no model selected (`use <model>`)"))
        };
        let result: Result<String> = (|| {
            Ok(match cmd.as_str() {
                "exit" | "quit" | "q" => return Err(anyhow::anyhow!("__exit__")),
                "help" | "?" => {
                    let mut s = String::from("Commands:\n");
                    for (_, h) in COMMANDS {
                        s.push_str(&format!("  {h}\n"));
                    }
                    s
                }
                "use" => {
                    let Some(m) = args.first() else { bail!("use <model>") };
                    let e = ctx.app.resolve(m)?;
                    current = Some(e.name.clone());
                    set_tensors(&mut ed, &current);
                    format!("using {} ({})\n", e.name, e.path.display())
                }
                "rescan" => {
                    ctx.app.rescan();
                    if let Some(h) = ed.helper() {
                        *h.models.lock().unwrap() = ctx.app.library().entries.iter().map(|e| e.name.clone()).collect();
                    }
                    format!("{} models\n", ctx.app.library().entries.len())
                }
                "jobs" => render::jobs(&ctx.app.jobs.list()),
                "job" => show_job(ctx, args.first().map(String::as_str).unwrap_or(""))?,
                "cancel" => {
                    let id = args.first().cloned().unwrap_or_default();
                    if ctx.app.jobs.cancel(&id) {
                        format!("cancelling {id}\n")
                    } else {
                        bail!("no running job {id}")
                    }
                }
                "web" => {
                    let Some(launch) = ctx.web.clone() else { bail!("this build has no web interface") };
                    let mut running = web_running.lock().unwrap();
                    if *running {
                        bail!("web UI already running");
                    }
                    let port = args.iter().position(|a| a == "--port").and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok()).unwrap_or(ctx.app.config.port);
                    let app = ctx.app.clone();
                    std::thread::spawn(move || {
                        if let Err(e) = launch(app, WebArgs { host: "127.0.0.1".into(), port, open: true }) {
                            eprintln!("web: {e:#}");
                        }
                    });
                    *running = true;
                    "web UI starting in the background (URL with token printed above)\n".into()
                }
                _ if bg => background(ctx, &need_model()?, &cmd, &args)?,
                _ => {
                    let mut argv: Vec<String> = vec!["neurondump".into()];
                    if ctx.global.json {
                        argv.push("--json".into());
                    }
                    match cmd.as_str() {
                        "models" | "list" => argv.push("list".into()),
                        "info" | "inspect" => {
                            argv.extend(["inspect".into(), need_model()?]);
                            argv.extend(args.clone());
                        }
                        "vocab" => {
                            argv.extend(["vocab".into(), need_model()?]);
                            if let Some(first) = args.first().filter(|a| !a.starts_with("--")) {
                                argv.extend(["--search".into(), first.clone()]);
                                argv.extend(args[1..].iter().cloned());
                            } else {
                                argv.extend(args.clone());
                            }
                        }
                        "search" => {
                            if args.is_empty() {
                                bail!("search <text>");
                            }
                            argv.extend(["search".into(), args.join(" ")]);
                            if let Some(m) = &snapshot {
                                argv.extend(["--model".into(), m.clone()]);
                            }
                        }
                        "layer" => {
                            let Some(n) = args.first() else { bail!("layer <N>") };
                            argv.extend(["tensors".into(), need_model()?, "--layer".into(), n.clone()]);
                        }
                        "trace" => {
                            argv.extend(["trace".into(), need_model()?]);
                            if args.first().is_some_and(|a| !a.starts_with("--")) {
                                let prompt: Vec<String> = args.iter().take_while(|a| !a.starts_with("--")).cloned().collect();
                                argv.extend(["--prompt".into(), prompt.join(" ")]);
                                argv.extend(args.iter().skip(prompt.len()).cloned());
                            } else {
                                argv.extend(args.clone());
                            }
                        }
                        "compare" => {
                            argv.extend(["compare".into(), need_model()?]);
                            argv.extend(args.clone());
                        }
                        "tensors" | "layers" | "template" | "stats" | "analyze" | "slim" | "export" => {
                            argv.extend([cmd.clone(), need_model()?]);
                            argv.extend(args.clone());
                        }
                        "config" => {
                            argv.push("config".into());
                            argv.extend(args.clone());
                        }
                        other => bail!("unknown command `{other}` (type `help`)"),
                    }
                    let cli = Cli::try_parse_from(&argv).map_err(|e| anyhow::anyhow!("{e}"))?;
                    let Some(c) = cli.command else { bail!("no command") };
                    let sub = Ctx { app: ctx.app.clone(), global: ctx.global.clone(), web: ctx.web.clone() };
                    run_command(&sub, &c)?
                }
            })
        })();
        match result {
            Ok(out) => page(&out),
            Err(e) if e.to_string() == "__exit__" => break,
            Err(e) => eprintln!("error: {e:#}"),
        }
    }
    if let Some(d) = hist.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = ed.save_history(&hist);
    Ok(())
}
