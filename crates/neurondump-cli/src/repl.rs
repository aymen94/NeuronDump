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

use neurondump_core::edits::EditOp;
use neurondump_core::graph::ModelGraph;
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
    ("graph", "graph [--level layer|block|tensor|fine] [--group-by g] [--format tree|dot|mermaid|json] [--expand id]"),
    ("node", "node <id>              what a graph node contains, stats, connections"),
    ("cut", "cut <id>...            add nodes to the cut selection (shows the static impact)"),
    ("uncut", "uncut <id>|all         remove nodes from the selection"),
    ("selection", "selection              the current cut selection"),
    ("impact", "impact [--force]       full static impact of the selection"),
    ("simulate", "simulate [prompt] [--tokens N] [--corpus f] [--force]   run the selection in memory (& for background)"),
    ("sweep", "sweep --corpus f [--level layer|head] [--layer N]   rank by damage (& for background)"),
    ("plan", "plan save <file> [--force]   write the selection as a prune plan for `slim`"),
    ("tokens", "tokens <text> [--chat]  tokenization, template tokens, embedding neighbours"),
    ("journey", "journey <idx> [prompt] [--chat] [--baseline]   follow a token through the layers (trace)"),
    ("generate", "generate [prompt] [--steps N] [--explain] [--temperature t] [--top-k k] [--top-p p] [--seed s]"),
    ("compare", "compare [prompt] [--tokens N] [--corpus f]   baseline vs the current edits"),
    ("edit", "edit scale <node> <f> | zero <node> | skip <layer> | ablate <ids> | steer <layer> \"<a>\" \"<b>\" <strength>"),
    ("edit ", "edit ban <token> | boost <token> <bias> | embed <token> scale <f>|replace <token> | patch <layer> [--head h] [--all] \"<src>\""),
    ("edit  ", "edit sampling <temp> [top_k] [top_p] [seed] | force <step> <token> | system <text> | template <file>"),
    ("edits", "edits [export <file> | import <file>]   the edit stack"),
    ("toggle", "toggle <id>            switch one edit on/off"),
    ("undo", "undo | redo | restore  edit history (restore = back to the original model)"),
    ("snapshot", "snapshot save|load <name>   named edit sets"),
    ("bake", "bake <file.gguf>       write the enabled edits into a new GGUF + Modelfile"),
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
    graph: Mutex<Option<Arc<ModelGraph>>>,
}

impl ReplHelper {
    /// Node ids for completion; fine ids (heads, neuron buckets, experts) once a layer prefix is typed.
    fn node_ids(&self, word: &str) -> Vec<String> {
        let Some(g) = self.graph.lock().unwrap().clone() else { return vec![] };
        let mut ids = g.ids();
        if let Some(rest) = word.strip_prefix("layer.") {
            if let Some((n, _)) = rest.split_once('.') {
                let lid = format!("layer.{n}");
                for b in g.children(&lid) {
                    if b.lazy {
                        ids.extend(g.fine_children(&b.id).into_iter().map(|c| c.id));
                    }
                }
            }
        }
        ids
    }
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
            (None, _) => {
                let mut names: Vec<String> = COMMANDS.iter().map(|c| c.0.trim().to_string()).collect();
                names.dedup();
                pairs(names)
            }
            (_, Some("--class")) => pairs(TokenClass::ALL.iter().map(|c| c.as_str().to_string()).collect()),
            (_, Some("--tensor")) | (_, Some("--role")) => pairs(self.tensors.lock().unwrap().clone()),
            (_, Some("--sort")) => pairs(["size", "type", "layer", "name", "offset"].map(String::from).to_vec()),
            (Some("graph"), Some("--format")) => pairs(["tree", "json", "dot", "mermaid"].map(String::from).to_vec()),
            (_, Some("--format")) => pairs(["json", "ndjson", "csv"].map(String::from).to_vec()),
            (_, Some("--table")) => pairs(["tensors", "vocab", "metadata", "layers"].map(String::from).to_vec()),
            (_, Some("--level")) => pairs(["section", "layer", "block", "tensor", "fine", "head", "expert"].map(String::from).to_vec()),
            (_, Some("--group-by")) => pairs(["section", "layer", "block", "quant", "kind", "importance"].map(String::from).to_vec()),
            (_, Some("--expand")) | (Some("node" | "cut" | "uncut"), _) => pairs(self.node_ids(word)),
            (Some("plan"), _) if words.len() == 1 => pairs(vec!["save".into()]),
            (Some("edit"), _) if words.len() == 1 => pairs(
                ["scale", "zero", "skip", "ablate", "steer", "ban", "boost", "embed", "patch", "sampling", "force", "system", "template"].map(String::from).to_vec(),
            ),
            (Some("edit"), Some("scale" | "zero" | "ablate")) => pairs(self.node_ids(word)),
            (Some("snapshot"), _) if words.len() == 1 => pairs(vec!["save".into(), "load".into()]),
            (Some("edits"), _) if words.len() == 1 => pairs(vec!["export".into(), "import".into()]),
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
fn background(ctx: &Ctx, model: &str, cmd: &str, rest: &[String], selection: &[String]) -> Result<String> {
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
        graph: Mutex::new(None),
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
                *h.graph.lock().unwrap() = ctx.app.graph(&mh).ok();
            }
        }
    };
    set_tensors(&mut ed, &current);
    println!("NeuronDump REPL — type `help` for commands, Tab to complete, `exit` to quit.");
    let web_running: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    // Cut selection for impact / simulate / plan (cleared by `use`).
    let mut selection: Vec<String> = Vec::new();
    // Last prompt given to tokens / journey / generate / compare.
    let mut last_prompt = String::new();

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
                    selection.clear();
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
                "cut" | "uncut" | "selection" | "impact" | "plan" => {
                    let model = need_model()?;
                    let m = ctx.app.open(&model)?;
                    let g = ctx.app.graph(&m)?;
                    let force = args.iter().any(|a| a == "--force");
                    match cmd.as_str() {
                        "cut" => {
                            if args.is_empty() {
                                bail!("cut <id>...");
                            }
                            for a in args.iter().filter(|a| !a.starts_with("--")) {
                                let id = g.resolve_id(a).ok_or_else(|| anyhow::anyhow!("unknown node {a:?} (Tab completes ids)"))?;
                                if !selection.contains(&id) {
                                    selection.push(id);
                                }
                            }
                        }
                        "uncut" => {
                            if args.first().is_some_and(|a| a == "all") {
                                selection.clear();
                            } else {
                                for a in &args {
                                    let id = g.resolve_id(a).unwrap_or_else(|| a.clone());
                                    selection.retain(|x| x != &id);
                                }
                            }
                        }
                        "plan" => {
                            let (Some("save"), Some(file)) = (args.first().map(String::as_str), args.get(1)) else { bail!("plan save <file>") };
                            let plan = ctx.app.plan_from_cut(&m, &selection, force)?;
                            std::fs::write(file, serde_json::to_vec_pretty(&plan)?).map_err(|e| anyhow::anyhow!("writing {file}: {e}"))?;
                            return Ok(format!("{}plan written to {file} (apply with `slim --plan {file} --out <name>.gguf`)\n", render::plan(&plan)));
                        }
                        _ => {}
                    }
                    if selection.is_empty() {
                        return Ok("selection is empty (`cut <id>`)\n".into());
                    }
                    let i = ctx.app.impact(&m, &selection, force)?;
                    if cmd == "impact" {
                        render::impact(&i)
                    } else {
                        format!(
                            "selection: {}\n{} params, {} -> {}, verdict {}{}\n",
                            selection.join(", "),
                            neurondump_core::util::human_count(i.params_removed + i.params_zeroed),
                            neurondump_core::util::human_bytes(i.original_size),
                            neurondump_core::util::human_bytes(i.new_size),
                            i.verdict.as_str(),
                            if i.refused { " (refused without --force)" } else { "" }
                        )
                    }
                }
                "tokens" | "journey" | "generate" | "compare" | "edit" | "edits" | "toggle" | "undo" | "redo" | "restore" | "snapshot" | "bake" if !bg => {
                    let model = need_model()?;
                    let m = ctx.app.open(&model)?;
                    let flag = |f: &str| args.iter().any(|a| a == f);
                    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
                    // Words that are not flags (or flag values) form the prompt.
                    let words = |skip_first: usize| -> Vec<String> {
                        let mut out = Vec::new();
                        let mut i = 0;
                        while i < args.len() {
                            let a = &args[i];
                            if a.starts_with("--") {
                                if !matches!(a.as_str(), "--chat" | "--explain" | "--baseline" | "--all") {
                                    i += 1;
                                }
                            } else if i >= skip_first {
                                out.push(a.clone());
                            }
                            i += 1;
                        }
                        out
                    };
                    let prompt_of = |w: Vec<String>, last: &str| -> Result<String> {
                        let p = if w.is_empty() { last.to_string() } else { w.join(" ") };
                        if p.is_empty() {
                            bail!("give a prompt (it is remembered for the next tokens / journey / generate / compare)");
                        }
                        Ok(p)
                    };
                    let chat = flag("--chat");
                    let view = |ctx: &Ctx| -> Result<String> { Ok(render::edit_view(&ctx.app.edit_view(&m)?)) };
                    match cmd.as_str() {
                        "tokens" => {
                            let p = prompt_of(words(0), &last_prompt)?;
                            last_prompt = p.clone();
                            let mut argv = vec!["neurondump".to_string()];
                            if ctx.global.json {
                                argv.push("--json".into());
                            }
                            argv.extend(["tokens".into(), model.clone(), p]);
                            if chat {
                                argv.push("--chat".into());
                            }
                            let cli = Cli::try_parse_from(&argv).map_err(|e| anyhow::anyhow!("{e}"))?;
                            run_command(ctx, &cli.command.ok_or_else(|| anyhow::anyhow!("no command"))?)?
                        }
                        "journey" => {
                            let idx: i64 = args.first().and_then(|a| a.parse().ok()).ok_or_else(|| anyhow::anyhow!("journey <token index> [prompt] (negative counts from the end)"))?;
                            let p = prompt_of(words(1), &last_prompt)?;
                            last_prompt = p.clone();
                            let params = json!({ "prompt": p, "chat": chat, "token": idx, "baseline": flag("--baseline") });
                            let v = ctx.run_job(JobKind::Journey, &model, params)?;
                            if ctx.global.json { serde_json::to_string_pretty(&v)? } else { render::journey(&serde_json::from_value(v)?) }
                        }
                        "generate" => {
                            let p = prompt_of(words(0), &last_prompt)?;
                            last_prompt = p.clone();
                            let num = |f: &str| value(f).and_then(|x| x.parse::<f64>().ok());
                            let params = json!({
                                "prompt": p, "chat": chat, "steps": num("--steps").map(|x| x as usize).unwrap_or(16), "explain": flag("--explain"),
                                "temperature": num("--temperature"), "sample_top_k": num("--top-k").map(|x| x as usize), "top_p": num("--top-p"),
                                "seed": num("--seed").map(|x| x as u64), "baseline": flag("--baseline"),
                            });
                            let v = ctx.run_job(JobKind::Generate, &model, params)?;
                            if ctx.global.json { serde_json::to_string_pretty(&v)? } else { render::generate(&serde_json::from_value(v)?, flag("--explain")) }
                        }
                        "compare" => {
                            let p = prompt_of(words(0), &last_prompt)?;
                            last_prompt = p.clone();
                            let tokens = value("--tokens").and_then(|x| x.parse::<usize>().ok()).unwrap_or(12);
                            let params = json!({ "prompt": p, "chat": chat, "tokens": tokens, "corpus_file": value("--corpus") });
                            let v = ctx.run_job(JobKind::Compare, &model, params)?;
                            let r: neurondump_core::app::CompareResult = serde_json::from_value(v)?;
                            render::edits_list(&r.edits, Some(&r.compiled)) + "\n" + &render::simulation(&r.simulation) + &render::compare_decisions(&r.simulation)
                        }
                        "edit" => {
                            let vocab = ctx.app.vocab(&m)?;
                            let tok = |s: Option<&String>| -> Result<u32> {
                                neurondump_core::edits::resolve_token(&vocab, s.ok_or_else(|| anyhow::anyhow!("missing token"))?)
                            };
                            let num = |s: Option<&String>, what: &str| -> Result<f32> {
                                s.and_then(|x| x.parse::<f32>().ok()).ok_or_else(|| anyhow::anyhow!("missing or bad {what}"))
                            };
                            let layer = |s: Option<&String>| -> Result<u32> { s.and_then(|x| x.trim_start_matches("layer.").parse().ok()).ok_or_else(|| anyhow::anyhow!("missing layer number")) };
                            let a = |i: usize| args.get(i);
                            let op = match a(0).map(String::as_str) {
                                Some("scale") => EditOp::Scale { target: a(1).cloned().ok_or_else(|| anyhow::anyhow!("edit scale <node> <factor>"))?, factor: num(a(2), "factor")? },
                                Some("zero") => EditOp::Zero { target: a(1).cloned().ok_or_else(|| anyhow::anyhow!("edit zero <node>"))? },
                                Some("skip") => EditOp::Skip { layer: layer(a(1))? },
                                Some("ablate") => EditOp::Ablate { cut: args[1..].to_vec() },
                                Some("steer") => EditOp::Steer {
                                    layer: layer(a(1))?,
                                    a: a(2).cloned().ok_or_else(|| anyhow::anyhow!("edit steer <layer> \"<a>\" \"<b>\" <strength>"))?,
                                    b: a(3).cloned().ok_or_else(|| anyhow::anyhow!("edit steer <layer> \"<a>\" \"<b>\" <strength>"))?,
                                    strength: num(a(4), "strength").unwrap_or(4.0),
                                },
                                Some("ban") => EditOp::Ban { token: tok(a(1))? },
                                Some("boost") => EditOp::Boost { token: tok(a(1))?, bias: num(a(2), "bias").unwrap_or(5.0) },
                                Some("embed") => match a(2).map(String::as_str) {
                                    Some("replace") => EditOp::Embed { token: tok(a(1))?, scale: None, replace_with: Some(tok(a(3))?) },
                                    Some("scale") => EditOp::Embed { token: tok(a(1))?, scale: Some(num(a(3), "factor")?), replace_with: None },
                                    _ => bail!("edit embed <token> scale <factor> | edit embed <token> replace <other token>"),
                                },
                                Some("patch") => {
                                    let rest = words(2);
                                    EditOp::Patch {
                                        layer: layer(a(1))?,
                                        head: value("--head").and_then(|h| h.parse().ok()),
                                        source: (!rest.is_empty()).then(|| rest.join(" ")).ok_or_else(|| anyhow::anyhow!("edit patch <layer> [--head h] [--all] \"<source prompt>\""))?,
                                        all_positions: flag("--all"),
                                    }
                                }
                                Some("sampling" | "temp" | "temperature") => EditOp::Sampling {
                                    temperature: num(a(1), "temperature")?,
                                    top_k: a(2).and_then(|x| x.parse().ok()),
                                    top_p: a(3).and_then(|x| x.parse().ok()),
                                    seed: a(4).and_then(|x| x.parse().ok()),
                                },
                                Some("force") => EditOp::Force { step: a(1).and_then(|x| x.parse().ok()).ok_or_else(|| anyhow::anyhow!("edit force <step> <token>"))?, token: tok(a(2))? },
                                Some("system") => EditOp::System { text: args[1..].join(" ") },
                                Some("template") => {
                                    let f = a(1).ok_or_else(|| anyhow::anyhow!("edit template <file with a Go template>"))?;
                                    EditOp::Template { text: std::fs::read_to_string(f).map_err(|e| anyhow::anyhow!("reading {f}: {e}"))? }
                                }
                                _ => bail!("edit scale|zero|skip|ablate|steer|ban|boost|embed|patch|sampling|force|system|template … (see `help`)"),
                            };
                            let id = ctx.app.push_edit(&m, op, None)?;
                            format!("added edit #{id}\n{}", view(ctx)?)
                        }
                        "edits" => match args.first().map(String::as_str) {
                            Some("export") => {
                                let f = args.get(1).ok_or_else(|| anyhow::anyhow!("edits export <file>"))?;
                                let file = neurondump_core::edits::EditsFile::new(&m, None, ctx.app.edit_stack(&m).edits);
                                std::fs::write(f, serde_json::to_vec_pretty(&file)?).map_err(|e| anyhow::anyhow!("writing {f}: {e}"))?;
                                format!("wrote {f} (apply with `neurondump edit {model} --apply {f}`)\n")
                            }
                            Some("import") => {
                                let f = args.get(1).ok_or_else(|| anyhow::anyhow!("edits import <file>"))?;
                                let text = std::fs::read_to_string(f).map_err(|e| anyhow::anyhow!("reading {f}: {e}"))?;
                                let w = ctx.app.import_edits(&m, neurondump_core::edits::EditsFile::parse(&text)?)?;
                                w.iter().map(|x| format!("warning: {x}\n")).collect::<String>() + &view(ctx)?
                            }
                            _ => view(ctx)?,
                        },
                        "toggle" => {
                            let id: u64 = args.first().and_then(|x| x.trim_start_matches('#').parse().ok()).ok_or_else(|| anyhow::anyhow!("toggle <edit id>"))?;
                            let on = ctx.app.edit(&m, |s| s.toggle(id))?;
                            format!("edit #{id} {}\n{}", if on { "on" } else { "off" }, view(ctx)?)
                        }
                        "undo" | "redo" | "restore" => {
                            let changed = ctx.app.edit(&m, |s| {
                                Ok(match cmd.as_str() {
                                    "undo" => s.undo(),
                                    "redo" => s.redo(),
                                    _ => {
                                        s.restore();
                                        true
                                    }
                                })
                            })?;
                            format!("{}{}", if changed { "" } else { "nothing to do\n" }, view(ctx)?)
                        }
                        "snapshot" => match (args.first().map(String::as_str), args.get(1)) {
                            (Some("save"), Some(n)) => format!("saved {}\n", ctx.app.save_snapshot(&m, n)?.display()),
                            (Some("load"), Some(n)) => {
                                let w = ctx.app.load_snapshot(&m, n)?;
                                w.iter().map(|x| format!("warning: {x}\n")).collect::<String>() + &view(ctx)?
                            }
                            _ => format!(
                                "snapshot save|load <name>; saved: {}\n",
                                neurondump_core::edits::list_snapshots(&ctx.app.config.output_dir, &m).join(", ")
                            ),
                        },
                        _ => {
                            let f = args.first().ok_or_else(|| anyhow::anyhow!("bake <file.gguf>"))?;
                            let abs = std::path::absolute(f).unwrap_or_else(|_| std::path::PathBuf::from(f));
                            let v = ctx.run_job(JobKind::Bake, &model, json!({ "out": abs }))?;
                            render::bake(&serde_json::from_value(v)?)
                        }
                    }
                }
                _ if bg => background(ctx, &need_model()?, &cmd, &args, &selection)?,
                "simulate" => {
                    if selection.is_empty() {
                        bail!("nothing selected (`cut <id>` first)");
                    }
                    let model = need_model()?;
                    let mut argv: Vec<String> = vec!["neurondump".into()];
                    if ctx.global.json {
                        argv.push("--json".into());
                    }
                    argv.extend(["ablate".into(), model, "--cut".into(), selection.join(",")]);
                    let prompt: Vec<String> = args.iter().take_while(|a| !a.starts_with("--")).cloned().collect();
                    if !prompt.is_empty() {
                        argv.extend(["--prompt".into(), prompt.join(" ")]);
                    }
                    argv.extend(args.iter().skip(prompt.len()).cloned());
                    let cli = Cli::try_parse_from(&argv).map_err(|e| anyhow::anyhow!("{e}"))?;
                    let Some(c) = cli.command else { bail!("no command") };
                    run_command(ctx, &c)?
                }
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
                        "node" => {
                            let Some(id) = args.first() else { bail!("node <id>") };
                            argv.extend(["node".into(), need_model()?, id.clone()]);
                        }
                        "tensors" | "layers" | "template" | "stats" | "analyze" | "slim" | "export" | "graph" | "sweep" => {
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
