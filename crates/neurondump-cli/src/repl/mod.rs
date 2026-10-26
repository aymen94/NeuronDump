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

mod helper;
mod util;
mod jobs;
use helper::*;
use util::*;
use jobs::*;

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
                        "diff" => {
                            if args.first().is_none_or(|a| a.starts_with("--")) {
                                bail!("diff <model> [--top N] [--metadata-only]");
                            }
                            argv.extend(["diff".into(), need_model()?]);
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
