//! CLI front end: one-shot subcommands, interface picker and the interactive REPL.
//! Contains no analysis logic — everything is a call into `neurondump-core`.

mod progress;
pub mod render;
mod repl;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Args, CommandFactory, Parser, Subcommand};
use serde_json::{json, Value};

use neurondump_core::analyze::PrunePlan;
use neurondump_core::compare::compare;
use neurondump_core::config::{config_path, Config, UiMode};
use neurondump_core::export::{render as export_render, Format, Table};
use neurondump_core::graph::{GroupBy, Level, ViewOptions};
use neurondump_core::inspect::{inspect, InspectOptions, InspectReport};
use neurondump_core::slim::SlimReport;
use neurondump_core::stats::StatsReport;
use neurondump_core::template::template_report;
use neurondump_core::tokenizer::{Encoder, VocabQuery};
use neurondump_core::tokens::TokenizeRequest;
use neurondump_core::trace::TraceReport;
use neurondump_core::{App, DiscoveryOptions, JobKind, NoProgress};

pub use progress::Bar;

#[derive(Debug, Clone)]
pub struct WebArgs {
    pub host: String,
    pub port: u16,
    pub open: bool,
}

/// Launches the web UI on an existing session (provided by the binary when built with
/// the `web` feature).
pub type WebLauncher = Arc<dyn Fn(Arc<App>, WebArgs) -> Result<()> + Send + Sync>;

#[derive(Debug, Parser)]
#[command(name = "neurondump", version, about = "Reverse-engineer, analyze and slim down local GGUF LLMs")]
pub struct Cli {
    #[command(flatten)]
    pub global: Global,
    #[command(subcommand)]
    pub command: Option<Cmd>,
}

#[derive(Debug, Clone, Args)]
pub struct Global {
    /// Machine-readable JSON output.
    #[arg(long, global = true)]
    pub json: bool,
    /// Worker threads for analysis (default: all cores).
    #[arg(long, global = true)]
    pub threads: Option<usize>,
    /// Print a timing breakdown to stderr.
    #[arg(long, global = true)]
    pub timings: bool,
    /// No progress bars or informational messages.
    #[arg(long, short, global = true)]
    pub quiet: bool,
    /// Extra directory to scan for .gguf files (repeatable).
    #[arg(long = "scan", global = true)]
    pub scan: Vec<PathBuf>,
    /// Interface to start when no subcommand is given.
    #[arg(long, global = true, env = "NEURONDUMP_UI")]
    pub ui: Option<String>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Cmd {
    /// Start the interactive REPL.
    Cli,
    /// Start the local web interface.
    Web {
        #[arg(long)]
        port: Option<u16>,
        /// Open the browser.
        #[arg(long)]
        open: bool,
        /// Bind address (default 127.0.0.1). Binding to other interfaces exposes your files.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Show or edit settings.
    Config {
        #[command(subcommand)]
        action: Option<ConfigAction>,
    },
    /// List all local models.
    List,
    /// Extract everything inside a model.
    Inspect {
        model: Option<String>,
        /// Inspect every discovered model.
        #[arg(long)]
        all: bool,
        /// Compute weight statistics (reads all tensor data).
        #[arg(long)]
        stats: bool,
        /// Do not summarize large arrays; include the full vocabulary in JSON.
        #[arg(long)]
        full: bool,
        /// Write `<model>.inspect.json` reports to this directory.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Print all metadata keys.
        #[arg(long)]
        metadata: bool,
    },
    /// Search models; with --model also its tokens, tensors and metadata.
    Search {
        query: String,
        #[arg(long, short)]
        model: Option<String>,
        /// Maximum results per kind.
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Search and classify the vocabulary.
    Vocab {
        model: String,
        #[arg(long, short)]
        search: Option<String>,
        /// Treat --search as a regular expression.
        #[arg(long)]
        regex: bool,
        /// special, reserved, byte, whitespace, digit, punctuation, emoji, cjk, code, word, subword, mixed
        #[arg(long)]
        class: Option<String>,
        #[arg(long)]
        script: Option<String>,
        /// Programming language for code tokens (rust, python, javascript, c_cpp, java, go, sql, shell).
        #[arg(long)]
        lang: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// Tokenize this text instead of listing tokens.
        #[arg(long)]
        tokenize: Option<String>,
    },
    /// Tensor map.
    Tensors {
        model: String,
        #[arg(long)]
        layer: Option<u32>,
        #[arg(long)]
        role: Option<String>,
        /// size, type, layer, name, offset
        #[arg(long, default_value = "offset")]
        sort: String,
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Per-layer structure (attention / linear / MoE / dense).
    Layers { model: String },
    /// Chat template decision tree.
    Template {
        model: String,
        /// Also print the raw template source.
        #[arg(long)]
        raw: bool,
    },
    /// Weight statistics.
    Stats {
        model: String,
        #[arg(long)]
        tensor: Vec<String>,
        #[arg(long)]
        layer: Vec<u32>,
        #[arg(long, default_value_t = 40)]
        limit: usize,
    },
    /// Find removable parts and write a prune plan.
    Analyze {
        model: String,
        /// Corpus files for vocabulary trimming.
        #[arg(long, num_args = 1..)]
        corpus: Vec<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Compute weight statistics if not cached (enables head/neuron/expert heuristics).
        #[arg(long)]
        stats: bool,
        /// Skip the layer weight-similarity pass.
        #[arg(long)]
        fast: bool,
        /// Allow first/last layers as removal candidates.
        #[arg(long)]
        include_edge_layers: bool,
    },
    /// Apply a prune plan and write a slimmed GGUF.
    Slim {
        model: String,
        #[arg(long)]
        plan: PathBuf,
        /// Comma-separated candidate ids or groups: vocab, layers, tied, quant, projector, metadata.
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// Output file name (written to <output_dir>/slim/).
        #[arg(long)]
        out: String,
        /// Text file for a llama-perplexity before/after comparison.
        #[arg(long)]
        perplexity: Option<PathBuf>,
    },
    /// Run the model and record per-layer internals.
    Trace {
        model: String,
        #[arg(long)]
        prompt: Option<String>,
        /// File of texts separated by blank lines (aggregated statistics).
        #[arg(long)]
        corpus: Option<PathBuf>,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        max_tokens: Option<usize>,
    },
    /// Compare models side by side.
    Compare {
        #[arg(num_args = 2.., required = true)]
        models: Vec<String>,
    },
    /// Export a report as json, ndjson or csv.
    Export {
        model: String,
        #[arg(long, default_value = "json")]
        format: String,
        /// Table for ndjson/csv: tensors, vocab, metadata, layers.
        #[arg(long, default_value = "tensors")]
        table: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// List background jobs (REPL / web sessions).
    Jobs,
    /// Structural graph: sections, layers, blocks, tensors, heads/neurons/experts.
    Graph {
        model: String,
        /// section, layer, block, tensor, fine
        #[arg(long, default_value = "layer")]
        level: String,
        /// section, layer, block, quant, kind, importance
        #[arg(long, default_value = "section")]
        group_by: String,
        /// tree, json, dot, mermaid
        #[arg(long, default_value = "tree")]
        format: String,
        /// Node ids to expand (comma-separated), e.g. layer.7,layer.7.attn
        #[arg(long, value_delimiter = ',')]
        expand: Vec<String>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// What a graph node contains, its stats, connections and removable flag.
    Node { model: String, id: String },
    /// "What if I cut this?": static impact, then a simulated run (feature `trace`).
    Ablate {
        model: String,
        /// Node ids (comma-separated): layer.7, layer.7.attn, layer.7.attn.head.3, token_embd, vocab:100-200, ...
        #[arg(long, value_delimiter = ',', required = true)]
        cut: Vec<String>,
        /// Static impact only (no inference).
        #[arg(long = "static")]
        static_only: bool,
        #[arg(long)]
        prompt: Option<String>,
        /// Texts separated by blank lines: perplexity and mean KL.
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// Greedy tokens to generate, baseline vs ablated.
        #[arg(long, default_value_t = 16)]
        tokens: usize,
        /// Allow `breaks_model` cuts.
        #[arg(long)]
        force: bool,
        /// Write the cut as a prune plan for `slim`.
        #[arg(long)]
        save_plan: Option<PathBuf>,
        #[arg(long)]
        max_tokens: Option<usize>,
    },
    /// Ablate each layer (or head / expert) one at a time and rank them by damage.
    Sweep {
        model: String,
        /// layer, head, expert
        #[arg(long, default_value = "layer")]
        level: String,
        /// Layer for --level head / expert.
        #[arg(long)]
        layer: Option<u32>,
        /// Texts separated by blank lines.
        #[arg(long)]
        corpus: PathBuf,
        /// Where to write the sweep JSON (default <model>.sweep.json).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Tokens per sample (default 64).
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Also write a prune plan with the --top least damaging entries.
        #[arg(long)]
        save_plan: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        top: usize,
        /// Allow `breaks_model` plans.
        #[arg(long)]
        force: bool,
    },
    /// Tokenization: ids, text, class, template tokens, embedding norms and neighbours.
    Tokens {
        model: String,
        text: String,
        /// Wrap the text in the chat template (shows the full templated prompt).
        #[arg(long)]
        chat: bool,
        #[arg(long)]
        system: Option<String>,
        /// Nearest neighbours per token by embedding cosine (0 = skip).
        #[arg(long, default_value_t = 5)]
        neighbors: usize,
    },
    /// Follow one token through the layers: logit lens, attention, FFN, decision breakdown (feature `trace`).
    Journey {
        model: String,
        #[command(flatten)]
        prompt: PromptArgs,
        /// Token index (negative counts from the end; -1 = last).
        #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
        token: i64,
        /// Apply these edits (edits.json); default: none.
        #[arg(long)]
        edits: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Generate tokens with a per-step timeline (feature `trace`).
    Generate {
        model: String,
        #[command(flatten)]
        prompt: PromptArgs,
        #[arg(long, default_value_t = 16)]
        steps: usize,
        /// Decision breakdown for every step.
        #[arg(long)]
        explain: bool,
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        top_p: Option<f32>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        edits: Option<PathBuf>,
    },
    /// Apply an edits.json in memory: baseline vs edited comparison, optionally bake a new GGUF.
    Edit {
        model: String,
        /// The edits file (REPL `edits export`, the web UI, or hand-written).
        #[arg(long)]
        apply: PathBuf,
        /// Prompt(s) to compare (repeatable).
        #[arg(long)]
        prompt: Vec<String>,
        #[arg(long)]
        chat: bool,
        /// Corpus (texts separated by blank lines) for perplexity and mean KL.
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// Tokens generated per prompt on both sides.
        #[arg(long, default_value_t = 16)]
        tokens: usize,
        /// Write the edits into a new GGUF (never the original).
        #[arg(long)]
        bake: Option<PathBuf>,
        /// Only bake, no comparison.
        #[arg(long)]
        no_compare: bool,
    },
}

#[derive(Debug, Clone, Args)]
pub struct PromptArgs {
    #[arg(long)]
    pub prompt: String,
    /// Wrap the prompt in the model's chat template.
    #[arg(long)]
    pub chat: bool,
    /// System prompt (implies --chat; empty = none).
    #[arg(long)]
    pub system: Option<String>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ConfigAction {
    /// Print the configuration.
    Show,
    /// Set a key: default_ui, port, scan_dirs, output_dir, threads.
    Set { key: String, value: String },
    /// Restore defaults.
    Reset,
    /// Print the config file path.
    Path,
}

pub struct Ctx {
    pub app: Arc<App>,
    pub global: Global,
    pub web: Option<WebLauncher>,
}

impl Ctx {
    fn bar(&self, label: &str) -> Bar {
        Bar::new(label, self.global.quiet)
    }

    fn run_job(&self, kind: JobKind, model: &str, params: Value) -> Result<Value> {
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

fn to_json<T: serde::Serialize>(v: &T) -> Result<String> {
    Ok(serde_json::to_string_pretty(v)? + "\n")
}

/// Executes one command and returns its output text.
pub fn run_command(ctx: &Ctx, cmd: &Cmd) -> Result<String> {
    let json = ctx.global.json;
    let app = &ctx.app;
    match cmd {
        Cmd::Cli | Cmd::Web { .. } => bail!("not available here"),
        Cmd::Config { action } => {
            let mut c = Config::load();
            match action.clone().unwrap_or(ConfigAction::Show) {
                ConfigAction::Show => Ok(if json { to_json(&c)? } else { format!("# {}\n{}", config_path().display(), c.to_toml()) }),
                ConfigAction::Set { key, value } => {
                    c.set(&key, &value)?;
                    let p = c.save()?;
                    Ok(format!("saved {}\n", p.display()))
                }
                ConfigAction::Reset => {
                    let p = Config::default().save()?;
                    Ok(format!("reset {}\n", p.display()))
                }
                ConfigAction::Path => Ok(format!("{}\n", config_path().display())),
            }
        }
        Cmd::List => {
            let lib = app.library();
            let rows = lib.summaries();
            if json {
                to_json(&json!({ "schema_version": 1, "models": rows, "warnings": lib.warnings }))
            } else {
                Ok(render::list(&rows, &lib.warnings))
            }
        }
        Cmd::Inspect { model, all, stats, full, out, metadata } => {
            let opts = InspectOptions { full: *full, stats: *stats };
            if *all {
                return inspect_all(ctx, &opts, out.as_ref());
            }
            let Some(model) = model else { bail!("give a model name or --all") };
            let t = Instant::now();
            let r = inspect_one(ctx, model, &opts)?;
            if ctx.global.timings {
                eprintln!(
                    "timings: architecture {} ms, tokenizer {} ms, templates {} ms, stats {} ms, total {} ms",
                    r.timings.architecture_ms, r.timings.tokenizer_ms, r.timings.template_ms, r.timings.stats_ms, t.elapsed().as_millis()
                );
            }
            let v = serde_json::to_value(&r)?;
            if let Some(dir) = out {
                std::fs::create_dir_all(dir)?;
                let p = dir.join(format!("{}.inspect.json", file_safe(&r.model.name)));
                std::fs::write(&p, serde_json::to_vec_pretty(&v)?)?;
                if !ctx.global.quiet {
                    eprintln!("wrote {}", p.display());
                }
            }
            if json {
                return to_json(&v);
            }
            let mut s = render::inspect(&r);
            if *metadata {
                s.push_str(&render::metadata(&r));
            }
            Ok(s)
        }
        Cmd::Search { query, model, limit } => {
            let models = app.library().summaries();
            let r = match model {
                Some(m) => {
                    let h = app.open(m)?;
                    let v = app.vocab(&h).ok();
                    neurondump_core::search::search(query, &models, Some((&h, v.as_deref())), *limit)
                }
                None => neurondump_core::search::search(query, &models, None, *limit),
            };
            if json {
                to_json(&r)
            } else {
                Ok(render::search(&r))
            }
        }
        Cmd::Vocab { model, search, regex, class, script, lang, limit, offset, tokenize } => {
            let m = app.open(model)?;
            let v = app.vocab(&m)?;
            if let Some(text) = tokenize {
                let enc = Encoder::new(&v)?;
                let e = enc.encode(text, true, None);
                if json {
                    return to_json(&e);
                }
                let mut s = String::new();
                for w in &enc.warnings {
                    s.push_str(&format!("warning: {w}\n"));
                }
                s.push_str(&format!("{} tokens\n", e.ids.len()));
                for (id, p) in e.ids.iter().zip(&e.pieces) {
                    s.push_str(&format!("{id:>8} -> '{}'\n", render::esc(p)));
                }
                return Ok(s);
            }
            let q = VocabQuery {
                q: search.clone(),
                regex: *regex,
                class: class.clone(),
                script: script.clone(),
                lang: lang.clone(),
                offset: Some(*offset),
                limit: Some(*limit),
            };
            let page = v.search(&q)?;
            if json {
                to_json(&json!({ "summary": v.summary, "page": page }))
            } else {
                Ok(render::vocab(&page))
            }
        }
        Cmd::Tensors { model, layer, role, sort, limit } => {
            let m = app.open(model)?;
            let rows = neurondump_core::inspect::tensor_rows(&m.gguf);
            let mut sel: Vec<_> = rows
                .iter()
                .filter(|r| layer.is_none_or(|l| r.layer == Some(l)))
                .filter(|r| role.as_ref().is_none_or(|ro| r.role == *ro || r.role.contains(ro.as_str())))
                .collect();
            match sort.as_str() {
                "size" => sel.sort_by(|a, b| b.n_bytes.cmp(&a.n_bytes)),
                "type" => sel.sort_by(|a, b| a.ggml_type.cmp(&b.ggml_type).then(a.name.cmp(&b.name))),
                "layer" => sel.sort_by(|a, b| a.layer.cmp(&b.layer).then(a.name.cmp(&b.name))),
                "name" => sel.sort_by(|a, b| a.name.cmp(&b.name)),
                "offset" => sel.sort_by_key(|r| r.offset),
                other => bail!("unknown sort {other:?} (size, type, layer, name, offset)"),
            }
            if let Some(l) = limit {
                sel.truncate(*l);
            }
            if json {
                to_json(&sel)
            } else {
                Ok(render::tensors(&sel))
            }
        }
        Cmd::Layers { model } => {
            let m = app.open(model)?;
            let r = inspect(&m, &InspectOptions::default(), &NoProgress)?;
            if json {
                to_json(&r.architecture.layers)
            } else {
                Ok(render::layers(&r))
            }
        }
        Cmd::Template { model, raw } => {
            let m = app.open(model)?;
            let t = template_report(&m);
            if json {
                to_json(&t)
            } else {
                Ok(render::template(&t, *raw))
            }
        }
        Cmd::Stats { model, tensor, layer, limit } => {
            let v = ctx.run_job(JobKind::Stats, model, json!({ "tensors": tensor, "layers": layer }))?;
            if json {
                return to_json(&v);
            }
            let r: StatsReport = serde_json::from_value(v)?;
            Ok(render::stats(&r, *limit))
        }
        Cmd::Analyze { model, corpus, out, stats, fast, include_edge_layers } => {
            let params = json!({
                "corpus_files": corpus,
                "compute_stats": stats,
                "skip_layer_similarity": fast,
                "include_edge_layers": include_edge_layers,
            });
            let v = ctx.run_job(JobKind::Analyze, model, params)?;
            let plan: PrunePlan = serde_json::from_value(v.clone())?;
            let out = out.clone().unwrap_or_else(|| PathBuf::from(format!("{}.prune.json", file_safe(&plan.model))));
            std::fs::write(&out, serde_json::to_vec_pretty(&v)?).with_context(|| format!("writing {}", out.display()))?;
            if json {
                return to_json(&v);
            }
            Ok(format!("{}\nplan written to {} (edit `selected` flags or use `slim --only`)\n", render::plan(&plan), out.display()))
        }
        Cmd::Slim { model, plan, only, out, perplexity } => {
            let plan = App::load_plan(plan)?;
            let params = json!({ "plan": plan, "only": only, "out": out, "perplexity_corpus": perplexity });
            let v = ctx.run_job(JobKind::Slim, model, params)?;
            if json {
                return to_json(&v);
            }
            let r: SlimReport = serde_json::from_value(v)?;
            Ok(render::slim(&r))
        }
        Cmd::Trace { model, prompt, corpus, top_k, max_tokens } => {
            let params = json!({ "prompt": prompt, "corpus_file": corpus, "top_k": top_k, "max_tokens": max_tokens });
            let v = ctx.run_job(JobKind::Trace, model, params)?;
            if json {
                return to_json(&v);
            }
            let r: TraceReport = serde_json::from_value(v)?;
            Ok(render::trace(&r))
        }
        Cmd::Compare { models } => {
            let handles: Vec<_> = models.iter().map(|m| app.open(m)).collect::<Result<_>>()?;
            let refs: Vec<_> = handles.iter().map(|h| h.as_ref()).collect();
            let r = compare(&refs);
            if json {
                to_json(&r)
            } else {
                Ok(render::compare(&r))
            }
        }
        Cmd::Export { model, format, table, out } => {
            let f = Format::parse(format)?;
            let t = Table::parse(table)?;
            let m = app.open(model)?;
            let r = inspect(&m, &InspectOptions { full: f == Format::Json, stats: false }, &NoProgress)?;
            let tokens = if t == Table::Vocab { Some(app.vocab(&m)?) } else { None };
            let text = export_render(&r, tokens.as_ref().map(|v| v.tokens.as_slice()), f, t)?;
            match out {
                Some(p) => {
                    std::fs::write(p, &text)?;
                    Ok(format!("wrote {} ({} bytes)\n", p.display(), text.len()))
                }
                None => Ok(text),
            }
        }
        Cmd::Jobs => {
            let jobs = app.jobs.list();
            if json {
                to_json(&jobs)
            } else {
                Ok(render::jobs(&jobs))
            }
        }
        Cmd::Graph { model, level, group_by, format, expand, out } => {
            let m = app.open(model)?;
            let g = app.graph(&m)?;
            let level = Level::parse(level).ok_or_else(|| anyhow::anyhow!("--level must be section, layer, block, tensor or fine"))?;
            let group_by = GroupBy::parse(group_by).ok_or_else(|| anyhow::anyhow!("--group-by must be section, layer, block, quant, kind or importance"))?;
            for e in expand {
                if g.resolve_id(e).is_none() {
                    bail!("unknown node {e:?} (ids are shown after # in `graph`)");
                }
            }
            let v = g.view(&ViewOptions { level, group_by, expand: expand.clone() });
            let text = match (format.as_str(), json) {
                ("json", _) | (_, true) => to_json(&v)?,
                ("tree", _) => v.to_tree(),
                ("dot", _) => v.to_dot(),
                ("mermaid", _) => v.to_mermaid(),
                (other, _) => bail!("unknown --format {other:?} (tree, json, dot, mermaid)"),
            };
            match out {
                Some(p) => {
                    std::fs::write(p, &text).with_context(|| format!("writing {}", p.display()))?;
                    Ok(format!("wrote {} ({} nodes, {} edges)\n", p.display(), v.nodes.len(), v.edges.len()))
                }
                None => Ok(text),
            }
        }
        Cmd::Node { model, id } => {
            let m = app.open(model)?;
            let g = app.graph(&m)?;
            let rid = g.resolve_id(id).ok_or_else(|| anyhow::anyhow!("unknown node {id:?} (try `graph {model} --level tensor`)"))?;
            let d = g.detail(&rid).ok_or_else(|| anyhow::anyhow!("unknown node {id:?}"))?;
            if json {
                to_json(&d)
            } else {
                Ok(render::node(&d))
            }
        }
        Cmd::Ablate { model, cut, static_only, prompt, corpus, tokens, force, save_plan, max_tokens } => {
            let m = app.open(model)?;
            let impact = app.impact(&m, cut, *force)?;
            let mut s = render::impact(&impact);
            if impact.cut.is_empty() {
                bail!("{s}nothing to cut: unknown id(s) {}", impact.unknown.join(", "));
            }
            if impact.refused {
                if json {
                    write_stdout(&to_json(&json!({ "impact": impact }))?);
                } else {
                    write_stdout(&s);
                }
                bail!("refused: cutting this is `breaks_model`; pass --force to simulate or plan it anyway");
            }
            let mut plan_path = None;
            if let Some(p) = save_plan {
                let plan = app.plan_from_cut(&m, cut, *force)?;
                std::fs::write(p, serde_json::to_vec_pretty(&plan)?).with_context(|| format!("writing {}", p.display()))?;
                s.push_str(&format!("plan written to {} (apply with `slim {model} --plan {} --out <name>.gguf`)\n", p.display(), p.display()));
                plan_path = Some(p.clone());
            }
            let mut simulation = None;
            if !*static_only {
                if impact.simulation_supported {
                    let params = json!({
                        "cut": cut, "force": force, "prompt": prompt, "corpus_file": corpus, "tokens": tokens, "max_tokens": max_tokens,
                    });
                    let v = ctx.run_job(JobKind::Ablate, model, params)?;
                    let r: neurondump_core::ablate::AblateResult = serde_json::from_value(v)?;
                    if let Some(sim) = &r.simulation {
                        s.push('\n');
                        s.push_str(&render::simulation(sim));
                    }
                    simulation = r.simulation;
                } else {
                    s.push_str("(no simulation: see the notes above; showing static impact only)\n");
                }
            }
            if json {
                return to_json(&json!({ "impact": impact, "simulation": simulation, "plan": plan_path }));
            }
            Ok(s)
        }
        Cmd::Sweep { model, level, layer, corpus, out, max_tokens, save_plan, top, force } => {
            let params = json!({ "level": level, "layer": layer, "corpus_file": corpus, "max_tokens": max_tokens });
            let v = ctx.run_job(JobKind::Sweep, model, params)?;
            let r: neurondump_core::trace::SweepReport = serde_json::from_value(v.clone())?;
            let out = out.clone().unwrap_or_else(|| PathBuf::from(format!("{}.sweep.json", file_safe(&r.model))));
            std::fs::write(&out, serde_json::to_vec_pretty(&v)?).with_context(|| format!("writing {}", out.display()))?;
            let mut s = render::sweep(&r);
            s.push_str(&format!("sweep written to {}\n", out.display()));
            if let Some(p) = save_plan {
                let m = app.open(model)?;
                let (plan, impact) = app.plan_from_sweep(&m, &r, *top, *force)?;
                std::fs::write(p, serde_json::to_vec_pretty(&plan)?).with_context(|| format!("writing {}", p.display()))?;
                s.push_str(&format!(
                    "plan with the {top} least damaging {}(s) written to {} — {} → {} ({})\n",
                    r.level,
                    p.display(),
                    neurondump_core::util::human_bytes(impact.original_size),
                    neurondump_core::util::human_bytes(impact.new_size),
                    impact.verdict.as_str()
                ));
            }
            if json {
                return to_json(&v);
            }
            Ok(s)
        }
        Cmd::Tokens { model, text, chat, system, neighbors } => {
            let m = app.open(model)?;
            let v = app.vocab(&m)?;
            let req = TokenizeRequest { text: text.clone(), chat: *chat || system.is_some(), system: system.clone(), template: None, neighbors: *neighbors };
            let banned = app.edit_view(&m).ok().and_then(|v| v.compiled).map(|c| c.intervention.mask_tokens).unwrap_or_default();
            let bar = ctx.bar("tokens");
            let r = neurondump_core::tokens::tokenize(&m, &v, &req, &banned, &bar);
            bar.finish();
            let r = r?;
            if json {
                to_json(&r)
            } else {
                Ok(render::tokens(&r))
            }
        }
        Cmd::Journey { model, prompt, token, edits, top_k } => {
            let e = load_edits(edits.as_ref())?;
            let params = json!({
                "prompt": prompt.prompt, "chat": prompt.chat || prompt.system.is_some(), "system": prompt.system,
                "token": token, "top_k": top_k, "edits": e.unwrap_or_default(),
            });
            let v = ctx.run_job(JobKind::Journey, model, params)?;
            if json {
                return to_json(&v);
            }
            Ok(render::journey(&serde_json::from_value(v)?))
        }
        Cmd::Generate { model, prompt, steps, explain, temperature, top_k, top_p, seed, edits } => {
            let e = load_edits(edits.as_ref())?;
            let params = json!({
                "prompt": prompt.prompt, "chat": prompt.chat || prompt.system.is_some(), "system": prompt.system,
                "steps": steps, "explain": explain, "temperature": temperature, "sample_top_k": top_k, "top_p": top_p, "seed": seed,
                "edits": e.unwrap_or_default(),
            });
            let v = ctx.run_job(JobKind::Generate, model, params)?;
            if json {
                return to_json(&v);
            }
            Ok(render::generate(&serde_json::from_value(v)?, *explain))
        }
        Cmd::Edit { model, apply, prompt, chat, corpus, tokens, bake, no_compare } => {
            let e = load_edits(Some(apply))?.unwrap_or_default();
            let mut out = String::new();
            let mut jv = json!({});
            {
                // Validate the edits (and show what they compile to) before running anything.
                let m = app.open(model)?;
                let g = app.graph(&m)?;
                let v = app.vocab(&m).ok();
                let c = neurondump_core::edits::compile(&g, v.as_deref(), &e)?;
                out.push_str(&render::edits_list(&e, Some(&c)));
            }
            if !*no_compare {
                let params = json!({ "prompts": prompt, "chat": chat, "corpus_file": corpus, "tokens": tokens, "edits": e });
                let v = ctx.run_job(JobKind::Compare, model, params)?;
                let r: neurondump_core::app::CompareResult = serde_json::from_value(v.clone())?;
                out.push('\n');
                out.push_str(&render::simulation(&r.simulation));
                out.push_str(&render::compare_decisions(&r.simulation));
                jv["compare"] = v;
            }
            if let Some(path) = bake {
                let abs = std::path::absolute(path).unwrap_or_else(|_| path.clone());
                let v = ctx.run_job(JobKind::Bake, model, json!({ "edits": e, "out": abs }))?;
                let r: neurondump_core::edits::BakeReport = serde_json::from_value(v.clone())?;
                out.push('\n');
                out.push_str(&render::bake(&r));
                jv["bake"] = v;
            }
            if json {
                return to_json(&jv);
            }
            Ok(out)
        }
    }
}

/// Reads an edits.json (portable patch) given on the command line.
fn load_edits(p: Option<&PathBuf>) -> Result<Option<Vec<neurondump_core::edits::Edit>>> {
    let Some(p) = p else { return Ok(None) };
    let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(Some(neurondump_core::edits::EditsFile::parse(&text)?.edits))
}

/// Inspect with statistics coming from the shared cache (computed with a progress bar).
fn inspect_one(ctx: &Ctx, model: &str, opts: &InspectOptions) -> Result<InspectReport> {
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

fn file_safe(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn inspect_all(ctx: &Ctx, opts: &InspectOptions, out: Option<&PathBuf>) -> Result<String> {
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

fn pick_interface(config: &mut Config) -> Result<UiMode> {
    use dialoguer::{Confirm, Select};
    let items = ["CLI (interactive)", "Web interface"];
    let sel = Select::new().with_prompt("How do you want to use NeuronDump?").items(&items).default(0).interact()?;
    let mode = if sel == 0 { UiMode::Cli } else { UiMode::Web };
    if Confirm::new().with_prompt("Remember my choice?").default(false).interact()? {
        config.default_ui = Some(mode);
        let p = config.save()?;
        eprintln!("saved to {} (change with `neurondump config set default_ui none`)", p.display());
    }
    Ok(mode)
}

/// Entry point used by the binary. Returns the process exit code.
pub fn main(web: Option<WebLauncher>) -> i32 {
    let cli = Cli::parse();
    match run(cli, web) {
        Ok(code) => code,
        Err(e) => {
            if e.downcast_ref::<neurondump_core::util::Cancelled>().is_some() {
                eprintln!("cancelled");
                return 130;
            }
            eprintln!("error: {e:#}");
            1
        }
    }
}

fn run(cli: Cli, web: Option<WebLauncher>) -> Result<i32> {
    let mut config = Config::load();
    if let Some(n) = cli.global.threads.or(config.threads) {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }

    // Decide the interface when no subcommand is given.
    let command = match cli.command.clone() {
        Some(c) => c,
        None => {
            let forced = cli.global.ui.as_deref().map(|u| UiMode::parse(u).ok_or_else(|| anyhow::anyhow!("--ui must be cli or web")));
            let mode = match forced {
                Some(m) => Some(m?),
                None => config.default_ui,
            };
            let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
            let mode = match mode {
                Some(m) => m,
                None if interactive => pick_interface(&mut config)?,
                None => {
                    let _ = Cli::command().print_help();
                    eprintln!("\nerror: no subcommand given and stdin is not a terminal");
                    return Ok(2);
                }
            };
            match mode {
                UiMode::Cli => Cmd::Cli,
                UiMode::Web => Cmd::Web { port: None, open: true, host: "127.0.0.1".into() },
            }
        }
    };

    let t = Instant::now();
    let discovery = DiscoveryOptions { extra_dirs: cli.global.scan.clone(), no_defaults: false };
    let app = App::new(config.clone(), discovery);
    if cli.global.timings {
        eprintln!("timings: discovery {} ms ({} models)", t.elapsed().as_millis(), app.library().entries.len());
    }
    let ctx = Ctx { app, global: cli.global.clone(), web };

    match &command {
        Cmd::Cli => {
            repl::run(&ctx)?;
            Ok(0)
        }
        Cmd::Web { port, open, host } => {
            let Some(launch) = &ctx.web else { bail!("this build has no web interface (enable the `web` feature)") };
            launch(ctx.app.clone(), WebArgs { host: host.clone(), port: port.unwrap_or(config.port), open: *open })?;
            Ok(0)
        }
        other => {
            let t = Instant::now();
            let out = run_command(&ctx, other)?;
            write_stdout(&out);
            if ctx.global.timings {
                eprintln!("timings: command {} ms", t.elapsed().as_millis());
            }
            Ok(0)
        }
    }
}
