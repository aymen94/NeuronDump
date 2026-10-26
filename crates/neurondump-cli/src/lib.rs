//! CLI front end: one-shot subcommands, interface picker and the interactive REPL.
//! Contains no analysis logic — everything is a call into `neurondump-core`.

mod progress;
pub mod render;
mod repl;
mod cli_args;
mod commands;
mod context;
mod dispatch;
pub use cli_args::*;
pub use commands::*;
pub use context::*;
pub use dispatch::*;

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
