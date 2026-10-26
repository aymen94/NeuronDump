//! Top-level clap types: global flags, web launcher hook, prompt arguments, config actions.

use super::*;

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
    /// Extra directory to scan for .gguf files (repeatable). Also: NEURONDUMP_MODEL_DIRS.
    #[arg(long = "dir", visible_alias = "scan", global = true)]
    pub scan: Vec<PathBuf>,
    /// Interface to start when no subcommand is given.
    #[arg(long, global = true, env = "NEURONDUMP_UI")]
    pub ui: Option<String>,
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
