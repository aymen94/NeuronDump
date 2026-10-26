//! Every subcommand and its flags (`Cmd`).

use super::*;

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
    /// Diff two models' weights and metadata (base vs fine-tune, Q4 vs Q8, original vs slimmed).
    Diff {
        a: String,
        b: String,
        /// Most-changed tensors to show (with --json: to keep in the report).
        #[arg(long)]
        top: Option<usize>,
        /// Compare metadata, vocabulary and tensor lists only (no weight reads).
        #[arg(long)]
        metadata_only: bool,
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
