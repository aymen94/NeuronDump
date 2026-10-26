//! REPL command table and the rustyline helper (completion of commands, models, node ids, hints).

use super::*;

pub(super) const COMMANDS: &[(&str, &str)] = &[
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
    ("diff", "diff <model> [--top N] [--metadata-only]   weight + metadata diff against another model"),
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

pub(super) struct ReplHelper {
    pub(super) models: Mutex<Vec<String>>,
    pub(super) tensors: Mutex<Vec<String>>,
    pub(super) graph: Mutex<Option<Arc<ModelGraph>>>,
}

impl ReplHelper {
    /// Node ids for completion; fine ids (heads, neuron buckets, experts) once a layer prefix is typed.
    pub(super) fn node_ids(&self, word: &str) -> Vec<String> {
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
            (Some("use" | "compare" | "diff"), _) => pairs(self.models.lock().unwrap().clone()),
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
