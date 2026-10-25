# CLI reference

## Global flags

These work with every command.

| Flag | Meaning |
|---|---|
| `--json` | machine-readable JSON output |
| `--threads N` | number of worker threads (default: all cores) |
| `--timings` | print how long each step took (stderr) |
| `-q`, `--quiet` | no progress bars or messages |
| `--scan <dir>` | also look for `.gguf` files in this folder (repeatable) |
| `--ui cli\|web` | interface to start when no command is given (or env `NEURONDUMP_UI`) |

Progress bars go to stderr, so `--json` output can be piped safely.

## Commands

### `list`
All local models with source, architecture, parameter count, quantization, context and size.

### `inspect <model>`
Full summary of one model.

| Option | |
|---|---|
| `--all` | inspect every model instead of one |
| `--stats` | also compute weight statistics (reads the whole file) |
| `--full` | don't shorten long arrays; include the whole vocabulary in JSON |
| `--metadata` | also print every metadata key |
| `--out <dir>` | save `<model>.inspect.json` there |

### `search <text>`
Searches model names, architectures and quantizations. With `--model <m>` it also searches
that model's tokens, tensors and metadata keys. `--limit N` sets the number of results per kind (default 10).

```sh
neurondump search qwen
neurondump search attn_q --model llama3.2:1b
```

### `layers <model>`
One row per layer: kind (attention / linear / MoE / dense), heads, sizes.

### `tensors <model>`
| Option | |
|---|---|
| `--layer N` | only this layer |
| `--role <r>` | e.g. `ffn_down`, `attn_q` |
| `--sort size\|type\|layer\|name\|offset` | |
| `--limit N` | |

### `vocab <model>`
| Option | |
|---|---|
| `-s`, `--search <text>` | match text, raw form or id |
| `--regex` | treat `--search` as a regular expression |
| `--class <c>` | `special reserved byte whitespace digit punctuation emoji cjk code word subword mixed` |
| `--script <s>` | `latin cyrillic han arabic hangul …` |
| `--lang <l>` | code tokens of `rust python javascript c_cpp java go sql shell` |
| `--limit N`, `--offset N` | paging |
| `--tokenize "<text>"` | show the token ids for a text (same ids as `llama-tokenize`) |

### `template <model>`
The chat template as a tree. `--raw` also prints the source.

### `stats <model>`
Weight statistics. `--tensor <name>` and `--layer N` (repeatable) limit the work. Results are cached.

### `analyze <model>`
Builds a prune plan. See [Analyze and slim](analyze-and-slim.md).

| Option | |
|---|---|
| `--corpus <files…>` | text you care about; enables vocabulary trimming |
| `--out <file>` | where to write the plan (default `<model>.prune.json`) |
| `--stats` | compute weight statistics if not cached (head/neuron candidates) |
| `--fast` | skip the layer-similarity pass |
| `--include-edge-layers` | allow removing the first/last layer |

### `slim <model> --plan <file> --out <name.gguf>`
| Option | |
|---|---|
| `--only <list>` | candidate ids or groups: `vocab layers tied quant projector metadata` |
| `--perplexity <file>` | compare quality before/after with `llama-perplexity` |

### `trace <model>`
Needs a build with `--features trace`. See [Trace](trace.md).

| Option | |
|---|---|
| `--prompt "<text>"` | one prompt |
| `--corpus <file>` | several texts separated by blank lines; results are averaged |
| `--top-k N` | how many tokens / neurons to show per layer |
| `--max-tokens N` | cut long inputs |

### `graph <model>`
The model as a graph. See [Graph and ablation](graph-and-ablation.md).

| Option | |
|---|---|
| `--level section\|layer\|block\|tensor\|fine` | how deep to go (default `layer`) |
| `--group-by section\|layer\|block\|quant\|kind\|importance` | how to group nodes |
| `--expand <id,...>` | open these nodes (for example `layer.7.attn` for its heads) |
| `--format tree\|json\|dot\|mermaid` | output format (default `tree`) |
| `--out <file>` | write to a file |

### `node <model> <id>`
What a node contains, its statistics, its connections and whether it can be removed.

### `ablate <model> --cut <id,...>`
Impact of removing or zeroing these nodes, then a simulated run if the build has `trace`.

| Option | |
|---|---|
| `--static` | only the instant estimate, no inference |
| `--prompt "<text>"` | prompt to compare (default "The name of the star that Earth orbits is") |
| `--corpus <file>` | texts separated by blank lines: perplexity and mean KL |
| `--tokens N` | greedy tokens to generate (default 16) |
| `--force` | allow `breaks_model` cuts |
| `--save-plan <file>` | write the cut as a prune plan for `slim` |

### `sweep <model> --corpus <file>`
Cuts each layer (`--level layer`) or each head of one layer (`--level head --layer N`) in turn
and ranks them by damage. `--out` writes the JSON, and `--save-plan <file> --top N` writes the N
least damaging as a prune plan.

### `tokens <model> "<text>"`
Tokenization with no inference (any model): token ids, readable text, class, which tokens the chat
template added, the norm of each token's `token_embd` row and output row, whether the model can
predict it, and its nearest tokens by embedding cosine.

| Option | |
|---|---|
| `--chat` | wrap the text in the chat template first and print the full templated prompt |
| `--system "<text>"` | system prompt for the template (implies `--chat`; `""` = none) |
| `--neighbors N` | nearest neighbours per token (default 5, `0` skips the embedding scan) |

```sh
neurondump tokens llama3.2:1b "fn main() {"
neurondump tokens qwen2.5-coder:7b "fn main() {" --chat --system "You are a Rust expert."
```

### `journey <model> --prompt "<text>"`
Follows one token through every layer (needs `--features trace`): residual norm, logit lens, the
probability and rank of the final prediction per layer, which earlier tokens it attends to, the most
active FFN neuron buckets, and the **decision breakdown** (final candidates and how much every layer,
head and neuron pushed the chosen token against the runner-up).

| Option | |
|---|---|
| `--token N` | token index; negative counts from the end (default `-1`, the last token) |
| `--chat`, `--system` | as for `tokens` |
| `--edits <edits.json>` | run with these edits |
| `--top-k N` | lens tokens per layer (default 5) |

```sh
neurondump journey qwen2.5-coder:7b --prompt "fn main() {"
neurondump journey llama3.2:1b --prompt "The name of the star that Earth orbits is" --token -1 --edits steer.json
```

### `generate <model> --prompt "<text>"`
Generates tokens with a per-step timeline: chosen token, probability, entropy, top alternative (and
the layer that pushed most, with `--explain`). Needs `--features trace`.

| Option | |
|---|---|
| `--steps N` | tokens to generate (default 16) |
| `--explain` | a decision breakdown for every step |
| `--temperature T`, `--top-k K`, `--top-p P`, `--seed S` | sampling (default greedy) |
| `--chat`, `--system` | chat formatting |
| `--edits <edits.json>` | run with these edits |

```sh
neurondump generate llama3.2:1b --prompt "The name of the star that Earth orbits is" --steps 8 --explain
```

### `edit <model> --apply <edits.json>`
Applies a portable edit patch in memory and compares baseline vs edited; optionally bakes a new
GGUF. See [Interventions](interventions.md) for the edit format.

| Option | |
|---|---|
| `--prompt "<text>"` | prompt to compare (repeatable; default "The name of the star that Earth orbits is") |
| `--chat` | wrap prompts in the chat template |
| `--corpus <file>` | texts separated by blank lines: perplexity and mean KL |
| `--tokens N` | tokens generated per side (default 16) |
| `--bake <out.gguf>` | also write the edits into a new GGUF (plus a Modelfile); never the original |
| `--no-compare` | only bake |

```sh
neurondump edit llama3.2:1b --apply edits.json --prompt "The name of the star that Earth orbits is" --tokens 12
neurondump edit llama3.2:1b --apply edits.json --bake llama-edited.gguf --no-compare
```

### `compare <model> <model> [...]`
Side by side: sizes, layers, tokenizer, vocabulary overlap, differing metadata.

### `export <model>`
| Option | |
|---|---|
| `--format json\|ndjson\|csv` | |
| `--table tensors\|vocab\|metadata\|layers` | for ndjson / csv |
| `--out <file>` | default: stdout |

### `config`
`config show`, `config set <key> <value>`, `config reset`, `config path`.
Keys: `default_ui`, `port`, `scan_dirs`, `output_dir`, `threads`.

### `web`
`--port N`, `--open`, `--host`. See [Web interface](web-ui.md) and [API](api.md).

## Interactive mode (REPL)

```sh
neurondump cli
```

Pick a model from the list (type to filter). The prompt then shows it:

```
neurondump[llama3.2:1b]> info
neurondump[llama3.2:1b]> layer 3
neurondump[llama3.2:1b]> vocab Rome --limit 5
neurondump[llama3.2:1b]> stats &          # & = run in the background
neurondump[llama3.2:1b]> jobs
neurondump[llama3.2:1b]> use nomic-embed-text
```

| Command | |
|---|---|
| `use <model>` | switch model |
| `search <text>` | search models + the current model's tokens, tensors, metadata |
| `models` | list models |
| `info`, `layers`, `layer N`, `tensors`, `vocab`, `template`, `stats` | same as the commands above, for the current model |
| `analyze`, `slim`, `trace <prompt>`, `compare <model>`, `export` | same as above |
| `graph [options]`, `node <id>` | the graph of the current model |
| `cut <id>...`, `uncut <id>\|all`, `selection`, `impact` | build a cut selection and see its static impact |
| `simulate [prompt] [--tokens N] [--corpus f]` | run the selection in memory (`&` for background) |
| `sweep --corpus f [--level head --layer N]` | rank layers or heads by damage (`&` for background) |
| `plan save <file>` | write the selection as a prune plan |
| `tokens <text> [--chat]` | tokenization of the current model (the text is remembered) |
| `journey <idx> [prompt] [--chat] [--baseline]` | follow token `idx` (negative from the end) through the layers, with the current edits |
| `generate [prompt] [--steps N] [--explain] [--temperature t] [--top-k k] [--top-p p] [--seed s]` | generation timeline |
| `compare [prompt] [--tokens N] [--corpus f]` | baseline vs the current edits |
| `edit scale <node> <factor>` | scale a layer, block, KV group, head, neuron bucket (`layer.7.ffn.n.0-63`), neuron (`layer.7.ffn.neuron.42`) or expert |
| `edit zero <node>`, `edit skip <layer>`, `edit ablate <ids>` | zero / skip / cut (any `ablate` id) |
| `edit steer <layer> "<a>" "<b>" <strength>` | steering vector from two prompts |
| `edit ban <token>`, `edit boost <token> <bias>` | logit edits (token as text or `#id`) |
| `edit embed <token> scale <f>` / `edit embed <token> replace <token>` | embedding row edits |
| `edit patch <layer> [--head h] [--all] "<source prompt>"` | activation patching |
| `edit sampling <temp> [top_k] [top_p] [seed]`, `edit force <step> <token>` | sampling edits |
| `edit system <text>`, `edit template <file>` | prompt edits |
| `edits [export <file> \| import <file>]` | the edit stack |
| `toggle <id>`, `undo`, `redo`, `restore` | switch an edit, history, back to the original model |
| `snapshot save\|load <name>` | named edit sets (saved to disk) |
| `bake <file.gguf>` | write the enabled edits into a new GGUF + Modelfile |
| `jobs`, `job <id>`, `cancel <id>` | background jobs |
| `web` | open the web interface on the same session |
| `rescan` | look for models again |
| `help`, `exit` | |

The edit stack is per model and shared with the web UI started by `web` in the same session.

```
neurondump[llama3.2:1b]> tokens The name of the star that Earth orbits is
neurondump[llama3.2:1b]> journey -1
neurondump[llama3.2:1b]> edit zero layer.7
neurondump[llama3.2:1b]> compare --tokens 8
neurondump[llama3.2:1b]> undo
neurondump[llama3.2:1b]> edit steer 6 "Dear Sir or Madam, I am writing to formally request" "hey dude, just wanna ask" 4
neurondump[llama3.2:1b]> generate Write a note to my boss: --steps 20
neurondump[llama3.2:1b]> snapshot save formal
neurondump[llama3.2:1b]> restore
neurondump[llama3.2:1b]> bake llama-formal.gguf
```

Tab completes commands, model names, tensor names, token classes and graph node ids. History is kept
between sessions. Long output is shown one page at a time.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | error (message on stderr) |
| 2 | no command given and no terminal (prints help) |
| 130 | cancelled |
