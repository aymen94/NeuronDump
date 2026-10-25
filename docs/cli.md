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
`--port N`, `--open`, `--host`. See [Web interface](web.md).

## Interactive mode (REPL)

```sh
neurondump cli
```

Pick a model from the list (type to filter). The prompt then shows it:

```
neurondump[llama3.2:1b]> info
neurondump[llama3.2:1b]> layer 3
neurondump[llama3.2:1b]> vocab Paris --limit 5
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
| `jobs`, `job <id>`, `cancel <id>` | background jobs |
| `web` | open the web interface on the same session |
| `rescan` | look for models again |
| `help`, `exit` | |

Tab completes commands, model names, tensor names and token classes. History is kept
between sessions. Long output is shown one page at a time.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | error (message on stderr) |
| 2 | no command given and no terminal (prints help) |
| 130 | cancelled |
