# Getting started

## 1. Install

The quickest way is a ready-made build from the
[releases page](https://github.com/aymen94/NeuronDump/releases) (see
[download steps](install.md#download-a-ready-made-build)).

To build it yourself, follow the [install guide](install.md) for Windows or Linux. In short:

```sh
cargo build --release                    # CLI + web interface
cargo build --release --features trace   # also lets you run models (needs more RAM)
```

The program is `target/release/neurondump` (`neurondump.exe` on Windows). The examples below
assume it is on your `PATH`, or that you created an alias:

```sh
alias neurondump=./target/release/neurondump
```

## 2. Find your models

```sh
neurondump list
```

NeuronDump looks in these places automatically:

| Source | Where |
|---|---|
| Ollama | `OLLAMA_MODELS`, or `~/.ollama/models` (`%USERPROFILE%\.ollama\models` on Windows) |
| llama.cpp | `LLAMA_CACHE`, or `~/.cache/llama.cpp` |
| Hugging Face | `HF_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`, or `~/.cache/huggingface/hub` (all OSes) |
| LM Studio | the folder in `~/.lmstudio-home-pointer` + `/models`, `~/.lmstudio/models`, or the old `~/.cache/lm-studio/models` |
| Jan | `<data folder>/llamacpp/models` and `<data folder>/models`; data folder is `%APPDATA%\Jan\data` (Windows), `~/Library/Application Support/Jan/data` (macOS), `~/.local/share/Jan/data` or `~/.config/Jan/data` (Linux); also the old `~/jan/models` |

Folders that don't exist are skipped. The `SOURCE` column of `list` shows which app a model
came from; a file reachable from two places is listed once. A `mmproj*.gguf` next to a model
is attached to it as its vision projector.

If you moved an app's models folder (e.g. a custom LM Studio or Jan folder), add it with
`--dir <folder>` (repeatable; `--scan` still works), the `NEURONDUMP_MODEL_DIRS` environment
variable (a path list: `;`-separated on Windows, `:`-separated elsewhere), or the config:

```sh
neurondump --dir ~/my-models list
NEURONDUMP_MODEL_DIRS="$HOME/models:/mnt/data/gguf" neurondump list
neurondump config set scan_dirs "C:/models, D:/more-models"
```

## 3. Name a model

All of these work:

```sh
neurondump inspect llama3.2:1b                      # Ollama name (tag defaults to "latest")
neurondump inspect someuser/somemodel:q8            # namespaced Ollama model
neurondump inspect ./downloads/model.gguf           # any GGUF file
neurondump inspect 17d2889ccc34                     # the id shown by `list`
```

If you mistype a name, NeuronDump suggests the closest match.

## 4. First look inside

```sh
neurondump inspect llama3.2:1b       # summary: architecture, layers, tokenizer, templates
neurondump layers llama3.2:1b        # one line per layer
neurondump tensors llama3.2:1b --layer 0
neurondump vocab llama3.2:1b --class code --limit 20
neurondump template llama3.2:1b
neurondump stats llama3.2:1b         # reads all weights, takes a few seconds
```

## 5. Choose an interface

```sh
neurondump            # asks: CLI (interactive) or Web interface, and can remember your choice
neurondump cli        # interactive prompt with tab completion
neurondump web --open # local web page
```

## Where files go

| What | Where |
|---|---|
| Settings | `~/.config/neurondump/config.toml` (`%APPDATA%\neurondump\config.toml` on Windows) |
| Cached results | `~/neurondump/cache/` |
| Slimmed models | `~/neurondump/slim/` |

Change the output folder with `neurondump config set output_dir <dir>`.

Next: [How it works](internals.md) · [Architecture](architecture.md) · [CLI reference](cli.md)
