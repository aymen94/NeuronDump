# NeuronDump

Look inside any local LLM, find the parts it doesn't need, and make it smaller.

**Website:** https://aymen94.github.io/NeuronDump/

NeuronDump reads **GGUF** model files (the format used by **Ollama** and **llama.cpp**) and
shows what is inside: layers, attention heads, vocabulary, chat template, weight statistics.
It can then remove unused parts and write a smaller GGUF that still runs in Ollama and llama.cpp.

It works with any model. It reads each file's own metadata; nothing is hard-coded per model.

## Install

You need [Git](https://git-scm.com) and [Rust](https://rustup.rs) 1.88 or newer.
Step-by-step instructions: **[Windows and Linux install guide](docs/install.md)**.

**Windows** (PowerShell):

```powershell
winget install Git.Git
winget install Rustlang.Rustup              # then reopen PowerShell
git clone https://github.com/aymen94/NeuronDump
cd NeuronDump
cargo install --path crates/neurondump
neurondump list
```

**Linux**:

```sh
sudo apt install build-essential git curl   # or dnf / pacman
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source "$HOME/.cargo/env"
git clone https://github.com/aymen94/NeuronDump
cd NeuronDump
cargo install --path crates/neurondump
neurondump list
```

Add `--features trace` to `cargo install` to also run models and watch each layer.

## Quick start

```sh
neurondump list                          # every local model
neurondump inspect llama3.2:1b           # what's inside a model
neurondump vocab llama3.2:1b --search hello
neurondump template llama3.2:1b          # the chat template as a tree
neurondump analyze llama3.2:1b --out plan.json
neurondump slim llama3.2:1b --plan plan.json --only vocab --out small.gguf
neurondump web --open                    # the same, in your browser
```

Run `neurondump` with no arguments to choose between the interactive CLI and the web interface.

## Documentation

| Page | What it covers |
|---|---|
| [Install and run](docs/install.md) | Windows and Linux, step by step |
| [Getting started](docs/getting-started.md) | find your models, first commands |
| [How it works](docs/internals.md) | what NeuronDump reads, and how |
| [CLI reference](docs/cli.md) | every command and flag, and the REPL |
| [Web interface](docs/web.md) | the browser UI and the HTTP API |
| [Analyze and slim](docs/analyze-and-slim.md) | making models smaller, safely |
| [Trace](docs/trace.md) | running a model and looking at each layer |
| [What's inside real models](docs/models.md) | findings on Llama, Qwen, Nomic… |
| [Building](docs/building.md) | build options and Windows notes |
| [Manual test plan](https://github.com/aymen94/NeuronDump/blob/main/TEST.md) | step-by-step checks |

A short illustrated explanation is on the [website](https://aymen94.github.io/NeuronDump/)
(source: [`docs/index.html`](docs/index.html)).

## Status

Early version, tested on Windows with Ollama models. Known gaps are listed at the end of
[TEST.md](TEST.md). Bug reports and ideas: [issues](https://github.com/aymen94/NeuronDump/issues).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
