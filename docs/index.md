# NeuronDump

Look inside any local LLM, find the parts it doesn't need, and make it smaller.

NeuronDump reads **GGUF** model files (the format used by **Ollama** and **llama.cpp**),
shows what's inside them, and can write a smaller version that still runs.

## What it does

| Command | You get |
|---|---|
| `list` | all models on your computer |
| `inspect` | architecture, layers, heads, sizes, warnings |
| `vocab` | the vocabulary, readable and sorted by kind (words, code, emoji, special…) |
| `template` | the chat template as a tree of decisions |
| `stats` | statistics of the weights, strength of each head and neuron |
| `analyze` | a list of parts that could be removed, with size and risk |
| `slim` | a new, smaller GGUF file and an Ollama Modelfile |
| `trace` | what each layer "thinks" while the model answers |

Everything is available from the command line and from a local web page. Every result can
be written as JSON with a `schema_version`; the JSON Schemas are in
[`schemas/`](https://github.com/aymen94/NeuronDump/tree/main/schemas).

## Install

Ready-made Windows and Linux builds are on the
[releases page](https://github.com/aymen94/NeuronDump/releases)
([download steps](install.md#download-a-ready-made-build)).

To build from source you need [Git](https://git-scm.com) and [Rust](https://rustup.rs) 1.88+.
Full steps: [install guide](install.md).

=== "Windows (PowerShell)"

    ```powershell
    winget install Git.Git
    winget install Rustlang.Rustup              # then reopen PowerShell
    git clone https://github.com/aymen94/NeuronDump
    cd NeuronDump
    cargo install --path crates/neurondump
    ```

=== "Linux"

    ```sh
    sudo apt install build-essential git curl   # or dnf / pacman
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"
    git clone https://github.com/aymen94/NeuronDump
    cd NeuronDump
    cargo install --path crates/neurondump
    ```

Add `--features trace` to be able to run models.

## Try it

```sh
neurondump list
neurondump inspect llama3.2:1b
neurondump analyze llama3.2:1b --corpus my-texts.txt --out plan.json
neurondump slim llama3.2:1b --plan plan.json --only vocab --out small.gguf
neurondump web --open
```

## Example

```text
$ neurondump inspect llama3.2:1b
Architecture llama  [decoder]  1.24B params, quantization Q8_0
Layers: 16
  full_attention+dense                  16 layers  0-15
  embeddings are tied (no output.weight)
Tokenizer    gpt2 (pre: llama-bpe), 128256 tokens, 280147 merges
  specials   bos=128000 <|begin_of_text|>  eos=128009 <|eot_id|>
```

Next: [how it works](how-it-works.md), step by step, or
[what's inside real models](models.md) for findings on DeepSeek-R1 and Qwen 3.8.

## Citing

If you use NeuronDump in academic work, please cite the repository:

```bibtex
@software{neurondump,
  author  = {Aymen},
  title   = {NeuronDump: inspecting, analyzing and slimming GGUF language models},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/aymen94/NeuronDump},
  license = {MIT}
}
```

## License

Licensed under the MIT License.
