# How it works (details)

## The GGUF file

A GGUF file has three parts:

```
┌──────────────┬─────────────────────────────┬──────────────────────────────┐
│ header       │ metadata (key = value)      │ tensor list (name, shape,    │ ... tensor data (the weights)
│ magic, count │ general.architecture=llama  │ type, offset)                │
│              │ llama.block_count=16        │ blk.0.attn_q.weight 2048x2048│
│              │ tokenizer.ggml.tokens=[...] │ ...                          │
└──────────────┴─────────────────────────────┴──────────────────────────────┘
```

The weights make up almost all of the file, so NeuronDump memory-maps it and reads only
what each command needs. `list` and `inspect` touch only the header, which is why they take
well under a second, even for a 16 GB model.

## Step by step

1. **Discovery.** NeuronDump reads Ollama's manifest files to map names such as
   `llama3.2:1b` to the weights file (a "blob"). The same manifest points to the chat
   template, the system prompt, the parameters and, for vision models, the image projector.
2. **Parsing.** It decodes the header, every metadata key and every tensor description.
   Unknown or damaged things become warnings; the parser never crashes.
3. **Architecture.** `general.architecture` names the family (llama, qwen2, nomic-bert…).
   Keys like `llama.block_count` and `llama.attention.head_count` give the sizes.
4. **Tensor roles.** Names follow a convention: `blk.3.attn_q.weight` is layer 3, the
   attention query matrix. From the tensors present in each layer, NeuronDump works out
   what kind of layer it is:

   | Layer has… | Kind |
   |---|---|
   | `attn_q`, `attn_k`, `attn_v` | full attention |
   | `ssm_*` | linear attention / state-space |
   | `ffn_*_exps` | mixture of experts |
   | `ffn_up`, `ffn_down` | dense feed-forward |

   This is how it finds hybrid models (for example 3 linear layers for every attention layer)
   without knowing them in advance.
5. **Tokenizer.** The vocabulary is stored as raw strings (`Ġhello`, `▁hello`, `<0x0A>`).
   NeuronDump decodes them into readable text and sorts each token into a class: word,
   code keyword, digit, CJK, emoji, special, reserved… It also re-implements the tokenizers
   themselves (BPE, SentencePiece, WordPiece), so it can turn text into the same ids as llama.cpp.
6. **Chat template.** The template that turns a conversation into model input is a small
   program, in Jinja (from the GGUF) or Go templates (from Ollama). NeuronDump parses both
   into a tree of `if` / `for` branches, the variables they read and the special tokens they emit.
7. **Statistics.** On request, it reads the weights in small chunks and computes min, max,
   mean, spread, sparsity, outliers and a histogram. It also computes the strength (norm) of
   every attention head and every feed-forward neuron.
8. **Analysis.** It turns all of that into a list of removal candidates. See
   [Analyze and slim](analyze-and-slim.md).
9. **Slimming.** It writes a new GGUF without the chosen parts, fixes every number that
   depends on them, then re-reads the result to check it.
10. **Tracing** (optional). It runs the model on a prompt and records what each layer does.
    See [Trace](trace.md).

## Code layout

| Crate | Job |
|---|---|
| `neurondump-gguf` | read, decode and write GGUF files |
| `neurondump-core` | all the logic: discovery, analysis, slimming, jobs |
| `neurondump-trace` | runs models (optional `trace` feature, uses candle) |
| `neurondump-cli` | command line and interactive prompt |
| `neurondump-web` | web server and web page |
| `neurondump` | the program that ties them together |

The CLI and the web interface contain no analysis code. Both call the same core library,
so they always give the same answers. Every result is JSON with a `schema_version`, and the
JSON Schemas are in [`schemas/`](https://github.com/aymen94/NeuronDump/tree/main/schemas).
