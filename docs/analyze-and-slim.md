# Analyze and slim

Making a model smaller takes two steps:

1. **analyze** writes a *plan*: a list of things that could be removed, each with its size and risk.
2. **slim** applies the parts of the plan you choose and writes a new GGUF.

The original model is never modified.

## 1. Analyze

```sh
neurondump analyze llama3.2:1b --corpus my-texts.txt --out plan.json
```

Example output:

```
ID               RISK          SAVES       APPLY  TITLE
vocab_trim       high          267.94 MiB  yes    Trim vocabulary to the 570 tokens used by the corpus
reserved_tokens  medium        527.00 KiB  yes    Remove 248 reserved/unused tokens
low_norm_heads   experimental  1.25 MiB    no     5 attention heads with < 50% of their layer's median norm
layer_11         high          61.64 MiB   yes    Remove layer 11 (similarity 0.0021)
```

### What it looks for

| Candidate | Idea | Risk |
|---|---|---|
| `vocab_trim` | keep only the tokens your corpus uses (plus special and byte tokens) | high: the model can't write other text |
| `reserved_tokens` | placeholder tokens that are never used (`<\|reserved_special_token_N\|>`, `[unusedN]`, `[PADn]`) | medium |
| `tied_embeddings` | `output.weight` is an exact copy of `token_embd.weight` | low |
| `layer_N` | a layer that changes little (needs a perplexity check) | high |
| `quant_headroom` | large 16/32-bit tensors that could be 8-bit (Q8_0) | low |
| `unused_projector` | drop the image part of a vision model, for text-only use | low |
| `large_metadata` | big metadata entries the model doesn't need to run | low |
| `low_norm_heads` / `low_norm_neurons` / `low_norm_experts` | very weak heads / neurons / experts | experimental: reported only |

Tips:
- Without `--corpus`, there is no vocabulary trimming.
- `--stats` adds the head/neuron/expert candidates (it reads all weights once; the result is cached).
- Layer candidates are better after `neurondump trace <model> --corpus <file>`: they then use
  how much each layer really changes the text, instead of a weight comparison.
- The first and last layers are never proposed unless you pass `--include-edge-layers`.
- Hybrid models (with a fixed pattern of layer kinds) refuse layer removal.

## 2. Slim

```sh
neurondump slim llama3.2:1b --plan plan.json --only vocab --out small.gguf
```

`--only` takes candidate ids (`layer_11`) or groups: `vocab`, `layers`, `tied`, `quant`,
`projector`, `metadata`. Without `--only`, it applies the candidates marked `"selected": true`
in the plan file, which you can edit by hand.

Output:

```
Wrote     ~/neurondump/slim/small.gguf
Modelfile ~/neurondump/slim/small.Modelfile
Size      1.23 GiB -> 987.47 MiB (21.6% smaller)
Applied   vocab_trim, reserved_tokens
Validate  OK (147 tensors, vocab 570, block_count 16)
```

What slim changes:
- **Vocabulary:** removes rows from the embedding and output matrices, renumbers the
  tokens, fixes the special-token ids and the BPE merge rules.
- **Layers:** removes the layer's tensors, renumbers `blk.N`, updates `block_count` and every per-layer setting.
- Then it **re-reads the new file** and checks that it is consistent.

Experimental candidates (heads, neurons, experts) are refused with an explanation: llama.cpp
expects every layer to have the same shape.

## 3. Check quality

```sh
neurondump slim llama3.2:1b --plan plan.json --only reserved_tokens --out a.gguf --perplexity eval.txt
```

If `llama-perplexity` is installed, this prints the perplexity before and after (lower is
better; equal means no loss). You can also just try it:

```sh
llama-completion -m ~/neurondump/slim/small.gguf -p "The quick brown fox" -n 20 --temp 0
```

## 4. Use it in Ollama

A `Modelfile` is written next to the GGUF, with the original template, system prompt and parameters:

```sh
cd ~/neurondump/slim
ollama create my-small-model -f small.Modelfile
ollama run my-small-model "Hello"
```
