# How it works

Here is the whole path, from a model name to a smaller model.

```text
 "llama3.2:1b"
      │
      ▼
 1. FIND      Ollama manifest  ──►  weights file (.gguf) + template + settings
      │
      ▼
 2. READ      header + metadata + tensor list   (the weights are not loaded)
      │
      ▼
 3. UNDERSTAND  layers, heads, vocabulary, chat template
      │
      ▼
 4. MEASURE   weight statistics          (optional: run the model = trace)
      │
      ▼
 5. ANALYZE   list of removable parts, each with size and risk  →  plan.json
      │
      ▼
 6. SLIM      write a new .gguf without those parts, then check it
      │
      ▼
 small.gguf + Modelfile  ──►  llama.cpp / ollama create
```

## 1. Find

Ollama keeps a small manifest file for each model. It says which big file holds the
weights and which small files hold the chat template, system prompt and settings.
NeuronDump also scans the llama.cpp, Hugging Face and LM Studio folders for `.gguf` files.

## 2. Read

A GGUF file starts with a table of contents: settings (`llama.block_count = 16`…) and
a list of every weight matrix with its name, shape and position. NeuronDump reads only this
table, so it is fast even for huge files.

## 3. Understand

Weight names follow a pattern. For example, `blk.3.attn_q.weight` is layer 3, attention,
"query" matrix. From the names in each layer, NeuronDump knows what kind of layer it is:

| Layer contains | Kind |
|---|---|
| `attn_q`, `attn_k`, `attn_v` | attention |
| `ssm_…` | linear attention (state-space) |
| `ffn_…_exps` | mixture of experts |
| `ffn_up`, `ffn_down` | normal feed-forward |

It also decodes the vocabulary into readable text and parses the chat template.

## 4. Measure

`stats` reads the weights in small pieces and computes how strong each part is.
`trace` runs the model on a sentence and records what every layer does:

```text
LAYER  LOGIT LENS (what the model would answer here)
10     "otope" 0.03   "namen" 0.03
11     " Earth" 0.08  " planets" 0.05   ← the topic appears here
14     " called" 0.39
15     " not" 0.17
```

## 5. Analyze

NeuronDump lists what could be removed:

| Candidate | Example | Risk |
|---|---|---|
| unused tokens | 248 reserved tokens never used | medium |
| vocabulary trim | keep only the tokens your texts need | high |
| duplicate output matrix | exact copy of the input embeddings | low |
| a whole layer | one that changes the text very little | high |
| 16-bit weights | could be stored in 8 bits | low |
| vision part | not needed for text-only use | low |

## 6. Slim

You choose candidates, and NeuronDump writes a new file. It renumbers tokens and layers,
fixes every setting that depends on them, then re-reads the file to check it:

```text
Size      1.23 GiB -> 987.47 MiB (21.6% smaller)
Validate  OK (147 tensors, vocab 570, block_count 16)
```

A Modelfile is written next to it, so `ollama create` works directly.

## CLI or web

Both use the same engine. `neurondump cli` opens an interactive prompt;
`neurondump web --open` opens the same features in your browser, on your own computer only.

Next: [getting started](getting-started.md).
