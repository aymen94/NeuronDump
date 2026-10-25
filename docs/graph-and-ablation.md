# Graph and ablation

NeuronDump can draw a model as a graph and answer "what happens if I cut this part?".
It first estimates the effect instantly. With `--features trace` it also runs the cut model in
memory and compares its output with the original.

## The graph

![The graph of qwen3.8:27b in the web interface (3D city view, layer 64 selected)](images/model_graph.png)

```sh
neurondump graph llama3.2:1b                       # sections and layers, as a tree
neurondump graph llama3.2:1b --level block         # attention / feed-forward blocks in every layer
neurondump graph llama3.2:1b --expand layer.7.attn # its KV groups; expand a group for its heads
neurondump graph llama3.2:1b --format dot --out g.dot && dot -Tsvg g.dot -o g.svg
neurondump node llama3.2:1b layer.7.attn.head.3    # what a node contains and how it is connected
```

| Level | Shows |
|---|---|
| `section` | Input, Body, Output, Tokenizer, Chat template, and the vision projector / MTP head if present |
| `layer` | every layer, plus the input and output tensors |
| `block` | attention (or SSM for hybrid models), feed-forward or MoE, and the residual adds |
| `tensor` | every tensor |
| `fine` | attention heads (in their KV groups), FFN neurons in buckets of 64 or more, MoE experts |

The fine level is large, so the web page and `--expand` only build it for the node you open.

Edges:
- **dataflow:** from the embeddings through every layer, including the residual stream, to the output;
- **tied:** the output head reuses the embedding matrix;
- **uses:** chat template → special tokens, tokenizer → embeddings, projector → input.

`--group-by` regroups the graph by `layer`, `block` (all attention together), `quant`, `kind`
(attention / SSM / MoE layers) or `importance`. `--format` is `tree`, `json`, `dot` or `mermaid`.

Every node also has:
- **removable:** `safe`, `risky` or `breaks_model`, with a reason;
- **importance score:** 0–1, and where it comes from. It is the measured damage from a sweep if
  you ran one, otherwise the layer change from `trace --corpus`, otherwise weight norms from
  `stats`, otherwise structural only.

## Static impact

```sh
neurondump ablate llama3.2:1b --cut layer.7 --static
neurondump ablate llama3.2:1b --cut layer.7.attn,layer.9.attn.head.3 --static
neurondump ablate llama3.2:1b --cut token_embd --static          # refused: breaks_model
```

Node ids are shown after `#` in the tree. Shorthands work too: `L7`, `blk.7`, `token_embd`,
`blk.7.attn_q`, and `vocab:100-200` for a range of tokens.

A cut is either:
- **removed** (layers, tokens, the projector, the chat template): the file gets smaller;
- **zeroed** (blocks, heads, neuron buckets, experts, single tensors): the weights are set to 0
  in place. Every shape stays the same, so llama.cpp still loads the file, but it is not smaller.

The report gives:
- removed and zeroed parameters, and the new file size;
- what gets cut off from the input or output;
- dependencies (tied embeddings, tokens used by the chat template, bos/eos);
- whether llama.cpp loads the result, and the layer renumbering;
- a verdict.

`breaks_model` cuts are refused unless you pass `--force`.

Hybrid models (for example qwen3.5, with three SSM layers then one attention layer) can only lose
whole aligned groups of layers. Removing a single layer would misalign the pattern.

## Simulated impact

```sh
neurondump ablate llama3.2:1b --cut layer.7 --prompt "The capital of Italy is" --tokens 12
neurondump ablate llama3.2:1b --cut layer.15 --corpus texts.txt
```

The cut is applied **in memory**: skipped layers pass the residual stream through, and zeroed parts
output 0. Nothing is written to disk. The report compares baseline and ablated runs:
- the top next tokens and their KL divergence;
- the greedy continuation;
- the logit lens per layer;
- perplexity and mean KL on the corpus;
- cosine similarity of the embeddings (embedding models);
- tokens/s.

Simulation uses the [trace](trace.md) forward pass, so it supports the same architectures.
It runs on the CPU without a KV cache, so tokens/s only compares the two runs.

## Sweeps

```sh
neurondump sweep llama3.2:1b --corpus texts.txt --save-plan plan.json --top 2
neurondump sweep llama3.2:1b --level head --layer 7 --corpus texts.txt
neurondump slim llama3.2:1b --plan plan.json --out llama-14.gguf
```

A sweep removes each layer (or each head of one layer) in turn and ranks them by the damage this
causes (mean KL divergence on the corpus, plus perplexity). That ranking becomes the importance
shown in the graph. `--save-plan` writes the `--top` least damaging entries as a prune plan for
`slim`.

## In the REPL and the web interface

- **REPL:** `graph`, `node <id>`, `cut <id>`, `uncut <id>`, `selection`, `impact`, `simulate [prompt]`,
  `sweep` and `plan save <file>` work on the current model. Tab completes node ids.
- **Web:** the **graph** tab shows the same graph.
  - The **view** menu picks how it is drawn:
    - **3D tower** (default): sections stacked bottom to top, with the layers as floors;
    - **3D city**: sections left to right, bars sized by bytes;
    - **3D force**: a force layout along the dataflow;
    - **2D**: a flat top-to-bottom diagram (drag to pan, wheel to zoom).

    All four are drawn with wamr3d (WebGPU, or WebGL 2 as a fallback), bundled with NeuronDump so it
    works offline. A browser with neither shows an error instead of the graph.
  - Double-click a node to expand it; click it for details.

    ![Section level, grouped by layer](images/model_graph2.png)

  - **Cut this** adds it to a selection that shows its impact live.
  - **Simulate**, **Sweep** and **Save as plan → Slim** start jobs.
  - After a sweep, the graph is colored by damage.
