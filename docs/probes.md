# Probes

Seven ways to look inside a model beyond the journey and the sweep. Three run the model (build
with `--features trace`, [supported architectures](limitations.md)); four read the weights only
and work on every GGUF.

| Probe | Runs the model | Answers | CLI | Web |
|---|---|---|---|---|
| [Causal tracing](#causal-tracing) | yes | where is a fact held (layer × token)? | `causal` | Probes |
| [Head roles](#attention-head-roles) | yes | which heads copy, look back, or sink? | `heads` | Probes, graph **color by role** |
| [Layer similarity](#layer-similarity) | yes | which layers barely change the representation? | `similarity` | Probes |
| [Weight spectrum](#weight-spectrum) | no | which matrices are low-rank or outlier-heavy? | `spectrum` | Probes, graph **color by rank** |
| [Neuron explorer](#neuron-explorer) | no | what does one FFN neuron read and write? | `neuron` | Probes, graph side panel |
| [Glitch tokens](#glitch-tokens) | no | which tokens were barely trained? | `glitch` | Probes |
| [Memory](#memory-per-context-length) | no | how much RAM / VRAM for a context length? | `memory` | Probes |

Every probe is a job (`/api/jobs`, kinds `causal`, `heads`, `similarity`, `spectrum`, `neuron`,
`glitch`), so results are cached and the web page shows progress and a cancel button. The memory
estimate is instant (`GET /api/models/{id}/memory`). Probes that run the model use the current edit
stack, like the journey.

## Causal tracing

Run a **clean** prompt and a **corrupted** one (another subject), then, inside the corrupted run,
restore the clean state at one layer and one token and measure how much of the clean answer comes
back. Doing it for every layer and token gives a map of where the model holds the fact (the method of
Meng et al., *Locating and Editing Factual Associations in GPT*, 2022).

```sh
neurondump causal qwen2.5:0.5b --prompt "The Eiffel Tower is located in the city of" \
  --corrupt "The Colosseum is located in the city of"
```

```
Causal tracing of qwen2.5:0.5b (residual, 104750 ms): target " Paris" — clean 0.448, corrupted 0.000
corrupted top: " Rome" 0.328  " __" 0.070 …
         0 1 2 3 4 5 6 7 8 910
  L1        █
  L4        █ ▓
  L7        █ █
  L12       █ ░
  L19       █ ░
  L20       ░               ▒
  L21                       █
  L23                       █
tokens: 0:"The" 1:" Col" 2:"os" 3:"se" 4:"um" 5:" is" 6:" located" 7:" in" 8:" the" 9:" city" 10:" of"
```

The classic picture: the subject's tokens carry the answer through the early and middle layers, then
around layer 20 it moves to the last token, which carries it to the output.

- `--component residual` (default) restores the residual stream after the layer; `attn` / `ffn`
  restore only what that block added, which separates where attention moves the fact from where
  the FFN stores it.
- `--target " Paris"` measures another token (default: the clean prediction).
- The cell value is the share of the gap restored: `(p_restored − p_corrupt) / (p_clean − p_corrupt)`.
  It can exceed 1 when the positions do not line up exactly.
- Prompts of different lengths are aligned **from the end**. Corrupt the subject with a similar
  number of tokens for the cleanest map.
- Cost: one forward pass per cell, layers × tokens. Each pass starts at the restored layer (the
  layers before it are the corrupted run, kept in memory), so it is about half a full pass on
  average. A 24-layer 0.5B model on 11 tokens takes about 1.5–2 minutes on CPU.

## Attention head roles

Feed random tokens repeated twice (`[BOS] r₁…rₙ r₁…rₙ`) and measure, for every head, the mean
attention to:

| Score | Attends to | Role when high |
|---|---|---|
| induction | the token *after* the earlier copy of the current token | `induction` ≥ 0.3: the in-context copying circuit |
| prev token | the previous position | `previous-token` ≥ 0.4 |
| duplicate | the earlier copy of the current token | `duplicate-token` ≥ 0.3 |
| self | the current position | `self` ≥ 0.4 |
| sink | the first position | `sink` ≥ 0.5 (only when no specific pattern) |

```sh
neurondump heads qwen2.5:0.5b --samples 4
```

```
mixed: 176 · sink: 103 · induction: 21 · self: 17 · previous-token: 11 · duplicate-token: 8
│ 16 ┆ 3  ┆ induction       ┆ 0.96 …
│ 8  ┆ 7  ┆ previous-token  ┆ 0.00 ┆ 0.91 …
│ 3  ┆ 0  ┆ duplicate-token ┆ … 0.91 …
```

Induction heads sit in the middle layers and previous-token heads before them, as expected from the
two-step copying circuit. On the web, **Color the graph by head role** colors heads (expand an
attention block), and layers / attention blocks get their most common specific role. Click a head in
the graph for its scores; **zero** in the table adds an edit, so you can check what a head does with
**Edits → compare**.

## Layer similarity

Run a corpus, keep the residual stream after every layer, and compare every pair of points with
linear CKA (centered kernel alignment, Kornblith et al. 2019) over the token positions. CKA is 1 when
two representations have the same geometry, whatever their rotation or scale.

```sh
neurondump similarity qwen2.5:0.5b --corpus texts.txt
```

The result is the CKA matrix (blocks along the diagonal are layers that compute alike) and, per
layer, CKA / cosine between its input and output and the size of its update relative to its input.
Layers with CKA in→out ≥ 0.99 are listed as the most redundant. They are the first to try with
`skip` (the web page has a **skip** button per layer), then measure with `compare` or a sweep:
similarity is a hint, the sweep is the measurement.

- Position 0 (often BOS, an attention sink with a huge norm) is left out so it does not dominate.
- `--max-positions` (default 256, max 1024) caps the kernel size (positions²).

## Weight spectrum

For every weight matrix: the largest singular value σmax (power iteration), the **stable rank**
‖W‖²F / σ²max, its share of the full rank, the outlier ratio max |w| / rms, and the excess kurtosis.
No forward pass. `token_embd` / `output` are skipped unless `--embeddings` (they are large); MoE
expert stacks are listed as skipped.

```sh
neurondump spectrum qwen2.5:0.5b --limit 5
```

```
│ blk.0.attn_q.weight      ┆ 896×896  ┆ Q5_0 ┆ 35.157 ┆ 2.9  ┆ 0.32% ┆ 18.4 ┆ 19.7 │
│ blk.0.attn_output.weight ┆ 896×896  ┆ Q5_0 ┆ 2.715  ┆ 18.8 ┆ 2.10% ┆ 42.8 ┆ 13.1 │
```

- **Low stable rank:** the energy sits in a few directions, so a low-rank approximation keeps most
  of the matrix (layer 0's query projection here: stable rank 2.9 of 896).
- **High outlier ratio / kurtosis:** a few huge weights. These matrices lose the most to aggressive
  quantization.

On the web, **Color the graph by stable rank** paints tensors, blocks and layers (mean) from red
(lowest, log scale) to pale.

## Neuron explorer

One FFN neuron in token space, from the weights:

- **reads:** tokens whose embedding (through the FFN norm) points along the neuron's input row
  (`ffn_gate`, or `ffn_up` without a gate);
- **promotes / suppresses:** tokens its output column (`ffn_down`) raises or lowers through the
  output norm and the unembedding.

```sh
neurondump neuron qwen2.5:0.5b 20 4058
```

```
reads: " Romans" +0.115  " Rome" +0.105  "古代" +0.095  " Roman" +0.094  "罗马" +0.087  "长安" +0.085 …
```

Both follow the **direct path** only (the embedding straight into the neuron, the neuron straight
into the output); everything in between is ignored, so treat them as a first look. The journey
measures neurons on real text. In the graph, a neuron bucket's side panel has **Explore neuron**.
MoE expert neurons are not supported yet.

## Glitch tokens

Under-trained token candidates: rows of `token_embd` (and `output`, when not tied) much smaller than
the rest, which happens when a token was (almost) never seen in training. Prompts containing them can
make a model behave erratically. Each candidate is checked for **reachability** (whether encoding its
text gives the token back).

```sh
neurondump glitch llama3.1:8b --top 15
```

```
│ 124    ┆ "<0xC0>" ┆ normal ┆ 0.000 ┆ -8.56 ┆ -3.67 ┆ no  │
│ 177    ┆ "<0xF5>" ┆ normal ┆ 0.000 ┆ -8.56 ┆ -3.67 ┆ no  │
│ 103003 ┆ "аракт"  ┆ normal ┆ 0.000 ┆ -8.56 ┆ -3.67 ┆ yes │
```

On llama3.1:8b the byte tokens that can never occur in UTF-8 (0xC0, 0xC1, 0xF5–0xFF) have all-zero
embeddings, and so does token 103003. Control, byte-type and unused tokens are not checked. `--z`
(default −3) sets the threshold; a model with no flagged tokens has no abnormal rows. On the web, **ban**
adds an edit so the token can never be predicted.

## Memory per context length

Weights plus the KV cache, for the cache types llama.cpp and Ollama offer (`f16`, `q8_0`, `q4_0`;
Ollama: `OLLAMA_KV_CACHE_TYPE`, llama.cpp: `--cache-type-k/v`). Read from the metadata: layers,
KV heads per layer, key and value lengths.

```sh
neurondump memory llama3.1:8b --ctx 65536
```

```
│ 8192             ┆ 1.00 GiB  ┆ 544.00 MiB ┆ 288.00 MiB ┆ 5.58 GiB  ┆ 5.11 GiB  ┆ 4.86 GiB │
│ 131072 (trained) ┆ 16.00 GiB ┆ 8.50 GiB   ┆ 4.50 GiB   ┆ 20.58 GiB ┆ 13.08 GiB ┆ 9.08 GiB │
```

Runtimes add compute buffers on top. Sliding-window layers (Gemma 2 / 3, in llama.cpp's layout)
are counted at most at their window, so gemma3:1b needs 139 MiB of f16 cache at 32K instead of
832 MiB; llama.cpp's `--swa-full` caches the whole context there. Recurrent and state-space layers
keep a fixed-size state instead of a KV cache. The notes under the table say when either applies.
See [Inside Gemma 3 1B](gemma3-research.md) for a whole model through every probe.
