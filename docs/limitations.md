# Limitations

What NeuronDump does not do (yet), so results are read correctly.

## Runtime features (trace, simulation, edits, journey, timeline)

- **Supported architectures:** `llama`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `gemma3`, `phi3`,
  `bert`, `nomic-bert`. Checked against llama.cpp / Ollama on the same files (see
  [Gemma 4, Gemma, Gemma 2 and Phi-3](new-models.md)): **llama, qwen2, qwen3 and gemma** generate the
  same tokens with first-token probabilities within ~0.03 (a near-tie can flip, as in gemma:7b);
  **nomic-bert** runs; **gemma2** and **phi3** have the gaps below. `gemma3` and `bert` were not run
  (no model installed).
- **No MoE and no SSM / hybrid forward pass.** `qwen3moe`, `qwen35` (Qwen 3.5 / 3.8: linear attention
  + attention), `gemma4` and other hybrids get static analysis, static impact and bake only. Router choices and
  SSM state contributions are never measured; expert edits are bake-only; expert sweeps are refused.
- **Embedding models** (`nomic-bert`, `bert`) have no next-token prediction: the logit lens, decision
  breakdown, generation and logit edits do not apply. Journeys still show attention and FFN activity;
  comparisons report the cosine between pooled embeddings.
- **gemma2** logit / attention soft-capping (50 / 30) is not applied: tokens and their order match
  llama.cpp, but probabilities are too sharp (gemma2:2b's top first token: 0.49 vs 0.33 in Ollama).
  Its sliding window on every other layer is not shown by `layers` (the GGUF does not store the
  pattern) and is not applied; prompts under 4096 tokens are unaffected.
- **phi3 with LongRoPE** (the 128k models, e.g. `phi3:latest`): `rope_factors_short` /
  `rope_factors_long` and the attention factor are not applied, so the trace drifts from llama.cpp from
  the first token (" called" 0.13 vs " the" 0.09). Phi-3 4k models have no such factors.
- **CPU only, no KV cache.** Every generated token reruns the whole sequence, so tokens/s is only for
  comparing two runs, and long generations or sweeps take minutes. All weights are loaded into RAM
  (an 8B Q4_K model needs about 5 GB).
- **Direct logit attribution** linearizes the final norm at the position (exact for the layer sums of
  pre-norm decoders; per-head values under a post-attention norm are an approximation), and logit
  biases from edits are not attributed to any layer. `attributed` is computed in exact f32 from the recorded activations; the model's own logits come from quantized matmuls (candle also quantizes the activations to 8 bits), so on quantized models the two can differ by a few tenths of a logit, which matters when the chosen token and the runner-up are close. Example: deepseek-r1
  (Q4_K_M) on 'Reply with the word "PASS" and nothing else.' reports a logit difference of 0.182 and
  an attribution of 0.048.
- **Patching** aligns the source and target sequences from the end; with `all_positions`, prompts of
  different lengths are patched over their common suffix only.
- **Steering vectors** are the difference of mean residuals of two prompts at one layer, without
  normalization; useful strengths depend on the model's residual norms (see the journey's
  `|residual|` column).
- **Embedding edits at run time** change the input side only; with tied embeddings the output head
  keeps its rows until the edit is baked.

## Edits and bake

- The edit stack lives in the running process (per model). It is lost on exit unless saved as a
  snapshot or exported.
- Bake cannot store run-time-only edits (ban, boost, steering, patching, forced tokens); they are
  listed as skipped. Sampling, system prompt and template go into the Modelfile.
- Baked scales of quantized tensors re-encode them as Q8_0 (the file can grow); baking a skipped layer
  zeroes it instead of removing it (use `slim` to remove layers).
- Chat rendering supports the Go `text/template` subset Ollama templates use (`if` / `else if` /
  `range` / `with`, variables, `eq` `ne` `and` `or` `not` `len` `slice` `index` `print` `printf`
  `json`); `define` / `template` / `block` are refused. Jinja templates are parsed into trees but not
  executed: raw GGUF files without an Ollama template use a built-in template for their family
  (llama3, chatml, gemma, phi3), or the raw text.

## Pruning

- **Structured pruning** (heads / FFN neurons / MoE experts) is detected and scored by `analyze`, but
  `slim` refuses to remove them: no architecture is confirmed to load non-uniform shapes. They can be
  **zeroed** in place (same shapes, same size) through ablation plans or bake. Zeroing an expert does
  not change the router, which can still pick it. This part is experimental.
- Dropping MTP / in-file vision tensors is not verified to load in llama.cpp.
- Requantization only targets Q8_0 and dequantizes a whole tensor in memory.
- Dequantization does not support IQ1_S / IQ1_M / IQ2_* / IQ3_*, TQ1_0 / TQ2_0 or the legacy
  repacked types; those tensors are skipped in stats, and trace works only where candle supports the type.

## Tokenizers

Checked against llama.cpp's `llama-tokenize` on 8 texts (accents, emoji, code, CJK, `\r\n`):
**identical ids** for SPM (gemma, gemma2, phi3) and BPE llama3 / qwen2 / qwen3 (including the
qwen3moe and qwen35 models). Other BPE pre-tokenizers fall back to the llama3 rules with a warning.

- **WPM** (BERT, nomic-bert) does not normalize like llama.cpp: accented words become `[UNK]`
  (llama.cpp strips the accents first) and Japanese kana with voiced marks differ; plain English is
  exact.
- **`gemma4`** tokenization is not implemented (`tokens` stops with an error; vocabulary and template
  views work). UGM / RWKV are not implemented either.

## Web and 3D

- **WebGPU needs a secure context:** HTTPS or `localhost` / `127.0.0.1`. With `--host 0.0.0.0`, a page
  opened from another device over plain HTTP gets WebGL 2 through the automatic fallback. A browser
  with neither WebGPU nor WebGL 2 cannot show the graphs or the embedding map (the static pages work).
- Every view has been used in Chrome on WebGPU: the 2D, tower, city and force layouts, the 3D
  dataflow view with the token flow, diff coloring, the embedding map, and the tokens, journey,
  timeline and edit pages (the screenshots in these docs). The 2D and tower / city / force views were
  also checked in headless Chrome on the WebGL 2 path (software rendering). A headless `--screenshot`
  can be taken before the WebAssembly finishes loading.
- The [browser demo](https://aymen94.github.io/NeuronDump/demo/)'s WebAssembly module was checked in
  Node on a real model (header read in slices, overview, tensors, graph, vocabulary, one tensor's
  stats). It needs a browser with WebAssembly and, for the graph, WebGPU or WebGL 2.
- The embedding map of very large vocabularies (250k+ tokens) is heavy on software rendering.
- The REPL `web` command starts one server per session; the model list refreshes on `/api/models`
  calls rather than OS file events.

## Other

- No automated test suite (a single unit test), fuzz targets or benchmarks. Clippy runs; a few
  style lints remain in `neurondump-gguf`.
- **New architectures can have tensor roles NeuronDump does not know** (shown as `unknown:…` and
  placed under *other*): `gemma4`'s `inp_gate`, `layer_output_scale`, `post_norm` and `proj`.
- **Chat templates rendered by Ollama itself:** for some new models (Gemma 4) Ollama's template layer
  is only `{{ .Prompt }}` and Ollama formats the chat in its own code, so `--chat` adds nothing. The
  GGUF's Jinja template is still shown by `template`.
- Graph importance without measurements is structural only. Run `trace --corpus`, `inspect --stats` or
  `sweep` for real numbers.
- `comfy-table` is built without terminal-width detection (Windows GNU toolchain workaround), so wide
  tables do not wrap.
