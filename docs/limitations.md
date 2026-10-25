# Limitations

What NeuronDump does not do (yet), so results are read correctly.

## Runtime features (trace, simulation, edits, journey, timeline)

- **Supported architectures:** `llama`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `gemma3`, `phi3`,
  `bert`, `nomic-bert`. Only llama, qwen2 and nomic-bert were run on the test machine (no gemma / phi /
  bert model installed).
- **No MoE and no SSM / hybrid forward pass.** `qwen3moe`, `qwen35` (Qwen 3.5 / 3.8: linear attention
  + attention) and other hybrids get static analysis, static impact and bake only. Router choices and
  SSM state contributions are never measured; expert edits are bake-only; expert sweeps are refused.
- **Embedding models** (`nomic-bert`, `bert`) have no next-token prediction: the logit lens, decision
  breakdown, generation and logit edits do not apply. Journeys still show attention and FFN activity;
  comparisons report the cosine between pooled embeddings.
- **gemma2** logit / attention soft-capping is not applied.
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

BPE pre-tokenizers are exact for llama3 / qwen2 / gpt2; others fall back to the llama3 rules with a
warning. SPM is implemented but untested here; UGM / RWKV are not implemented.

## Web and 3D

- **WebGPU needs a secure context:** HTTPS or `localhost` / `127.0.0.1`. With `--host 0.0.0.0`, a page
  opened from another device over plain HTTP gets WebGL 2 through the automatic fallback. A browser
  with neither WebGPU nor WebGL 2 cannot show the graphs or the embedding map (the static pages work).
- The 2D and tower / city / force views of 0.1.2 were checked in headless Chrome on the WebGL 2 path
  (software rendering). The 3D dataflow view, the token flow, the embedding map and the token
  inspector / edit pages are new and have only been built, not yet run in a browser: follow
  [TEST.md](https://github.com/aymen94/NeuronDump/blob/main/TEST.md) sections 14–16. A headless
  `--screenshot` can be taken before the WebAssembly finishes loading.
- The embedding map of very large vocabularies (250k+ tokens) is heavy on software rendering.
- The REPL `web` command starts one server per session; the model list refreshes on `/api/models`
  calls rather than OS file events.

## Other

- No automated tests, fuzz targets or benchmarks; clippy is not run.
- Graph importance without measurements is structural only. Run `trace --corpus`, `inspect --stats` or
  `sweep` for real numbers.
- `comfy-table` is built without terminal-width detection (Windows GNU toolchain workaround), so wide
  tables do not wrap.
