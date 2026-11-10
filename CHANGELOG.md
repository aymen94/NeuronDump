# Changelog

## Unreleased

- **Gemma 3 forward pass:** sliding-window layers get their own RoPE base (`rope.freq_base_swa`,
  10,000) and the window mask, in llama.cpp's layout (five local layers per global one; Gemma 2
  alternates). gemma3:1b now matches Ollama (first tokens 0.186 / 0.449 / 0.887 against 0.163 /
  0.452 / 0.877; before: 0.103 / 0.223 / 0.648). Linear RoPE scaling (`rope.scaling.factor`) is applied
  to the global layers.
- **Memory estimate:** sliding-window layers are counted at most at their window (gemma3:1b at 32K:
  139 MiB of f16 KV cache instead of 832 MiB).
- **Research:** [Inside Gemma 3 1B](docs/gemma3-research.md), every probe on one model.

## 0.1.5

- **Probes** ([docs/probes.md](docs/probes.md)), each a CLI command, a job and a section of the new
  web **Probes** page:
  - `causal`: causal tracing. Restores clean states (residual, attention or FFN) inside a
    corrupted run, layer × token, to show where a fact is held. Passes resume at the restored
    layer, about 2× faster than full passes.
  - `heads`: attention head roles (induction, previous-token, duplicate-token, self, sink) from
    repeated random tokens, with graph **color by role**.
  - `similarity`: linear CKA between the residual stream after every layer on a corpus, plus the
    most redundant layers (with skip buttons).
  - `spectrum`: largest singular value, stable rank, outlier ratio and kurtosis of every weight
    matrix, with graph **color by rank**.
  - `neuron`: what one FFN neuron reads (embeddings) and writes (unembedding) in token space.
  - `glitch`: under-trained token candidates (abnormally small embedding / output rows) with a
    reachability check.
  - `memory`: weights + KV cache per context length for f16 / q8_0 / q4_0 caches
    (`GET /api/models/{id}/memory`).
- **Journey:** compare two prompts side by side (logit lens per layer, where they diverge, push per layer).
- **3D graph controls:** a panel over the canvas with press-and-hold orbit / pan / zoom, view
  presets, auto-rotate, speed / inertia / field-of-view sliders, keyboard navigation (WASD / arrows,
  Q/E, R/F, +/−, Space). Three navigation modes: Orbit, Map, and Trackball (free rotation; a new
  `TrackballControls` add-on for wamr3d). Saved camera views per model and **Copy link** (`cam=` in
  the URL). Minimap with the camera and click-to-move. Layout tools: size by parameters / bytes /
  importance, explode, slice plane along x / y / z, isolate the selection with N hops of neighbours.
  Display: labels and label count, connections and their strength, container outlines.
- **Renderer choice:** a **renderer** menu on the graph page (Auto / WebGPU / WebGL 2) rebuilds both
  3D views on the chosen API and is remembered.
- **Header:** a GitHub **Repo** link to the project.
- **Gemma 4 tokenization:** `tokenizer.ggml.model = gemma4` is now encoded with llama.cpp's
  SPM-style BPE rules (spaces escaped to `▁`, merges over raw UTF-8, newline-only pre-split), so
  `tokens`, the token inspector and the embedding map work on Gemma 4 files. Ids verified against
  `llama-server /tokenize` (20 texts, including newline runs, tabs, CJK, emoji and combining marks).

## 0.1.4

- **macOS builds:** release archives for `aarch64-apple-darwin` and `x86_64-apple-darwin`.
- **Model discovery:** Jan models, LM Studio home pointer, `XDG_CACHE_HOME` for Hugging Face,
  `NEURONDUMP_MODEL_DIRS`, `--dir` (alias of `--scan`); `mmproj` files attach to their model.
- **Model diff:** `diff <a> <b>` (`--json`, `--top N`, `--metadata-only`) and a Diff page: streamed
  relative L2 / cosine / max |diff| per tensor, layer, block and expert across any quant types,
  metadata / vocab / chat template diff, graph coloring by diff. `slim` now records removed layers
  (`neurondump.removed_layers`) so slimmed models realign.
- **Browser demo:** the web UI on a WebAssembly build of the static inspection core
  (`crates/neurondump-wasm`, `scripts/build-wasm-demo.sh`), published with the docs at `/demo/`.
  Files are read lazily with `Blob.slice`; nothing is uploaded. `neurondump-core` gains a default
  `native` feature; `neurondump-gguf` can parse a header prefix and load tensor data on demand.

## 0.1.3

- **Phase 14 – 3D model graph:** 2D / 3D toggle with a 3D dataflow view drawn by wamr3d (instanced
  meshes, picking, search, fly-to, expand on demand), automatic WebGPU → WebGL 2 fallback, token flow
  animation, and a 3D embedding map of the vocabulary.
- **Phase 15 – Token inspector:** `tokens` (tokenization with the chat template, embedding norms,
  nearest tokens), `journey` (logit lens, attention heatmaps, FFN activity and decision breakdown per
  layer, head and neuron) and `generate` (step-by-step timeline).
- **Phase 16 – Live edits:** in-memory edits (scale, zero, skip, embedding, ban / boost, patching,
  steering, sampling, prompt and template), baseline vs edited comparison, undo / redo / restore,
  snapshots, portable `edits.json`, and bake into a new GGUF.

## 0.1.2

- **Phase 10 – Model graph:** `graph` and `node`, from sections down to heads, neuron buckets and experts.
- **Phase 11 – Static impact:** what removing or zeroing a part does to the file, with a verdict.
- **Phase 12 – Simulated ablation and sweeps:** run a cut in memory and rank layers or heads by damage.
- **Phase 13 – Web graph:** interactive graph page with cuts, impact, simulation, sweeps and plan → slim.

## 0.1.1

- Trace support included in release builds.

## 0.1.0

- **Phases 0–9:** model discovery, GGUF parser, `inspect`, architecture and tokenizer analysis,
  chat templates, weight statistics, `analyze` and `slim`, `trace`, job system, interactive REPL,
  web interface, exports and `compare`.
