# Changelog

Features by development phase.

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
