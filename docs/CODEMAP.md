# Code map

Find the right file without reading code. One line per module; keep it updated when adding modules.

**Overview.** `neurondump-gguf` parses / dequantizes / writes GGUF. `neurondump-core` holds all
logic (discovery, inspection, graph, stats, analyze / slim, edits, diff, jobs) behind the `App`
facade; its `native` feature adds everything needing files or threads. `neurondump-trace` is the
candle forward pass, plugged into core's `TraceBackend`. `neurondump-cli` and `neurondump-web` are
thin front ends; `neurondump` wires them into the binary; `neurondump-wasm` runs the static part
in the browser demo with the same web frontend.

## neurondump-gguf

- `lib.rs` — re-exports of the GGUF API.
- `error.rs` — `GgufError` / `Result`.
- `reader.rs` — bounds-checked little/big-endian cursor over raw bytes.
- `types.rs` — ggml tensor types, block layouts, `general.file_type` names.
- `file/mod.rs` — `GgufFile`, `TensorInfo`, parse options, storage (mmap, bytes, lazy window), open.
- `file/parse.rs` — header parsing: metadata, tensor infos, alignment, bounds warnings.
- `file/access.rs` — metadata lookups, tensors by name, totals, tensor byte ranges.
- `value/mod.rs` — metadata `Value` / `ValueType` and scalar accessors.
- `value/array.rs` — typed metadata arrays, views, previews.
- `value/serialize.rs` — JSON serialization and display of values.
- `dequant/mod.rs` — supported types, `quantize_q8_0`.
- `dequant/kernels.rs` — per-type dequantization kernels (one big match).
- `dequant/stream.rs` — streaming `Dequantizer`, row-range dequantization on `GgufFile`.
- `writer.rs` — GGUF v3 writer used by slim and bake.

## neurondump-core

- `lib.rs` — module list, re-exports, `native` feature gating.
- `util.rs` — ids, hashing, human sizes, progress / cancellation, wasm-safe `Instant`.
- `model.rs` — `ModelHandle`: opened GGUF plus Ollama side layers.
- `api.rs` — JSON of read-only endpoints, shared by web server and wasm.
- `inspect.rs` — `inspect`: full report of one model file.
- `export.rs` — JSON / NDJSON / CSV exports.
- `search.rs` — search across models, tokens, tensors, metadata.
- `compare.rs` — side-by-side comparison of several models.
- `config.rs` — user config file (native).
- `jobs.rs` — job records, progress, cancellation, stored results (native).
- `discovery/mod.rs` — `ModelEntry`, `SourceKind`, summaries, options, quantization label.
- `discovery/library.rs` — `Library`: scan sources, rescan on change, resolve names, dedup.
- `discovery/files.rs` — loose `.gguf` sources: llama.cpp, Hugging Face, LM Studio, Jan, dirs.
- `discovery/ollama.rs` — Ollama manifests / blobs store.
- `arch/mod.rs` — architecture report types, hyperparameters.
- `arch/roles.rs` — tensor name → layer, role, component.
- `arch/analyze.rs` — `analyze_architecture`: layers and their kinds.
- `tokenizer/mod.rs` — `Vocab` from GGUF.
- `tokenizer/types.rs` — token types / classes, special tokens, vocab summary.
- `tokenizer/text.rs` — raw token text → readable text (byte maps, SPM spaces).
- `tokenizer/classify.rs` — token classification, special-token keys.
- `tokenizer/search.rs` — vocabulary search and paging.
- `tokenizer/keywords.rs` — programming keyword tables.
- `tokenizer/encode/mod.rs` — `Encoder`: text → ids, pre-tokenization.
- `tokenizer/encode/algos.rs` — BPE, SentencePiece, WordPiece algorithms.
- `template/mod.rs` — chat template decision trees: types, report.
- `template/lex.rs` — template pieces, variables and specials of tags.
- `template/builder.rs` — nested branch tree builder.
- `template/parse.rs` — Jinja and Go template parsers.
- `chat/mod.rs` — chat prompt rendering, built-in templates.
- `chat/ast.rs` — Go template values and parsed tree.
- `chat/parse.rs` — Go `text/template` lexer and parser.
- `chat/eval.rs` — Go template evaluator.
- `stats/mod.rs` — weight statistics types and options.
- `stats/tensor.rs` — streaming stats of one tensor; tensor cosine.
- `stats/model.rs` — `compute_stats`: all tensors, head / neuron / expert norms.
- `graph/mod.rs` — structural model graph (module root).
- `graph/kinds.rs` — node / edge kinds, removability, levels, grouping.
- `graph/types.rs` — nodes, edges, `ModelGraph`, views, node details.
- `graph/helpers.rs` — ids, labels, block / stage classification.
- `graph/build.rs` — `ModelGraph::build` from the architecture (one large function).
- `graph/measure.rs` — apply stats / traces / sweeps to nodes.
- `graph/fine.rs` — lazy heads / neuron buckets / experts, id resolution.
- `graph/query.rs` — node lookups, ancestors, details.
- `graph/view.rs` — level / grouping / expansion views with lifted edges.
- `graph/export.rs` — tree, DOT, Mermaid exports.
- `tokens/mod.rs` — token inspector types.
- `tokens/neighbors.rs` — embedding rows, nearest tokens.
- `tokens/tokenize.rs` — `tokenize` with template, spans, norms, neighbours.
- `tokens/embedding_map.rs` — PCA embedding map for the 3D view.
- `trace/mod.rs` — `TraceBackend` trait, backend registry, entry points.
- `trace/report.rs` — trace request / report, logit lens.
- `trace/intervention.rs` — ablation masks, steering, patching, sampling.
- `trace/journey.rs` — journey and generation types, decision breakdown.
- `trace/simulate.rs` — simulation and sweep types.
- `analyze/mod.rs` — prune plan, candidates, options (native).
- `analyze/find.rs` — `analyze`: find removable parts (one large function).
- `slim/mod.rs` — slim options / report, file-name checks.
- `slim/write.rs` — `slim`: write the new GGUF (one large function).
- `slim/tensors.rs` — vocab subsets, row gathering, zeroing.
- `slim/check.rs` — Modelfile, validation, llama-perplexity.
- `ablate/mod.rs` — cut / impact types.
- `ablate/cut.rs` — cut ids → targets and `Ablation`.
- `ablate/zero.rs` — zero-in-place operations, requantization cost.
- `ablate/impact.rs` — `static_impact` of a cut.
- `ablate/run.rs` — simulated ablation and sweeps.
- `ablate/plan.rs` — cut / sweep → prune plan.
- `edits/mod.rs` — live edits (module root).
- `edits/ops.rs` — edit ops, edits.json, token arguments.
- `edits/stack.rs` — edit stack with undo / redo.
- `edits/compile.rs` — edits → `Intervention`.
- `edits/bake.rs` — bake edits into a new GGUF.
- `edits/snapshots.rs` — named snapshots on disk.
- `diff/mod.rs` — diff report types, slim layer realignment.
- `diff/weights.rs` — streamed weight diff, per node / expert aggregation.
- `diff/meta.rs` — metadata, vocab, template differences.
- `app/mod.rs` — `App` facade: library, model cache, result cache.
- `app/params.rs` — job parameter types, edit view.
- `app/cache.rs` — cached vocab, stats, graph, sweeps, impact, plans, embedding map.
- `app/jobs.rs` — `App::run` for every job kind, `App::submit`.
- `app/edits.rs` — per-model edit stacks, snapshots, prompt / sampling.

## neurondump-trace

- `lib.rs` — module root, `SUPPORTED`, `ModelTracer`.
- `spec.rs` — per-architecture forward-pass description.
- `weights.rs` — weights from the GGUF: linear, norm, layer blocks.
- `model.rs` — loaded model, records, loading, RoPE, logit lens.
- `masks.rs` — intervention masks per layer / head.
- `forward.rs` — instrumented forward pass `Model::run` (one large function).
- `decode.rs` — masks building, generation, decision breakdown.
- `math.rs` — sampling, softmax / KL / entropy, cosine, corpus metrics.
- `backend.rs` — `CandleBackend` (`TraceBackend` impl, trace).
- `simulate.rs` / `journey.rs` / `generate.rs` / `sweep.rs` — one backend operation each.

## neurondump-cli

- `lib.rs` — `main`, interface picker, run.
- `cli_args.rs` — global flags, web hook, prompt args, config actions.
- `commands.rs` — all subcommands (`Cmd`).
- `dispatch.rs` — `run_command`: one match over subcommands.
- `context.rs` — `Ctx`, JSON output, inspect helpers.
- `progress.rs` — progress bar.
- `render/mod.rs` — table helpers; `render/{model,analysis,graph,tokens,edits,diff}.rs` — text output per area.
- `repl/mod.rs` — REPL loop (one large function).
- `repl/helper.rs` — command table, completion; `repl/util.rs` — args, paging; `repl/jobs.rs` — background jobs.

## neurondump-web

- `lib.rs` — router, serve, browser launch.
- `state.rs` — shared state, API errors, auth.
- `assets.rs` — embedded `frontend/` files.
- `models.rs` / `jobs.rs` / `graph.rs` / `edits.rs` — HTTP endpoints per area.
- `frontend/app.js` — hash router, nav, theme.
- `frontend/api.js` — API client, local (wasm) backend hook.
- `frontend/ui.js` — DOM helpers; `frontend/session.js` — per-model session state.
- `frontend/views.js` — shared job helpers, re-exports pages.
- `frontend/views/{model,analysis,global}.js` — model pages, analysis pages, compare / jobs / search.
- `frontend/graph.js` — graph page (one closure: controls, colors, cuts, sweeps).
- `frontend/graph3d.js` + `graph3d-layout.js` — 2D / tower / city / force renderer and layouts.
- `frontend/flow3d.js` + `flow3d-layout.js` — 3D dataflow renderer and layout.
- `frontend/wamr.js` — wamr3d boot; `frontend/embedmap.js` — 3D embedding map.
- `frontend/inspector.js` — tokens page; `frontend/journey.js` — journey and timeline pages.
- `frontend/edits.js` — edits page and drawer; `frontend/diff.js` — diff page.
- `frontend/vendor/` — vendored wamr3d (do not edit).

## neurondump-wasm, neurondump

- `neurondump-wasm/src/lib.rs` — browser `Session`: lazy header parse, API routing, tensor stats.
- `neurondump-wasm/www/demo.js` — demo boot: banner, file drop, local backend.
- `neurondump/src/main.rs` — binary wiring (CLI, web, trace backend).
