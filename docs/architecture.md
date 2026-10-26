# Architecture

How the code is organized. For what NeuronDump reads from a GGUF file and why, see
[How it works (details)](internals.md).

## Crates

| Crate | Job | Depends on |
|---|---|---|
| [`neurondump-gguf`](https://crates.io/crates/neurondump-gguf) | zero-copy GGUF parser (memory-mapped), dequantization, Q8_0 quantization, GGUF v3 writer | `memmap2`, `half` |
| [`neurondump-core`](https://crates.io/crates/neurondump-core) | **all the logic**: discovery, architecture, tokenizer, templates, stats, analyze, slim, graph, impact, chat rendering, token inspector, edit overlay, bake, jobs | `neurondump-gguf`, `rayon`, `serde` |
| [`neurondump-trace`](https://crates.io/crates/neurondump-trace) | the forward pass (candle, CPU): trace, simulation, sweeps, token journey, generation, interventions | `candle-core`, `neurondump-core` |
| [`neurondump-cli`](https://crates.io/crates/neurondump-cli) | one-shot commands, the interactive REPL, terminal rendering | `neurondump-core`, `clap`, `rustyline` |
| [`neurondump-web`](https://crates.io/crates/neurondump-web) | axum server, JSON API, embedded frontend (plain ES modules + vendored wamr3d) | `neurondump-core`, `axum`, `rust-embed` |
| [`neurondump`](https://crates.io/crates/neurondump) | the binary: wires the front ends and registers the trace backend | features `cli`, `web`, `trace` |

All six crates are published on crates.io with the same version. The library crates have API
docs on docs.rs ([gguf](https://docs.rs/neurondump-gguf), [core](https://docs.rs/neurondump-core),
[trace](https://docs.rs/neurondump-trace), [cli](https://docs.rs/neurondump-cli),
[web](https://docs.rs/neurondump-web)); `neurondump` is binary-only, so it has none.

Front ends contain no analysis code. The CLI, the REPL and the web API call the same
`neurondump_core::App` methods and render the same serializable results, so they always agree.
Every result is JSON with a `schema_version`.

## Core API

`App` (`crates/neurondump-core/src/app.rs`) is the facade:

| Area | Methods |
|---|---|
| models | `library()`, `rescan()`, `resolve(query)`, `open(query)`, `open_id(id)` (web: ids only), `vocab(m)` |
| measurements | `stats(m, compute, p)`, `latest_corpus_trace(m)`, `latest_sweep(m, level, layer)` |
| graph | `graph(m)` (cached per model, rebuilt after jobs that add measurements) |
| impact / plans | `impact(m, cut, force)`, `plan_from_cut`, `plan_from_sweep` |
| edits | `edit_stack(m)`, `edit(m, f)` (any change; rolled back if the result does not compile), `push_edit`, `edit_view`, `import_edits`, `save_snapshot`, `load_snapshot`, `prompt_text` |
| token inspector | `embedding_map(m, p)` (memory + disk cache) |
| jobs | `run(kind, model, params, p)` (synchronous, used by the CLI), `submit(kind, model, params)` (background thread, used by the web and the REPL `&`) |

Module map of `neurondump-core`:

| Module | Contents |
|---|---|
| `discovery` | Ollama manifests, extra scan dirs, ids, "did you mean" |
| `model` | `ModelHandle`: the mmapped GGUF plus the Ollama layers (template, system, params) |
| `arch`, `inspect`, `stats` | architecture and tensor roles, the inspect report, weight statistics |
| `tokenizer` | vocabulary decoding, token classes, BPE / SPM / WPM encoders |
| `template` | Jinja and Go template **parsers** (decision trees) |
| `chat` | a small Go `text/template` **evaluator** for Ollama templates, plus built-in templates (llama3, chatml, gemma, phi3) |
| `analyze`, `slim` | prune plans and the slim writer |
| `graph` | the structural graph, views, node details, fine nodes (heads, neuron buckets, experts) |
| `ablate` | static impact, `Ablation` from a cut, file zero ops, sweeps, plans |
| `trace` | backend trait and every request / report type of the forward pass |
| `tokens` | tokenize report (template tokens, embedding norms, neighbours) and the PCA embedding map |
| `edits` | edit ops, the edit stack (undo / redo), compile to an intervention, bake, snapshots |
| `jobs` | job manager, progress, cancellation, results on disk |

## Job system

Long work runs as a job: `inspect`, `stats`, `analyze`, `slim`, `trace`, `ablate`, `sweep`,
`journey`, `generate`, `compare`, `bake`.

1. `App::submit` creates a `JobInfo` (queued), spawns a thread and calls `App::run`.
2. `run` reports progress through the `Progress` trait (percent + step name, throttled) and checks
   cancellation between steps; a cancel request makes the next check return `Cancelled`.
3. Results are written to `<output_dir>/jobs/<id>.json`; the web streams progress with Server-Sent
   Events (`/api/jobs/{id}/events`) and serves the result at `/api/jobs/{id}/result`.
4. Deterministic jobs are **cached** in `<output_dir>/cache/<digest>/<kind>-<hash>.json`, keyed by the
   model digest and the full parameters. `journey`, `generate` and `compare` include the edits in
   their parameters (injected from the edit stack when the caller gives none), so a cached result
   always matches the edits it was computed with. `slim` and `bake` are never cached.

## Graph and layout

`ModelGraph::build` turns architecture metadata and tensor roles into nodes
(Model → Section → Layer → Block → Tensor, plus lazily generated fine nodes) and edges
(dataflow, contains, tied, uses). `view(level, group_by, expand)` returns the visible nodes, lifts
edges to the nearest visible ancestor and adds synthetic group nodes for `group_by`.

The server sends only this structure. **Layouts are computed in the browser** on top of wamr3d
(see [3D graph](3d-graph.md)): the flat 2D diagram and the tower / city / force layouts in
`graph3d.js`, and the 3D dataflow layout in `flow3d.js`. Every view uses the graph's node ids, so a
selection, a cut, an edit mark or a token-path highlight is the same set of ids in all of them.

## Forward pass and interventions

`neurondump-trace` builds a generic llama-family decoder and a BERT-family encoder straight from
the mmapped GGUF (quantized matmuls through candle's `QMatMul`). One function, `Model::run`, serves
every feature; options select what it measures or records:

| Option | Used by |
|---|---|
| `measure` | trace (entropy, top neurons, logit lens per layer) |
| `lens_probs`, `all_logits` | simulation and sweeps (target probability per layer, perplexity, KL) |
| `record: Some(pos)` | journey, decision breakdown, explained generation: attention rows of every head, attention output before the projection, mixer and FFN contributions, FFN activations, residual after each layer |
| `keep` | steering vectors and patching sources (every layer's residual rows and attention outputs) |

An `Intervention` (`trace::Ablation`, the old ablation type extended) is compiled to `Masks`:
skipped layers, zeroed blocks, per-layer mixer / FFN factors, per-head and per-neuron factors,
embedding row edits, logit bias and bans, steering vectors and patch sources. The masks are applied
to activations during the pass, which is equivalent to scaling the touched weights; nothing
else is copied and the GGUF is never written. Details: [Interventions](interventions.md).

**Direct logit attribution** (`Model::decision`): the final norm is linearized with its actual scale
at the position, so every residual contribution `c` pushes the chosen token against the runner-up by
`((W_U[chosen] − W_U[runner_up]) ⊙ g) · c / rms(h_final)`. Embedding + every mixer + every FFN adds up
to the model's logit difference; per-head values project each head's slice through `attn_output` on
its own; per-neuron values use the matching columns of `ffn_down`. `attributed` is computed in exact f32 from the recorded activations; the model's own logits come from quantized matmuls (candle also quantizes the activations to 8 bits), so on quantized models the two can differ by a few tenths of a logit, which matters when the chosen token and the runner-up are close.

## Edit overlay

```
EditOp (JSON, `op` tag) ─┐
                         ├─ EditStack (per model, in App): edits, undo / redo checkpoints, ids
Edit {id, enabled, label}┘        │
                                  ├─ compile(graph, vocab, edits) → Compiled
                                  │     intervention (masks) · sampling · system · template · touched node ids
                                  ├─ journey / generate / compare jobs (edited side)
                                  └─ bake(…) → new GGUF + Modelfile
```

- `compile` resolves node ids through the graph (layer, block, KV group, head, neuron bucket,
  single neuron `layer.L.ffn.neuron.N`, expert, tensor, section, `vocab:` range). Ablation ops
  (`zero`, `skip`, `ablate`) reuse `ablate::resolve_cut` + `ablation_for`, so an ablation is just an
  edit on the same stack.
- The stack is shared by every front end of one process: edits made in the REPL show up in the web UI
  started with the REPL `web` command, and the other way round.
- Snapshots are `edits.json` files in `<output_dir>/edits/<digest>/<name>.json`; the same format is the
  portable patch for `neurondump edit --apply`.
- `bake` dequantizes only the touched tensors, applies the ops, re-encodes them (F32 / F16 / BF16 stay;
  quantized types become Q8_0 unless the change is a pure zeroing that keeps the type), writes every
  other tensor unchanged, then validates the file and writes a Modelfile with the edited template,
  system prompt and sampling.
