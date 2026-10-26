# HTTP API

`neurondump web` serves a JSON API under `/api`. Every call needs the session token printed at
startup, in the `x-neurondump-token` header, or as `?token=` for links and Server-Sent Events
(which cannot set headers). Without it: `401 {"error": "missing or invalid session token"}`.

```sh
T=<token>; H="x-neurondump-token: $T"; U=http://127.0.0.1:7878/api
M=17d2889ccc34            # model id of llama3.2:1b (ids come from /api/models)
curl -s -H "$H" $U/info
```

Models are referred to by **id**, never by path. Errors are `{"error": "…"}` with 400 (bad request),
404 (unknown model / node / job), 409 (job not finished) or 500.

## Server and models

| Method | Path | Returns |
|---|---|---|
| GET | `/api/info` | `{"version","schema_version","output_dir","trace_architectures":[…],"job_kinds":[…]}` |
| GET | `/api/models` | `{"models":[{id,name,source,architecture,params,quantization,size,…}],"warnings":[]}` |
| POST | `/api/models/rescan` | `{"models": 9}` |
| GET | `/api/models/{id}` | overview: model, file, architecture, tokenizer, Ollama layers, projector, warnings |
| GET | `/api/models/{id}/metadata?full=` | metadata keys (`full=true`: arrays not summarized) |
| GET | `/api/models/{id}/vocab?q=&regex=&class=&script=&lang=&offset=&limit=` | `{"summary", "page": {"total","tokens":[…]}}` |
| GET | `/api/models/{id}/vocab/texts` | `{"texts":["!", "\"", …], "classes":["special",…]}`: every token's decoded text by id |
| GET | `/api/models/{id}/tensors?layer=&role=&q=&sort=&offset=&limit=` | `{"total","tensors":[…]}` |
| GET | `/api/models/{id}/layers` | per-layer kinds |
| GET | `/api/models/{id}/template` | template decision trees |
| GET | `/api/models/{id}/tensors/{name}/stats` | statistics of one tensor |
| GET | `/api/search?q=&model=&limit=` | models, plus tokens / tensors / metadata of `model` |
| GET | `/api/compare?models=a,b` | comparison report |
| GET | `/api/export/{id}?format=json\|csv\|ndjson&table=` | download |

## Graph

| Method | Path | |
|---|---|---|
| GET | `/api/models/{id}/graph?level=&group_by=&expand=` | the graph view: `{"nodes":[…],"edges":[…],"level","group_by","expand"}`. Layouts are computed by the browser. |
| GET | `/api/models/{id}/nodes/{node}` | `{"node","ancestors","children","incoming","outgoing"}` |
| POST | `/api/models/{id}/impact` | static impact of a cut |
| POST | `/api/models/{id}/plan-from-ablation` | a prune plan from a cut or a sweep job |

```sh
curl -s -H "$H" "$U/models/$M/graph?level=layer&expand=layer.7,layer.7.attn"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"cut":["layer.7"],"force":false}' $U/models/$M/impact
# {"verdict":"risky","params_removed":60817408,"new_size":1256447808,"llama_cpp_loads":true,…}
```

## Token inspector

### `POST /api/models/{id}/tokenize`

Static (no inference, any model).

```sh
curl -s -H "$H" -X POST -H "content-type: application/json" \
  -d '{"text":"fn main() {","chat":true,"system":null,"neighbors":5}' $U/models/$M/tokenize
```

```json
{
  "schema_version": 1, "model": "llama3.2:1b", "tokenizer": "gpt2",
  "input": "fn main() {",
  "chat": { "source": "ollama", "system": null, "text": "<|start_header_id|>system<|end_header_id|>…", "prompt_span": [96, 107], "template": "…" },
  "full_text": "<|start_header_id|>system<|end_header_id|>…",
  "tokens": [
    { "index": 0, "id": 128000, "raw": "<|begin_of_text|>", "text": "<|begin_of_text|>", "class": "special", "token_type": "control",
      "special": true, "added": true, "embedding_norm": 0.43, "output_norm": 0.43, "in_output": true, "banned": false, "neighbors": [] },
    { "index": 12, "id": 8998, "raw": "fn", "text": "fn", "class": "code", "special": false, "added": false, "span": [96, 98],
      "embedding_norm": 0.51, "in_output": true, "neighbors": [ { "id": 5079, "text": "func", "cosine": 0.41 } ] }
  ],
  "tied_embeddings": true, "has_output_head": true, "embedding_tensor": "token_embd.weight", "warnings": [], "elapsed_ms": 812
}
```

(Values are illustrative.) `chat` wraps the text in the model's chat template (Ollama's, else a
built-in one for llama3 / chatml / gemma / phi3 families); `added` marks tokens that come from the
template or the tokenizer (BOS); `banned` reflects `ban` edits on the stack.

### `GET /api/models/{id}/embedding-map`

The vocabulary projected to 3D with PCA. Binary `application/octet-stream`: little-endian `float32`,
4 values per token (`x, y, z, class`), token id = index. Headers: `x-neurondump-count`,
`x-neurondump-stride: 4`, `x-neurondump-explained` (variance share of the 3 axes). The first call
computes it (seconds); later calls come from the cache.

```sh
curl -s -D - -o map.bin -H "$H" $U/models/$M/embedding-map | grep -i x-neurondump
curl -s -H "$H" "$U/models/$M/embedding-map?format=json"
# {"count":128256,"dim":2048,"sample":16384,"tensor":"token_embd.weight","explained":[…],"stride":4,"classes":["special",…]}
```

## Edits

All edit endpoints return the **edit view**:

```json
{
  "model": "llama3.2:1b", "digest": "…",
  "edits": [ { "id": 1, "enabled": true, "label": "zero layer.7", "created": 1792000000, "op": "zero", "target": "layer.7" } ],
  "can_undo": true, "can_redo": false, "active": 1,
  "compiled": { "intervention": { "skip_layers": [7], … }, "sampling": null, "system": null, "template": null, "touched": ["layer.7"], "notes": [], "active": 1 },
  "error": null,
  "snapshots": ["formal-steer"]
}
```

| Method | Path | Body | |
|---|---|---|---|
| GET | `/api/models/{id}/edits` | | the stack |
| POST | `/api/models/{id}/edits` | an op, e.g. `{"op":"scale","target":"layer.7.attn.head.3","factor":0}` (optional `label`) | push an edit |
| POST | `/api/models/{id}/edits` | `{"schema_version":1,"edits":[…]}` (an `edits.json`) | replace the stack (import; undoable). The view gets `warnings` when the file was made for another model |
| DELETE | `/api/models/{id}/edits` | | restore all (back to the original model; undoable) |
| DELETE | `/api/models/{id}/edits/{n}` | | remove edit `n` |
| POST | `/api/models/{id}/edits/{n}/toggle` | | switch edit `n` on / off |
| POST | `/api/models/{id}/edits/undo` | | undo (`"changed": false` when there is nothing to undo) |
| POST | `/api/models/{id}/edits/redo` | | redo |
| POST | `/api/models/{id}/edits/restore` | | restore all |
| GET | `/api/models/{id}/edits/export` | | download `<model>.edits.json` |
| GET | `/api/models/{id}/snapshots` | | `{"snapshots":[…]}` |
| POST | `/api/models/{id}/snapshots` | `{"name":"formal","action":"save"}` or `{"name":"formal","action":"load"}` | save / load a named snapshot |
| POST | `/api/models/{id}/bake` | `{"out":"llama-edited.gguf"}` | starts a `bake` job → `202 {"id":"bake-…"}`; written to `<output_dir>/bake/` |

Ops and their fields are listed in [Interventions](interventions.md#edit-operations). A push that does
not compile (unknown node, token out of range, a bad template) returns 400 and leaves the stack as it was.

```sh
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"op":"zero","target":"layer.7"}' $U/models/$M/edits
curl -s -H "$H" -X POST $U/models/$M/edits/undo
curl -s -H "$H" -X POST -H "content-type: application/json" \
  -d '{"op":"steer","layer":6,"a":"Dear Sir or Madam, I am writing to formally request","b":"hey dude, just wanna ask","strength":4}' $U/models/$M/edits
curl -s -H "$H" -X POST $U/models/$M/edits/1/toggle
curl -s -H "$H" -X DELETE $U/models/$M/edits
```

## Jobs

| Method | Path | |
|---|---|---|
| POST | `/api/jobs` | `{"kind","model":"<id>","params":{…}}` → `202 {"id"}` |
| GET | `/api/jobs` | every job |
| GET | `/api/jobs/{id}` | one job: status, progress, step, error, cached |
| DELETE | `/api/jobs/{id}` | cancel → `{"cancelling": true}` |
| GET | `/api/jobs/{id}/events` | Server-Sent Events: `progress` …, then `done` |
| GET | `/api/jobs/{id}/result` | the result JSON (409 while running) |

Kinds: `inspect`, `stats`, `analyze`, `slim`, `trace`, `ablate`, `sweep`, `journey`, `generate`,
`compare`, `bake`, `diff` (`model` is A; `params`: `{ "other": "<id of B>", "top": N, "metadata_only": false }`). The API never accepts server file paths: corpus files are stripped (send texts),
`slim` / `bake` outputs must be plain file names.

`journey`, `generate` and `compare` use the model's **current edit stack** unless `params.edits` is
given (a list of edits; `[]` = none); `journey` and `generate` also take `"baseline": true` to ignore
the edits. The edits are part of the cache key.

### `journey`

```json
{ "kind": "journey", "model": "<id>", "params": { "prompt": "fn main() {", "chat": false, "token": -1, "top_k": 5 } }
```

`token` is the position (negative counts from the end); `append` adds token ids after the prompt
(the timeline uses it for generated tokens). Result:

```json
{
  "tokens": [ { "id": 128000, "text": "<|begin_of_text|>", "special": true }, … ],
  "position": 4, "next": null,
  "prediction": [ { "id": 198, "text": "\n", "prob": 0.61 }, … ],
  "first_top1": 12, "first_top5": 9,
  "layers": [ { "index": 0, "residual_norm": 7.4, "lens": […], "target_prob": 0.0, "target_rank": 5012,
                "attention": [[…per key…] …per head…], "attended": [[0, 0.71], …],
                "ffn_buckets": [["layer.0.ffn.n.512-575", 0.21], …], "top_neurons": [[1031, 2.9], …], "skipped": false } ],
  "decision": { "chosen": {…}, "runner_up": {…}, "logit_diff": 1.9, "attributed": 1.8, "embedding": 0.02,
                "layers": [ { "layer": 0, "attn": 0.01, "ffn": -0.03 }, … ],
                "top_heads": [ { "layer": 13, "head": 7, "value": 0.42 }, … ],
                "top_neurons": [ { "layer": 15, "neuron": 4012, "activation": 3.1, "value": 0.31 }, … ] },
  "highlight": ["layer.13.attn.head.7", "layer.15", "layer.15.ffn", …],
  "edited": false, "warnings": [], "elapsed_ms": 4200
}
```

### `generate`

```json
{ "kind": "generate", "model": "<id>", "params": { "prompt": "The name of the star that Earth orbits is", "steps": 12, "explain": false,
  "temperature": 0.8, "sample_top_k": 40, "top_p": 0.9, "seed": 1 } }
```

Result: `steps: [{index, position, token:{id,text,prob}, entropy, alternatives:[…], forced, decision?}]`,
`generated`, `generated_ids`, `sampling`, `tokens_per_s`. `explain: true` adds a decision breakdown to
every step.

### `compare`

```json
{ "kind": "compare", "model": "<id>", "params": { "prompt": "The name of the star that Earth orbits is", "tokens": 12, "corpus": ["Paris is the capital of France."] } }
```

Result: `{"edits":[…], "compiled":{…}, "simulation": SimReport}`. Each `simulation.prompts[]` entry has
`baseline` / `ablated` (= edited) sides with `top_k` and `generated`, `kl`, `top1_changed`, `lens`
(logit lens per layer), `edited_prompt` (when a prompt edit changed the text) and
`decision_baseline` / `decision_edited`; `simulation.corpus` has the perplexities and mean KL.

### `bake`

`{"out": "name.gguf"}` (plus optional `edits`). Result: `output`, `modelfile`, `original_size`,
`new_size`, `applied`, `skipped` (with reasons), `tensors_changed`, `requantized`, `notes`, `validation`.

### Older kinds

`ablate`: `{"cut":["layer.7"],"prompt":"…","corpus":[…],"tokens":12,"force":false,"static":false}`;
`sweep`: `{"level":"layer"|"head","layer":7,"corpus":[…]}`; `trace`: `{"prompt"|"corpus","top_k","max_tokens"}`;
`analyze`: `{"corpus_text","compute_stats","skip_layer_similarity"}`; `slim`: `{"plan","only","out"}`;
`stats`, `inspect`.

## Assets

`/` serves the UI; `/assets/{path}` the embedded files with their MIME types: `.js` as
`text/javascript`, `.css`, `.html`, and **`.wasm` as `application/wasm`** (needed by
`WebAssembly.instantiateStreaming` for wamr3d).

```sh
curl -sI http://127.0.0.1:7878/assets/vendor/wamr3d/wamr3d_wasm_bg.wasm | grep -i content-type
# content-type: application/wasm
```
