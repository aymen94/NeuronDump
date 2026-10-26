# NeuronDump manual test plan

Run from the repo root in **Git Bash**. Known gaps are in [docs/limitations.md](docs/limitations.md).

```sh
cargo build --release --features trace
alias nd=./target/release/neurondump
# a model's GGUF path, for llama.cpp tools:
blob() { nd --json list | python -c "import json,sys;print(next(m['path'] for m in json.load(sys.stdin)['models'] if m['name']==sys.argv[1]))" "$1"; }
```

**Models** (`ollama pull <name>`): `llama3.2:1b` (main model for exact numbers), `qwen2.5:0.5b`,
`nomic-embed-text`, `qwen3:8b`, `gemma2:2b`, `oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU` (hybrid; the
name starts with **o**), `qwen3.8:27b` (hybrid + vision). Diff pairs: `huihui_ai/qwen2.5-abliterate:0.5b`.
Model ids are derived from names: `llama3.2:1b` = `17d2889ccc34`, `qwen2.5:0.5b` = `7f1452142d64`,
`nomic-embed-text:latest` = `84bf674d64ec`. Numbers can change slightly if Ollama updates a model.

Shared files:
```sh
cat > /tmp/corpus5.txt <<'EOF'
The quick brown fox jumps over the lazy dog. It was a sunny day and the children were playing in the park.

Paris is the capital of France. Berlin is the capital of Germany, and Madrid is the capital of Spain.

Rust is a systems programming language focused on safety, speed and concurrency.

Water boils at one hundred degrees Celsius at sea level. Ice melts at zero degrees.

Once upon a time there was a little girl who lived in a village near the forest.
EOF
P="The name of the star that Earth orbits is"
```

---

## 1. Discovery, `list`, `inspect`

```sh
nd --timings list
nd --dir ~/models list                                  # extra folder (alias --scan)
NEURONDUMP_MODEL_DIRS="$HOME/models:/nope" nd list      # path list; missing folders skipped quietly
nd --timings inspect llama3.2:1b
nd inspect nomic-embed-text
nd inspect oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU
nd inspect --all
```

**Expected:**
- `list` shows every model from Ollama, LM Studio, Jan, the Hugging Face cache and llama.cpp's cache,
  with the app in the **SOURCE** column; a file reachable twice is listed once; a `mmproj*.gguf` next to a
  model shows as `+vision` on that model. Under 1 s.
- `inspect llama3.2:1b`: GGUF v3, 147 tensors, `llama [decoder]`, 1.24B params, 16 layers, 32 / 8 heads,
  vocab 128256, tied embeddings, `bos=128000`, `eos=128009`, two templates. About 0.5 s.
- `nomic-embed-text`: `nomic-bert [encoder]`, 12 layers, 30522 tokens. qwen3.5: `qwen35 [hybrid]`,
  full attention at layers 3, 7, …, 23, linear attention elsewhere.
- `inspect --all`: `N model(s), 0 failed`.

**Edge cases:** `nd inspect llama3.2:1bb` → "did you mean", exit 1; `amazonasgabriel/…` → suggests the
`o…` name; a truncated GGUF (`head -c 5000000 x.gguf`) → warnings, exit 0; `head -c 100` → error, no
panic; a text file → "not a GGUF file".

## 2. Layers, tensors, vocabulary, templates

```sh
nd layers oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU
nd tensors llama3.2:1b --layer 3 --sort size
nd tensors qwen2.5:0.5b --layer 0 | grep bias
nd vocab llama3.2:1b --search Rome --limit 5
nd vocab llama3.2:1b --class code --lang rust --limit 10
nd vocab llama3.2:1b --tokenize 'Hello world, fn main() { println!("hi"); }  123456'
llama-tokenize -m "$(blob llama3.2:1b)" -p 'Hello world, fn main() { println!("hi"); }  123456' --ids --log-disable
nd template llama3.2:1b
```

**Expected:** qwen3.5 alternates 3 linear + 1 full layer; `ffn_*` tensors come first by size (17 MiB
each); qwen2.5 has `attn_q/k/v.bias`. `--tokenize` ids are **identical** to `llama-tokenize`
(`[128000, 9906, 1917, 11, 5279, 1925, 368, 314, 14069, 17667, 6151, 5146, 335, 220, 220, 4513, 10961]`).
Repeat the tokenizer check on `qwen2.5:0.5b`, `gemma2:2b` and `phi3` (identical) and `nomic-embed-text`
(differs only on accented / CJK text). On Windows pass text with `-p`, not `--stdin` (it drops `\r`).
The template prints a decision tree (`if custom_tools is defined`, `for t in tools`, emits
`<|start_header_id|>` … `<|eot_id|>`) followed by the Ollama Go template tree.

**Edge cases:** `--sort bogus`, `--class bogus` → error listing the valid values; `--search '[' --regex`
→ regex error, no panic.

## 3. Stats, jobs, cache

```sh
time nd stats llama3.2:1b --limit 10
time nd stats llama3.2:1b --limit 3        # "(cached result)", instant
nd stats llama3.2:1b --tensor blk.0.attn_q.weight
```

**Expected:** about 3 s with a progress bar; per-head table 32 heads / 8 KV, FFN 8192 neurons; the
second run is cached. Ctrl-C on `nd stats qwen3.8:27b` stops at once with no partial cache file.

## 4. Picker, config, REPL

```sh
nd                                    # interactive picker on a TTY
nd < /dev/null ; echo "exit=$?"       # non-TTY: help text, exit=2
nd config set default_ui cli && nd    # starts the REPL; reset with: nd config reset
```

REPL (`nd cli`, pick `llama3.2:1b`): `info`, `layer 3`, `vocab Rome --limit 5`, `stats &`, `jobs`,
`job <id>`, `use nomic-embed-text`, `diff qwen2.5:0.5b`, `graph --level section`, `cut layer.7`,
`impact`, `web`, `exit`. Tab completes commands, models and node ids; history survives a restart.

**Expected:** prompt `neurondump[llama3.2:1b]>`; `stats &` starts a job and `job <id>` renders it;
long outputs page; `cut layer.7` prints `60.8M params, 1.23 GiB -> 1.17 GiB, verdict risky`; `web` opens
the UI on the same session. Unknown command / model → clear errors; Ctrl-C at the prompt does not exit.

## 5. Web API and security

Start `nd web --port 7878 --open`; then `T=<token>; B=http://127.0.0.1:7878; H="x-neurondump-token: $T"`.

```sh
curl -s -o /dev/null -w "%{http_code}\n" $B/api/models                                   # 401
curl -s -H "$H" "$B/api/models/17d2889ccc34/vocab?q=rome&limit=5" | head -c 200; echo
J=$(curl -s -H "$H" -H "content-type: application/json" -d '{"kind":"stats","model":"84bf674d64ec","params":{}}' $B/api/jobs | python -c "import json,sys;print(json.load(sys.stdin)['id'])")
curl -s -N "$B/api/jobs/$J/events?token=$T"                                             # progress … done
curl -s -w " %{http_code}\n" -H "$H" -H "content-type: application/json" -d '{"kind":"inspect","model":"C:/Windows/win.ini"}' $B/api/jobs   # 404
curl -s -o /dev/null -w "%{http_code}\n" "$B/assets/..%2F..%2FCargo.toml"                # 400
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" $B/assets/vendor/wamr3d/wamr3d_wasm_bg.wasm   # 200 application/wasm
```

**Expected:** 401 without a token, 202 + `{"id": …}` for jobs, an SSE stream ending in `event: done`,
409 for a result while running, 404 for paths sent as model ids, 400 for traversal and for slim / bake
names like `../evil.gguf`. `nd web --host 0.0.0.0` prints a warning.

## 6. Web UI

Check every page on `llama3.2:1b` (Chrome or Edge):
- **Models:** filter, sortable columns, Rescan, theme switch (System / Light / Dark, remembered).
- **overview, vocab** (fast paging over 128k tokens), **tensors** (click → stats), **layers**, **template**.
- **graph:** 3D dataflow by default (badge `WebGPU`, or `WebGL 2` with `?backend=webgl`); 2D, tower,
  city, force; double-click down to heads and neuron buckets; search flies to `blk.3.ffn_down`;
  cut → live impact; color by removable / kind / quant / importance / damage / flow / diff.
- **tokens / journey / timeline** (trace build): chips, neighbours, embedding map, heatmaps, decision
  breakdown, **Show path in the graph**, **▶ token flow**.
- **Edits** drawer and page (section 9). **Diff** page (section 10). **Jobs**: live progress, cancel.
- Header search (`/`), click-to-copy ids, the running-jobs badge.
- Without `?token=` → "No session token".

## 7. `analyze` + `slim`

```sh
nd analyze llama3.2:1b --corpus /tmp/corpus5.txt --out /tmp/llama.prune.json
nd slim llama3.2:1b --plan /tmp/llama.prune.json --only vocab --out llama-vocab.gguf
nd slim llama3.2:1b --plan /tmp/llama.prune.json --only reserved_tokens --out llama-noreserved.gguf --perplexity README.md
llama-completion -m ~/neurondump/slim/llama-vocab.gguf -p "The quick brown fox" -n 12 --temp 0 -no-cnv
cd ~/neurondump/slim && ollama create llama-noreserved -f llama-noreserved.Modelfile && ollama run llama-noreserved "Why is the sky blue?"; ollama rm llama-noreserved; cd -
```

**Expected:** the plan lists `vocab_trim`, `reserved_tokens` (248 tokens) and 3 `layer_N` candidates
(never layer 0 or 15). `llama-vocab.gguf`: `1.23 GiB -> 987 MiB`, `Validate OK (147 tensors, vocab …,
block_count 16)`; it continues "jumps over the lazy dog." in llama.cpp. `llama-noreserved`: perplexity
unchanged (≈50.9 on README.md), runs in Ollama. A plan from another model, a bad output name or an
invalid plan are refused; experimental candidates (`low_norm_heads`) are refused with a reason.

## 8. Trace, static impact, simulation, sweep (trace build)

```sh
nd trace llama3.2:1b --prompt "$P"
nd ablate llama3.2:1b --cut layer.7 --static
nd ablate llama3.2:1b --cut token_embd --static ; echo "exit=$?"
nd ablate oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --cut layer.5 --static
nd ablate llama3.2:1b --cut layer.7 --prompt "$P" --tokens 12
for c in layer.7 layer.15; do nd ablate llama3.2:1b --cut $c --corpus /tmp/corpus5.txt --tokens 0 --prompt Hi | grep -E "^corpus"; done
nd sweep llama3.2:1b --corpus /tmp/corpus5.txt --save-plan /tmp/sweep.plan.json --top 2
nd trace oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --prompt hi      # unsupported, static still works
```

**Expected:**
- **trace:** the lens shows `" Earth"` at layer 11, `" called" 0.39` at layer 14, final `" not" 0.166`.
- **static:** `layer.7` → RISKY, `1.23 GiB -> 1.17 GiB`, renumbering 8→7…; `token_embd` → BREAKS_MODEL,
  refused, exit 1 (allowed with `--force`); qwen3.5 single layer → BREAKS_MODEL (remove a whole
  period of 4 instead).
- **simulation:** `layer.7`: `KL 0.5473`, top-1 `" not"` → `" a" 0.157`; corpus `perplexity 7.554 ->
  10.915` (layer 7) and `-> 15.976` (layer 15: the last layer hurts more).
- **sweep:** about 4 minutes; least damaging layers 6 and 12; the plan slims to 14 layers and still
  answers in Ollama.

## 9. Token inspector and live edits (trace build)

```sh
nd tokens llama3.2:1b "fn main() {" --chat
nd journey qwen2.5:0.5b --prompt "fn main() {"
nd generate llama3.2:1b --prompt "$P" --steps 8
echo '{"schema_version":1,"edits":[{"id":1,"enabled":true,"op":"zero","target":"layer.7"}]}' > /tmp/zero7.json
nd edit llama3.2:1b --apply /tmp/zero7.json --prompt "$P" --tokens 12
nd edit llama3.2:1b --apply /tmp/zero7.json --bake /tmp/llama-zero7.gguf --no-compare
nd diff llama3.2:1b /tmp/llama-zero7.gguf
```

**Expected:** `--chat` shows the templated prompt with template tokens marked; the journey's lens
converges in the last layers and the decision breakdown names heads and neurons; `generate` gives
`" not a fixed or constant name. It"`. Zeroing layer 7 gives the same numbers as `ablate --cut
layer.7`; the bake validates (147 tensors, same size) and the diff shows only layer 7's
`attn_output` and `ffn_down` changed.

REPL / web: `edit zero layer.7` → `compare` changes → `undo` → `compare` identical (KL 0);
`toggle`, `redo`, `restore all`, snapshots, `edits export` / import; `edit ban " not"` removes it from the
edited top-k; **Bake** lists run-time-only edits (ban, steer, patch, force) as skipped. Edited nodes are
outlined orange in the graph; the header shows `Edits (n)`.

## 10. Diff

```sh
nd diff qwen2.5:0.5b huihui_ai/qwen2.5-abliterate:0.5b --top 10
nd --json diff qwen2.5:0.5b qwen2.5-coder:0.5b --top 5 | python -c "import json,sys;r=json.load(sys.stdin);print(r['compared'],r['identical'],round(r['total']['rel_l2'],3))"
nd diff gemma4:e2b gemma4:e4b --metadata-only | head -8
```

**Expected:** abliterated: 290 compared, **227 identical**, relative L2 0.018, changes only in
`ffn_down` / `attn_output` (plus `attn_v` where the quant type differs); coder: `290 0 0.559`;
gemma4: weights not compared, layers only in B, shape mismatches and metadata listed. Web: **Diff**
page → **Color the graph of A by this diff** → red feed-forward blocks, side panel numbers.
See [What fine-tunes change](docs/diff-research.md) for all values.

## 11. Cross-check against llama.cpp / Ollama

```sh
nd generate gemma2:2b --prompt "$P" --steps 8 --json
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"gemma2:2b","prompt":"'"$P"'","raw":true,"stream":false,"logprobs":true,"top_logprobs":5,"options":{"temperature":0,"num_predict":8}}'
```

**Expected:** same text and first-token probabilities within ~0.01 for `llama3.2:1b` and `qwen2.5:0.5b`;
gemma2 same text but sharper probabilities (no soft-capping); `phi3` 128k drifts (LongRoPE). Details in
[Gemma 4, Gemma, Gemma 2 and Phi-3](docs/new-models.md).

## 12. Browser demo

```sh
sh scripts/build-wasm-demo.sh && python -m http.server -d docs/demo 8000
```

Open `http://localhost:8000`, drop a `.gguf` (e.g. a copy of `$(blob qwen2.5:0.5b)` named `*.gguf`).
**Expected:** the banner says inspection only and nothing is uploaded; overview, vocab, tensors, layers,
template and the 2D / 3D graph work; stats of one tensor read only that tensor; server-only pages are
hidden.

## 13. Exports and compare

```sh
nd export llama3.2:1b --format csv --table vocab --out /tmp/vocab.csv && wc -l /tmp/vocab.csv   # 128257
nd compare qwen3:8b qwen3-embedding:8b
nd graph llama3.2:1b --level block --format dot --out /tmp/llama.dot && dot -Tsvg /tmp/llama.dot -o /tmp/llama.svg
```

**Expected:** every JSON output has `"schema_version": 1`; the CSV quotes fields with commas or quotes;
`compare` qwen3 vs qwen3-embedding vocab overlap ≈ 0.998; DOT renders with green / amber / red nodes.
