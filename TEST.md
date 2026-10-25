# NeuronDump manual test plan

Run from the repo root in **Git Bash** (PowerShell differences are noted where they matter).

```sh
cargo build --release --features trace      # everything (trace included)
alias nd=./target/release/neurondump
# path of a model's GGUF file, for llama.cpp tools:
blob() { nd --json list | python -c "import json,sys;print(next(m['path'] for m in json.load(sys.stdin)['models'] if m['name']==sys.argv[1]))" "$1"; }
```

The expected results below were recorded with these Ollama models:
`llama3.2:1b`, `nomic-embed-text`, `qwen2.5-coder:7b`, `qwen3:8b`, `qwen3-embedding:8b`,
`qwen3.8:27b` and the community model `oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU`. Install the
ones you want with `ollama pull <name>`. Exact numbers can differ slightly if Ollama updates a
model.

Notes on model names:

- The community model is **`oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU`** (with a leading `o`).
  `amazonasgabriel/...` does not exist and makes a good "wrong name" test.
- For a smaller qwen2 model, `ollama pull qwen2.5:0.5b` and use it wherever `qwen2.5-coder:7b` appears.
- Raw `.gguf` paths: any slim output (e.g. `~/neurondump/slim/llama-vocab.gguf`, created in
  section 7), or a Hugging Face download, e.g.
  `mkdir -p ~/models && curl -L -o ~/models/smollm2-135m.gguf https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q8_0.gguf`
  (check the repo/file name on huggingface.co if it 404s).
- Model ids (used by the web API) are derived from the name, so they are the same on every machine: `llama3.2:1b` = `17d2889ccc34`,
  `nomic-embed-text:latest` = `84bf674d64ec`, `oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU` = `4d459e28845e`.
  `nd list` shows all of them.

---

## 1. Discovery + parser + `list` / `inspect` (Phases 1–2)

**What to test:** every local model is found and its header parses in under a second.

```sh
nd --timings list
nd --json list | python -m json.tool | head -30
nd --timings inspect llama3.2:1b
nd inspect nomic-embed-text
nd inspect oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU
nd inspect qwen2.5-coder:7b
nd inspect ~/neurondump/slim/llama-vocab.gguf      # raw file (after section 7)
nd --scan ~/models list                            # extra scan directory
nd inspect --all
nd inspect --all --out /tmp/nd-reports && ls /tmp/nd-reports
```

**Expected:**
- `list` shows 7 models: `llama3.2:1b`, `nomic-embed-text:latest`,
  `oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU`, `qwen2.5-coder:7b`, `qwen3-embedding:8b`,
  `qwen3.8:27b` (arch `qwen35 +vision`), `qwen3:8b`. Source `ollama`; quantization Q8_0 /
  F16 / Q4_K_M; sizes 1.23 GiB, 261.58 MiB, 988.04 MiB, 4.36 GiB, 4.36 GiB, 15.66 GiB, 4.87 GiB.
  Discovery takes about 10 ms; `list` stays under 1 s.
- `inspect llama3.2:1b`: GGUF v3 little-endian, 147 tensors, 30 keys, `llama [decoder]`,
  1.24B params, **16 layers** (`full_attention+dense 0-15`), head_count 32 / kv 8, gqa_group 4,
  embedding 2048, FFN 8192, context 131072, **vocab 128256** (gpt2 / llama-bpe, 280147 merges),
  "embeddings are tied", 248 reserved tokens, specials `bos=128000 <|begin_of_text|>`,
  `eos=128009 <|eot_id|>`, two templates (jinja: 23 decisions, ollama go_template: 15).
  The `--timings` total is about 0.5 s.
- `nomic-embed-text`: `nomic-bert [encoder]`, 12 layers, bert tokenizer with 30522 tokens,
  994 reserved, Params `{"num_ctx":8192}`, no template.
- `qwen3.5-0.8b`: `qwen35 [hybrid]`, 24 layers: `full_attention+dense` at 3,7,11,15,19,23 and
  `linear_attention+dense` for the other 18. Components include `linear_attention`, `mtp` and `vision`.
- `inspect --all`: 7 `ok` lines, `7 model(s), 0 failed`, exit status 0.

**Edge cases to try:**
```sh
nd inspect llama3.2:1bb ; echo "exit=$?"                          # did you mean: llama3.2:1b? exit=1
nd inspect amazonasgabriel/qwen3.5-0.8b:q8-8gbGPU ; echo "exit=$?" # suggests oamazonasgabriel/...
nd inspect ./does-not-exist.gguf ; echo "exit=$?"
head -c 5000000 ~/neurondump/slim/nomic-mini.gguf > /tmp/trunc.gguf && nd inspect /tmp/trunc.gguf   # truncated: warnings, still exit 0
head -c 100 ~/neurondump/slim/nomic-mini.gguf > /tmp/trunc2.gguf && nd inspect /tmp/trunc2.gguf     # "metadata count … exceeds file size", exit 1, no panic
printf 'hello world' > /tmp/not.gguf && nd inspect /tmp/not.gguf                                     # "not a GGUF file"
# unknown architecture (same-length rename of every "nomic-bert" string):
python -c "d=open(r'$(cygpath -w ~/neurondump/slim/nomic-mini.gguf)','rb').read();open(r'$(cygpath -w /tmp)\\mystery.gguf','wb').write(d.replace(b'nomic-bert',b'zzzzz-bert'))"
nd inspect /tmp/mystery.gguf          # arch zzzzz-bert, still fully inspected, exit 0
OLLAMA_MODELS=/nonexistent nd list    # still lists ~/.ollama models (the env var only adds a dir)
```
(The `nomic-mini.gguf` / `llama-vocab.gguf` files are created in section 7.)

---

## 2. Architecture + tensor roles (Phase 3, items 2/3/6)

**What to test:** layers, roles and per-layer kinds come from tensor names, for every architecture.

```sh
nd layers llama3.2:1b
nd layers oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU
nd layers qwen3.8:27b
nd tensors llama3.2:1b --layer 3 --sort size
nd tensors qwen2.5-coder:7b --role attn_q --limit 5
nd tensors qwen3:8b --layer 0 | grep norm
nd --json inspect qwen3.8:27b | python -c "import json,sys;r=json.load(sys.stdin);print(r['projector']['architecture']['name'], r['projector']['file']['n_tensors'])"
```

**Expected:** llama has 16 identical layers with 9 tensors each. qwen3.5-0.8b alternates 3 linear
layers (`ssm_*` roles) and 1 full-attention layer. qwen3.8:27b has 65 layers, the last one
holding `nextn.*` roles. In `tensors --layer 3 --sort size`, `ffn_down`, `ffn_gate` and `ffn_up` come first
(17 MiB each). qwen2.5-coder layer 0 has `attn_q.bias`, `attn_k.bias` and `attn_v.bias`. qwen3 has
`attn_q_norm` and `attn_k_norm`. The qwen3.8 projector prints `clip 334`.

**Edge cases:** `nd tensors llama3.2:1b --sort bogus` → error listing the valid sorts.
`nd tensors llama3.2:1b --layer 99` → `0 tensor(s)`.

---

## 3. Tokenizer + chat templates (Phase 3, items 4/5)

**What to test:** readable vocabulary, classes, special tokens, tokenizer parity with
llama.cpp, and template decision trees.

```sh
nd vocab llama3.2:1b --search Rome --limit 5
nd vocab llama3.2:1b --class code --lang rust --limit 10
nd vocab llama3.2:1b --class reserved --limit 3
nd vocab llama3.2:1b --search '^ fn$' --regex
nd vocab qwen2.5-coder:7b --class cjk --limit 5
nd vocab nomic-embed-text --class reserved --limit 3
nd vocab llama3.2:1b --tokenize 'Hello world, fn main() { println!("hi"); }  123456'
llama-tokenize -m "$(blob llama3.2:1b)" -p 'Hello world, fn main() { println!("hi"); }  123456' --ids --log-disable
nd --json vocab qwen2.5-coder:7b --tokenize 'def foo(x):  return x**2  # 2024' | python -c "import json,sys;print(json.load(sys.stdin)['ids'])"
llama-tokenize -m "$(blob qwen2.5-coder:7b)" -p 'def foo(x):  return x**2  # 2024' --ids --log-disable
nd vocab nomic-embed-text --tokenize 'Hello, World! unbelievable'
llama-tokenize -m "$(blob nomic-embed-text:latest)" -p 'Hello, World! unbelievable' --ids --log-disable
nd template llama3.2:1b
nd template llama3.2:1b --raw | head -40
nd template oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU
nd --json template qwen2.5-coder:7b | python -c "import json,sys;[print(t['name'],t['syntax'],t['decisions'],t['variables']) for t in json.load(sys.stdin)['templates']]"
```

**Expected:**
- `--class code --lang rust` lists tokens such as `"in"`, `" for"`, `" return"`, with their languages.
- Reserved tokens are `<|reserved_special_token_N|>` (llama), `[PADnnnnn]` (qwen2.5) and `[unusedN]` (nomic).
- `--tokenize` ids are **identical** to `llama-tokenize`: llama
  `[128000, 9906, 1917, 11, 5279, 1925, 368, 314, 14069, 17667, 6151, 5146, 335, 220, 220, 4513, 10961]`;
  qwen `[750, 15229, 2075, 1648, 220, 470, 856, 334, 17, 220, 671, 220, 17, 15, 17, 19]`;
  nomic `[101, 7592, 1010, 2088, 999, 23653, 102]`.
- `template llama3.2:1b` prints a tree: `if custom_tools is defined`, `if messages[0]['role'] == 'system'`
  with an `else` branch, `for t in tools`, and emits `→ <|start_header_id|> <|end_header_id|>`
  and `<|eot_id|>`. It is followed by the Ollama Go template tree (`if .System`, `range $i, $_ := .Messages`, …).
- The qwen3.5 template emits `<think> </think> <tool_call> …`.

**Edge cases:** `nd vocab llama3.2:1b --class bogus` → error listing the classes.
`nd vocab llama3.2:1b --search '[' --regex` → regex error, no panic.
`nd vocab nomic-embed-text --search '##ing' --limit 3`.

---

## 4. Weight stats, job system, cache (Phases 3 item 7 and 6)

**What to test:** streaming statistics, per-head/neuron norms, cached re-runs.

```sh
time nd stats llama3.2:1b --limit 10
time nd stats llama3.2:1b --limit 3          # second run: "(cached result)", instant
nd stats nomic-embed-text --limit 5
nd stats llama3.2:1b --tensor blk.0.attn_q.weight --tensor blk.0.ffn_down.weight
nd --json stats llama3.2:1b | python -c "import json,sys;r=json.load(sys.stdin);print(len(r['tensors']),len(r['heads']),len(r['heads'][0]['q']),len(r['ffn_neurons'][0]['norms']))"
nd inspect llama3.2:1b --stats | tail -40
ls ~/neurondump/cache/
```

**Expected:** the first `stats llama3.2:1b` takes about 3 s in release (CPU and disk bound) and shows a
progress bar on stderr. Tensor `rope_freqs.weight` ranges 1.000…32.000. Q8_0 weights have mean ≈ 0
and std ≈ 0.01–0.05. The per-head table shows 32 heads per layer, 8 KV heads. The FFN table shows
8192 neurons per layer. JSON check prints `147 16 32 8192`. The second run prints
`(cached result)` and returns almost immediately. `~/neurondump/cache/<sha256-…>/` holds `stats-*.json`.

**Edge cases:**
- `nd stats qwen3.8:27b` reads 15 GiB. Press Ctrl-C mid-run: it stops at once with no partial cache file.
- `nd --threads 1 stats nomic-embed-text` is slower but gives the same numbers.
- `nd --quiet stats nomic-embed-text` shows no progress bar.

---

## 5. Interface picker + REPL (Phases 0 and 7a)

**What to test:** mode selection, remembered choice, non-TTY behaviour, REPL commands.

```sh
nd                                   # interactive picker: "How do you want to use NeuronDump?"
nd config show
nd config set default_ui cli && nd   # starts the REPL directly
nd config set default_ui none
nd --ui web                          # web even with a saved cli choice
NEURONDUMP_UI=cli nd
nd < /dev/null ; echo "exit=$?"      # non-TTY: help text + exit=2, no prompt
nd | cat ; echo "exit=${PIPESTATUS[0]}"
nd config path                       # %APPDATA%\neurondump\config.toml on Windows
nd config set port 9000 && nd config show && nd config reset
nd config set bogus 1                # error: unknown config key
```

REPL (`nd cli`): pick `llama3.2:1b` in the fuzzy picker (type "llama"), then:
```
help
info
layers
layer 3
vocab Rome --limit 5
vocab --class special
tensors --sort size --limit 5
template
stats &
jobs
job <id from jobs>
use nomic-embed-text
compare llama3.2:1b
export --format csv --table layers
trace The name of the star that Earth orbits is
web
exit
```
Also try Tab completion: `us<Tab>`, `use ll<Tab>`, `vocab --class sp<Tab>`, `stats --tensor blk.0.<Tab>`.
↑ recalls history after a restart (history is stored next to the config file).

**Expected:** the prompt shows `neurondump[llama3.2:1b]>`. `stats &` prints `started job stats-…`;
`jobs` shows its progress; `job <id>` renders the stats tables. Long outputs page with
`-- more (50/…) Enter: next, q: quit --`. `web` prints a URL with `?token=` and opens the browser
while the REPL keeps working. A piped REPL (`printf 'info\nexit\n' | nd cli`) runs without prompting.

Graph and ablation commands (Phase 10–12) in the same REPL:
```
use llama3.2:1b
graph --level section
node L7
cut layer.7 layer.7.attn.head.3
selection
impact
simulate The name of the star that Earth orbits is --tokens 4
simulate &
plan save C:/Users/<you>/llama-cut.plan.json
uncut all
sweep --level head --layer 7 --corpus /tmp/corpus5.txt &
```
Tab completes node ids after `node`, `cut`, `uncut` and `--expand` (type `layer.7.` to get
its heads and neuron buckets). **Expected:** `cut` prints a one-line summary
(`60.8M params, 1.23 GiB -> 1.17 GiB, verdict risky`). `impact` prints the full static impact, with
`ignored layer.7.attn.head.3: an ancestor is already cut`. `simulate &` starts an `ablate-…` job.
`job <id>` renders the impact and the simulation. `cut nope` → `unknown node "nope"`. `use` clears the selection.
The REPL is a native program: in piped input, give `plan save` a Windows path (a `/tmp/...` path fails with
`writing /tmp/...: The system cannot find the path specified`).

**Edge cases:** an unknown command → `unknown command`; `use nope` → did-you-mean error;
`cancel bogus` → `no running job`; Ctrl-C at the prompt does not exit; Ctrl-D exits.

---

## 6. Web backend + frontend + SSE (Phase 7b)

**What to test:** the API, the security rules and the UI views.

Start it: `nd web --port 7878 --open`. It prints:
```
NeuronDump web UI: http://127.0.0.1:7878/?token=<32 hex>
API token header: x-neurondump-token: <32 hex>
```
In another shell: `T=<that token>; B=http://127.0.0.1:7878; H="x-neurondump-token: $T"`

```sh
curl -s -o /dev/null -w "%{http_code}\n" $B/api/models                       # 401 (no token)
curl -s -o /dev/null -w "%{http_code}\n" -H "x-neurondump-token: wrong" $B/api/models   # 401
curl -s -H "$H" $B/api/info
curl -s -H "$H" $B/api/models | python -m json.tool | head -20
curl -s -X POST -H "$H" $B/api/models/rescan
curl -s -H "$H" $B/api/models/17d2889ccc34 | head -c 400; echo
curl -s -w "\n%{time_total}s\n" -H "$H" "$B/api/models/17d2889ccc34/vocab?q=rome&limit=5"
curl -s -H "$H" "$B/api/models/17d2889ccc34/vocab?q=%5E%20fn%24&regex=true&class=code"
curl -s -H "$H" "$B/api/models/17d2889ccc34/vocab?class=reserved&offset=100&limit=2"
curl -s -H "$H" "$B/api/models/17d2889ccc34/tensors?layer=3&sort=size&limit=3"
curl -s -H "$H" "$B/api/models/17d2889ccc34/layers" | head -c 300; echo
curl -s -H "$H" "$B/api/models/17d2889ccc34/template" | head -c 300; echo
curl -s -H "$H" "$B/api/models/17d2889ccc34/metadata?full=false" | head -c 300; echo
curl -s -H "$H" "$B/api/models/17d2889ccc34/tensors/blk.0.attn_q.weight/stats" | head -c 300; echo
J=$(curl -s -H "$H" -H "content-type: application/json" -d '{"kind":"stats","model":"84bf674d64ec","params":{}}' $B/api/jobs | python -c "import json,sys;print(json.load(sys.stdin)['id'])"); echo $J
curl -s -N "$B/api/jobs/$J/events?token=$T"                                    # SSE progress … event: done
curl -s -H "$H" $B/api/jobs
curl -s -H "$H" $B/api/jobs/$J
curl -s -H "$H" $B/api/jobs/$J/result | head -c 300; echo
J2=$(curl -s -H "$H" -H "content-type: application/json" -d '{"kind":"stats","model":"15ee989e4060","params":{}}' $B/api/jobs | python -c "import json,sys;print(json.load(sys.stdin)['id'])")
sleep 2; curl -s -X DELETE -H "$H" $B/api/jobs/$J2; sleep 1; curl -s -H "$H" $B/api/jobs/$J2   # status "cancelled"
curl -s -H "$H" "$B/api/compare?models=17d2889ccc34,84bf674d64ec" | head -c 400; echo
curl -s -D - -o /tmp/tensors.csv "$B/api/export/17d2889ccc34?format=csv&table=tensors&token=$T" | grep -i content-
curl -s -o /tmp/report.json "$B/api/export/17d2889ccc34?format=json&token=$T" && ls -la /tmp/report.json
```

**Expected status codes:** 401 without/with a wrong token; 200 for the GETs; `POST /api/jobs` → **202**
with `{"id": …}`; SSE streams `event: progress` lines then one `event: done`; result is 409
while the job is running and 200 once done; DELETE → `{"cancelling":true}`, and the job then
reports `"status":"cancelled"`. Vocab queries take under 50 ms after the first (cached) call. The export
has `content-disposition: attachment; filename="llama3.2_1b.tensors.csv"`. The CSV matches
`nd export llama3.2:1b --format csv --table tensors` byte for byte (`diff` them).

**Security edge cases:**
```sh
curl -s -w " %{http_code}\n" -H "$H" $B/api/models/..%2F..%2Fetc                     # 404 unknown model id
curl -s -w " %{http_code}\n" -H "$H" -H "content-type: application/json" -d '{"kind":"inspect","model":"C:/Windows/win.ini","params":{}}' $B/api/jobs   # 404: paths are not ids
curl -s -o /dev/null -w "%{http_code}\n" "$B/assets/..%2F..%2FCargo.toml"             # 400
curl -s -w " %{http_code}\n" -H "$H" -H "content-type: application/json" -d '{"kind":"bogus","model":"17d2889ccc34"}' $B/api/jobs   # 400
nd web --host 0.0.0.0 --port 7879     # prints a WARNING about exposing the machine
```
A slim job with `"out":"../evil.gguf"` → 400 `output must be a plain file name` (see section 7).

**Search API:**
```sh
curl -s -H "$H" "$B/api/search?q=qwen" | python -m json.tool | head -20                 # 5 models
curl -s -H "$H" "$B/api/search?q=attn_q&model=17d2889ccc34" | python -c "import json,sys;print(json.load(sys.stdin)['totals'])"   # [0, 0, 16, 0]  (models, tokens, tensors, metadata)
nd search rome --model llama3.2:1b
```

**UI (open the printed URL):** check each view:
- Models: the list of 7, with a Rescan button.
- Model → overview: hyperparameters, components, types, layer kinds, tokenizer, Ollama layers, warnings, export links.
- vocab: search, regex checkbox, class/lang filters, prev/next paging over 128,256 tokens with no lag.
- tensors: filters and paging. Clicking a name opens its stats with a histogram.
- layers: the colored layer map. qwen3.5 shows orange linear and blue full-attention cells in a 3:1 pattern. Clicking a cell filters tensors.
- template: collapsible tree plus source.
- stats: per tensor, plus a "compute all" job with a live progress bar.
- analyze: corpus textarea → job → plan with checkboxes → Slim button → result.
- trace: prompt → job → logit-lens table.
- Compare: tick 2+ models.
- Jobs: live progress over SSE, cancel buttons, result links.
- Opening `http://127.0.0.1:7878/` without `?token=` shows the "No session token" error.
- Header search: press `/`, type `rome` on a model page, then Enter. You get grouped results with
  highlighted matches; a token result opens vocab with the query filled in. Typing on the results page updates them live.
- Models page: the filter box narrows the list; clicking `size` sorts by size (click again to reverse).
- Click an id → "Copied …" toast. The System / Light / Dark switch changes the theme (System is selected by default and follows the OS); the choice survives a reload.
- Start a stats job: a number badge appears next to Jobs and disappears when it finishes.

---

## 7. `analyze` + `slim` with round-trip validation (Phase 4)

**What to test:** prune plans, slimmed GGUFs that validate, load in llama.cpp and import into Ollama.

```sh
cat > /tmp/corpus.txt <<'EOF'
The quick brown fox jumps over the lazy dog. Rust is a systems programming language.
fn main() { println!("Hello, world!"); let x: Vec<u32> = (0..10).collect(); }
Machine learning models are made of layers, attention heads and feed-forward networks.
EOF
nd analyze llama3.2:1b --corpus /tmp/corpus.txt --out /tmp/llama.prune.json
nd analyze llama3.2:1b --stats --fast --out /tmp/llama-stats.prune.json
nd analyze nomic-embed-text --corpus /tmp/corpus.txt --out /tmp/nomic.prune.json
nd analyze oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --out /tmp/qwen35.prune.json
nd analyze qwen3.8:27b --fast --out /tmp/q38.prune.json

nd slim llama3.2:1b --plan /tmp/llama.prune.json --only vocab --out llama-vocab.gguf
nd slim llama3.2:1b --plan /tmp/llama.prune.json --only layer_11 --out llama-15l.gguf
nd slim llama3.2:1b --plan /tmp/llama.prune.json --only reserved_tokens --out llama-noreserved.gguf --perplexity /tmp/corpus.txt
nd slim nomic-embed-text --plan /tmp/nomic.prune.json --only vocab,quant --out nomic-mini.gguf
nd slim llama3.2:1b --plan /tmp/llama-stats.prune.json --only low_norm_heads --out x.gguf
ls -la ~/neurondump/slim/
nd inspect ~/neurondump/slim/llama-vocab.gguf | head -12
```

Load in llama.cpp (this build uses `llama-completion` for one-shot prompts; `llama-cli` is chat-only):
```sh
llama-completion -m ~/neurondump/slim/llama-vocab.gguf -p "The quick brown fox" -n 12 --temp 0 -no-cnv
llama-completion -m "$(blob llama3.2:1b)" -p "The quick brown fox" -n 12 --temp 0 -no-cnv
llama-completion -m ~/neurondump/slim/llama-15l.gguf -p "The name of the star that Earth orbits is" -n 10 --temp 0 -no-cnv
llama-cli -m ~/neurondump/slim/llama-noreserved.gguf          # interactive chat, type a question, Ctrl-C to quit
llama-perplexity -m ~/neurondump/slim/llama-noreserved.gguf -f /tmp/corpus.txt -c 64
```

Load in Ollama:
```sh
cd ~/neurondump/slim
cat llama-vocab.Modelfile | head
ollama create llama-vocab -f llama-vocab.Modelfile
ollama run llama-vocab "The quick brown fox"
ollama create llama-noreserved -f llama-noreserved.Modelfile && ollama run llama-noreserved "Why is the sky blue?"
ollama create nomic-mini -f nomic-mini.Modelfile
curl -s http://localhost:11434/api/embed -d '{"model":"nomic-mini","input":"hello world"}' | head -c 200
nd list                                          # the new Ollama models now appear (and can be inspected)
ollama rm llama-vocab llama-noreserved nomic-mini   # clean up
```

**Expected:**
- The llama plan lists `vocab_trim` (high, about 268 MiB), `reserved_tokens` (medium, 248 tokens, about 527 KiB),
  `layer_N` ×3 (high, about 62 MiB each, never layer 0 or 15), and the note "embeddings are already tied".
  With `--stats` it adds `low_norm_heads` (experimental, APPLY = no).
- The qwen3.5 plan shows `quant_headroom` (27 F16 tensors, ≈63 MiB, low) and 3 `layer_N` candidates
  marked APPLY = `no` ("layer kinds follow an index pattern (hybrid model)…").
  qwen3.8 shows `reserved_tokens` (243) and `unused_projector` (≈888 MiB, low).
- `llama-vocab.gguf`: `Size 1.23 GiB -> 987 MiB (21.6% smaller)`, `Validate OK (147 tensors, vocab 570, block_count 16)`.
- `llama-15l.gguf`: `Validate OK (138 tensors, vocab 128256, block_count 15)`.
- `llama-noreserved.gguf` prints `Perplexity original 9.7153 -> slim 9.7153` plus `context 36 (73 corpus tokens)`.
  The context is sized to the corpus. Use a longer text for a meaningful number, e.g. `--perplexity README.md` (≈50.9 both).
- `low_norm_heads` → `Refused low_norm_heads: …non-uniform shapes…`, `Applied -`.
- **Compare:** the original and `llama-vocab` both continue "jumps over the lazy dog." and then diverge
  (the slim model can only emit 570 tokens). The 15-layer model loads and generates, with visibly
  worse quality (`-p "The name of the star that Earth orbits is"` continued " not a country that exists in the world of art"). `llama-noreserved`
  must be indistinguishable from the original. The Ollama runs reply normally, with the original
  template, stop tokens and parameters. `nomic-mini` returns a 768-dim embedding.

**Edge cases:**
```sh
nd slim llama3.2:1b --plan /tmp/nomic.prune.json --out a.gguf          # "plan was made for nomic-embed-text…"
nd slim llama3.2:1b --plan /tmp/llama.prune.json --only vocab --out ../evil.gguf   # plain-file-name error
nd slim llama3.2:1b --plan /does/not/exist.json --out a.gguf
echo '{}' > /tmp/bad.json && nd slim llama3.2:1b --plan /tmp/bad.json --out a.gguf  # invalid plan JSON
```
Plans made from ablations (section 11–12) add three candidate kinds that `slim` applies:
`zero_weights` (weights set to 0 in place, `Zeroed N tensor(s)`), `drop_tensors` and `strip_template`
(no `TEMPLATE` in the Modelfile). Removing a next-n layer also decrements `nextn_predict_layers`.

Hand-edit `/tmp/llama.prune.json` (set `"selected": true` on `reserved_tokens`) and run
`slim` without `--only`: only that candidate is applied.

---

## 8. Runtime tracing (Phase 5, feature `trace`)

**What to test:** per-layer internals, the logit lens, corpus aggregation, embedding contributions.

```sh
nd trace llama3.2:1b --prompt "The name of the star that Earth orbits is"
nd trace qwen2.5-coder:7b --prompt "def fibonacci(n):"
nd trace qwen3:8b --prompt "The opposite of hot is"
nd trace nomic-embed-text --prompt "search_query: what is rust?"
printf 'The cat sat on the mat.\n\nRust is fast.\n\nParis is in France.\n' > /tmp/tc.txt
nd trace llama3.2:1b --corpus /tmp/tc.txt
nd analyze llama3.2:1b --out /tmp/llama-trace.prune.json        # layer candidates now say "trace: cosine(...)"
nd --json trace llama3.2:1b --prompt "Hi" --top-k 3 | python -c "import json,sys;r=json.load(sys.stdin);print(len(r['layers']), r['prediction'][0])"
```

**Expected:** for llama3.2:1b, 10 tokens starting with `<|begin_of_text|>` (`The`, ` name`, ` of`, ` the`,
` star`, ` that`, ` Earth`, ` orbits`, ` is`). The logit lens is noise for layers 0–10. The topic appears
at layer 11 (`" Earth" 0.08  " planets" 0.05 … " astronomers" 0.04`), layer 14 is sure of the phrasing
(`" called" 0.39`), and the last layer changes its mind: `next token: " not" 0.166  " the" 0.070  " a" 0.067
" called" 0.051`. The 1B model never puts " Sun" in its top 5 here. cos(in,out) rises from ≈0.33 (layer 0)
to ≈0.94 (layers 12–13) and drops to 0.51 in the last layer. Layer 1 has one huge neuron (index 1417,
≈116). Runtime is about 3.5 s.
For qwen2.5-coder, the top next token is `" \n"`; compare with
`llama-completion -m "$(blob qwen2.5-coder:7b)" -p "def fibonacci(n):" -n 1 --temp 0 -no-cnv`.
nomic prints `token contributions to pooled embedding` per token (≈0.9 each) and no logit lens.
Corpus mode reports `3 sample(s)`. After it, `analyze` uses the cached corpus trace.
The JSON check prints `16 {…}`.

**Edge cases:**
```sh
nd trace oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --prompt hi   # error: trace unsupported for architecture "qwen35" (supported: …)
nd trace llama3.2:1b                                            # error: trace needs a prompt or a corpus
cargo build --release && ./target/release/neurondump trace llama3.2:1b --prompt hi   # without the feature: "rebuild with --features trace"
nd trace llama3.2:1b --prompt "$(python -c 'print("word "*500)')" --max-tokens 32   # truncation warning
```
In the web UI trace view, start a trace on `qwen3:8b` and cancel it from the Jobs page.

---

## 9. Output formats, schemas, compare (Phases 8 and 9)

```sh
nd export llama3.2:1b --format json --out /tmp/llama.json
nd export llama3.2:1b --format ndjson --table tensors | head -2
nd export llama3.2:1b --format csv --table vocab --out /tmp/vocab.csv && wc -l /tmp/vocab.csv
nd export nomic-embed-text --format csv --table metadata
nd export llama3.2:1b --format csv --table layers
nd compare llama3.2:1b qwen2.5-coder:7b qwen3:8b
nd --json compare qwen3:8b qwen3-embedding:8b | python -c "import json,sys;r=json.load(sys.stdin);print(r['vocab_overlap'])"
ls schemas/
# optional: pip install jsonschema
python -c "import json,jsonschema;jsonschema.validate(json.load(open('$(cygpath -w /tmp/llama.json)')),json.load(open('schemas/inspect.schema.json')))" && echo valid
```

**Expected:** every JSON output has `"schema_version": 1`. The vocab CSV has 128,257 lines (header + tokens)
and quotes fields with commas or quotes. In `compare`, `qwen3:8b` vs `qwen3-embedding:8b` overlap ≈ 0.998.
llama vs qwen ≈ low (different BPE). Metadata differences are listed.
`schemas/` holds 8 `*.schema.json` files, and the inspect export validates.

**Edge cases:** `nd export llama3.2:1b --format xml` → error. `nd compare llama3.2:1b` → clap error (needs 2+ models).
`nd export llama3.2:1b --format csv --table vocab | head -2` → no panic (closed pipe handled).

---

## 10. Model graph (Phase 10)

**What to test:** the structural graph of any model (sections, layers, blocks, tensors, and
heads / neuron buckets / experts on demand), its connections, grouping, and the four output formats.

The phase-3 substitutions apply here: `qwen2.5:0.5b` was not installed, so `qwen2.5-coder:7b`
(the same `qwen2` architecture) stands in for it. Importance values depend on what is cached:
the numbers below came after sections 4 and 8 (weight stats and a corpus trace of `llama3.2:1b`)
and before section 12 (sweeps).

```sh
nd graph llama3.2:1b --level section
nd graph llama3.2:1b                                   # level layer (default)
nd graph llama3.2:1b --level block | head -20
nd graph llama3.2:1b --expand layer.7.attn             # lazy fine level: KV groups
nd graph llama3.2:1b --expand layer.7.attn.kv.1        # the 4 heads of KV group 1
nd graph llama3.2:1b --expand layer.7.ffn | grep neurons | head -3
nd graph llama3.2:1b --level fine | tail -1
nd graph llama3.2:1b --group-by importance
nd graph llama3.2:1b --level tensor --group-by quant | head
nd graph qwen2.5-coder:7b --level block | head -14
nd graph nomic-embed-text --level block | head -16
nd graph oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --level section
nd graph oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --group-by kind | head -10
nd graph oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --level block --expand layer.0 | grep -A5 "layer 0 "
nd graph qwen3.8:27b --level section
nd graph qwen3.8:27b | grep -E "layer 6[34] "

nd node llama3.2:1b layer.7.attn
nd node llama3.2:1b layer.7.attn.head.5
nd node llama3.2:1b L7              # shorthands: L7, 7, blk.7, layer7, token_embd, blk.7.attn_q
nd node llama3.2:1b token_embd
nd node llama3.2:1b template

for m in llama3.2:1b qwen2.5-coder:7b nomic-embed-text oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU; do
  f=$(echo $m | tr '/:' '__')
  for fmt in json dot mermaid; do nd graph $m --level block --format $fmt --out /tmp/$f.$fmt; done
done
nd --json graph llama3.2:1b --level block | python -c "import json,sys;v=json.load(sys.stdin);print(len(v['nodes']),len(v['edges']))"
```

Render the DOT file with Graphviz (`winget install Graphviz.Graphviz`, `sudo apt install graphviz`):
```sh
dot -Tsvg /tmp/llama3.2_1b.dot -o /tmp/llama-graph.svg
dot -Tsvg /tmp/oamazonasgabriel_qwen3.5-0.8b_q8-8gbGPU.dot -o /tmp/qwen35-graph.svg
```
Open the SVGs in a browser. Paste a `.mermaid` file into https://mermaid.live to check it.

**Expected:**
- **llama3.2:1b section level:** 5 nodes: Input (266 MiB), Body (986 MiB), Output, Tokenizer and
  Chat template. Everything is `breaks_model` except the template (`risky`).
- **Layer level:** 26 nodes and 22 connections. `rope_freqs` and `token_embd` sit under Input.
  Output holds `output_norm`, `output head (tied to token_embd)` (virtual) and `next-token logits`.
- **Block level:** every layer holds `attention` and `feed-forward` blocks (▸ = expandable) and
  two `residual add` ops. The JSON view has 90 nodes and 187 edges.
- **Expanding `layer.7.attn`:** shows its 5 tensors and 8 `KV group N (heads 4N–4N+3)` nodes
  (32 query / 8 KV heads). Expanding a KV group shows its 4 heads. The FFN shows 128 buckets of
  64 neurons (`neurons 0-63` …). Level `fine` has 2926 nodes.
- **qwen2.5-coder:7b:** 28 layers, biases included. Block level: 149 nodes and 317 edges.
- **nomic-embed-text:** encoder. Output is `pooled embedding` (no logits). Input holds `token_embd`,
  `token_types` and `token_embd_norm`. Block level: 69 nodes and 165 edges.
- **qwen3.5 (hybrid), section level:**
  - `Multi-token prediction` is `safe`, and `Vision encoder (in file)` (192 MiB) is `risky`.
  - **Every layer is `breaks_model`** ("hybrid model: layer kinds follow a fixed pattern (every 4th layer is full attention)").
  - Linear layers hold a `linear attention (SSM)` block. Every 4th layer (3, 7, … 23) holds `attention`.
  - `--group-by kind` shows the groups `full_attention+dense` (6 layers) and `linear_attention+dense` (18).
  - Block level: 131 nodes. The 334 vision tensors only appear at `--level tensor` or with `--expand vision`.
- **qwen3.8:27b:**
  - Layers 0–63 are `breaks_model`; layer 64 is the next-n layer (`risky`).
  - Has a `Vision projector (separate file)` section (888 MiB, `safe`).
- **`nd node`:** prints path, size, quant, layer kind, removable + reason, importance + source,
  weight stats (if cached), activation (if traced), info, children, and `in <- / out ->` connections.
  - `layer.7.attn.head.5` shows `columns_of_attn_output: [320, 384]`, an importance from
    `stats: |Q|·|O| norm…` and an attention entropy from the trace.
  - Unknown ids → `unknown node "…"`.
- **DOT:** renders to SVG with clusters per section/layer. Colors are green (safe), amber (risky) and red (breaks_model).
  The tied edge is dashed red, `uses` edges are dotted, and residual edges are grey.
- **Mermaid:** `flowchart TD` with nested `subgraph`s and `classDef safe/risky/breaks_model`.

**Edge cases:** `nd graph llama3.2:1b --level nope` and `--group-by nope` → usage errors.
`--expand nope` → `unknown node`. `--format xml` → error.

---

## 11. Static impact: "what if I cut this?" (Phase 11)

**What to test:** instant, architecture-independent impact of removing or zeroing any node(s),
with a clear verdict. `breaks_model` cuts are refused unless `--force`.

```sh
nd ablate llama3.2:1b --cut layer.7 --static                    # one middle layer
nd ablate llama3.2:1b --cut token_embd --static ; echo "exit=$?"  # refused
nd ablate llama3.2:1b --cut token_embd --static --force | tail -8
nd ablate llama3.2:1b --cut layer.7.attn --static               # all attention of one layer
nd ablate llama3.2:1b --cut layer.7.attn.head.3,layer.7.ffn.n.0-63,layer.9.attn --static --save-plan /tmp/zero.plan.json
nd ablate llama3.2:1b --cut layer.3,layer.4 --static
nd ablate llama3.2:1b --cut t.output_norm.weight --static
nd ablate llama3.2:1b --cut vocab:128000-128010 --static
nd ablate llama3.2:1b --cut layer.7,layer.7.attn.head.3,nope --static
nd ablate qwen2.5-coder:7b --cut layer.10.attn.head.2 --static
nd ablate oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --cut layer.5 --static
nd ablate oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --cut layer.4,layer.5,layer.6,layer.7 --static
nd ablate qwen3.8:27b --cut projector,template --static
nd --json ablate llama3.2:1b --cut layer.7 --static | python -c "import json,sys;i=json.load(sys.stdin)['impact'];print(i['verdict'],i['new_size'])"
```

Apply a zero-weights plan and check it really zeroes:
```sh
nd slim llama3.2:1b --plan /tmp/zero.plan.json --out llama-zero.gguf
nd --json stats ~/neurondump/slim/llama-zero.gguf --tensor blk.9.attn_output.weight --tensor blk.7.attn_output.weight --tensor blk.7.ffn_up.weight \
  | python -c "import json,sys
for t in json.load(sys.stdin)['tensors']:
    rn,cn=t.get('row_norms'),t.get('col_norms')
    print(t['name'],'l2=%.2f'%t['l2'],'rows0-63=0:',rn and all(x==0 for x in rn[:64]),'cols192-255=0:',cn and all(x==0 for x in cn[192:256]))"
llama-completion -m ~/neurondump/slim/llama-zero.gguf -p "The name of the star that Earth orbits is" -n 10 --temp 0 -no-cnv
```

**Expected:**
- **`layer.7`:** `remove`, 60.8M params (61.64 MiB), 4.92% of the model, `1.23 GiB -> 1.17 GiB`, verdict **RISKY**.
  `llama.cpp loads the slimmed file`, `Renumber 8->7 … 15->14`, `slim would remove layer(s) {7}; block_count 16 -> 15`.
- **`token_embd`:** verdict `BREAKS_MODEL — refused without --force`, error `refused: …` and exit 1. It shows:
  - `no path from the input embeddings to the output remains`;
  - `67 node(s) are cut off` (listed as `layer.0.add_ffn, …` in layer order);
  - `Depends tied embeddings: token_embd is also the output head`.

  With `--force` it exits 0, with the same verdict.
- **`layer.7.attn`:** mode `zero`, 10.5M params zeroed (0.85%), size unchanged, **RISKY**,
  `slim would zero weights in place in 1 tensor(s)` (the attention output projection).
- **Head + neuron bucket + block:** 655K params zeroed, 3–4 tensors.
  - **After slim:** `Applied zero_weights`, `Size 1.23 GiB (unchanged)`, `Zeroed 4 tensor(s)` and `Validate OK (147 tensors …)`.
  - **Stats check:** `blk.9.attn_output.weight l2=0.00`. `blk.7.attn_output` has columns 192–255 zero (head 3). `blk.7.ffn_up` has rows 0–63 zero.
  - **llama.cpp:** the original continues "The name of the star that Earth orbits is" with " not a fixed or constant name. It is a"
    (the same text as NeuronDump's in-memory baseline); the zeroed file answers " called the Sun. The Sun is a massive ball".
- **`layer.3,layer.4`:** RISKY. The two removed layers are bypassed together, so the path from input to output stays.
- **`output_norm`:** BREAKS_MODEL (no path to the output).
- **`vocab:128000-128010`:** BREAKS_MODEL. It shows:
  - `the chat template uses 3 of the cut tokens ([128006, 128007, 128009])`;
  - `bos/eos token(s) [128000, 128009] are cut`.
- **Mixed ids:** `ignored layer.7.attn.head.3: an ancestor is already cut`, `unknown id: nope`.
- **qwen2.5-coder head:** `File size 4.36 GiB -> 4.37 GiB (requantization +6.12 MiB)` and
  `slim would requantize 1 tensor(s) to Q8_0 to zero unaligned columns`. A 128-wide head does not
  line up with Q4_K's 256-value blocks.
- **qwen3.5 `layer.5`:** BREAKS_MODEL, `hybrid pattern (period 4) broken: remove whole aligned groups of 4 layers instead`,
  `llama.cpp would NOT load the slimmed file`, `simulation unsupported for architecture "qwen35"`.
- **qwen3.5 `layer.4-7`** (one whole period): **RISKY**, `removed layers form whole periods of 4`,
  `llama.cpp loads the slimmed file`, `988.04 MiB -> 903.69 MiB`.
- **qwen3.8 `projector,template`:** RISKY, `Other files 888.01 MiB no longer needed (projector)`,
  `slim would drop the vision projector from the Modelfile` / `strip tokenizer.chat_template and the Modelfile TEMPLATE`.

---

## 12. Simulated ablation + sweep (Phase 12, feature `trace`)

**What to test:** the cut applied in memory (nothing written):
- baseline vs ablated top-k, KL, greedy text, logit lens per layer;
- perplexity on a corpus, plus speed and memory;
- layer and head sweeps ranked by damage, then plan → slim → Ollama.

```sh
cat > /tmp/corpus5.txt <<'EOF'
The quick brown fox jumps over the lazy dog. It was a sunny day and the children were playing in the park.

Paris is the capital of France. Berlin is the capital of Germany, and Madrid is the capital of Spain.

Rust is a systems programming language focused on safety, speed and concurrency.

Water boils at one hundred degrees Celsius at sea level. Ice melts at zero degrees.

Once upon a time there was a little girl who lived in a village near the forest.
EOF
nd ablate llama3.2:1b --cut layer.7  --prompt "The name of the star that Earth orbits is" --tokens 12
nd ablate llama3.2:1b --cut layer.15 --prompt "The name of the star that Earth orbits is" --tokens 12
for c in layer.7 layer.15 layer.0; do nd ablate llama3.2:1b --cut $c --corpus /tmp/corpus5.txt --tokens 0 --prompt Hi | grep -E "^corpus|^KL"; done
nd ablate llama3.2:1b --cut layer.7.attn.head.3 --prompt "The name of the star that Earth orbits is" --tokens 4 | grep -E "KL|tokens/s"
nd ablate nomic-embed-text --cut layer.5 --prompt "search_query: what is rust?" --tokens 0 | grep -E "cosine|embedding"
nd ablate oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --cut layer.4,layer.5,layer.6,layer.7 | tail -3   # static only + note

nd sweep llama3.2:1b --corpus /tmp/corpus5.txt --out /tmp/llama.sweep.json --save-plan /tmp/llama-sweep.plan.json --top 2
nd sweep llama3.2:1b --level head --layer 7 --corpus /tmp/corpus5.txt --out /tmp/llama-heads7.sweep.json
nd graph llama3.2:1b --group-by importance | head -12        # importance now comes from the sweep
nd node llama3.2:1b layer.3 | grep importance

nd slim llama3.2:1b --plan /tmp/llama-sweep.plan.json --out llama-sweep14.gguf
cd ~/neurondump/slim && ollama create llama-sweep14 -f llama-sweep14.Modelfile
for m in llama3.2:1b llama-sweep14; do echo "== $m"; ollama run $m "What is the capital of Italy? Answer in one sentence."; done
ollama rm llama-sweep14
```

**Expected:**
- **`layer.7`, prompt "The name of the star that Earth orbits is":**
  - baseline `" not" 0.166  " the" 0.070  " a" 0.067`, ablated `" a" 0.157  " not" 0.118`;
    `KL(baseline || ablated) = 0.5473 nats; top-1 CHANGED`.
  - Greedy text changes from `" not a fixed or constant name. It is a name that"` to
    `" a bit of a mystery, but I'll give you some"`.
  - The lens table marks `7 (cut)`. Layers before 7 are identical; after it the lens drifts
    (layer 11 `" Earth" 0.08` → `" nicknamed" 0.03`), and `P(" not")` is 0.166 → 0.118 at the end.
  - `weights in use: 1.22 GiB -> 1.16 GiB`. Takes about 35 s on CPU.
- **`layer.15` on that one prompt:** KL 1.6020, top-1 CHANGED to `" called" 0.387`: without the last layer
  the output is layer 14's lens (`" called" 0.39`), so the last layer is what turns " called" into " not".
  Greedy text: `" called the star itself, but it is also called the star"`.
  The corpus is the fair comparison, and there too **the last layer hurts much more**:
  - **layer 7:** `perplexity 7.554 -> 10.915, mean KL 0.2623`;
  - **layer 15:** `7.554 -> 15.976, mean KL 0.6710`;
  - **layer 0:** `7.554 -> 23257.776, mean KL 7.94, top-1 agreement 0.0%`.
- **Head 3 of layer 7:** KL 0.0043, top-1 unchanged (a single head barely matters).
- **nomic:** `embedding |v|  22.815 → 22.722` and `embedding cosine(baseline, ablated) = 0.9872`. No top-k (no output head).
- **qwen3.5:** static impact only, ending with `(no simulation: see the notes above; showing static impact only)`,
  since there is no qwen35 forward pass.
- **Layer sweep:** 5 samples, 96 positions, baseline perplexity 7.554, about 4 minutes on CPU (85 forward passes).
  - **Most damaging:** layer 0 (KL 7.94) and layer 1 (4.35), then 15 and 14.
  - **Least damaging:** layer 6 (0.13) and layer 12 (0.26). The `DAMAGE` column is a bar chart.
  - `plan with the 2 least damaging layer(s) … 1.23 GiB → 1.11 GiB (risky)`.
- **Head sweep (32 heads × 5 texts):** about 8 minutes on CPU. Head 13 hurts most (KL 0.019,
  perplexity +5.7%), then head 2. Most heads change the perplexity by less than 1%. A one-sentence
  corpus (as in the web test below) takes about 1 minute. Afterwards `nd node llama3.2:1b layer.7.attn.head.13`
  shows importance `sweep: KL … when removed`, and the web graph does too.
- **Slim:** `Applied layer_6, layer_12`, `9.8% smaller`, `Validate OK (129 tensors, vocab 128256, block_count 14)`.
  `ollama run llama3.2:1b` answers "The capital of Italy is Rome." and `ollama run llama-sweep14` answers "Rome is the capital of Italy.".

**Edge cases:**
```sh
nd sweep llama3.2:1b --level head --corpus /tmp/corpus5.txt             # error: --level head needs --layer N
nd sweep llama3.2:1b --level expert --layer 0 --corpus /tmp/corpus5.txt # error: no MoE forward pass yet
nd sweep nomic-embed-text --corpus /tmp/corpus5.txt                     # error: no output head (embedding model)
nd sweep oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --corpus /tmp/corpus5.txt   # simulation unsupported for "qwen35"
cargo build --release && ./target/release/neurondump ablate llama3.2:1b --cut layer.7   # no trace feature: static + note
```

---

## 13. Web graph view + ablation endpoints (Phase 13)

**What to test:** the interactive graph, node panel, cut selection with live impact, simulation,
sweep overlay, and plan → slim from the browser. Everything is served from the embedded assets, so
it works offline:
wamr3d 0.9.4 (MIT, WebGPU with a WebGL 2 fallback) draws all four views. It is in
`crates/neurondump-web/frontend/vendor/wamr3d/` and is loaded only when the graph tab opens.

Open `nd web --open`, pick `llama3.2:1b`, then the **graph** tab. The toolbar starts with the
**2D / 3D** toggle; in 3D the **3D layout** menu offers `dataflow` (default since Phase 14, see
section 14), `tower`, `city` and `force`. The choice is remembered, and `#/model/<id>/graph?view=2d`
(or `?view=tower|city|force|flow`) forces one.

0. **3D tower** (3D layout → tower):
   - Input is at the bottom, then the 16 layers as stacked floors (Body), then Output, Tokenizer and
     Chat template on top. The status line says `26 nodes · 22 connections`.
   - Containers are drawn as outlines; clicking one selects it. Hovering shows a tooltip (label, kind,
     params, bytes, removable).
   - Drag to orbit, scroll or `−` / `+` / `100%` / `Fit` to zoom.
   - Click, double-click to expand, search, cut outlines and color-by all work the same as in 2D.
   - **3D city:** sections left to right, with bars whose height follows their size.
   - **3D force:** a force layout along the dataflow.
   - In a browser with neither WebGPU nor WebGL 2, the canvas shows an error box
     ("The graph needs WebGPU or WebGL 2…"). There is no fallback, since 2D uses the same renderer.
1. Switch to **2D** for steps 1–10: a flat diagram read top to bottom, seen straight on (left drag
   pans, wheel zooms, no rotation). It shows Input → layer 0 … layer 15 → Output, colored by
   removable flag. The dashed red arrow is the tied head. Expanded layers sit side by side as columns
   of their blocks. Steps 2–9 also work in the 3D views.
2. Double-click `layer 7` → its blocks appear. Double-click `attention` → 8 KV groups. Double-click
   `KV group 0` → heads 0–3. "Collapse all" goes back.
3. Click `head 3`: the side panel shows the path, size, removable flag, importance, weight stats,
   activation and info (`columns_of_attn_output`). Click **Cut this** to add it to the selection.
4. The selection panel shows the live static impact (`risky · 0 removed + 262.1K zeroed …`).
5. Type `blk.3.ffn_down` in the graph search and press Enter → layer 3's FFN expands, and the tensor
   is highlighted while everything else is dimmed.
6. Set tokens to 4 and press **Simulate** → a progress line with cancel. After about 25 s the result shows:
   - the baseline vs ablated table (top-k, generated text, tokens/s) and KL;
   - the logit lens per layer (expandable);
   - the corpus perplexity (the sample corpus is prefilled).
7. **Save this cut as plan → Slim** → the usual plan table. Pick a name and press Slim.
8. Cut `layer 6`, press **Sweep this level (layers)** → the graph recolors by damage (color by =
   `damage`), with a sorted table. "save the 2 least damaging as a plan" → Slim.
9. Change level (section / layer / block / tensor), group by (section / layer / block / quant / kind /
   importance) and color by (removable / section / kind / quant / importance / damage).
10. Repeat on `oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU`:
    - Simulate is disabled, and its tooltip gives the reason;
    - cutting a single layer shows `breaks_model` with the pattern explanation.

API (token from the `nd web` output):
```sh
T=<token>; H="x-neurondump-token: $T"; U=http://127.0.0.1:7878/api; M=17d2889ccc34   # llama3.2:1b
curl -s -H "$H" "$U/models/$M/graph?level=layer" | python -c "import json,sys;v=json.load(sys.stdin);print(v['level'],len(v['nodes']),len(v['edges']))"
curl -s -H "$H" "$U/models/$M/graph?level=layer&group_by=kind"
curl -s -H "$H" "$U/models/$M/graph?level=layer&expand=layer.7,layer.7.attn" | python -c "import json,sys;print(len(json.load(sys.stdin)['nodes']))"
curl -s -H "$H" "$U/models/$M/nodes/layer.7.attn.head.3"
curl -s -o /dev/null -w "%{http_code}\n" -H "$H" "$U/models/$M/nodes/nope"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"cut":["layer.7"]}' "$U/models/$M/impact"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"cut":["token_embd"]}' "$U/models/$M/impact" | python -c "import json,sys;i=json.load(sys.stdin);print(i['verdict'],i['refused'])"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"cut":["layer.6","layer.12"]}' "$U/models/$M/plan-from-ablation"
curl -s -H "$H" -X POST -H "content-type: application/json" \
  -d '{"kind":"ablate","model":"'$M'","params":{"cut":["layer.7"],"prompt":"The name of the star that Earth orbits is","tokens":4}}' "$U/jobs"
J=$(curl -s -H "$H" -X POST -H "content-type: application/json" \
  -d '{"kind":"sweep","model":"'$M'","params":{"level":"head","layer":7,"corpus":["Paris is the capital of France. Berlin is the capital of Germany."],"corpus_file":"/etc/passwd"}}' "$U/jobs" \
  | python -c "import json,sys;print(json.load(sys.stdin)['id'])")
curl -sN -H "$H" "$U/jobs/$J/events" | grep -E "^event" | sort | uniq -c
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"sweep_job":"'$J'","top":4}' "$U/models/$M/plan-from-ablation" | python -c "import json,sys;r=json.load(sys.stdin);print(r['plan']['notes'][-1][:120], r['impact']['verdict'])"
curl -s -o /dev/null -w "%{http_code}\n" "$U/models/$M/graph"            # no token
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" http://127.0.0.1:7878/assets/vendor/wamr3d/wamr3d_wasm_bg.wasm
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" http://127.0.0.1:7878/assets/vendor/wamr3d/compat.js
```

**Expected:**
- **graph:** `layer 26 43`, and 43 nodes once layers 7 and 7.attn are expanded.
- **nodes:** head 3 → `risky`, parent `layer.7.attn.kv.0`. `nope` → 404.
- **impact:** `layer.7` → `risky` with `new_size` 1256447808. `token_embd` → `breaks_model True`.
- **plan-from-ablation:** candidates `layer_6, layer_12`.
- **jobs:**
  - an ablate job → `{"id":"ablate-…"}`;
  - the sweep job streams about 35 `progress` events and one `done`;
  - `corpus_file` is stripped, so it uses the inline text;
  - the plan from it → `4 least damaging of 32 head(s) …` with verdict `risky`.
- **auth and assets:** no token → 401. `wamr3d_wasm_bg.wasm` → `200 application/wasm`, `compat.js` → `200 text/javascript`.

---

## 14. 3D model graph with wamr3d (Phase 14)

**What to test:** the 2D / 3D toggle, the 3D dataflow view (instancing, picking, search, fly-to, level
of detail), the `auto` backend and its WebGL 2 fallback, the token flow animation, the embedding map,
and that the vendored engine is served correctly.

**Status:** built only; the numbers below marked *record* have not been measured yet. Fill them in on
the first run.

Setup: `cargo build --release --features trace`, `ollama pull qwen2.5:0.5b` (not installed on the
original test machine), then `nd web --open`. Ids: `llama3.2:1b` = `17d2889ccc34`,
`qwen2.5:0.5b` = `7f1452142d64`, `nomic-embed-text:latest` = `84bf674d64ec`,
`oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU` = `4d459e28845e`. (The qwen3.5 community model's name starts
with **`o`**: `amazonasgabriel/…` does not exist.)

```sh
sh scripts/vendor-wamr3d.sh && git status --short crates/neurondump-web/frontend/vendor   # only README.txt may change
T=<token>; U=http://127.0.0.1:7878
curl -sI $U/assets/vendor/wamr3d/wamr3d_wasm_bg.wasm | grep -iE "^HTTP|content-type"
curl -sI $U/assets/vendor/wamr3d/compat.js | grep -iE "^HTTP|content-type"
curl -sI $U/assets/flow3d.js | grep -iE "^HTTP|content-type"
```

**Expected (assets):**
- the vendor script prints `vendored wamr3d 0.9.4 …` and leaves every vendored file unchanged;
- `.wasm` → `HTTP/1.1 200 OK` and `content-type: application/wasm`; `.js` → `text/javascript; charset=utf-8`.

**Browser, `llama3.2:1b`, graph tab** (Chrome or Edge on this machine, `http://127.0.0.1:7878`):

1. **Toggle:** the toolbar starts with `[2D | 3D]`; 3D is selected by default with layout `dataflow`.
   The badge next to it says **`WebGPU`** (Chrome with a GPU). The status line says `26 nodes · 22 connections`
   at layer level (same as 2D).
2. **3D dataflow:** Input at the bottom, `layer 0` … `layer 15` as slabs stacked upwards, Output on top,
   Tokenizer and Chat template to the right near the input. Dataflow edges run up the column.
3. **2D:** press `2D` → the flat diagram of section 13. Back to `3D`: the camera framing is restored.
   The choice survives a reload; `#/model/17d2889ccc34/graph?view=2d` forces 2D, `?view=tower` the tower.
4. **Picking / hover:** hover `layer 7` → tooltip `layer 7 layer · 60.8M · 58.00 MiB · risky`
   (sizes as in the 2D view). Click → side panel opens; the slab gets a blue outline.
5. **Expand / collapse (level of detail):** double-click `layer 7` → its attention block to the left,
   feed-forward to the right, two residual adds on the column, an outline around the layer.
   Double-click `attention` → 8 KV groups on a ring; double-click `KV group 0` → 4 heads on a small ring.
   Double-click `feed-forward` → 128 neuron buckets on a grid (> 48). **Collapse all** returns to 26 nodes.
6. **Search + fly-to:** type `blk.3.ffn_down`, Enter → layer 3's FFN expands, the tensor is outlined and
   the camera flies to it; everything else is dimmed. Type `layer.7.attn` → one match, camera flies to it.
   **Focus** in the side panel flies to the selected node.
7. **Marks shared with 2D:** cut `layer 6` (**Cut this**) → grey, shrunk, red outline; switch to 2D → the
   same red outline. Uncut.
8. **Edit marks:** side panel of `layer 7` → **Edit: zero** → orange outline in 3D and 2D, the header
   button reads `Edits (1)` (orange). `undo` in the drawer removes the mark.
9. **Color by:** removable / section / kind / quant / importance / damage / flow all recolor the instances
   without rebuilding the view (camera does not move).
10. **Token flow:** press **▶ token flow** (3D dataflow) with no journey yet → `tracing…`, then particles
    run from Input up to Output in a loop (~4 s) and layers glow orange-red, most strongly the layers with
    the largest push (see the journey's decision table). With layer 13 expanded, its attention and FFN
    blocks glow separately. **■ stop flow** restores the colors. The button is disabled in 2D / tower / city / force.
11. **Fallback to WebGL 2:** open `#/model/17d2889ccc34/graph?backend=webgl` (or Firefox without WebGPU, or
    `chrome --disable-features=WebGPU` / `chrome://flags/#enable-unsafe-webgpu` disabled) → badge
    **`WebGL 2`**, everything in steps 1–10 still works. Then run `nd web --host 0.0.0.0` and open
    `http://<this machine's LAN IP>:7878/?token=…` from another device: the page is not a secure
    context, the badge shows `WebGL 2` (WebGPU is off over plain HTTP), and the graph works.
12. **No 3D at all:** a browser with WebGL disabled (`chrome --disable-gpu --disable-software-rasterizer
    --disable-webgl`) shows the message "3D needs WebGPU or WebGL 2, and this browser provides neither…"
    in the canvas, and the badge `no WebGPU / WebGL 2`; the side panel, cut selection and impact still work.

**All test models:** repeat steps 1, 2, 4, 5, 6 and 10 on:
- `qwen2.5:0.5b`: 24 layers; heads ring of 14 query heads (2 KV groups of 7). *record node count*.
- `oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU`: hybrid layers; expanding an SSM layer puts the
  `linear attention (SSM)` block on the left. Token flow: `token flow needs a trace: … unsupported for
  architecture "qwen35"` toast (no forward pass). Embedding map works.
- `nomic-embed-text`: 12 layers, Output (pooling); token flow uses residual-norm growth (embedding model).

**Embedding map** (tokens page → **Load embedding map**):
- `llama3.2:1b`: `128256 tokens · PCA explains …%` (*record*); first load computes the PCA
  (*record seconds*), a reload is instant (`~/neurondump/cache/<digest>/embedmap-v1.bin` exists).
  Badge shows the backend. Orbit, zoom; hover shows `"text" #id · class`; click inspects a token.
- Class toggles hide classes (uncheck `reserved` → the 248 reserved tokens vanish).
- Search `Sun` → matches enlarged in accent color and the camera flies to them.
- After **Tokenize**, the prompt's tokens are enlarged and joined by a line in order.
- Repeat on `qwen2.5:0.5b` (151,936 tokens), `oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU` and
  `nomic-embed-text` (30,522 tokens, F16).

```sh
curl -s -D - -o /tmp/map.bin -H "x-neurondump-token: $T" $U/api/models/17d2889ccc34/embedding-map | grep -i x-neurondump
python -c "import os;print(os.path.getsize('/tmp/map.bin'), 128256*16)"
curl -s -H "x-neurondump-token: $T" "$U/api/models/17d2889ccc34/embedding-map?format=json"
```

**Expected:** `x-neurondump-count: 128256`, `x-neurondump-stride: 4`, an `x-neurondump-explained`
triple; the file is `2052096` bytes (= 128256 × 16); the JSON has `"dim":2048,"sample":16384,"stride":4`.

---

## 15. Token inspector (Phase 15)

**What to test:** tokenization with the template, per-token embedding information, the token journey
(logit lens, attention heatmaps, FFN, decision breakdown), the generation timeline, CLI / REPL / API.

```sh
nd tokens llama3.2:1b "fn main() {"
nd tokens llama3.2:1b "fn main() {" --chat
nd tokens llama3.2:1b "fn main() {" --chat --system ""
nd tokens qwen2.5:0.5b "fn main() {" --chat --system "You are a Rust expert."
nd tokens nomic-embed-text "search_query: what is rust?"
nd tokens oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU "Hello" --chat
nd --json tokens llama3.2:1b "Hello" --neighbors 0 | python -m json.tool | head -30

nd journey qwen2.5:0.5b --prompt "fn main() {"
nd journey qwen2.5:0.5b --prompt "fn main() {" --token 2
nd journey llama3.2:1b --prompt "The name of the star that Earth orbits is"
nd journey nomic-embed-text --prompt "search_query: what is rust?"
nd journey oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --prompt "Hi"        # unsupported
nd journey llama3.2:1b --prompt "Hi" --token 99                         # out of range

nd generate llama3.2:1b --prompt "The name of the star that Earth orbits is" --steps 8
nd generate llama3.2:1b --prompt "The name of the star that Earth orbits is" --steps 8 --explain | tail -40
nd generate llama3.2:1b --prompt "Once upon a time" --steps 12 --temperature 0.8 --top-p 0.9 --seed 1
nd generate llama3.2:1b --prompt "Once upon a time" --steps 12 --temperature 0.8 --top-p 0.9 --seed 1   # identical (cached)
```

**Expected (tokens):**
- Raw `"fn main() {"` on llama3.2: `<|begin_of_text|>` (special, `ADDED template`), then `fn`, ` main`,
  `()`, ` {` (*record exact split*), class `code` / `word` / `punctuation`; `|EMB|` (*record*);
  `PREDICTABLE yes`; 5 nearest tokens each (` main` → `main`, ` Main`, … *record*).
- `--chat`: `chat template: ollama (system: none)` and the templated prompt starts with
  `<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023` and ends with
  `<|start_header_id|>assistant<|end_header_id|>\n\n`. Every token outside the user text is `template`.
- qwen2.5 with a system prompt: `<|im_start|>system\nYou are a Rust expert.<|im_end|>\n<|im_start|>user\nfn main() {<|im_end|>\n<|im_start|>assistant\n`.
- nomic: `[CLS]` … `[SEP]` added; `|OUT|` empty, `PREDICTABLE no` (no output head).
- qwen3.5: no Ollama template → `chat template: builtin:chatml`.

**Expected (journey on `qwen2.5:0.5b`, prompt `fn main() {`):**
- The table has 24 layers. The **logit lens converges in the late layers**: early layers show unrelated
  tokens with low P(target); the final prediction (a newline or indentation token, *record*) reaches
  rank 0 only in the last layers (`target first top-1 at layer` ≥ 18, *record*) and its probability
  climbs over the last few layers.
- `ATTENDS TO` of the last token points at `{`, `main` and the first token (attention sink).
- The decision breakdown lists `decision at position …: "<pred>" … vs runner-up "…"`, with
  `logit diff` and `attributed` close but not equal (the attribution is exact f32, the logits come from
  quantized matmuls; *record* the gap), the per-layer
  bars largest in the late layers, and top heads / neurons.
- `--token 2` inspects ` main` and shows `next token in the sequence: "("…` (*record*).
- llama3.2 "The name of the star that Earth orbits is": final `" not" 0.166` (as in section 8); the
  decision is " not" vs runner-up " the"; first top-1 at layer 15 (the lens is `" called"` at layer 14).
- nomic: attention and FFN columns filled, logit lens empty, note "embedding model: there is no
  next-token prediction…".
- qwen3.5: `error: simulation unsupported for architecture "qwen35" …`.
- `--token 99`: `token index 99 is out of range (the sequence has N tokens …)`.

**Expected (generate):** greedy " not a fixed or constant name. It" (section 12's baseline text, first
8 tokens); a table of 8 steps with P (0.166 for " not"), entropy (*record*) and the top
alternative; `--explain` prints a decision block per step. The sampled run is reproducible with the
same seed, and the second identical command prints `(cached result)`.

**Web:**
1. **tokens** page on `llama3.2:1b`: the text is prefilled; **Tokenize** shows chips (class-colored
   borders; template tokens dimmed with **chat template** ticked), the templated prompt with the user
   text highlighted, and the table. Click a chip → details with neighbours, **journey of this token**,
   **show on map**, **ban**, **boost +5**, **embedding ×0** (the last three update the header `Edits (n)`).
2. **journey** on `qwen2.5:0.5b` with `fn main() {`: the layer table, the attention heatmap
   (layers × tokens; click a row → heads × tokens for that layer, top heads outlined in purple),
   the decision breakdown with **zero** buttons. Click another token chip → its journey (cached when repeated).
3. **Show path in the graph** → the graph opens with the contributing layers expanded and purple
   outlines on them, their blocks and heads, in 3D and in 2D; **▶ token flow** animates that token.
4. **timeline**: 12 step cards with probability and entropy bars; click step 3 → its decision
   breakdown (a journey job with the generated tokens appended); tick **explain** → instant breakdowns.
5. On qwen3.5 the journey and timeline pages show the "Runtime tracing does not support the "qwen35"
   architecture" notice and the buttons are disabled; tokens still works.

```sh
H="x-neurondump-token: $T"; A=$U/api; M=7f1452142d64      # qwen2.5:0.5b
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"text":"fn main() {","chat":true,"neighbors":3}' $A/models/$M/tokenize \
  | python -c "import json,sys;r=json.load(sys.stdin);print(r['chat']['source'],len(r['tokens']),[t['text'] for t in r['tokens'] if not t['added']])"
J=$(curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"kind":"journey","model":"'$M'","params":{"prompt":"fn main() {","token":-1}}' $A/jobs | python -c "import json,sys;print(json.load(sys.stdin)['id'])")
curl -sN -H "$H" "$A/jobs/$J/events" | grep -c "^event: progress"
curl -s -H "$H" $A/jobs/$J/result | python -c "import json,sys;r=json.load(sys.stdin);print(len(r['layers']),r['first_top1'],r['first_top5'],r['decision']['chosen']['text'],round(r['decision']['logit_diff'],3),round(r['decision']['attributed'],3))"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"kind":"generate","model":"'$M'","params":{"prompt":"fn main() {","steps":6}}' $A/jobs
```

**Expected:** `ollama 5 ['fn', ' main', '()', ' {'…]` (*record*); the SSE stream has progress events;
the result prints `24 <layer> <layer> <token> <x> <y>` with the logit difference and the attribution
close but not equal (*record*; deepseek-r1 Q4_K_M showed 0.182 vs 0.048 on another prompt); the generate job returns `202 {"id":"generate-…"}`.

---

## 16. Live edits: change, compare, restore, bake (Phase 16)

**What to test:** every edit type, compare (text, top-k, KL, perplexity, lens, decision), undo / redo /
restore / toggle, snapshots, `edits.json`, bake → Ollama, and that ablation is part of the same stack.

```sh
cat > /tmp/zero7.json <<'EOF'
{"schema_version":1,"edits":[{"id":1,"enabled":true,"op":"zero","target":"layer.7"}]}
EOF
cat > /tmp/steer.json <<'EOF'
[{"op":"steer","layer":6,"a":"Dear Sir or Madam, I am writing to formally request your assistance.","b":"hey dude, just wanna ask u something lol","strength":4}]
EOF
cat > /tmp/off.json <<'EOF'
{"schema_version":1,"edits":[{"id":1,"enabled":false,"op":"zero","target":"layer.7"}]}
EOF
nd edit llama3.2:1b --apply /tmp/zero7.json --prompt "The name of the star that Earth orbits is" --tokens 12
nd edit llama3.2:1b --apply /tmp/off.json   --prompt "The name of the star that Earth orbits is" --tokens 12   # = baseline
nd edit llama3.2:1b --apply /tmp/zero7.json --corpus /tmp/corpus5.txt --tokens 0 --prompt Hi | grep -E "^corpus"
nd edit llama3.2:1b --apply /tmp/steer.json --prompt "Write a short note to my boss:" --tokens 24
nd edit llama3.2:1b --apply /tmp/zero7.json --bake /tmp/llama-zero7.gguf --no-compare
nd edit oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU --apply /tmp/zero7.json        # compare unsupported, see error
echo '[{"op":"scale","target":"layer.99","factor":2}]' > /tmp/bad.json && nd edit llama3.2:1b --apply /tmp/bad.json   # unknown node
```

**Expected:**
- **zero layer 7:** the compare prints the same numbers as `ablate --cut layer.7` in section 12
  (top-1 " not" 0.166 → " a" 0.157, KL 0.5473, the " a bit of a mystery, but I'll give you some" text), plus a
  decision table showing which layers' push changed. The corpus line reads
  `perplexity 7.554 -> 10.915, mean KL 0.2623` (same as the ablation).
- **all edits off:** KL `0.0000`, top-1 unchanged, identical generated text on both sides
  (that is the "undo → identical to baseline" check from the CLI).
- **steering:** the edited side's continuation becomes more formal (*record both texts*); KL > 0.
- **bake:** `baked llama3.2:1b -> /tmp/llama-zero7.gguf`, `size 1.23 GiB -> 1.23 GiB` (zeroing keeps
  shapes), `applied: zero layer.7`, `tensors changed: 2` (attn_output, ffn_down of layer 7), `validate: OK (147 tensors)`, a
  Modelfile next to it. Then:
  ```sh
  cd /tmp && ollama create llama-zero7 -f llama-zero7.Modelfile && ollama run llama-zero7 "What is the capital of Italy?" ; ollama rm llama-zero7
  ```
  The baked model runs in Ollama and answers the question (*record*; the in-memory run changes the
  continuation, see above).
- qwen3.5: the edit list prints, then `simulation unsupported for architecture "qwen35"`.
- unknown node: `edit #1 (scale layer.99 x2): unknown node "layer.99"`, exit 1.

**REPL** (`nd cli`, pick `llama3.2:1b`):

```
compare The name of the star that Earth orbits is --tokens 8
edit zero layer.7
compare --tokens 8
undo
compare --tokens 8
redo
toggle 1
compare --tokens 8
toggle 1
edit ban " not"
compare --tokens 8
edit steer 6 "Dear Sir or Madam, I am writing to formally request your assistance." "hey dude, just wanna ask u something lol" 4
generate Write a short note to my boss: --steps 20
edits
snapshot save formal
restore
edits
compare --tokens 8
snapshot load formal
edits export /tmp/formal.json
edit scale layer.7.attn.head.3 0
edit scale layer.7.ffn.neuron.42 0
edit embed Earth scale 0
edit patch 9 "The capital of France is"
edit sampling 0.8 40 0.9 1
edit force 0 " the"
edit system You are a pirate. Answer like one.
compare --tokens 8
journey -1
bake /tmp/llama-formal.gguf
web
```

**Expected:**
1. The first `compare` (empty stack) shows identical sides, KL 0.
2. `edit zero layer.7` prints `added edit #1` and the stack `#1 [on ] zero layer.7`, `changed nodes: layer.7`.
   `compare` shows the layer-7 numbers above. **`undo` → the stack is empty and `compare` is identical
   to the first one (KL 0, same text)**. `redo` brings #1 back; `toggle 1` switches it off (`[off]`) → identical again.
3. `edit ban " not"` resolves the token (`ban token <id>`); `compare` shows " not" gone from the edited
   top-k (probability 0) and a different continuation.
4. `generate` after the steering edit runs with the stack (`[edited]` in the header line).
5. `snapshot save formal` → `saved …/edits/<digest>/formal.json`; `restore` → empty stack; `compare`
   identical to baseline; `snapshot load formal` → the stack is back; `edits export` writes the file.
6. Every `edit …` line is accepted; `edit embed Earth scale 0` resolves `Earth` to one token (or
   reports how many tokens it is); `compare` shows the edited prompt text with the pirate system prompt
   (prompt edits switch both sides to chat format), the forced first token " the" and the sampling note.
7. `bake` lists the skipped edits (`ban`, `patch`, `steer`, `force`: run-time only) and the notes
   (sampling / system written to the Modelfile, tied embeddings changed by the embedding edit).
8. `web` opens the UI on the same session: the **Edits** button shows the REPL's edits.

**Web** (`llama3.2:1b`):
1. Header **Edits** → drawer. Add `zero` `layer.7` → listed with a checkbox; header `Edits (1)` in orange;
   graph outlines layer 7 in orange (3D and 2D).
2. **Compare** with the prompt `The name of the star that Earth orbits is`, tokens 12 → baseline / edited side by side,
   top-k, `KL 0.2…`, top-1 unchanged, the decision table, the logit lens diff (expandable).
3. **↶ undo** → stack empty; **Compare** again → identical texts, KL `0.0000`.
4. **↷ redo**, untick the checkbox (off) → compare identical; tick it again.
5. Add a steering edit (defaults are the formal / casual prompts, layer 6, strength 4) → compare with
   `Write a short note to my boss:` and 24 tokens → a more formal edited text.
6. Tokens page → a token → **ban** → the drawer lists `ban token …`; compare shows it gone from the edited top-k.
7. **restore all** → empty; **↶ undo** → everything back.
8. Snapshot: type `formal`, save → appears in the load menu; restore all; load `formal` → back.
9. **export edits.json** downloads `llama3.2_1b.edits.json`; **import** the same file → the same stack
   (ids renumbered).
10. Graph: cut `layer 6` → **Apply as edit** → `ablate layer.6` on the stack.
11. **Bake** `edited.gguf` → progress, then `wrote …/bake/edited.gguf`, `validate: OK`, the
    `ollama create edited -f "…edited.Modelfile"` command, and the skipped run-time edits.

```sh
M=17d2889ccc34; A=$U/api
curl -s -H "$H" $A/models/$M/edits | python -c "import json,sys;v=json.load(sys.stdin);print(len(v['edits']),v['can_undo'])"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"op":"zero","target":"layer.7"}' $A/models/$M/edits | python -c "import json,sys;v=json.load(sys.stdin);print(v['active'],v['compiled']['touched'])"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"op":"scale","target":"nope","factor":2}' -o /dev/null -w "%{http_code}\n" $A/models/$M/edits
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"kind":"compare","model":"'$M'","params":{"prompt":"The name of the star that Earth orbits is","tokens":4}}' $A/jobs
curl -s -H "$H" -X POST $A/models/$M/edits/1/toggle | python -c "import json,sys;print(json.load(sys.stdin)['edits'][0]['enabled'])"
curl -s -H "$H" -X POST $A/models/$M/edits/undo | python -c "import json,sys;v=json.load(sys.stdin);print(v['changed'],v['edits'][0]['enabled'])"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"name":"t1"}' $A/models/$M/snapshots | python -c "import json,sys;print(json.load(sys.stdin)['snapshots'])"
curl -s -H "$H" -X DELETE $A/models/$M/edits | python -c "import json,sys;print(len(json.load(sys.stdin)['edits']))"
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"name":"t1","action":"load"}' $A/models/$M/snapshots | python -c "import json,sys;print(len(json.load(sys.stdin)['edits']))"
curl -s -H "$H" "$A/models/$M/edits/export" -o /tmp/e.json && python -m json.tool /tmp/e.json | head -12
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"out":"../evil.gguf"}' -o /dev/null -w "%{http_code}\n" $A/models/$M/bake
curl -s -H "$H" -X POST -H "content-type: application/json" -d '{"out":"api-baked.gguf"}' $A/models/$M/bake
curl -s -H "$H" -X POST $A/models/$M/edits/restore | python -c "import json,sys;print(len(json.load(sys.stdin)['edits']))"
```

**Expected:** `0 False` (fresh session), `1 ['layer.7']`, `400`, `202 {"id":"compare-…"}`, `False`,
`True True`, `['t1']`, `0`, `1`, the edits.json with `schema_version`, `model`, `digest`, `edits`;
`400` for `../evil.gguf`; `202 {"id":"bake-…"}` (file in `~/neurondump/bake/api-baked.gguf`); `0`.

---
## Checklist

### 1. Discovery / parser / list / inspect
- [ ] `nd list` shows all 7 Ollama models with arch, params, quant, size
- [ ] `nd --json list` is valid JSON with `schema_version`
- [ ] `inspect llama3.2:1b`: 16 layers, vocab 128256, GQA 32/8, tied embeddings, under 1 s
- [ ] `inspect nomic-embed-text` → encoder, 30522 tokens
- [ ] `inspect oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU` → hybrid 18 linear + 6 full
- [ ] `inspect qwen3.8:27b` → manifest list resolved, projector linked
- [ ] raw `.gguf` path inspects
- [ ] `--scan <dir>` picks up extra `.gguf` files
- [ ] `inspect --all` → 0 failed
- [ ] wrong names give "did you mean"
- [ ] truncated / non-GGUF / tiny files → warning or error, never a panic
- [ ] unknown architecture (`mystery.gguf`) inspects with warnings only

### 2. Architecture / roles
- [ ] `layers` shows per-layer kinds (hybrid pattern visible)
- [ ] `tensors --layer/--role/--sort` filter correctly
- [ ] qwen2 biases and qwen3 q/k norms detected

### 3. Tokenizer / templates
- [ ] vocab search, regex, class and lang filters work
- [ ] reserved tokens are recognized for llama, qwen and nomic
- [ ] `--tokenize` matches `llama-tokenize` for llama3.2, qwen2.5-coder and nomic
- [ ] Jinja template tree (llama3.2, qwen3.5)
- [ ] Ollama Go template tree (llama3.2, qwen2.5-coder)

### 4. Stats / jobs / cache
- [ ] `stats` with progress bar, sensible numbers
- [ ] the second run is cached
- [ ] Ctrl-C cancels cleanly
- [ ] `inspect --stats` includes the stats section

### 5. Picker / REPL
- [ ] the picker appears with no args on a TTY
- [ ] "remember my choice" is saved and honored
- [ ] `--ui` / `NEURONDUMP_UI` override the saved choice
- [ ] non-TTY with no args → help + exit 2
- [ ] `config show/set/reset/path`
- [ ] REPL: fuzzy model picker, prompt shows the model
- [ ] REPL: all commands from `help` work
- [ ] REPL: Tab completion (commands, models, classes, tensors)
- [ ] REPL: history persists
- [ ] REPL: background job (`stats &`), `jobs`, `job`, `cancel`
- [ ] REPL: `web` launches the UI on the same session

### 6. Web
- [ ] the URL with token opens the UI; without a token → error message
- [ ] every curl endpoint returns the expected status (401 / 200 / 202 / 404 / 409)
- [ ] SSE progress, then `done`
- [ ] job cancel via DELETE
- [ ] compare and export downloads work
- [ ] path traversal / raw paths / bad slim names rejected
- [ ] `--host 0.0.0.0` prints a warning
- [ ] all UI views render
- [ ] vocab paging is fast on 128k–248k vocabularies
- [ ] analyze → checkboxes → slim from the UI
- [ ] header search (`/` shortcut, grouped results, links, live update) and `nd search`
- [ ] models filter + sortable columns, click-to-copy, theme toggle, running-jobs badge

### 7. Analyze / slim
- [ ] analyze plans for llama, nomic, qwen3.5, qwen3.8
- [ ] vocab-trim slim validates OK and runs in `llama-completion`
- [ ] layer-removal slim validates OK and runs
- [ ] reserved-token slim + perplexity comparison
- [ ] nomic vocab+quant slim validates
- [ ] experimental candidates are refused with an explanation
- [ ] the Modelfile works with `ollama create` + `ollama run`
- [ ] plan/model mismatch and bad output names are rejected

### 8. Trace
- [ ] llama3.2:1b: topic in the lens from layer 11 (" Earth"), " called" 0.39 at layer 14, final " not" 0.166
- [ ] qwen2.5-coder next token matches llama.cpp
- [ ] qwen3:8b trace runs
- [ ] nomic: embedding contributions
- [ ] corpus mode aggregates, and analyze then uses trace similarity
- [ ] qwen35 → "trace unsupported", static analysis still works
- [ ] a build without `--features trace` explains how to enable it
- [ ] web trace view + cancel

### 9. Formats / schemas / compare
- [ ] json / ndjson / csv exports for tensors, vocab, metadata, layers
- [ ] CLI export == web download
- [ ] the JSON validates against `schemas/inspect.schema.json`
- [ ] `compare` of 2–3 models, vocab overlap matrix

### 10. Graph
- [ ] `graph` tree at section / layer / block / tensor / fine on llama, qwen2.5-coder, nomic, qwen3.5, qwen3.8
- [ ] hybrid layer kinds: SSM vs attention blocks, `--group-by kind`, layers `breaks_model`
- [ ] lazy expansion: KV groups → heads, neuron buckets, experts (MoE only)
- [ ] `--group-by` section / layer / block / quant / kind / importance
- [ ] `node` for a block, a head, a shorthand id and an unknown id
- [ ] json / dot / mermaid exports; `dot -Tsvg` renders; mermaid.live renders

### 11. Static impact
- [ ] middle layer → risky, renumbering, new size
- [ ] token_embd → breaks_model, refused without `--force` (exit 1), allowed with it
- [ ] all attention of one layer → zeroed in place, risky
- [ ] head / neuron bucket / block plan → slim → zeros verified with `stats` → llama.cpp runs
- [ ] Q4_K head cut reports the Q8_0 requantization
- [ ] vocab range hitting template / bos / eos tokens → dependencies + breaks_model
- [ ] hybrid: single layer refused, whole aligned period risky
- [ ] projector + template → other files freed, slim file ops listed

### 12. Simulation / sweep
- [ ] middle vs last layer on llama: prompt KL, greedy text, lens table; corpus perplexity (last layer hurts more)
- [ ] head ablation barely changes the output; nomic shows embedding cosine
- [ ] unsupported architecture → static only with an explanation
- [ ] layer sweep ranking, `--save-plan --top 2` → slim → `ollama create` → `ollama run` answers
- [ ] head sweep of one layer; importance in `node` / graph comes from the sweep
- [ ] sweep error cases (missing --layer, expert level, embedding model, qwen35)

### 13. Web graph
- [ ] graph renders offline (vendored wamr3d), level / group / color switchers, Fit, Collapse all
- [ ] view select: 3D tower (default), 3D city, 3D force, 2D; choice remembered; `?view=` works
- [ ] 2D: flat top-to-bottom diagram, drag pans, wheel zooms, expanded layers as columns
- [ ] 3D: orbit, zoom −/+/100%/Fit, hover tooltip, click outline selects a container, double-click expands
- [ ] 3D: search, cut outline, color by (incl. damage after a sweep) match the 2D view
- [ ] no WebGPU / WebGL 2 → error box in the canvas (no fallback)
- [ ] double-click expand / collapse down to heads; click → side panel
- [ ] Cut this / Uncut / × in the selection; live static impact; force checkbox
- [ ] search highlights visible nodes and expands to hidden tensors
- [ ] Simulate (SSE progress, cancel, result view, cut nodes greyed)
- [ ] Sweep this level / Sweep heads of layer N → damage overlay + table → plan → Slim
- [ ] every `curl` in section 13 returns the expected status and values

### 14. 3D graph (wamr3d)
- [ ] `scripts/vendor-wamr3d.sh` re-vendors 0.9.4 with no file changes
- [ ] `.wasm` served as `application/wasm` (`curl -I`), `.js` as `text/javascript`
- [ ] 2D / 3D toggle; 3D dataflow is the default; choice remembered; `?view=` works
- [ ] llama3.2:1b: badge `WebGPU`; same page with `?backend=webgl` (or a browser without WebGPU) shows `WebGL 2` and works
- [ ] `--host 0.0.0.0` from another device over HTTP → `WebGL 2`, graph works
- [ ] no WebGPU / WebGL 2 → clear message in the canvas
- [ ] dataflow layout: input at the bottom, layers up the column, output on top, side sections beside
- [ ] expand / collapse down to heads, buckets (grid), experts; Collapse all
- [ ] picking (instances and container outlines), hover tooltip, side panel
- [ ] search highlights and flies to matches; Focus button flies to a node
- [ ] cut / edited / token-path marks shared between 2D and 3D
- [ ] color by works without moving the camera (incl. `flow`)
- [ ] token flow: particles, layer glow by push, blocks glow separately when expanded, stop restores colors
- [ ] all of the above on llama3.2:1b, qwen2.5:0.5b, oamazonasgabriel/qwen3.5-0.8b:q8-8gbGPU (token flow → unsupported toast), nomic-embed-text
- [ ] embedding map on all four models: loads, cached on reload, class colors/toggles, search, click, prompt path
- [ ] `/embedding-map` binary is count × 16 bytes; `?format=json` describes it

### 15. Token inspector
- [ ] `tokens` raw and `--chat` (templated prompt, template tokens marked), `--system ""` removes the system prompt
- [ ] nearest neighbours, embedding / output norms, predictable flag; nomic has no output head
- [ ] qwen3.5 without an Ollama template uses `builtin:chatml`
- [ ] `journey` on qwen2.5:0.5b `fn main() {`: logit lens converges in the late layers
- [ ] journey attention (per layer, per head heatmap), FFN buckets, decision breakdown (logit diff and attribution close; gap from quantized matmuls)
- [ ] journey errors: unsupported architecture, index out of range; nomic shows the embedding note
- [ ] `generate` greedy matches section 12's baseline text; sampling reproducible with a seed; `--explain`
- [ ] web tokens page: chips, details, actions (journey / map / ban / boost / embedding ×0)
- [ ] web journey: heatmaps, decision with zero buttons, Show path in the graph (2D + 3D) + token flow
- [ ] web timeline: step cards, click a step → decision; explain
- [ ] API: tokenize, journey job (SSE), generate job

### 16. Live edits
- [ ] zero a middle layer → change visible (same numbers as `ablate --cut layer.7`) → undo → output identical to baseline
- [ ] all edits off → KL 0, identical text
- [ ] steering example makes the continuation more formal
- [ ] banning a token removes it from the edited top-k
- [ ] scale head / neuron, embed, patch, sampling, force, system prompt edits all accepted and compared
- [ ] toggle, redo, restore all (and undo after restore)
- [ ] snapshot save / load (REPL, web, API); export / import edits.json
- [ ] bake → validate OK → `ollama create` → `ollama run` works; run-time-only edits listed as skipped
- [ ] bad edits refused (unknown node 400 / exit 1), bad bake names refused (400)
- [ ] REPL edits visible in the web UI started with `web`
- [ ] graph **Apply as edit** puts a cut on the stack; edited nodes outlined orange in 2D / 3D
- [ ] edit drawer on every model page, `Edits (n)` badge

---

## Incomplete or stubbed features

The full list is in [docs/limitations.md](docs/limitations.md). In short:

- **Phases 14–16 were only built (`cargo build`), not run:** no expected value marked *record* in
  sections 14–16 has been measured, and the new 3D dataflow view, token flow, embedding map, token
  inspector pages and edit UI have not been opened in a browser yet. The numbers that are given come
  from code paths shared with sections 8 and 12 (same forward pass and masks).
- **Runtime features** (trace, simulation, journey, timeline, live edits) cover llama, qwen2, qwen3,
  gemma/gemma2/gemma3, phi3, bert and nomic-bert only; no MoE or hybrid (qwen35) forward pass, so MoE
  routing and SSM state are never measured and expert edits are bake-only. CPU only, no KV cache.
- **Bake** cannot store run-time-only edits (ban, boost, steering, patching, forced tokens); scaled
  quantized tensors are re-encoded as Q8_0.
- **Chat rendering** executes Ollama's Go templates (common subset) and built-in family templates; Jinja
  templates are not executed.
- **Structured pruning** (heads / FFN neurons / MoE experts) is detected and scored by `analyze`, and `slim`
  still refuses to remove them; they can be zeroed in place (ablation plans, bake). Experimental.
- **WebGPU** needs a secure context (`localhost` / HTTPS); remote plain-HTTP access uses WebGL 2.
- **Dequantization** doesn't support IQ1_S/IQ1_M/IQ2_*/IQ3_*, TQ1_0/TQ2_0 or the legacy repacked types.
- **Tokenizers**: BPE pre-tokenizers are exact for llama3 / qwen2 / gpt2; SPM untested; UGM/RWKV missing.
- **No automated tests, fuzz target or criterion benches**, as requested. Clippy was not run.
