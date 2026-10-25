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
nd vocab llama3.2:1b --search Paris --limit 5
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
vocab Paris --limit 5
vocab --class special
tensors --sort size --limit 5
template
stats &
jobs
job <id from jobs>
use nomic-embed-text
compare llama3.2:1b
export --format csv --table layers
trace The capital of France is
web
exit
```
Also try Tab completion: `us<Tab>`, `use ll<Tab>`, `vocab --class sp<Tab>`, `stats --tensor blk.0.<Tab>`.
↑ recalls history after a restart (history is stored next to the config file).

**Expected:** the prompt shows `neurondump[llama3.2:1b]>`. `stats &` prints `started job stats-…`;
`jobs` shows its progress; `job <id>` renders the stats tables. Long outputs page with
`-- more (50/…) Enter: next, q: quit --`. `web` prints a URL with `?token=` and opens the browser
while the REPL keeps working. A piped REPL (`printf 'info\nexit\n' | nd cli`) runs without prompting.

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
curl -s -w "\n%{time_total}s\n" -H "$H" "$B/api/models/17d2889ccc34/vocab?q=paris&limit=5"
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
nd search paris --model llama3.2:1b
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
- Header search: press `/`, type `paris` on a model page, then Enter. You get grouped results with
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
llama-completion -m ~/neurondump/slim/llama-15l.gguf -p "The capital of France is" -n 10 --temp 0 -no-cnv
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
  worse quality (one test run answered "The capital of France is not Paris"). `llama-noreserved`
  must be indistinguishable from the original. The Ollama runs reply normally, with the original
  template, stop tokens and parameters. `nomic-mini` returns a 768-dim embedding.

**Edge cases:**
```sh
nd slim llama3.2:1b --plan /tmp/nomic.prune.json --out a.gguf          # "plan was made for nomic-embed-text…"
nd slim llama3.2:1b --plan /tmp/llama.prune.json --only vocab --out ../evil.gguf   # plain-file-name error
nd slim llama3.2:1b --plan /does/not/exist.json --out a.gguf
echo '{}' > /tmp/bad.json && nd slim llama3.2:1b --plan /tmp/bad.json --out a.gguf  # invalid plan JSON
```
Hand-edit `/tmp/llama.prune.json` (set `"selected": true` on `reserved_tokens`) and run
`slim` without `--only`: only that candidate is applied.

---

## 8. Runtime tracing (Phase 5, feature `trace`)

**What to test:** per-layer internals, the logit lens, corpus aggregation, embedding contributions.

```sh
nd trace llama3.2:1b --prompt "The capital of France is"
nd trace qwen2.5-coder:7b --prompt "def fibonacci(n):"
nd trace qwen3:8b --prompt "The opposite of hot is"
nd trace nomic-embed-text --prompt "search_query: what is rust?"
printf 'The cat sat on the mat.\n\nRust is fast.\n\nParis is in France.\n' > /tmp/tc.txt
nd trace llama3.2:1b --corpus /tmp/tc.txt
nd analyze llama3.2:1b --out /tmp/llama-trace.prune.json        # layer candidates now say "trace: cosine(...)"
nd --json trace llama3.2:1b --prompt "Hi" --top-k 3 | python -c "import json,sys;r=json.load(sys.stdin);print(len(r['layers']), r['prediction'][0])"
```

**Expected:** for llama3.2:1b, 6 tokens starting with `<|begin_of_text|>`. The logit lens is noise for
layers 0–10 and **" Paris" ≈ 0.84 from layer 11** onward. Final `next token: " Paris" 0.72`.
cos(in,out) rises from ≈0.3 (layer 0) to ≈0.93 (layer 12) and drops in the last layer. Layer 1 has one
huge neuron (index 1417, ≈190). Runtime is about 3–4 s.
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
- [ ] llama3.2:1b: " Paris" in the logit lens from layer ~11, final prediction " Paris"
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

---

## Incomplete or stubbed features

- **Structured pruning** (heads / FFN neurons / MoE experts) is *detected and scored* but
  `slim` refuses to apply it: no architecture is confirmed to load non-uniform shapes.
- **Trace** covers llama, qwen2, qwen3, gemma/gemma2/gemma3, phi3, bert and nomic-bert only. Only
  llama, qwen2 and nomic-bert were run here, because no gemma/phi/bert model is installed. No MoE or hybrid
  (qwen35) forward pass, so router choices are never filled. gemma2 soft-capping is skipped.
  Inference is CPU-only and loads all weights into RAM (8B Q4_K ≈ 5 GB).
- **Dequantization** doesn't support IQ1_S/IQ1_M/IQ2_*/IQ3_*, TQ1_0/TQ2_0 or the legacy repacked
  types. Their tensors are skipped in stats with a reason; trace falls back only where candle supports the type.
- **Tokenizers**: BPE pre-tokenizers are exact for llama3 / qwen2 / gpt2. Others fall back to the
  llama3 rules with a warning. SPM is implemented but untested here (no SPM model installed).
  UGM/RWKV are not implemented (vocab trimming is unavailable for those).
- **Requantization** in slim only targets Q8_0, and it dequantizes a whole tensor in memory.
- **No automated tests, fuzz target or criterion benches**, as requested. Clippy was not run.
- The REPL `web` command can only start one server per session. The web watch mode checks the
  manifest directories on each `/api/models` call instead of using OS file events.
- `comfy-table` is built without terminal-width detection (Windows GNU toolchain workaround), so
  wide tables do not wrap to the terminal.
