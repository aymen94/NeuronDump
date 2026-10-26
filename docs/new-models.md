# Gemma 4, Gemma, Gemma 2 and Phi-3

New model families run through NeuronDump 0.1.4, and every runtime result checked against
llama.cpp and Ollama 0.40.5 on the same files. All models are the Ollama builds (`ollama pull`).

## Checked against llama.cpp

### Tokenizers

`neurondump tokens <model> "<text>"` against llama.cpp's `llama-tokenize -m <file> -p "<text>"` on 8
texts (English, extra spaces, accents and emoji, Python code, numbers, Japanese / Chinese, tabs and
`\r\n`):

| Tokenizer | Models | Identical ids |
|---|---|---|
| SPM (SentencePiece) | gemma:7b, gemma2:2b, phi3 | **8 / 8** each |
| BPE llama3 | llama3.2:1b | 8 / 8 |
| BPE qwen2 / qwen3 | qwen2.5:0.5b, deepseek-r1, qwen3-coder (MoE), qwen3.5 0.8B (hybrid) | 8 / 8 each |
| WPM (BERT) | nomic-embed-text | 6 / 8: accented words become `[UNK]` (llama.cpp strips the accents first), Japanese kana with voiced marks differ |
| gemma4 | gemma4:e2b, gemma4:e4b | not implemented: `tokens` stops with an error |

(On Windows, feed `llama-tokenize` the text with `-p`, not `--stdin`: stdin turns `\r\n` into `\n`.)

### Forward pass

`neurondump generate <model> --prompt "The name of the star that Earth orbits is" --steps 8`
(greedy) against Ollama's `/api/generate` with `raw: true`, `temperature: 0` and `logprobs`:

| Model | Same 8 tokens | First token: NeuronDump vs Ollama | Verdict |
|---|---|---|---|
| llama3.2:1b | yes: " not a fixed or constant name. It" | " not" 0.166 vs 0.173 | exact |
| qwen2.5:0.5b | yes: " ____\nA. Alpha Centauri" | " __" 0.155 vs 0.144 | exact |
| gemma:7b | first 5 of 8: "A) The **Sun**" vs "A) The **Moon**" | ":" 0.512 vs 0.481 | correct; step 5 is a near-tie in Ollama (" Moon" 0.412 vs " Sun" 0.407) that quantization noise flips |
| gemma2:2b | yes: " \*\*the Sun\*\*. …" | " \*\*" **0.491 vs 0.325** | right tokens, probabilities too sharp: logit soft-capping (50 / 30) is not applied |
| phi3 (mini, 128k) | no: " called the Sun." vs " the Sun." | " called" 0.133 vs " the" 0.091 | **drifts**: LongRoPE factors are not applied (see below) |

llama, qwen2, qwen3 (see [What's inside real models](models.md)), gemma and nomic-bert are therefore
reliable; gemma2 ranks tokens right but overstates probabilities; phi3 128k models are only
approximate. gemma3 and bert were not run (no model installed).

## Gemma 4 (e2b, e4b)

`gemma4:latest` is the same file as `gemma4:e4b`. Static analysis works (inspect, layers, graph,
stats, template, `diff --metadata-only`); trace does not support `gemma4` yet. Not tried: `slim`.

| | gemma4:e2b | gemma4:e4b |
|---|---|---|
| parameters | 4.65B | 7.52B |
| of which embeddings | **2.75B (59 %)** | **3.49B (46 %)** |
| without embeddings ("effective") | 1.90B | 4.03B |
| layers | 35 | 42 |
| full / sliding attention | 7 / 28 (every 5th is full) | 7 / 35 (every 6th is full) |
| query / KV heads | 8 / **1** | 8 / 2 |
| head size full / sliding | 512 / 256 | 512 / 256 |
| layers reusing an earlier KV cache | 20 | 18 |
| FFN width | 6144 (layers 0–14), **12288** (15–34) | 10240 everywhere |
| file | 3.26 GiB + vision projector | 5.12 GiB + vision projector |

- **"E2B / E4B" means effective size.** Most of the parameters are a second embedding table,
  `per_layer_token_embd` (8960 × 262,144 in e2b = 35 layers × 256 values): every token brings its own
  small vector into **every layer**, mixed in through `blk.N.inp_gate` (hidden → 256) and
  `blk.N.proj` (256 → hidden).
  Without the embeddings the models are 1.9B and 4.0B, hence the names.
- **Shared KV cache.** The last 20 (e2b) or 18 (e4b) layers have no KV cache of their own
  (`attention.shared_kv_layers`). In e2b exactly those 20 layers get a **twice as wide FFN** (12288 vs
  6144), as if the saved attention budget went into the feed-forward.
- **Long context on a budget:** 131,072 tokens, but only 7 layers see all of it; the others use a
  512-token window, with RoPE base 10,000 instead of 1,000,000 and half-size heads.
- **Vocabulary:** 262,144 tokens with 514,906 BPE merges (Gemma 1 / 2: 256,000 SPM tokens). Final
  logits are soft-capped at 30.
- **The chat template lives in Ollama, not in the file's Ollama layer:** the Go template is just
  `{{ .Prompt }}`; Ollama formats Gemma 4 chats in its own code. The GGUF's Jinja template is large
  (127 decisions: tools, `<|channel>` and audio markers).
- **What NeuronDump does not know yet:** four tensor roles (`inp_gate`, `layer_output_scale`,
  `post_norm`, `proj`) are shown as unknown, and the `gemma4` tokenizer is not implemented.

## Gemma 2 2B

- 26 layers, 8 query / 4 KV heads of 256, a 256,000-token vocabulary that is **23 % of the
  parameters** (tied input / output), Q4_0.
- Soft-capping: attention logits at 50, final logits at 30. NeuronDump's trace skips both, which is
  why its probabilities are sharper than Ollama's (0.49 vs 0.33 for the top token) while the order and
  the generated text match.
- Every other layer uses a 4096-token sliding window, but the GGUF does not store the pattern
  (llama.cpp hard-codes it), so `layers` lists 26 full-attention layers. Prompts under 4096 tokens
  are unaffected.

**Trace** of "The name of the star that Earth orbits is":

- The residual stream grows steadily from 106 to 1438 (×14 over 26 layers); in Gemma 7B it stays
  almost flat (below).
- The logit lens says `" is"` up to layer 6, then `" called"` from layer 7 to 23, with a detour
  through German `" sogenannte"` ("so-called", 0.48) at layer 13 and `" star"` (0.35) at layer 20.
- Only the last two layers change the plan: `" the"` (0.56) at layer 24, then `" **"` (0.49), the
  Markdown bold of a chat-tuned model. "Sun" never appears in the lens, yet the model writes
  "**the Sun**" two tokens later.

## Gemma 7B

- 28 layers, 16 heads with **16 KV heads** (no grouping) of 256, FFN 24,576, a 256,000-token
  vocabulary (9 % of 8.54B parameters), Q4_0.
- **The residual stream barely grows:** 88 at layer 0, 70 through the middle, 99 at layer 26, then
  142 at the last layer. Layer 0 has one huge neuron (21041, activation 127) and layer 1 another (908,
  49), the same kind of early "massive activation" as Qwen3's layer-6 neuron
  ([What's inside real models](models.md)).
- The logit lens mostly holds `" is"` from layer 2 to 19, then `" currently"` (0.88, layer 21), `" called"`
  (layers 23–25) and finally `":"` (0.51): the model sets up a **quiz** ("A) The Moon / Sun, B) …")
  rather than answering.

## Phi-3 mini (128k)

- 32 layers, 32 heads with 32 KV heads of 96, FFN 8192, a 32,064-token SPM vocabulary, **untied**
  output head, Q4_0, 2.03 GiB.
- The 128k context comes from **LongRoPE**: per-dimension RoPE factors (`rope_factors_short` /
  `rope_factors_long`, 48 values each) and an attention factor of 1.19 over an original context of
  4096. NeuronDump's trace applies only `rope_freqs` (the Llama 3 style), so neither is used and its
  outputs drift from llama.cpp from the first token. Static analysis is unaffected; a 4k Phi-3 has no
  such factors.

## Reproduce

```sh
neurondump tokens gemma2:2b "naïve café — 🚀" --json
llama-tokenize -m <path from `neurondump list --json`> -p "naïve café — 🚀" --ids
neurondump generate gemma2:2b --prompt "The name of the star that Earth orbits is" --steps 8 --json
curl http://127.0.0.1:11434/api/generate -d '{"model":"gemma2:2b","prompt":"The name of the star that Earth orbits is","raw":true,"stream":false,"logprobs":true,"top_logprobs":5,"options":{"temperature":0,"num_predict":8}}'
neurondump trace gemma:7b --prompt "The name of the star that Earth orbits is"
neurondump layers gemma4:e2b
neurondump graph gemma4:e4b --level layer
```
