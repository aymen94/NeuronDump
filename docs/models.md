# What's inside real models

Findings from running NeuronDump on common Ollama models. Every number comes from
`inspect`, `layers`, `vocab`, `template`, `compare`, `stats`, `analyze`, `trace` or `ablate`.

## qwen3:8b vs deepseek-r1 (8B)

`deepseek-r1:latest` is **DeepSeek R1 0528 Qwen3 8B**: DeepSeek's reasoning model distilled into
Qwen3 8B. Comparing the two shows what the fine-tune changed and what it kept.

### Same structure

- Both are `qwen3`, 8.19B parameters, Q4_K_M, 4.87 GiB. Each has 36 identical full-attention layers,
  with 32 query and 8 key-value heads.
- `compare` finds the same 399 tensor names, shapes and types, and the same 151,936-token
  vocabulary. Only 9 metadata keys differ.
- Both keep every `attn_v` matrix in F16 (288 MiB) while the rest is mostly Q4_K. `analyze`
  suggests Q8_0 for those, which would save 135 MiB in either model.

### What DeepSeek changed

- **Every weight, slightly.** A byte comparison finds **none of the 399 tensors identical**, but
  the changes are small: `token_embd` L2 norm 550.0 → 558.3, `output_norm` 131.3 → 129.0,
  `blk.0.attn_q` 106.64 → 106.70. It is a full fine-tune, not new layers.
- **Longer context:** 40,960 → **131,072 tokens** with YaRN (scaling factor 4, from an original 32,768).
- **4 tokens:**
  - two are renamed: `<|endoftext|>` → `<｜begin▁of▁sentence｜>` and `<|im_end|>` → `<｜end▁of▁sentence｜>`;
  - two padding slots (151669, 151670) become `<｜User｜>` and `<｜Assistant｜>`;
  - that leaves 265 unused `[PADnnnnn]` tokens instead of 267.
- **Chat template:** Qwen3 uses `<|im_start|>` / `<|im_end|>` with `<tool_call>` tags (28 decisions).
  DeepSeek uses `<｜User｜>` / `<｜Assistant｜>` (24 decisions), and it **drops earlier reasoning**:
  for past assistant messages it keeps only the text after `</think>`.
- **Broken tool-call markers:** the DeepSeek template emits `<｜tool▁calls▁begin｜>` and similar
  markers, but they are **not in the vocabulary**, so each one is split into 11 ordinary tokens.
- **Settings and license:** Ollama defaults go from Qwen3's `top_k 20, repeat_penalty 1` to only
  `temperature 0.6, top_p 0.95`. The license changes from Apache 2.0 to MIT.

### Same circuits, answer one layer later

`trace` with "The capital of Italy is":

| | qwen3:8b | deepseek-r1 |
|---|---|---|
| " Rome" first appears | layer 29 (0.52) | layer 30 (0.74) |
| " Rome" certain (1.00) | layer 30 | layer 31 |
| final prediction | " Rome" 0.83 | " Rome" 0.63 (then " __" 0.10) |
| huge neuron, layer 6 | neuron 5723, 770 | neuron 5723, 677 |
| huge neuron, last layer | neuron 9510, 647 | neuron 9510, 837 |

- **Shared circuits:** the most active neurons have the same indices layer after layer, and both
  models make their internal state about 32–33× larger at layer 6 with the same single neuron.
- **One layer later:** DeepSeek reaches the answer one layer later than Qwen3 and is less sure at
  the end. A reasoning model expects a chat template, not a raw sentence.

### What happens if you cut a layer

`ablate --cut <layer> --prompt "The capital of Italy is"` (in memory, nothing written):

| Cut | qwen3:8b | deepseek-r1 |
|---|---|---|
| layer 6 (huge neuron) | KL **8.76**, answer becomes `"
"` 0.13 | KL **8.22**, answer becomes noise (`"Italy"` 0.003) |
| layer 20 | KL 0.015, " Rome" 0.83 (almost no change) | KL 0.17, " Rome" 0.78 |
| layer 35 (last) | KL 0.68, " Rome" **0.995** | KL 1.22, " Rome" **0.95** |

- **Layer 6 is essential** in both models: without it nothing sensible comes out.
- **Middle layers are cheap to lose in Qwen3** but cost about 10× more in DeepSeek. The fine-tune
  spread its work across more layers.
- **The last layer lowers confidence:** removing it makes both models *more* certain of " Rome".
  It spreads probability to other continuations, and the KL measures that change.

## qwen3.8:27b

- `qwen35`: a **hybrid** model with 27.32B parameters, Q4_K_M, 15.66 GiB.
- 64 main layers, always 3 linear-attention (state-space) layers then 1 full-attention layer
  (48 + 16).
- Layer 64 is an extra **next-token-prediction layer** (`nextn.*` tensors, used for faster
  generation). It is a full-attention layer plus a 52M-parameter projection.
- 24 query heads share 4 key-value heads (groups of 6), 256-dimensional heads.
  Native context is **262,144 tokens**.
- 248,320-token vocabulary, of which 243 are unused.
- Input and output embeddings are separate (682 MiB and 995 MiB).
- Ollama's manifest is a **manifest list** pointing to more than one version of the model;
  NeuronDump picks the GGUF one.
- The image part is a separate 888 MiB file (a CLIP projector, 461M parameters). Leaving it out
  for text-only use saves that much.
- Layer removal is refused (it would break the 3:1 pattern). Trace doesn't support `qwen35` yet.
