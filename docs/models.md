# What's inside real models

Findings from running NeuronDump on common Ollama models. Every number comes from
`inspect`, `layers`, `vocab`, `template`, `compare`, `stats`, `analyze`, `trace` or `ablate`.
The trace and ablation numbers were measured with NeuronDump 0.1.3 (candle, CPU) on the prompt
"The name of the star that Earth orbits is".

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

### Same circuits, same timing

`trace` with "The name of the star that Earth orbits is" (9 tokens, the same ids in both models):

![deepseek-r1 traced on the prompt: the logit lens per layer](images/model_trace.png)

| | qwen3:8b | deepseek-r1 |
|---|---|---|
| the answer first appears | layer 29: `"太阳"` 0.84 + `" Sun"` 0.15 | layer 29: `"太阳"` 0.90 + `" Sun"` 0.05 |
| most certain | layer 30: `"太阳"` 0.91 | layer 30: `"太阳"` 0.94 |
| layers 33–34 | `" called"` 0.38 / 0.34 | `" called"` 0.78 / 0.55 |
| final prediction | `" the"` 0.45 (then `" not"` 0.07) | `" the"` 0.35 (then `" called"` 0.07) |
| huge neuron, layer 6 | neuron 5723, 428 | neuron 5723, 376 |
| huge neuron, last layer | neuron 9510, 653 | neuron 9510, 1003 |

- **The answer is found in Chinese.** Both models know the star is the Sun by layer 29, and the logit
  lens shows it mostly as `"太阳"` (Chinese for "sun") with `" Sun"` second. Earlier layers have
  `"叫做"` ("is called") from layer 11 on (and `" called"` in Qwen3), then a run of blanks (`" ____"`, up to 0.98)
  from layer 21 in Qwen3 and layer 17 in DeepSeek: the models treat the sentence as a fill-in-the-blank.
- **The last layers choose the wording, not the fact.** Layers 32–34 move away from the answer to
  `" called"` and blanks, and the last layer settles on `" the"`: the next words are "the Sun".
- **Shared circuits:** the most active neurons have the same indices layer after layer (11629 in
  layer 1, 5723 in layer 6, 9510 in the last layer), and both models make their internal state about
  17–19× larger at layer 6 (|h| 47.7 → 926.7 and 47.0 → 810.8) with the same single neuron.
- **DeepSeek is less sure in English.** It reaches the answer at the same layer, but its final
  distribution is flatter (" the" 0.35 vs 0.45) and it keeps more weight on " called", `":\n"` and
  `"?\n"`. A reasoning model expects a chat template, not a raw sentence.

For comparison, llama3.2:1b never finds the answer on this prompt: its lens reaches `" Earth"`,
`" planets"` and `" called"` (0.39 at layer 14), and it continues "…is not a fixed or constant name"
(see [Trace](trace.md)).

### What happens if you cut a layer

`ablate --cut <layer> --prompt "The name of the star that Earth orbits is" --tokens 0` (in memory,
nothing written):

| Cut | qwen3:8b | deepseek-r1 |
|---|---|---|
| layer 6 (huge neuron) | KL **9.20**, answer becomes `"<\|endoftext\|>"` 0.13 and noise (`"_CID"`, `"BitFields"`) | KL **6.37**, answer becomes noise (`"Earth"` 0.001, `"مسلم"` 0.001) |
| layer 20 | KL 0.11, `" the"` **0.61** (more certain) | KL 0.22, `" the"` 0.45, `" Sol"` appears (0.04) |
| layer 35 (last) | KL 1.20, top-1 becomes `" called"` 0.34 | KL 2.36, top-1 becomes `" called"` 0.55 |

- **Layer 6 is essential** in both models: without it nothing sensible comes out.
- **Middle layers are cheap to lose**, and DeepSeek's cost is again about twice Qwen3's. Cutting
  layer 20 even makes both models more confident in " the".
- **The last layer does the wording:** without it, both models stop at layer 34's `" called"`
  (exactly what the logit lens shows for layer 34), so the last layer is what turns "is called…" into
  "is the Sun".

To look deeper at one of these decisions (which heads and neurons push " the" over " called"), run
`neurondump journey qwen3:8b --prompt "The name of the star that Earth orbits is"`, or open the
**journey** page; [Interventions](interventions.md) shows how to change a layer or a head and compare.

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
