# What's inside real models

Findings from running NeuronDump on common Ollama models. Every number comes from
`inspect`, `layers`, `vocab`, `template`, `compare`, `analyze` or `trace`.

## deepseek-r1 (8B)

- `deepseek-r1:latest` is **DeepSeek R1 0528 Qwen3 8B**: architecture `qwen3`, 8.19B parameters,
  Q4_K_M, 36 identical full-attention layers, 32 query / 8 key-value heads.
- **It is Qwen3 8B underneath.** `compare deepseek-r1 qwen3:8b` finds the same 399 tensor names,
  shapes and types, and the same 151,936-token vocabulary. Only 9 metadata keys differ.
- Context is stretched from 40,960 to **131,072 tokens** with YaRN (scaling factor 4, from an
  original 32,768).
- The vocabulary differs in 4 tokens. Two are renamed (`<|endoftext|>` →
  `<｜begin▁of▁sentence｜>`, `<|im_end|>` → `<｜end▁of▁sentence｜>`). Two padding slots
  (151669, 151670) become `<｜User｜>` and `<｜Assistant｜>`. That leaves 265 unused `[PADnnnnn]` tokens.
- The chat template **drops earlier reasoning**: for past assistant messages it keeps only the
  text after `</think>`.
- The template also emits tool-call markers (`<｜tool▁calls▁begin｜>` and others) that are **not in
  the vocabulary**. Each one is split into 11 ordinary tokens.
- Weights are mostly Q4_K, but every `attn_v` matrix is kept in F16 (288 MiB). `analyze` suggests
  Q8_0 for those, which would save 135 MiB.
- Trace: for "The capital of France is", " Paris" first appears at **layer 27** and reaches
  1.00 at layer 31. The last layer then changes the answer to " a" (0.34).
- At layer 6 a single neuron (5723, value 676) pushes the internal state from 45 to 1,425 in size,
  about 30× larger. The last layer has a similar neuron (9510, value 687).

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
