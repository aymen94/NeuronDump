# What's inside real models

Findings from running NeuronDump on common Ollama models. Every number comes from
`inspect`, `stats` or `trace`.

## llama3.2:1b

- Architecture `llama`, 1.24B parameters, Q8_0.
- 16 identical layers, all full attention.
- 32 query heads share 8 key/value heads (grouped-query attention, groups of 4).
- **Tied embeddings**: no separate output matrix. The token embedding table
  (128,256 × 2,048) is 21% of the file.
- 128,256-token vocabulary, of which **248 are unused reserved tokens**.
- Two chat templates: the Jinja one (23 decisions, with tool calling and date injection) and
  Ollama's Go template (15 decisions).
- Trace: the answer " Paris" to "The capital of France is" appears at **layer 11**.
  Layer 1 has one neuron about 190× stronger than the others.
- Trimming the vocabulary to a small text made the file 21.6% smaller, and it still ran in llama.cpp.

## qwen2.5-coder:7b

- `qwen2`, 28 layers, 28 query / 4 key-value heads, mostly Q4_K with some Q6_K.
- Separate input and output embedding matrices (14% of all parameters).
- Bias vectors on the attention projections; fill-in-the-middle (code completion) tokens.
- 399 `[PADnnnnn]` padding tokens fill the table up to 152,064 rows. They are never used, but small.

## qwen3:8b and qwen3-embedding:8b

- `qwen3`, 36 layers, 32 / 8 heads, with an extra normalization on each head's queries and keys.
- The embedding model is the same kind of network. It uses the last token's state as the
  embedding, has no output head and has a slightly smaller vocabulary. The two vocabularies
  overlap 99.8%.

## qwen3.5-0.8b (community model)

- `qwen35`: a **hybrid** model. 18 linear-attention (state-space) layers and 6 full-attention
  layers, always 3 linear then 1 full.
- Also holds a **multi-token-prediction head** (`mtp.*`, used for faster generation) and an
  **image encoder** (`v.*`, 100M parameters), inside the same file.
- 248,320-token vocabulary.
- Layer removal is refused (it would break the 3:1 pattern). Trace isn't supported yet.

## qwen3.8:27b

- Ollama stores it as a **manifest list** with two versions (MLX and GGUF); NeuronDump picks the GGUF one.
- 48 linear + 16 full-attention layers, plus a 65th "next-n prediction" layer.
- The image part is a separate 888 MiB file (a CLIP projector). Omitting it for text-only use
  saves that much.

## nomic-embed-text

- `nomic-bert`: an **encoder** that reads the whole text at once (no left-to-right order).
- 12 layers, F16, mean pooling.
- WordPiece vocabulary of 30,522 tokens, of which **994 are `[unusedN]` placeholders**.
- Its tokenization matches llama.cpp's exactly.
