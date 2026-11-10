# Inside Gemma 3 1B

A deep look at `gemma3:1b` (Ollama, Q4_K_M, 768 MiB) with every NeuronDump probe: where it keeps a
fact, which heads copy text, what its layers do, and what its tokenizer hides. Every number below
comes from the commands in [Reproduce](#reproduce), run on CPU.

## The shape: one KV head, a huge vocabulary, mostly local attention

```sh
neurondump inspect gemma3:1b
```

| | |
|---|---|
| layers | 26, width 1152, FFN 6912 (GELU, gated) |
| attention | **4 query heads sharing 1 KV head**, head size 256 |
| vocabulary | **262,144 tokens**, embeddings tied with the output |
| parameters | 1.0B, of which **302M (30%) are the embedding table** |
| attention span | 22 layers see only the last **512 tokens** (sliding window, RoPE base 10,000); layers **5, 11, 17, 23** are global (RoPE base 1,000,000) |

That last row matters for everything that follows: the model alternates five local layers with one
global layer, and the global layers turn out to do most of the interesting work.

### Memory

```sh
neurondump memory gemma3:1b
```

| context | KV cache f16 | total f16 | total q4_0 cache |
|---|---|---|---|
| 8,192 | 43 MiB | 805 MiB | 775 MiB |
| 32,768 (trained) | 139 MiB | 901 MiB | 802 MiB |
| 131,072 | 523 MiB | 1.26 GiB | 910 MiB |

One KV head and a 512-token window on 22 of 26 layers make the cache tiny: 1 KiB per token per
global layer, and the local layers stop growing at 512 tokens. Counted naively (every layer at full
context) the cache at 32K would be 832 MiB, six times more. (llama.cpp's `--swa-full` turns the
saving off.)

## Before measuring: the forward pass had to be fixed

NeuronDump's runtime used Gemma 2's settings for Gemma 3: one RoPE base for every layer and no
sliding window. Against Ollama on "The name of the star that Earth orbits is", the text matched but
the probabilities did not:

| token | Ollama | NeuronDump before | NeuronDump after the fix |
|---|---|---|---|
| " the" | 0.163 (runner-up " called") | 0.103 (runner-up ":") | **0.186** (runner-up " called") |
| " **" | 0.452 | 0.223 | **0.449** |
| "Sun" | 0.877 | 0.648 | **0.887** |

The fix gives the local layers their own RoPE base (`rope.freq_base_swa`, 10,000) and the 512-token
window mask, in llama.cpp's layout (five local layers, then one global). The remaining gap on the
first token is the usual one from quantized activations. All results below use the fixed pass.

## Where "the Eiffel Tower is in Paris" lives

Causal tracing restores the clean run's residual stream, one layer and one token at a time, inside
the run of a corrupted prompt ("The **Colosseum** is located in the city of", which answers
" Rome" 0.987):

```sh
neurondump causal gemma3:1b --prompt "The Eiffel Tower is located in the city of" \
  --corrupt "The Colosseum is located in the city of"
```

```
target " Paris" — clean 0.973, corrupted 0.000
         0 1 2 3 4 5 6 7 8 9
  L0        ▓
  L1        █ █
  L5        █ █
  L10       ░ █
  L14         █
  L15         ░
  L16
  L17                     █
  L25                     █
tokens: 0:"<bos>" 1:"The" 2:" Col" 3:"osseum" 4:" is" 5:" located" 6:" in" 7:" the" 8:" city" 9:" of"
```

- **Layers 0–14:** the fact rides on the subject's tokens. Restoring the last subject token
  ("osseum" / " Tower") alone brings " Paris" back fully (effect 1.00 at layer 11).
- **Layer 16:** no single token is enough; the information is spread over several positions.
- **Layer 17 on:** it sits on the last token, " of". Layer 17 is a **global** layer.

The token journey shows the same handover from the inside:

| layer | logit lens at " of" | P(" Paris") |
|---|---|---|
| 15 | " " 0.066 | 0.001 |
| 16 | " " 0.730, " Paris" third | 0.008 |
| **17** | **" Paris" 0.930**, " France" 0.049 | **0.930** |
| 25 | " Paris" 0.973 | 0.973 |

At layer 17 the last token attends back to " Eiffel" (0.17) and " Tower" (0.05), and that layer's FFN
gives the largest single push of the whole decision: **+6.48 logits** for " Paris" over the
runner-up " **". Restoring only layer 17's attention output at the last token brings back 41% of the
answer (`--component attn`); no other single block does more than 6%, so the fact is not stored in
one place. It is moved, at one global layer.

Two neurons from the decision breakdown, read with the neuron explorer:

```sh
neurondump neuron gemma3:1b 17 4949
neurondump neuron gemma3:1b 25 3311
```

| neuron | push on " Paris" | reads | promotes |
|---|---|---|---|
| L17 N4949 (activation +7.0) | **+3.44** | " francês", " französischen", " Parisian", " CNRS", "法国" | " France", " French", "法國", " 프랑" |
| L25 N3311 (activation −10.8) | **−6.12** | " francês", " француз", " francés", "France" | " French", " France", " Francia", "法国" |

Both are **"French" neurons**, multilingual on both sides. The layer 17 one fires positively and
pushes the answer toward France. The one in the last layer fires *negatively*, so it subtracts
French-ness at the end and takes 6 logits back from " Paris" (it is the largest negative push in the
breakdown). The model ends at 0.973 instead of 0.99 at layer 20: the last layers temper the
prediction.

The biggest single head is local: **L20 H3** adds +9.7 logits on its own, a head with no pattern on
random tokens (sink 0.37, self 0.22). It amplifies an answer already in place rather than finding it.

## Seven induction heads, all in global layers

```sh
neurondump heads gemma3:1b --samples 4
```

```
mixed: 65 · sink: 23 · induction: 7 · previous-token: 4 · duplicate-token: 3 · self: 2
L11 H1 induction 0.88 · L5 H2 induction 0.87 · L17 H0 induction 0.86 · L5 H1 0.64 · L5 H3 0.58 · L17 H3 0.47 · L17 H2 0.33
L2 H3 previous-token 0.68 · L3 H1 duplicate-token 0.65 · L1 H2 duplicate-token 0.62
```

**All 7 induction heads are in the global layers 5, 11 and 17.** The test sequences are 49 tokens,
well inside the 512-token window, so the local layers could see the earlier copy; they just do not do
this job. The strongest previous-token and duplicate-token heads (L2 H3, L3 H1, L1 H2) sit before
them, in layers 1–3, as in the classic two-step circuit. Global layers also lean harder on the first
token (mean sink 0.54 against 0.36 for local layers).

Do the induction heads really do the copying? Zero them and ask the model to repeat a list:

```sh
neurondump edit gemma3:1b --apply induction-heads.json --tokens 8 \
  --prompt "pencil orbit lantern fjord cactus marble violin quartz harbor tulip. pencil orbit lantern fjord cactus"
```

| heads zeroed (7 of 104) | next token | greedy text | KL |
|---|---|---|---|
| none | " marble" 0.996 | " marble violin quartz harbor tulip." | — |
| the 7 induction heads | "\n\n" 0.077, " flower" 0.049 — " marble" not in the top 5 | "\n\nThis is a collection of images that" | **4.05 nats** |
| 7 other heads (L1, 3, 7, 9, 13, 19, 21, head 0) | " marble" 0.999 | " marble violin quartz harbor tulip." | 0.011 |

And on the Eiffel prompt the same 7 heads barely matter (" Paris" 0.973 → 0.966, KL 0.005). Copying
and factual recall use different machinery: 6.7% of the heads carry all the in-context copying and
none of the fact.

## What the layers do

```sh
neurondump similarity gemma3:1b --corpus texts.txt
```

| layers | CKA in→out | reading |
|---|---|---|
| 0 | 0.84 (cosine 0.05) | rewrites the embeddings completely (the update is 9.8× the input) |
| 4, **5**, 10, **11** | 0.59, **0.50**, 0.54, **0.44** | the largest changes of geometry, around the first two global layers |
| 12–18 | 0.85–0.94 | steady refinement |
| 19–23 | 0.98–0.99 | almost no change of geometry |
| 25 | 0.85 | the final reshaping for the output |

No layer passes the 0.99 bar, so none is an obvious candidate to cut, but layers 19–23 come closest.
The CKA matrix splits the model into blocks: the embeddings with layers 0–3, a loose middle (4–10),
then 11–16 and a tight block 17–24, with layer 25 on its own. The last two blocks start exactly at
the global layers 11 and 17.

## Weights

```sh
neurondump spectrum gemma3:1b
```

The most compressible matrices are the **attention output projections**: `attn_output` in layers
21, 11, 13, 22, 20, 18, 17, 24, 19 have a stable rank of 8–13 out of 1024 (under 1.3% of full rank).
The largest outliers are in `blk.1.attn_q` and the FFN down projections. Across matrix types, global
and local layers look alike (attn_q 1.7% vs 2.5%, ffn_down 10.2% vs 9.6%), except the key projections:
15.0% of full rank in global layers against 10.3% in local ones.

## The tokenizer: names with small embeddings

```sh
neurondump glitch gemma3:1b --top 25
```

Gemma 3's embedding rows are remarkably uniform (norm 0.946 ± 0.026), so a few stand out sharply.
The 1,172 rows below z −3 are not byte tokens or junk: the most extreme are **character names**,

| token | norm | z |
|---|---|---|
| " Bartholomew" | 0.728 | −8.3 |
| " Anya" | 0.728 | −8.3 |
| " Elias" | 0.739 | −7.9 |
| " Thorne" | 0.740 | −7.9 |
| " Silas", " Reginald", " Finch", " Amelia", " Maya", " Agnes" | 0.76–0.80 | −7.2 … −5.7 |

followed by emoji ("📃", "🐅", "👺", "🚑") and two zero-width characters (U+200D, U+200B). All of them
are reachable (the tokenizer produces them). A small norm with tied embeddings means a smaller logit
when the token is predicted and a weaker signal when it is read. Why these names in particular is not
something the weights can tell; they are names that recur heavily in generated fiction, which is
consistent with an unusual training treatment of exactly those tokens, but that is a hypothesis.

## Summary

- Gemma 3 1B is mostly local attention with a global layer every sixth, and **the global layers do
  the long-range work**: they hold all 7 induction heads (zeroing them destroys copying and leaves
  facts intact), they are where the representation changes most, and **layer 17** is where the
  Eiffel Tower fact moves from the subject to the prediction.
- The fact is not stored in one block: restoring single FFN or attention outputs barely helps,
  restoring the residual stream on the subject token does it fully.
- A pair of multilingual "French" neurons push the answer up at layer 17 and back down at layer 25.
- One KV head and the sliding window keep the KV cache at 139 MiB for the full 32K context.

## Reproduce

```sh
ollama pull gemma3:1b
neurondump inspect gemma3:1b
neurondump memory gemma3:1b
neurondump generate gemma3:1b --prompt "The name of the star that Earth orbits is" --steps 8
neurondump causal gemma3:1b --prompt "The Eiffel Tower is located in the city of" --corrupt "The Colosseum is located in the city of"   # ~2.5 min
neurondump causal gemma3:1b --prompt "The Eiffel Tower is located in the city of" --corrupt "The Colosseum is located in the city of" --component attn
neurondump journey gemma3:1b --prompt "The Eiffel Tower is located in the city of"
neurondump neuron gemma3:1b 17 4949
neurondump neuron gemma3:1b 25 3311
neurondump heads gemma3:1b --samples 4
neurondump similarity gemma3:1b --corpus texts.txt      # 4 short paragraphs
neurondump spectrum gemma3:1b
neurondump glitch gemma3:1b --top 25
```

`induction-heads.json` (the control file zeroes heads 0 of layers 1, 3, 7, 9, 13, 19 and 21):

```json
{"schema_version": 1, "model": "gemma3:1b", "edits": [
  {"id": 1, "op": "zero", "target": "layer.5.attn.head.1"},
  {"id": 2, "op": "zero", "target": "layer.5.attn.head.2"},
  {"id": 3, "op": "zero", "target": "layer.5.attn.head.3"},
  {"id": 4, "op": "zero", "target": "layer.11.attn.head.1"},
  {"id": 5, "op": "zero", "target": "layer.17.attn.head.0"},
  {"id": 6, "op": "zero", "target": "layer.17.attn.head.2"},
  {"id": 7, "op": "zero", "target": "layer.17.attn.head.3"}
]}
```

The same probes are on the web **probes** page; **Color the graph by head role** shows the induction
heads lighting up in layers 5, 11 and 17.
