# Interventions: ablation, edits, steering, patching

Change something inside a model, see what happens, and put it back. Every intervention runs **in
memory**; the original GGUF is never modified. Only **bake** writes anything, and it always writes a
**new** file.

Runtime interventions need a build with `--features trace` and a supported architecture (see
[Limitations](limitations.md)). Static impact and bake work for any model.

## The edit stack

Each model has an edit stack. An edit is one operation with an id, a label and an on/off switch.

| Action | Web (edit panel) | REPL | API |
|---|---|---|---|
| add | **Add edit** form, or the **Edit:** buttons in the graph side panel | `edit …` | `POST /api/models/{id}/edits` |
| switch one off / on | checkbox | `toggle <id>` | `POST …/edits/{n}/toggle` |
| remove one | × | — | `DELETE …/edits/{n}` |
| undo / redo | ↶ / ↷ | `undo`, `redo` | `POST …/edits/undo`, `…/redo` |
| back to the original model | **restore all** | `restore` | `POST …/edits/restore` or `DELETE …/edits` |
| named snapshot | save / load | `snapshot save\|load <name>` | `POST …/snapshots` |
| portable patch | export / import `edits.json` | `edits export\|import <file>` | `GET …/edits/export`, `POST …/edits` with `{"edits": […]}` |
| write a new GGUF | **Bake** | `bake <file>` | `POST …/bake` |

Undo and redo cover every change, including restore and import: **restore all, then undo** brings
the edits back. A change that would not compile (an unknown node, a token outside the vocabulary)
is refused and the stack stays as it was.

The stack lives in the running process and is shared by its front ends: the web UI started with the
REPL `web` command sees the REPL's edits and the other way round. Snapshots are saved to disk
(`<output_dir>/edits/<digest>/<name>.json`) and survive restarts.

## Edit operations

Node ids are the graph's ids (`neurondump graph <model> --expand …` shows them).

| `op` | Fields | Effect at run time |
|---|---|---|
| `scale` | `target`, `factor` | multiplies the node's contribution to the residual stream: a **layer** (both blocks), a **block** (`layer.7.attn`, `layer.7.ffn`), a **KV group** (its heads), a **head** (`layer.7.attn.head.3`), a **neuron bucket** (`layer.7.ffn.n.0-63`), a **single neuron** (`layer.7.ffn.neuron.42`) or an **expert** (bake only) |
| `zero` | `target` | `scale` by 0; also accepts tensors, sections and `vocab:a-b` token ranges (same as `ablate`) |
| `skip` | `layer` | the layer becomes the identity |
| `ablate` | `cut` | exactly the cut of `neurondump ablate --cut` |
| `embed` | `token`, `scale?`, `replace_with?` | the token's input embedding row is scaled, or replaced by another token's row |
| `ban` | `token` | the token can no longer be predicted (logit −∞) |
| `boost` | `token`, `bias` | adds `bias` to the token's logit (negative discourages it) |
| `patch` | `layer`, `head?`, `source`, `all_positions?` | **activation patching**: the layer's output (or one head's attention output) is replaced by the one from the `source` prompt |
| `steer` | `layer`, `a`, `b`, `strength` | **steering vector**: adds `strength × (mean h_layer(a) − mean h_layer(b))` to the residual stream after the layer, at every position |
| `sampling` | `temperature`, `top_k?`, `top_p?`, `seed?` | sampling of the edited side (temperature 0 = greedy) |
| `force` | `step`, `token` | the token chosen at that generation step, whatever the logits say |
| `system` | `text` | replaces the system prompt (empty = none) |
| `template` | `text` | replaces the chat template (Go template syntax, as in Ollama: `{{ .System }}`, `{{ .Prompt }}`, `{{ range .Messages }}`) |

Example `edits.json` (also accepted: a bare list of edits, or a list of ops). Token ids are
vocabulary ids, as shown by `neurondump tokens`; the REPL and the web panel also accept a token's
text and resolve it.

```json
{
  "schema_version": 1,
  "model": "llama3.2:1b",
  "edits": [
    { "id": 1, "enabled": true, "label": "zero layer 7", "op": "zero", "target": "layer.7" },
    { "id": 2, "enabled": true, "op": "steer", "layer": 6, "a": "Dear Sir or Madam, I am writing to formally request", "b": "hey dude, just wanna ask", "strength": 4 },
    { "id": 3, "enabled": false, "op": "ban", "token": 22463 }
  ]
}
```

## How they run

The enabled edits compile to one intervention for the forward pass:

- **Scales and zeros** multiply activations: the head's slice of the attention output before
  `attn_output`, the neuron's activation before `ffn_down`, or the block's output after its
  post-norm. That is mathematically the same as scaling the touched weights, without copying any.
  A factor of 0 also skips computing the block.
- **Embedding edits** patch only the touched rows when the input is embedded. With tied embeddings
  the output head keeps the original rows at run time (bake changes both; see below).
- **Logit edits** are applied to the next-token logits before sampling, the logit lens and the KL.
- **Steering** runs prompts `a` and `b` on the clean model once, takes the mean residual after the
  layer, and adds the scaled difference at every position.
- **Patching** runs `source` on the clean model once and copies its activations, aligned from the
  end of the sequence: the last position only, or every position with `all_positions`.
- **Prompt edits** (`system`, `template`) change the text: prompts are then always wrapped in the
  chat template, on both sides of a comparison, so the only difference is the edit.

## Compare: before and after

Every comparison runs **baseline** (no edits, greedy) and **edited** on the same prompt(s):

- generated text side by side (`tokens` per side);
- top-k next tokens and **KL(baseline ‖ edited)**;
- **perplexity** and mean KL on a corpus, and top-1 agreement;
- the **logit lens** of both sides per layer;
- the **decision breakdown** of both sides and which layers changed their push;
- the changed nodes, outlined in orange in the 2D and 3D graph.

```sh
neurondump edit llama3.2:1b --apply edits.json --prompt "The name of the star that Earth orbits is" --tokens 12
neurondump edit llama3.2:1b --apply edits.json --corpus texts.txt --tokens 0
```

With no enabled edit, both sides are identical (KL 0, same text): that is how restore is checked.

## Ablation is a special case

`neurondump ablate`, the graph's **Cut this** + **Simulate** and sweeps are the zeroing / skipping
subset of the same machinery. In the graph, **Apply as edit** puts the current cut on the edit
stack as an `ablate` edit, where it can be toggled, undone and baked like any other.

## Bake

```sh
neurondump edit llama3.2:1b --apply edits.json --bake llama-edited.gguf --no-compare
# REPL:  bake llama-edited.gguf
# web:   edit panel → Bake (written to <output_dir>/bake/<name>)
ollama create llama-edited -f llama-edited.Modelfile
ollama run llama-edited "What is the capital of Italy?"
```

Bake writes the enabled edits into a **new** GGUF (it refuses to overwrite the source):

| Edit | Baked as |
|---|---|
| scale / zero of a block or layer | scales `attn_output` / `ffn_down` (and biases), or the post-norm weight when the block has one |
| skip a layer | zeroes the layer's block outputs (the layer becomes the identity; the file size stays the same: use `slim` to remove layers) |
| scale a head | scales the head's columns of `attn_output` |
| scale neurons | scales their columns of `ffn_down` |
| scale an expert | scales its slice of `ffn_down_exps` |
| embedding edit | rewrites the token's rows of `token_embd` (with tied embeddings this also changes the output head) |
| ablate / zero of tensors | the same zero operations as `slim` |
| sampling, system prompt, template | written to the Modelfile (`PARAMETER`, `SYSTEM`, `TEMPLATE`) |
| ban, boost, patch, steer, force | **not baked** (they only exist at run time); listed as skipped with the reason |

Only touched tensors are rewritten. Float tensors keep their type; quantized tensors that are scaled
are re-encoded as Q8_0 (reported as requantized); pure zeroing keeps the type when the zeroed ranges
are block-aligned. The result is validated like a slim output, and the edits are stored in the
`neurondump.edits` metadata key.
