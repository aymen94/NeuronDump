# Trace

Trace runs the model on your text and records what happens in every layer.
It needs a build with the `trace` feature:

```sh
cargo build --release --features trace
```

## Example

```sh
neurondump trace llama3.2:1b --prompt "The name of the star that Earth orbits is"
```

```
LAYER  |h|    COS(in,out)  LOGIT LENS (top-5)
0      5.1    0.3262       " is" 1.00  " was" 0.00 ...
...
10     87.2   0.8888       "otope" 0.03  "namen" 0.03  " nicknamed" 0.02 ...
11     88.7   0.9146       " Earth" 0.08  " planets" 0.05  " planet" 0.04  " astronomers" 0.04 ...
13     92.5   0.9408       " Earth" 0.13  "Earth" 0.10  " called" 0.07 ...
14     95.1   0.9079       " called" 0.39  " not" 0.08  " actually" 0.06 ...
15     52.6   0.5122       " not" 0.17  " the" 0.07  " a" 0.07  " called" 0.05 ...
next token: " not" 0.166
```

## The columns

| Column | Meaning |
|---|---|
| `\|h\|` | size of the model's internal state after the layer |
| `COS(in,out)` | how similar the state is before and after the layer. Close to 1 means the layer changes little, which makes it a removal candidate |
| `ATTN ENTROPY` | how spread out attention is (low means each word looks at a few others) |
| `TOP NEURONS` | the feed-forward neurons that fired the most |
| `LOGIT LENS` | what the model would answer if it stopped at this layer |

The same view in the web interface (trace tab), for deepseek-r1 (8B) on the same prompt. The logit
lens shows the Chinese word for "is called" (`"叫做"`) from layer 11, a run of blanks (`" ____"`) from
layer 17, and the answer at layer 29 (`"太阳"` 0.90, Chinese for "sun", then `" Sun"` 0.05). The next
token is `" the"` 0.347, as in "the Sun". More on this in [What's inside real models](models.md).

![Trace page in the web interface](images/model_trace.png)

In the llama example the topic arrives at layer 11 (" Earth", " planets", " astronomers"), and by
layer 14 the model has settled on " called" (0.39). The last layer turns that into " not" ("…is not a
fixed or constant name"): a 1B model does not answer " Sun" here. Larger models do; see
[What's inside real models](models.md).

llama3.1:8b gets there: `" Sun"` is top-1 from layer 21 (0.45) to layer 26. The last layers then
prefer `" called"` (0.336, then `" the"` 0.159 and `" Sol"` 0.144), so the model writes "…is called
the Sun" rather than answering in one word.

![Trace of llama3.1:8b on the same prompt](images/model_trace2.png)

## One token in depth

`trace` shows the last position. To follow **any** token, see the attention of every head and the
decision breakdown (which layers, heads and neurons pushed the prediction), use
`neurondump journey` and the web **journey** / **timeline** pages ([Web interface](web-ui.md#journey),
[CLI](cli.md#journey-model-prompt-text)). To change the model and compare, see [Interventions](interventions.md).

## Several texts

```sh
neurondump trace llama3.2:1b --corpus texts.txt     # texts separated by blank lines
```

Values are averaged over all texts. The result is saved, and the next `analyze` uses it to
rank which layers can be removed.

## Embedding models

For embedding models like `all-minilm`, there is no next token. Instead, trace shows how much each
input token contributes to the final embedding.

## Supported architectures

`llama`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `gemma3`, `phi3`, `bert`, `nomic-bert`.

Checked against llama.cpp / Ollama on the same files: llama, qwen2, qwen3 and gemma give the same
tokens and nearly the same probabilities; gemma2's probabilities are too sharp (soft-capping is not
applied) and Phi-3 128k models drift (LongRoPE is not applied). Details in
[Gemma 4, Gemma, Gemma 2 and Phi-3](new-models.md).

Other models (for example hybrid Qwen 3.5) report "trace unsupported". All the other commands
still work on them. Tracing runs on the CPU and loads the whole model into memory: an 8B
model at 4 bits needs about 5 GB of RAM.
