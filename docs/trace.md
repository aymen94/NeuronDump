# Trace

Trace runs the model on your text and records what happens in every layer.
It needs a build with the `trace` feature:

```sh
cargo build --release --features trace
```

## Example

```sh
neurondump trace llama3.2:1b --prompt "The capital of France is"
```

```
LAYER  |h|    COS(in,out)  LOGIT LENS (top-5)
0      7.5    0.2950       " is" 1.00  " was" 0.00 ...
...
10     139.0  0.8812       "olit" 0.08  " capitals" 0.02 ...
11     140.3  0.9093       " Paris" 0.84  "Paris" 0.05  " Brussels" 0.04 ...
15     69.6   0.5596       " Paris" 0.72  " a" 0.04 ...
next token: " Paris" 0.722
```

## The columns

| Column | Meaning |
|---|---|
| `\|h\|` | size of the model's internal state after the layer |
| `COS(in,out)` | how similar the state is before and after the layer. Close to 1 means the layer changes little, which makes it a removal candidate |
| `ATTN ENTROPY` | how spread out attention is (low means each word looks at a few others) |
| `TOP NEURONS` | the feed-forward neurons that fired the most |
| `LOGIT LENS` | what the model would answer if it stopped at this layer |

In the example the model "knows" the answer at layer 11: that's where " Paris" first appears.

## Several texts

```sh
neurondump trace llama3.2:1b --corpus texts.txt     # texts separated by blank lines
```

Values are averaged over all texts. The result is saved, and the next `analyze` uses it to
rank which layers can be removed.

## Embedding models

For models like `nomic-embed-text`, there is no next token. Instead, trace shows how much each
input token contributes to the final embedding.

## Supported architectures

`llama`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `gemma3`, `phi3`, `bert`, `nomic-bert`.

Other models (for example hybrid Qwen 3.5) report "trace unsupported". All the other commands
still work on them. Tracing runs on the CPU and loads the whole model into memory: an 8B
model at 4 bits needs about 5 GB of RAM.
