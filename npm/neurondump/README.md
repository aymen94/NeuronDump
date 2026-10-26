# neurondump

Look inside any local LLM, find the parts it doesn't need, and make it smaller.

NeuronDump reads **GGUF** model files (the format used by **Ollama** and **llama.cpp**), shows
what is inside (layers, attention heads, vocabulary, chat template, weight statistics) and can
write a smaller GGUF that still runs in Ollama and llama.cpp.

```sh
npm install -g neurondump
neurondump list
```

This installs the prebuilt binary (with the `trace` feature) for your system: Windows x64,
macOS (Apple Silicon or Intel) or Linux x64 (glibc). Other platforms: `cargo install neurondump`.

Documentation: https://aymen94.github.io/NeuronDump/ · Source: https://github.com/aymen94/NeuronDump
