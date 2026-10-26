# Building

```sh
cargo build --release                      # CLI + web interface (default)
cargo build --release --features trace     # + running models (candle, CPU)
cargo build --release --no-default-features --features cli   # CLI only, no web server
```

Requires Rust 1.88 or newer (`rustup update` if needed).

## Windows

**MSVC toolchain** (the rustup default with Visual Studio Build Tools): nothing special.

**GNU toolchain** (`x86_64-pc-windows-gnu`) without a full MinGW install: some dependencies
need a tool called `dlltool` that isn't included. To fix it:

1. Install the LLVM tools:
   ```sh
   rustup component add llvm-tools
   ```
2. Copy `llvm-ar.exe` to `.cargo/tools/llvm-dlltool.exe` in this repository. It is in
   `~/.rustup/toolchains/<your gnu toolchain>/lib/rustlib/x86_64-pc-windows-gnu/bin/`.
   (`llvm-ar` behaves as `dlltool` when renamed.)
3. Copy `.cargo/config.windows-gnu.example.toml` to `.cargo/config.toml` and replace `<REPO>`
   and `<TOOLCHAIN>` with your paths.

Why the extra linker setting: with `llvm-dlltool`, two different crates must not both
generate import libraries for the same Windows DLL, or the program crashes on start. That's
also why `comfy-table` is built without its `tty` feature. To check, run:

```sh
cargo tree -i windows-link      # only windows-sys 0.61 should appear
```

## Project layout

```
crates/
  neurondump-gguf/    GGUF reader, dequantization, writer
  neurondump-core/    all logic
  neurondump-trace/   model execution (feature "trace")
  neurondump-cli/     command line + REPL
  neurondump-web/     web server; frontend/ holds the HTML/JS, embedded in the binary
  neurondump-wasm/    static inspection in WebAssembly for the browser demo; www/demo.js
  neurondump/         the executable
schemas/              JSON Schemas of every output
docs/                 this documentation + the GitHub Pages site
```

## Publishing to crates.io

All crates share the version in the root `Cargo.toml` (`[workspace.package]`). The internal
dependencies in `[workspace.dependencies]` have both a `path` and a `version`; bump both when
you release. Then, from the repository root:

```sh
cargo login                     # once, with a token that has publish-new and publish-update
cargo publish --workspace       # publishes in dependency order
```

The order is `neurondump-gguf` → `neurondump-core` → `neurondump-cli` / `neurondump-trace` /
`neurondump-web` → `neurondump`. To publish one crate at a time, use `cargo publish -p <crate>`
in that order.

| Message | Fix |
|---|---|
| `dependency ... does not specify a version` | add `version = "..."` next to `path` in `[workspace.dependencies]` |
| `no matching package named ... found` / `timed out waiting for ... to be available` | crates.io hasn't indexed the previous crate yet: wait a few minutes and run `cargo publish -p <next crate>` |
| `429 Too Many Requests` | crates.io limits how many new crates (or versions) you can publish in a short time: wait until the time in the message |
| `403 Forbidden: authentication failed` | the token is wrong, expired or truncated: create a new one and run `cargo login` again |
| `dlltool.exe: program not found` while verifying (Windows GNU) | set up `llvm-dlltool` as in [Windows](#windows), or add `--no-verify` |

## Documentation

The docs are the Markdown files in `docs/`, built with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Every push to `main`
that changes them is published to GitHub Pages by `.github/workflows/docs.yml`.
To preview locally:

```sh
pip install -r docs/requirements.txt
mkdocs serve          # http://127.0.0.1:8000, reloads on save
```

## Browser demo

The [browser demo](https://aymen94.github.io/NeuronDump/demo/) is the regular web UI running on a WebAssembly
build of the static part of `neurondump-core` (GGUF parsing, metadata, tensors, architecture,
graph, vocabulary, chat template, per-tensor weight statistics). `neurondump-core` is built
with `default-features = false`, which leaves out the `native` feature: the app / job system,
config, analysis, slimming, edits, ablation, diff, and anything that opens files. The trace
runtime (`neurondump-trace`) is not included.

The page reads the dropped file with `Blob.slice`: the header first (8 MiB, grown while the
header does not fit), then a single tensor's bytes when you ask for its statistics (up to
512 MiB). Nothing is uploaded.

Build it into `docs/demo/` (ignored by git; the docs workflow builds it before `mkdocs`):

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <version of wasm-bindgen in Cargo.lock> --locked
scripts/build-wasm-demo.sh
mkdocs serve          # then open http://127.0.0.1:8000/demo/
```

Any static file server works too (`python -m http.server -d docs/demo`); the page must be served
over HTTP, not opened as a `file://` URL. `wasm-opt` (binaryen) is used when installed.

