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
  neurondump/         the executable
schemas/              JSON Schemas of every output
docs/                 this documentation + the GitHub Pages site
```

## Documentation

The docs are the Markdown files in `docs/`, built with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Every push to `main`
that changes them is published to GitHub Pages by `.github/workflows/docs.yml`.
To preview locally:

```sh
pip install -r docs/requirements.txt
mkdocs serve          # http://127.0.0.1:8000, reloads on save
```
