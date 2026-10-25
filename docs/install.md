# Install and run

- [Windows](#windows)
- [Linux](#linux)
- [Optional tools](#optional-tools)
- [Problems](#problems)

NeuronDump is built from source with Rust (1.88 or newer). The first build downloads the
dependencies and takes a few minutes; after that it is under a minute.

Tested on Windows 11 (GNU Rust toolchain) and Ubuntu 26.04 (under WSL2): build, `cargo install`,
`list`, `inspect`, `vocab --tokenize`, `stats`, `trace`, `analyze`, `slim`, the REPL and the web
server. The Windows MSVC toolchain and other Linux distributions use the standard steps below
but have not been tested yet.

---

## Windows

### 1. Install the tools

Open **PowerShell** and run:

```powershell
winget install Git.Git
winget install Rustlang.Rustup
```

Close and reopen PowerShell, then check:

```powershell
cargo --version
```

Rust on Windows uses Microsoft's C++ build tools. If `rustup` offers to install the
"Visual Studio prerequisites", accept. If a build later fails with `link.exe not found`,
install them yourself:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

> Using the GNU toolchain (`x86_64-pc-windows-gnu`) instead? See the extra step in
> [building.md](building.md#windows).

### 2. Build

```powershell
git clone https://github.com/aymen94/NeuronDump
cd NeuronDump
cargo build --release
```

To also run models (the `trace` command), build with `cargo build --release --features trace`.

### 3. Run

From the project folder:

```powershell
.\target\release\neurondump.exe list
```

Or install it so `neurondump` works from any folder:

```powershell
cargo install --path crates/neurondump              # add --features trace if you want it
neurondump list
```

### 4. First commands

```powershell
neurondump list
neurondump inspect llama3.2:1b
neurondump vocab llama3.2:1b --search 'hello'
neurondump web --open        # opens your browser
neurondump                   # asks: CLI or web
```

PowerShell notes:
- Put text with spaces or symbols in **single quotes**: `--prompt 'The capital of France is'`.
- Environment variables: `$env:NEURONDUMP_UI = "web"`, `$env:OLLAMA_MODELS = "D:\ollama\models"`.
- Settings are saved in `%APPDATA%\neurondump\config.toml`, and results in `%USERPROFILE%\neurondump\`.

### Where are my models?

NeuronDump finds Ollama models in `%USERPROFILE%\.ollama\models` automatically. If you moved
them, set `OLLAMA_MODELS` the same way you did for Ollama, or pass the folder:

```powershell
neurondump --scan D:\ollama\models list
```

---

## Linux

The steps are the same on every distribution except the first command.

### 1. Install the tools

```sh
# Debian / Ubuntu
sudo apt install build-essential git curl
# Fedora
sudo dnf install gcc git curl
# Arch
sudo pacman -S base-devel git curl
```

Then install Rust:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
cargo --version
```

If you already have Rust, make sure it is recent: `rustup update`.

### 2. Build

```sh
git clone https://github.com/aymen94/NeuronDump
cd NeuronDump
cargo build --release
```

To also run models: `cargo build --release --features trace`.

### 3. Run

```sh
./target/release/neurondump list
```

Or install it into `~/.cargo/bin` (already on your `PATH` after installing Rust):

```sh
cargo install --path crates/neurondump              # add --features trace if you want it
neurondump list
```

### 4. First commands

```sh
neurondump list
neurondump inspect llama3.2:1b
neurondump vocab llama3.2:1b --search hello
neurondump web --open
neurondump
```

Settings are saved in `~/.config/neurondump/config.toml`, and results in `~/neurondump/`.

### Where are my models?

NeuronDump looks in all of these:

| Ollama install | Models folder |
|---|---|
| desktop / user install | `~/.ollama/models` |
| official install script (system service) | `/usr/share/ollama/.ollama/models` |
| some packages | `/var/lib/ollama/.ollama/models` |
| custom | whatever `OLLAMA_MODELS` points to |

With the system-service install, the files belong to the `ollama` user. If `neurondump list`
shows nothing or says "permission denied", give yourself read access:

```sh
sudo usermod -aG ollama "$USER"     # then log out and back in
```

### Web interface on a server

`neurondump web` listens only on the machine itself. To use it from your laptop, don't open
it to the network; forward the port over SSH instead:

```sh
ssh -L 7878:127.0.0.1:7878 you@server      # on your laptop
neurondump web                             # on the server
```

Then open the printed `http://127.0.0.1:7878/?token=…` link in your laptop's browser.
On a machine without a desktop, `--open` cannot start a browser. Copy the link instead.

### WSL (Linux inside Windows)

Follow the Linux steps. Ollama usually runs on the Windows side, so `neurondump list` inside WSL
finds nothing at first. Point it at your Windows models:

```sh
neurondump --scan /mnt/c/Users/<windows-user>/.ollama/models list
# or remember it:
neurondump config set scan_dirs /mnt/c/Users/<windows-user>/.ollama/models
```

Model ids are the same as on Windows. Reading from `/mnt/c` is slower: `stats llama3.2:1b`
took 8 s there, against about 3 s natively.

---

## Optional tools

| Tool | Used for | Install |
|---|---|---|
| [Ollama](https://ollama.com/download) | getting models, running slimmed models | Windows: `winget install Ollama.Ollama` · Linux: `curl -fsSL https://ollama.com/install.sh \| sh` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-tokenize`, `llama-completion`, `llama-perplexity` (comparisons, `slim --perplexity`) | Windows: `winget install ggml.llamacpp` · Linux: build from source or use your package manager |

Get a first model to look at:

```sh
ollama pull llama3.2:1b
```

---

## Problems

| Message | Fix |
|---|---|
| `cargo: command not found` / not recognized | reopen the terminal; on Linux run `source "$HOME/.cargo/env"` |
| `linker 'cc' not found` (Linux), or rustup warns `no default linker (cc) was found` | install `build-essential` / `gcc` (step 1), then build again |
| `link.exe not found` (Windows) | install the Visual Studio Build Tools (step 1) |
| `dlltool.exe: program not found` (Windows) | you are on the GNU toolchain: see [building.md](building.md#windows) |
| `package requires rustc 1.88` | `rustup update` |
| `list` shows no models | check where Ollama keeps them and use `OLLAMA_MODELS` or `--scan <folder>`; in WSL see [WSL](#wsl-linux-inside-windows) |
| `trace support is not compiled in` | rebuild with `--features trace` |
| trace is very slow or runs out of memory | it runs on the CPU and loads the whole model: use a smaller model |
