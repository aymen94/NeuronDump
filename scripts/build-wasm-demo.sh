#!/usr/bin/env sh
# Builds the in-browser demo (static GGUF inspection in WebAssembly) into docs/demo/, which
# GitHub Pages serves with the docs. The page is the regular web UI (crates/neurondump-web/
# frontend) plus crates/neurondump-wasm/www/demo.js, which swaps the HTTP API for the wasm module.
#
# Needs: the wasm32-unknown-unknown target (rustup target add wasm32-unknown-unknown) and
# wasm-bindgen-cli at the version in Cargo.lock (cargo install wasm-bindgen-cli --version X --locked).
# Optional: wasm-opt (binaryen) for a smaller module.
#
# Usage: scripts/build-wasm-demo.sh [out_dir]   (default docs/demo)
set -eu
cd "$(dirname "$0")/.."
OUT=${1:-docs/demo}
FRONTEND=crates/neurondump-web/frontend

want=$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p')
have=$(wasm-bindgen --version 2>/dev/null | cut -d' ' -f2 || true)
if [ "$want" != "$have" ]; then
  echo "wasm-bindgen-cli $want is required (found: ${have:-none}):" >&2
  echo "  cargo install wasm-bindgen-cli --version $want --locked" >&2
  exit 1
fi

CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_PROFILE_RELEASE_LTO=true \
  cargo build --release --locked -q -p neurondump-wasm --target wasm32-unknown-unknown

rm -rf "$OUT"
mkdir -p "$OUT/assets"
cp -R "$FRONTEND/." "$OUT/assets/"
rm "$OUT/assets/index.html"
cp crates/neurondump-wasm/www/demo.js "$OUT/assets/"
wasm-bindgen --target web --no-typescript --out-dir "$OUT/assets/pkg" \
  target/wasm32-unknown-unknown/release/neurondump_wasm.wasm
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Os --enable-bulk-memory --enable-nontrapping-float-to-int \
    -o "$OUT/assets/pkg/neurondump_wasm_bg.wasm" "$OUT/assets/pkg/neurondump_wasm_bg.wasm"
fi
# Same page shell as the server's UI, with relative asset paths and the demo entry point.
sed -e 's#/assets/app.js#assets/demo.js#' -e 's#/assets/#assets/#g' \
  -e 's#<title>NeuronDump</title>#<title>NeuronDump · browser demo</title>#' \
  "$FRONTEND/index.html" > "$OUT/index.html"

echo "demo built in $OUT ($(du -sh "$OUT/assets/pkg/neurondump_wasm_bg.wasm" | cut -f1) wasm)"
