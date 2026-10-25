#!/usr/bin/env sh
# Vendors wamr3d (npm package `wamr3d`) into the embedded web assets, so the web UI works
# offline with no CDN. Run from anywhere; needs npm and tar. The version is pinned:
#   scripts/vendor-wamr3d.sh            # the pinned version
#   WAMR3D_VERSION=0.9.4 scripts/vendor-wamr3d.sh
set -eu

VERSION="${WAMR3D_VERSION:-0.9.4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/crates/neurondump-web/frontend/vendor/wamr3d"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "packing wamr3d@$VERSION"
(cd "$TMP" && npm pack "wamr3d@$VERSION" --silent >/dev/null)
tar -xzf "$TMP"/wamr3d-"$VERSION".tgz -C "$TMP"
SRC="$TMP/package"
[ -f "$SRC/wamr3d_wasm_bg.wasm" ] || { echo "error: the package has no wamr3d_wasm_bg.wasm" >&2; exit 1; }

rm -rf "$DEST"
mkdir -p "$DEST/addons/controls"
# Runtime only: the JS entry points, the engine bindings and the .wasm. No typings.
for f in package.json index.js compat.js math.js webgl.js webgpu.js wamr3d_wasm.js wamr3d_wasm_bg.wasm; do
  cp "$SRC/$f" "$DEST/$f"
done
for d in compat webgl webgpu; do
  cp -R "$SRC/$d" "$DEST/$d"
done
cp "$SRC/addons/controls/OrbitControls.js" "$DEST/addons/controls/"
cp -R "$SRC/addons/controls/orbit" "$DEST/addons/controls/orbit"
find "$DEST" -name '*.d.ts' -delete

cat > "$DEST/../README.txt" <<EOF
Vendored so the web interface works offline (no CDN):

- wamr3d/ — wamr3d $VERSION (Rust/WebAssembly 3D engine with a three.js-style API, WebGPU with
  an automatic WebGL 2 fallback), MIT, see wamr3d/package.json. Refresh with
  scripts/vendor-wamr3d.sh (pins the version; WAMR3D_VERSION=0.9.4 to change it).
EOF
echo "vendored wamr3d $VERSION into $DEST ($(du -sh "$DEST" | cut -f1)); rebuild to embed it: cargo build --release"
