#!/usr/bin/env sh
# Builds the npm tarballs for a GitHub release, ready to `npm publish` by hand. Nothing is published.
# The binaries are the release assets (checked against their .sha256), repacked as
# @aymen94/neurondump-windows, -macos (arm64 + x64) and -linux from npm/os, plus the
# `neurondump` package from npm/neurondump, which depends on all three and runs the matching one.
#
# The OS tarballs are written with tar (mode 755) instead of `npm pack`, so the Unix binaries
# stay executable even when packing on Windows.
#
# Needs: curl, tar, unzip, node, npm.
set -eu

VERSION=${1:?usage: scripts/npm-pack.sh <version>   e.g. 0.1.4}
VERSION=${VERSION#v}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/target/npm"
URL="https://github.com/aymen94/NeuronDump/releases/download/v$VERSION"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# Downloads and checks one release asset, then puts its binary at $2 (a path in a package).
fetch() {
  target=$1 dest=$2
  case $target in *windows*) asset="neurondump-$target.zip" exe=neurondump.exe ;;
                  *) asset="neurondump-$target.tar.gz" exe=neurondump ;; esac
  echo "downloading $asset"
  (cd "$TMP" && curl -fsSLO "$URL/$asset" && curl -fsSLO "$URL/$asset.sha256" && sha -c "$asset.sha256" >/dev/null)
  mkdir -p "$TMP/x/$target"
  case $asset in *.zip) unzip -q "$TMP/$asset" -d "$TMP/x/$target" ;;
                 *) tar -xzf "$TMP/$asset" -C "$TMP/x/$target" ;; esac
  mkdir -p "$(dirname "$dest")"
  mv "$TMP/x/$target/neurondump-$target/$exe" "$dest"
}

# Writes the package.json, README and LICENSE of one OS package, then its tarball.
pack_os() {
  name=$1 label=$2 os=$3 cpu=$4 extra=${5:-}
  pkg="$TMP/$name/package"
  cp -R "$ROOT/npm/os/." "$pkg/"
  cp "$ROOT/LICENSE" "$pkg/"
  cat > "$pkg/package.json" <<EOF
{
  "name": "@aymen94/neurondump-$name",
  "version": "$VERSION",
  "description": "neurondump for $label: reverse-engineer, analyze and slim down local GGUF LLMs",
  "license": "MIT",
  "homepage": "https://aymen94.github.io/NeuronDump/",
  "repository": {
    "type": "git",
    "url": "git+https://github.com/aymen94/NeuronDump.git"
  },
  "main": "index.js",
  "bin": {
    "neurondump": "bin/neurondump.js"
  },
  "os": ["$os"],
  "cpu": [$cpu]$extra
}
EOF
  printf '# @aymen94/neurondump-%s\n\n[neurondump](https://www.npmjs.com/package/neurondump) for %s.\n\n```sh\nnpm install -g @aymen94/neurondump-%s\nneurondump list\n```\n\nThe `neurondump` package installs the right one for your system automatically.\n' \
    "$name" "$label" "$name" > "$pkg/README.md"
  (cd "$TMP/$name" && tar -czf "$OUT/aymen94-neurondump-$name-$VERSION.tgz" \
    --owner=0 --group=0 --numeric-owner --mode=755 package)
}

rm -rf "$OUT"
mkdir -p "$OUT"

fetch x86_64-pc-windows-msvc "$TMP/windows/package/bin/neurondump.exe"
fetch aarch64-apple-darwin "$TMP/macos/package/bin/arm64/neurondump"
fetch x86_64-apple-darwin "$TMP/macos/package/bin/x64/neurondump"
fetch x86_64-unknown-linux-gnu "$TMP/linux/package/bin/neurondump"

pack_os windows "Windows x64" win32 '"x64"'
pack_os macos "macOS (Apple Silicon and Intel)" darwin '"arm64", "x64"'
pack_os linux "Linux x64 (glibc)" linux '"x64"' ',
  "libc": ["glibc"]'

echo "packing the neurondump package"
cp -R "$ROOT/npm/neurondump" "$TMP/main"
cp "$ROOT/LICENSE" "$TMP/main/"
node -e '
  const fs = require("fs"), file = process.argv[1], v = process.argv[2];
  const pkg = JSON.parse(fs.readFileSync(file, "utf8"));
  pkg.version = v;
  for (const dep of Object.keys(pkg.optionalDependencies)) pkg.optionalDependencies[dep] = v;
  fs.writeFileSync(file, JSON.stringify(pkg, null, 2) + "\n");
' "$TMP/main/package.json" "$VERSION"
(cd "$TMP/main" && npm pack --silent --pack-destination "$OUT" >/dev/null)

echo
ls -l "$OUT"
