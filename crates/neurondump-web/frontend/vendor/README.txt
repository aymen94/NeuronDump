Vendored so the web interface works offline (no CDN):

- wamr3d/ — wamr3d 0.9.4 (Rust/WebAssembly 3D engine with a three.js-style API, WebGPU with
  an automatic WebGL 2 fallback), MIT, see wamr3d/package.json. Refresh with
  scripts/vendor-wamr3d.sh (pins the version; WAMR3D_VERSION=0.9.4 to change it).
