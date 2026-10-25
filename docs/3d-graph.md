# 3D graph

The graph page draws the model with **wamr3d**, a Rust/WebAssembly 3D engine with a three.js-style
API. NeuronDump uses its object API (`wamr3d/compat`: scene, camera, instanced meshes, line
segments, raycaster) and its `OrbitControls` addon. Everything runs in the browser; the server only
sends the graph structure (`/api/models/{id}/graph`) and, for the embedding map, the PCA
coordinates.

## 2D / 3D toggle

| View | Renderer | Layout |
|---|---|---|
| **2D** | `graph3d.js` | flat diagram read top to bottom, seen straight on (pan + zoom, no rotation) |
| **3D · dataflow** (default) | `flow3d.js` | the dataflow column described below, with the token flow animation |
| 3D · tower / city / force | `graph3d.js` | the earlier 3D layouts: stacked floors, a skyline of bars, a force layout |

Both renderers use the graph's node ids, so the selection, cuts, edit marks, search matches and the
token-path highlight are shared. A hidden node (a head inside a collapsed layer, a tensor touched by
an edit) is shown on its closest visible ancestor. The choice is remembered; `#/model/<id>/graph?view=2d`
or `?view=flow|tower|city|force` forces one.

## Backend: `auto`

Renderers are created with `new WebGPURenderer({ backend: 'auto' })`: **WebGPU** when the browser has
it, otherwise wamr3d's **WebGL 2** renderer (or the browser's software WebGPU adapter). The active
backend is shown next to the toggle (`WebGPU`, `WebGPU (software)` or `WebGL 2`), from
`renderer.backend.name`. If neither API exists, the canvas shows a message saying so instead of the
graph.

To test the fallback, add `backend=webgl` to the page's query (`#/model/<id>/graph?backend=webgl`;
`backend=webgpu` forces WebGPU), or use a browser without WebGPU.

!!! note "WebGPU needs a secure context"
    Browsers only expose WebGPU on HTTPS pages and on `localhost` / `127.0.0.1`. The default
    `neurondump web` address (`http://127.0.0.1:7878`) qualifies. With `--host 0.0.0.0`, a page
    opened from another device over plain HTTP is not a secure context: WebGPU is unavailable and
    the `auto` backend uses WebGL 2. Everything still works.

## Offline: vendored wamr3d

wamr3d's dist files (the JS modules and `wamr3d_wasm_bg.wasm`) are vendored in
`crates/neurondump-web/frontend/vendor/wamr3d/` and embedded in the binary, so no CDN is needed.
`scripts/vendor-wamr3d.sh` refreshes them with `npm pack` at a pinned version (0.9.4;
`WAMR3D_VERSION=x.y.z` to change it). The server sends `.wasm` files as `application/wasm`, which
`WebAssembly.instantiateStreaming` requires.

## Dataflow layout

Computed in `flow3d.js` from the visible nodes, deterministically (the same view always gives the
same picture):

- **Vertical axis = dataflow:** Input at the bottom, then layer 0 … N up the column (4.2 units
  apart), then Output.
- **Side sections** (tokenizer, chat template, vision projector, in-file vision encoder) sit beside
  the input, the MTP head and unrecognized tensors beside the output.
- **Blocks around their layer** when it is expanded: the mixer (attention or SSM) to the left and
  slightly below, the FFN (or MoE) to the right and slightly above, the residual adds on the
  column, anything else in front.
- **Fine nodes** (KV groups, heads, neuron buckets, experts) on a ring around their block, or a grid
  when there are more than 48; a block's tensors on a smaller ring above it.
- **Group-by views** put each group in its own column.
- **Sizes** follow parameters (cube root, relative to the same kind); containers (expanded layers,
  blocks, sections) are drawn as outlines around their children.

## Rendering

- **One `InstancedMesh` per node kind** (boxes for sections / layers / blocks / tensors, spheres for
  heads / KV groups / buckets / experts / tokens, octahedra for ops), with per-instance colors. A
  fully expanded large model is a handful of draw calls, not one per node.
- **Edges** are two batched `LineSegments` (dataflow, and the fainter residual / uses edges), colored
  from source to target to show direction; tied weights in red.
- **Container outlines** and **marks** are batched line segments: blue = selected or a search match,
  red = cut, orange = changed by an edit, purple = on the selected token's path.
- **Color by** removable flag, section, layer kind, quantization, importance, sweep damage, or
  **flow** (the last journey's push per layer). Cut or simulated-removed nodes are grey and shrunk.
- **Labels** are HTML, limited to about 140, without overlaps.
- The frame is redrawn only when something changes (camera, data, marks, animation).

## Interaction

| | |
|---|---|
| orbit / pan / zoom | left drag / right drag / wheel; `−` `+` `100%` `Fit` buttons |
| pick | click: wamr3d's raycaster against the instanced meshes (instance id → node), then the containers' boxes |
| hover | tooltip with label, kind, size, removable flag, edited mark |
| expand / collapse | double-click, or **Expand** in the side panel (level of detail: start at section / layer level and open what you need) |
| search | Enter: highlights matches and flies to them; a hidden tensor expands its parents |
| fly-to | **Focus** in the side panel animates the camera to the node |
| edits | **Edit:** buttons in the side panel (zero, ×0.5, ×2, scale, skip layer) |

## Token flow animation

**▶ token flow** (3D dataflow only) plays the last [token journey](web-ui.md#journey): particles run
up the column from Input to Output, and every layer glows with its strength for that token — the
direct-logit-attribution push of the layer on the prediction, split into mixer and FFN when the
layer is expanded (residual-norm growth for embedding models). The glow pulses as the particles pass.
Without a journey yet, the button runs one on the session prompt first.

## Embedding map

The tokens page draws the vocabulary as a 3D point cloud: every `token_embd` row projected on its
first three principal components. The PCA runs in Rust (block power iteration on up to 16,384
sampled rows, then every row is projected) and is cached on disk; the browser receives a binary
`Float32Array` (`x, y, z, class` per token) from `/api/models/{id}/embedding-map` and the token texts
from `/api/models/{id}/vocab/texts`.

wamr3d draws points one pixel wide, so the map uses one instanced octahedron per token, colored by
token class (with per-class toggles). Picking projects the tokens on the screen and takes the
nearest one, which is cheaper than ray tests against 150,000 instances. Search enlarges the matching
tokens and flies to them; a tokenized prompt is enlarged and joined in order.

## Performance notes

- Instancing keeps the draw calls to one per node kind plus a few line batches, however many nodes
  are visible.
- Fine nodes are loaded only when their block is expanded (`expand=` on the graph request).
- The embedding map (128k–152k tokens) is one instanced mesh, 8 triangles per token; the PCA is
  computed once per model and cached in `<output_dir>/cache/<digest>/embedmap-v1.bin`.
- wamr3d itself loads only when a 3D page opens.
