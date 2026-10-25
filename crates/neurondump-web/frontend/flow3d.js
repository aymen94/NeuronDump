// 3D dataflow view of the model graph, drawn with wamr3d. The layout is computed here
// from the /graph JSON: the vertical axis is the dataflow (input → layer 0 … N → output),
// blocks sit around their layer (mixer left, FFN right), fine nodes (heads, neuron
// buckets, experts) on rings or grids around their block, side sections beside the
// column. Leaves are drawn with one instanced mesh per node kind (no per-node draw
// calls); edges and container outlines are batched line segments.

import { count, bytes } from './ui.js';
import { loadWamr, createRenderer, cssVar, rayBox, flyTo } from './wamr.js';

const GAP = 4.2; // between layers
const SIDE = 16; // x of the side sections
const SPHERE_KINDS = new Set(['head', 'kv_group', 'neuron_bucket', 'expert', 'token']);
const MAIN = ['input', 'body', 'output'];
const SIDE_ORDER = ['tokenizer', 'template', 'projector', 'vision', 'mtp', 'extra'];

// --- layout -------------------------------------------------------------------------------

export function layoutFlow(view) {
  const byId = new Map(view.nodes.map((n) => [n.id, n]));
  const kids = new Map();
  for (const n of view.nodes) {
    if (n.parent && byId.has(n.parent)) {
      if (!kids.has(n.parent)) kids.set(n.parent, []);
      kids.get(n.parent).push(n.id);
    }
  }
  const maxP = new Map();
  for (const n of view.nodes) maxP.set(n.kind, Math.max(maxP.get(n.kind) || 1, n.params || n.bytes || 1));
  const w = (n) => Math.cbrt((n.params || n.bytes || 0) / maxP.get(n.kind)); // 0..1
  const size = (n) => {
    const k = w(n);
    switch (n.kind) {
      case 'layer': return [2.2 + 4 * k, 0.55, 2.2 + 4 * k];
      case 'section': case 'group': case 'model': return [3.2, 1.2, 3.2];
      case 'block': return [1.2 + 1.8 * k, 0.9, 1.2 + 1.8 * k];
      case 'tensor': return Array(3).fill(0.35 + 0.9 * k);
      case 'op': return [0.45, 0.45, 0.45];
      case 'kv_group': return Array(3).fill(0.75);
      case 'expert': return Array(3).fill(0.55);
      case 'head': return Array(3).fill(0.42);
      default: return Array(3).fill(0.32);
    }
  };
  const pos = new Map();
  const layerOf = (id) => byId.get(id)?.layer ?? 0;

  const ring = (ids, x, y, z, r0, place) => {
    const c = ids.length;
    if (c > 48) {
      const cols = Math.ceil(Math.sqrt(c));
      const sp = 0.75;
      ids.forEach((id, i) => place(id, x + ((i % cols) - (cols - 1) / 2) * sp, y, z + (Math.floor(i / cols) - (Math.ceil(c / cols) - 1) / 2) * sp));
      return;
    }
    const R = Math.max(r0, (c * 0.62) / (2 * Math.PI));
    ids.forEach((id, i) => {
      const a = (i / Math.max(1, c)) * Math.PI * 2;
      place(id, x + R * Math.cos(a), y, z + R * Math.sin(a));
    });
  };

  const place = (id, x, y, z) => {
    const n = byId.get(id);
    const ch = kids.get(id);
    if (!ch) {
      const [sx, sy, sz] = size(n);
      pos.set(id, { x, y, z, sx, sy, sz });
      return;
    }
    const sorted = ch.slice().sort((a, b) => layerOf(a) - layerOf(b));
    const isLayerRun = ch.filter((c) => byId.get(c).kind === 'layer').length >= ch.length * 0.7;
    if (isLayerRun) {
      sorted.forEach((c, i) => place(c, x, y + i * GAP, z));
      return;
    }
    if (n.kind === 'layer') {
      for (const c of ch) {
        const b = byId.get(c);
        const blk = b.block || '';
        if (b.kind === 'op') place(c, x, y + (c.endsWith('add_ffn') ? 0.7 : -0.7), z);
        else if (blk === 'attention' || blk === 'ssm') place(c, x - 3.4, y - 0.7, z);
        else if (blk === 'ffn' || blk === 'moe') place(c, x + 3.4, y + 0.7, z);
        else place(c, x, y, z + 3.2);
      }
      return;
    }
    if (n.kind === 'section' && (id === 'input' || id === 'output')) {
      ch.forEach((c, i) => place(c, x + (i - (ch.length - 1) / 2) * 2.6, y, z));
      return;
    }
    // Blocks, KV groups, side sections, groups: tensors on an inner ring above, fine nodes
    // (or anything else) on a ring around the node.
    const tensors = ch.filter((c) => byId.get(c).kind === 'tensor');
    const rest = ch.filter((c) => byId.get(c).kind !== 'tensor');
    if (n.kind === 'group') {
      sorted.forEach((c, i) => place(c, x, y + i * 2.6, z));
      return;
    }
    if (tensors.length) ring(tensors, x, y + (rest.length ? 1.3 : 0), z, rest.length ? 1.2 : 1.4, place);
    if (rest.length) ring(rest, x, y, z, n.kind === 'kv_group' ? 0.9 : 2.0, place);
  };

  const roots = view.nodes.filter((n) => !n.parent || !byId.has(n.parent)).map((n) => n.id);
  const body = roots.includes('body') ? 'body' : null;
  const nLayers = body && kids.has(body) ? kids.get(body).length : 1;
  const top = Math.max(1, nLayers - 1) * GAP;
  const yOf = { input: -GAP * 1.6, body: 0, output: top + GAP * 1.6 };
  for (const r of roots) {
    if (MAIN.includes(r)) place(r, 0, yOf[r], 0);
  }
  const side = roots.filter((r) => !MAIN.includes(r));
  const known = side.filter((r) => SIDE_ORDER.includes(r)).sort((a, b) => SIDE_ORDER.indexOf(a) - SIDE_ORDER.indexOf(b));
  known.forEach((r, i) => {
    const nearOut = r === 'mtp' || r === 'extra';
    place(r, SIDE, nearOut ? yOf.output - i * 0.0 : yOf.input + i * 4.5, nearOut ? 4 : (i % 2 ? 5 : -5));
  });
  // Group-by views and anything unknown: columns along X.
  side.filter((r) => !SIDE_ORDER.includes(r)).forEach((r, i) => place(r, -SIDE - i * 9, 0, 0));

  // Containers: bounds of their descendants, bottom-up.
  const boxes = new Map();
  const bounds = (id) => {
    if (pos.has(id)) {
      const p = pos.get(id);
      return [p.x - p.sx / 2, p.y - p.sy / 2, p.z - p.sz / 2, p.x + p.sx / 2, p.y + p.sy / 2, p.z + p.sz / 2];
    }
    const ch = kids.get(id);
    if (!ch) return null;
    const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (const c of ch) {
      const cb = bounds(c);
      if (!cb) continue;
      for (let i = 0; i < 3; i++) { b[i] = Math.min(b[i], cb[i]); b[i + 3] = Math.max(b[i + 3], cb[i + 3]); }
    }
    if (!Number.isFinite(b[0])) return null;
    const pad = 0.45;
    for (let i = 0; i < 3; i++) { b[i] -= pad; b[i + 3] += pad; }
    boxes.set(id, { x: (b[0] + b[3]) / 2, y: (b[1] + b[4]) / 2, z: (b[2] + b[5]) / 2, sx: b[3] - b[0], sy: b[4] - b[1], sz: b[5] - b[2] });
    return b;
  };
  roots.forEach(bounds);

  // The spine the token flow follows: input, each layer, output.
  const at = (id) => pos.get(id) || boxes.get(id);
  const spine = [];
  if (at('input')) spine.push({ id: 'input', p: at('input') });
  const layerIds = view.nodes.filter((n) => n.kind === 'layer').sort((a, b) => a.layer - b.layer).map((n) => n.id);
  for (const id of layerIds) spine.push({ id, layer: byId.get(id).layer, p: at(id) });
  if (at('output')) spine.push({ id: 'output', p: at('output') });
  return { byId, kids, pos, boxes, spine: spine.filter((s) => s.p) };
}

// --- renderer -----------------------------------------------------------------------------

export async function createFlow3D(container, opts = {}) {
  const { T, OrbitControls } = await loadWamr();
  const { renderer, backend } = await createRenderer(T);

  const root = document.createElement('div');
  root.className = 'graph3d';
  const labels = document.createElement('div');
  labels.className = 'graph3d-labels';
  const tip = document.createElement('div');
  tip.className = 'graph3d-tip';
  tip.hidden = true;
  root.append(renderer.domElement, labels, tip);
  container.append(root);

  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(45, 1, 0.1, 5000);
  scene.add(new T.HemisphereLight(0xffffff, 0x8890a0, 1.6));
  const sun = new T.DirectionalLight(0xffffff, 1.3);
  sun.position.set(40, 80, 50);
  scene.add(sun);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = true;

  const geos = {
    box: new T.BoxGeometry(1, 1, 1),
    sphere: new T.SphereGeometry(0.5, 14, 10),
    octa: new T.OctahedronGeometry(0.6, 0),
  };
  const geoOf = (kind) => (SPHERE_KINDS.has(kind) ? geos.sphere : kind === 'op' ? geos.octa : geos.box);
  const content = new T.Group();
  scene.add(content);

  let view = null;
  let L = null; // layout
  let colorOf = () => '#cccccc';
  let marks = { cut: new Set(), removed: new Set(), match: null, selected: null, edited: new Set(), highlight: new Set() };
  const meshes = []; // { mesh, kind, ids }
  const where = new Map(); // id -> { m, i }
  let markLines = null;
  let labelList = [];
  let dirty = true;
  let radius = 10;
  const tmpM = new T.Matrix4();
  const tmpC = new T.Color();
  const q0 = new T.Quaternion();
  let flow = null; // token flow animation state

  function theme() {
    renderer.setClearColor(cssVar('--bg-alt', '#f4f5f7'));
    dirty = true;
  }
  const themeObs = new MutationObserver(() => { theme(); if (view) rebuild(false); });
  themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  function resize() {
    const w = Math.max(1, root.clientWidth); const h = Math.max(1, root.clientHeight);
    renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    dirty = true;
  }
  const ro = new ResizeObserver(resize);
  ro.observe(root);

  function disposeContent() {
    for (const c of [...content.children]) {
      if (c.userData.ownGeometry) c.geometry.dispose();
      c.material?.dispose?.();
    }
    content.clear();
    meshes.length = 0;
    where.clear();
    markLines = null;
    labelList = [];
    labels.replaceChildren();
  }

  const boxEdges = (b, out) => {
    const x0 = b.x - b.sx / 2; const x1 = b.x + b.sx / 2; const y0 = b.y - b.sy / 2; const y1 = b.y + b.sy / 2; const z0 = b.z - b.sz / 2; const z1 = b.z + b.sz / 2;
    const c = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
    for (const [a, d] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) out.push(c[a], c[d]);
  };

  function lines(pts, colors, opacity = 1) {
    if (!pts.length) return null;
    const p = new Float32Array(pts.length * 3); const c = new Float32Array(pts.length * 3);
    pts.forEach((q, i) => { p.set(q, i * 3); c.set(colors[i], i * 3); });
    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.Float32BufferAttribute(p, 3));
    geo.setAttribute('color', new T.Float32BufferAttribute(c, 3));
    const ls = new T.LineSegments(geo, new T.LineBasicMaterial({ vertexColors: true, transparent: opacity < 1, opacity }));
    ls.userData.ownGeometry = true;
    content.add(ls);
    return ls;
  }

  const rgb = (css) => { tmpC.set(css); return [tmpC.r, tmpC.g, tmpC.b]; };

  function rebuild(refit = true) {
    disposeContent();
    L = layoutFlow(view);
    // One instanced mesh per node kind.
    const byKind = new Map();
    for (const [id] of L.pos) {
      const k = L.byId.get(id).kind;
      if (!byKind.has(k)) byKind.set(k, []);
      byKind.get(k).push(id);
    }
    for (const [kind, ids] of byKind) {
      const mesh = new T.InstancedMesh(geoOf(kind), new T.MeshLambertMaterial({ color: 0xffffff }), ids.length);
      mesh.userData.kind = kind;
      const m = { mesh, kind, ids };
      ids.forEach((id, i) => where.set(id, { m, i }));
      meshes.push(m);
      content.add(mesh);
    }
    applyInstances();

    // Dataflow edges, colored from source (muted) to target (accent) to show direction.
    const center = (id) => L.pos.get(id) || L.boxes.get(id);
    const accent = rgb(cssVar('--accent', '#2457c5')); const muted = rgb(cssVar('--muted', '#777777')); const red = rgb('#d9534f');
    const ep = []; const ec = []; const faint = []; const fc = [];
    for (const e of view.edges) {
      if (e.kind === 'contains') continue;
      const a = center(e.from); const b = center(e.to);
      if (!a || !b) continue;
      const tgt = e.kind === 'tied' ? red : accent;
      const into = (e.kind === 'uses' || e.label === 'residual') ? [faint, fc] : [ep, ec];
      into[0].push([a.x, a.y, a.z], [b.x, b.y, b.z]);
      into[1].push(muted, tgt);
    }
    lines(ep, ec, 0.9);
    lines(faint, fc, 0.35);
    // Container outlines in one batch.
    const op = []; const border = rgb(cssVar('--border', '#bbbbbb'));
    for (const b of L.boxes.values()) boxEdges(b, op);
    lines(op, op.map(() => border), 0.9);
    buildMarks();

    // Labels: containers, sections and layers first, then the biggest leaves.
    const sized = (id) => L.byId.get(id).bytes || 0;
    const coarse = [...L.pos.keys()].filter((id) => ['section', 'layer', 'group'].includes(L.byId.get(id).kind));
    const ids = [...L.boxes.keys(), ...coarse, ...[...L.pos.keys()].filter((id) => !coarse.includes(id)).sort((a, b) => sized(b) - sized(a))];
    for (const id of ids) {
      const d = document.createElement('span');
      d.textContent = L.byId.get(id).label;
      d.className = L.boxes.has(id) ? 'strong' : '';
      labels.append(d);
      labelList.push({ id, el: d });
    }
    const all = [...L.pos.values(), ...L.boxes.values()];
    radius = Math.max(6, ...all.map((q) => Math.hypot(q.x, q.y - (L.spine.at(-1)?.p.y || 0) / 2, q.z) + 4));
    controls.maxDistance = radius * 8;
    controls.minDistance = 0.5;
    if (refit) fit();
    dirty = true;
  }

  function nodeColor(id) {
    const n = L.byId.get(id);
    if (marks.cut.has(id) || marks.removed.has(id)) return '#9aa0a6';
    let c = colorOf(n);
    if (marks.match && !marks.match.has(id)) {
      tmpC.set(c).lerp(new T.Color(cssVar('--bg-alt', '#f4f5f7')), 0.8);
      return '#' + tmpC.getHexString();
    }
    return c;
  }

  function applyInstances() {
    for (const { mesh, ids } of meshes) {
      ids.forEach((id, i) => {
        const p = L.pos.get(id);
        const shrink = marks.cut.has(id) || marks.removed.has(id) ? 0.55 : 1;
        tmpM.compose({ x: p.x, y: p.y, z: p.z }, q0, { x: p.sx * shrink, y: p.sy * shrink, z: p.sz * shrink });
        mesh.setMatrixAt(i, tmpM);
        tmpC.set(nodeColor(id));
        mesh.setColorAt(i, tmpC);
      });
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.computeBoundingBox?.();
    }
    dirty = true;
  }

  /** Visible id for `id`: itself, or the closest visible ancestor by id prefix (`layer.7.attn.head.3` → `layer.7.attn`). */
  function visibleId(id) {
    let cur = id;
    while (cur) {
      if (L.pos.has(cur) || L.boxes.has(cur)) return cur;
      const i = cur.lastIndexOf('.');
      if (i < 0) return null;
      cur = cur.slice(0, i);
    }
    return null;
  }

  function buildMarks() {
    if (markLines) { content.remove(markLines); markLines.geometry.dispose(); markLines.material.dispose(); markLines = null; }
    const pts = []; const cols = [];
    const add = (id, css, grow) => {
      const v = visibleId(id);
      const b = v && (L.pos.get(v) || L.boxes.get(v));
      if (!b) return;
      const before = pts.length;
      boxEdges({ ...b, sx: b.sx * grow + 0.1, sy: b.sy * grow + 0.1, sz: b.sz * grow + 0.1 }, pts);
      const c = rgb(css);
      for (let i = before; i < pts.length; i++) cols.push(c);
    };
    for (const id of marks.highlight) add(id, '#8e44ad', 1.18);
    for (const id of marks.edited) add(id, '#f28e2b', 1.12);
    for (const id of marks.cut) add(id, '#d9534f', 1.08);
    if (marks.match) for (const id of marks.match) add(id, cssVar('--accent', '#2457c5'), 1.08);
    if (marks.selected) add(marks.selected, cssVar('--accent', '#2457c5'), 1.25);
    markLines = lines(pts, cols, 1);
    dirty = true;
  }

  function fit(ids) {
    const list = (ids?.length ? ids.map(visibleId).filter(Boolean) : [...L.pos.keys(), ...L.boxes.keys()]).map((id) => L.pos.get(id) || L.boxes.get(id)).filter(Boolean);
    if (!list.length) return;
    const lo = [Infinity, Infinity, Infinity]; const hi = [-Infinity, -Infinity, -Infinity];
    for (const q of list) {
      lo[0] = Math.min(lo[0], q.x - q.sx / 2); lo[1] = Math.min(lo[1], q.y - q.sy / 2); lo[2] = Math.min(lo[2], q.z - q.sz / 2);
      hi[0] = Math.max(hi[0], q.x + q.sx / 2); hi[1] = Math.max(hi[1], q.y + q.sy / 2); hi[2] = Math.max(hi[2], q.z + q.sz / 2);
    }
    const c = new T.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
    const r = Math.max(1.5, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
    const fov = (camera.fov * Math.PI) / 180;
    const dist = (r / Math.sin(Math.min(fov, 2 * Math.atan(Math.tan(fov / 2) * camera.aspect)) / 2)) * 1.05;
    const dir = ids?.length ? camera.position.clone().sub(controls.target).normalize() : new T.Vector3(1, 0.35, 1.25).normalize();
    camera.near = Math.max(0.05, dist / 300);
    camera.far = dist * 20 + radius * 4;
    camera.updateProjectionMatrix();
    return flyTo(camera, controls, c, c.clone().add(dir.multiplyScalar(dist)), ids?.length ? 600 : 1, () => { dirty = true; });
  }

  // --- labels ---------------------------------------------------------------------------
  const v3 = new T.Vector3();
  function updateLabels() {
    const w = root.clientWidth; const h = root.clientHeight;
    const placed = [];
    let shown = 0;
    for (const { id, el } of labelList) {
      const q = L.boxes.get(id) || L.pos.get(id);
      const isBox = L.boxes.has(id);
      if (isBox) v3.set(q.x - q.sx / 2, q.y + q.sy / 2, q.z + q.sz / 2); else v3.set(q.x, q.y + q.sy / 2, q.z);
      v3.project(camera);
      const x = (v3.x + 1) / 2 * w; const y = (1 - v3.y) / 2 * h;
      const lw = el.textContent.length * 6.4 + 8; const lh = 16;
      const rect = isBox ? [x - lw, y - lh - 2, x, y - 2] : [x - lw / 2, y - lh - 2, x + lw / 2, y - 2];
      const visible = v3.z < 1 && x > -lw && x < w + lw && y > 0 && y < h + lh && shown < 140
        && !(marks.match && !marks.match.has(id) && !isBox)
        && !placed.some((r) => r[0] < rect[2] && rect[0] < r[2] && r[1] < rect[3] && rect[1] < r[3]);
      if (visible) {
        placed.push(rect); shown++;
        el.style.transform = `translate(${Math.round(rect[0])}px, ${Math.round(rect[1])}px)`;
        el.hidden = false;
      } else el.hidden = true;
    }
  }

  // --- picking --------------------------------------------------------------------------
  const ray = new T.Raycaster();
  const ndc = new T.Vector2();
  function pick(ev) {
    if (!L) return null;
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObjects(meshes.map((m) => m.mesh), false);
    const hit = hits.find((x) => x.instanceId !== undefined);
    if (hit) {
      const m = meshes.find((x) => x.mesh === hit.object);
      if (m) return m.ids[hit.instanceId];
    }
    // Containers: the smallest box the ray passes through.
    let best = null;
    for (const [id, b] of L.boxes) {
      const t = rayBox(ray.ray.origin, ray.ray.direction, b);
      if (t !== null) {
        const vol = b.sx * b.sy * b.sz;
        if (!best || vol < best.vol) best = { id, vol };
      }
    }
    return best?.id ?? null;
  }
  let down = null;
  const onDown = (ev) => { down = { x: ev.clientX, y: ev.clientY }; };
  const onUp = (ev) => {
    if (!down || ev.button !== 0) return;
    const moved = Math.hypot(ev.clientX - down.x, ev.clientY - down.y);
    down = null;
    if (moved > 5) return;
    const id = pick(ev);
    if (id) { marks.selected = id; buildMarks(); opts.onTap?.(id); }
  };
  const onDbl = (ev) => { const id = pick(ev); if (id) opts.onDblTap?.(id); };
  let hoverPending = null;
  const onMove = (ev) => {
    if (hoverPending) return;
    hoverPending = requestAnimationFrame(() => {
      hoverPending = null;
      if (ev.buttons) { tip.hidden = true; return; }
      const id = pick(ev);
      const n = id && L.byId.get(id);
      renderer.domElement.style.cursor = n ? 'pointer' : '';
      if (!n) { tip.hidden = true; return; }
      const r = root.getBoundingClientRect();
      const extra = flow?.values?.get(n.layer) !== undefined && n.kind === 'layer' ? ` · flow ${flow.values.get(n.layer).toFixed(2)}` : '';
      tip.replaceChildren(
        Object.assign(document.createElement('b'), { textContent: n.label }),
        document.createTextNode(` ${n.kind}${n.params || n.bytes ? ` · ${count(n.params)} · ${bytes(n.bytes)}` : ''}${n.removable ? ` · ${n.removable}` : ''}${marks.edited.has(id) ? ' · edited' : ''}${extra}`));
      tip.style.transform = `translate(${Math.round(ev.clientX - r.left + 12)}px, ${Math.round(ev.clientY - r.top + 12)}px)`;
      tip.hidden = false;
    });
  };
  const onLeave = () => { tip.hidden = true; };
  const cv = renderer.domElement;
  cv.addEventListener('pointerdown', onDown);
  cv.addEventListener('pointerup', onUp);
  cv.addEventListener('dblclick', onDbl);
  cv.addEventListener('pointermove', onMove);
  cv.addEventListener('pointerleave', onLeave);
  controls.addEventListener('change', () => { dirty = true; });

  // --- token flow -------------------------------------------------------------------------
  // Particles run up the spine; each layer glows with its activation strength, pulsing as
  // the particles pass it.
  function startFlow(values, label) {
    stopFlow();
    if (!L?.spine.length) return;
    const pts = L.spine.map((s) => s.p);
    const len = [0];
    for (let i = 1; i < pts.length; i++) len.push(len[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y, pts[i].z - pts[i - 1].z));
    const total = len.at(-1) || 1;
    const P = 36;
    const mesh = new T.InstancedMesh(geos.sphere, new T.MeshBasicMaterial({ color: cssVar('--accent', '#2457c5') }), P);
    content.add(mesh);
    flow = { t0: performance.now(), dur: 3800, pts, len, total, mesh, P, values, label, base: new Map() };
    // Glow targets: collapsed layers, or the mixer / FFN blocks of expanded layers.
    flow.targets = [];
    for (const s of L.spine) {
      if (s.layer === undefined) continue;
      const v = values.get(s.layer);
      if (v === undefined) continue;
      const u = len[L.spine.indexOf(s)] / total;
      if (where.has(s.id)) flow.targets.push({ id: s.id, u, v: v.total });
      else {
        for (const [suffix, key] of [['attn', 'attn'], ['ssm', 'attn'], ['ffn', 'ffn'], ['moe', 'ffn']]) {
          const id = `${s.id}.${suffix}`;
          if (where.has(id)) flow.targets.push({ id, u, v: v[key] });
        }
      }
    }
  }
  function stopFlow() {
    if (!flow) return;
    content.remove(flow.mesh);
    flow.mesh.material.dispose();
    flow = null;
    if (L) applyInstances();
  }
  function stepFlow(now) {
    const t = ((now - flow.t0) % flow.dur) / flow.dur;
    const at = (u) => {
      const d = u * flow.total;
      let i = 1;
      while (i < flow.len.length - 1 && flow.len[i] < d) i++;
      const a = flow.pts[i - 1]; const b = flow.pts[i];
      const k = (d - flow.len[i - 1]) / Math.max(1e-6, flow.len[i] - flow.len[i - 1]);
      return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, z: a.z + (b.z - a.z) * k };
    };
    for (let k = 0; k < flow.P; k++) {
      let u = t - (k / flow.P) * 0.18;
      if (u < 0) u += 1;
      const p = at(u);
      const s = 0.28 * (1 - k / flow.P) + 0.08;
      const jitter = Math.sin(k * 12.9898) * 0.35;
      tmpM.compose({ x: p.x + jitter, y: p.y, z: p.z + Math.cos(k * 4.1) * 0.35 }, q0, { x: s, y: s, z: s });
      flow.mesh.setMatrixAt(k, tmpM);
    }
    flow.mesh.instanceMatrix.needsUpdate = true;
    const hot = new T.Color('#ff5a1f');
    const touched = new Set();
    for (const g of flow.targets) {
      const w = where.get(g.id);
      if (!w) continue;
      const pulse = Math.exp(-(((t - g.u) / 0.06) ** 2));
      tmpC.set(nodeColor(g.id)).lerp(hot, Math.min(1, g.v * (0.35 + 0.65 * pulse)));
      w.m.mesh.setColorAt(w.i, tmpC);
      touched.add(w.m.mesh);
    }
    for (const m of touched) if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }

  renderer.setAnimationLoop((now) => {
    const moving = controls.update();
    if (flow) { stepFlow(now ?? performance.now()); dirty = true; }
    if (!moving && !dirty) return;
    dirty = false;
    renderer.render(scene, camera);
    if (L) updateLabels();
  });

  theme();
  resize();

  return {
    backend,
    setData(nextView, nextColorOf, refit = true) {
      view = nextView; colorOf = nextColorOf || colorOf;
      const playing = flow && { values: flow.values, label: flow.label };
      stopFlow();
      rebuild(refit);
      if (playing) startFlow(playing.values, playing.label);
    },
    recolor(nextColorOf) { colorOf = nextColorOf || colorOf; if (L) applyInstances(); },
    /** cut / removed / edited / highlight: Sets of ids; match: Set or null; selected: id or null. */
    setMarks(next) {
      const shape = next.cut || next.removed;
      marks = { ...marks, ...next };
      if (!L) return;
      if (shape || next.match !== undefined) applyInstances();
      buildMarks();
    },
    fit: (ids) => L && fit(ids),
    /** Fly the camera to a node. */
    focus(id) {
      const v = L && visibleId(id);
      const b = v && (L.pos.get(v) || L.boxes.get(v));
      if (!b) return false;
      const c = new T.Vector3(b.x, b.y, b.z);
      const dist = Math.max(b.sx, b.sy, b.sz) * 3 + 5;
      const dir = camera.position.clone().sub(controls.target).normalize();
      flyTo(camera, controls, c, c.clone().add(dir.multiplyScalar(dist)), 700, () => { dirty = true; });
      return true;
    },
    zoomBy(f) {
      const off = camera.position.clone().sub(controls.target);
      camera.position.copy(controls.target).add(off.setLength(Math.min(controls.maxDistance, Math.max(controls.minDistance, off.length() / f))));
      controls.update(); dirty = true;
    },
    resetZoom() { if (L) fit(); },
    /** values: Map(layer -> { total, attn, ffn } in 0..1). */
    playFlow: startFlow,
    stopFlow,
    get flowing() { return !!flow; },
    show(visible) { root.hidden = !visible; if (visible) resize(); },
    dispose() {
      renderer.setAnimationLoop(null);
      ro.disconnect();
      themeObs.disconnect();
      cv.removeEventListener('pointerdown', onDown);
      cv.removeEventListener('pointerup', onUp);
      cv.removeEventListener('dblclick', onDbl);
      cv.removeEventListener('pointermove', onMove);
      cv.removeEventListener('pointerleave', onLeave);
      stopFlow();
      disposeContent();
      controls.dispose();
      Object.values(geos).forEach((g) => g.dispose());
      renderer.dispose();
      root.remove();
    },
  };
}
