// Model graph renderer (wamr3d, vendored in vendor/wamr3d) over the /graph JSON: a flat 2D
// diagram, or in 3D a tower, a city skyline or a force-directed cloud.

import { count, bytes } from './ui.js';
import { loadWamr, createRenderer, cssVar } from './wamr.js';
import { structure, LAYOUTS, containerBounds } from './graph3d-layout.js';
import { createControlPanel, loadSettings } from './graph3d-controls.js';
import { createMinimap } from './graph3d-minimap.js';

export const MODES = { '2d': '2D', tower: '3D tower', city: '3D city', force: '3D force' };

let NAV = null; // navigation mode -> controls class
async function loadLib() {
  const w = await loadWamr();
  NAV = { orbit: w.OrbitControls, map: w.MapControls, trackball: w.TrackballControls };
  return w.T;
}

/**
 * Creates the 3D view inside `container`. Throws when wamr3d cannot start (no WebGPU and no
 * WebGL 2); the caller then keeps the 2D view.
 * opts: { onTap(id), onDblTap(id), backend: 'auto' | 'webgpu' | 'webgl', viewsKey (localStorage key of the saved camera views) }
 */
export async function createGraph3D(container, opts = {}) {
  const T = await loadLib();
  const { renderer, backend } = await createRenderer(T, opts.backend);

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
  const sun = new T.DirectionalLight(0xffffff, 1.4);
  sun.position.set(30, 60, 40);
  scene.add(sun);

  const settings = loadSettings();
  let controls = null;
  const navKind = () => (mode === '2d' ? 'orbit' : NAV[settings.nav] ? settings.nav : 'orbit');
  // Creates the controls of the current navigation mode, keeping the camera where it is.
  // orbit: left drag orbits around a fixed up; map: left drag pans the ground, right drag
  // orbits; trackball: free rotation over the poles (wamr3d's TrackballControls).
  function makeControls() {
    const kind = navKind();
    if (controls?.kind === kind) return;
    const target = controls ? controls.target.clone() : new T.Vector3();
    controls?.dispose();
    // Orbit and map keep the world up; a trackball may have tilted it.
    camera.up.set(0, 1, 0);
    controls = new NAV[kind](camera, renderer.domElement);
    controls.kind = kind;
    controls.target.copy(target);
    if (kind === 'trackball') controls.keys = []; // A/S/D would clash with WASD panning
    else controls.screenSpacePanning = kind === 'orbit';
    controls.maxDistance = radius * 8;
    controls.minDistance = 0.5;
    controls.addEventListener('change', () => { dirty = true; });
    controls.update();
  }

  // 2D: straight-on view, left drag pans, no rotation; 3D: the chosen navigation mode.
  function applyModeControls() {
    makeControls();
    const flat = mode === '2d';
    if (controls.kind === 'orbit') {
      controls.enableRotate = !flat;
      controls.mouseButtons = flat
        ? { LEFT: T.MOUSE.PAN, MIDDLE: T.MOUSE.DOLLY, RIGHT: T.MOUSE.PAN }
        : { LEFT: T.MOUSE.ROTATE, MIDDLE: T.MOUSE.DOLLY, RIGHT: T.MOUSE.PAN };
      controls.touches = flat ? { ONE: T.TOUCH.PAN, TWO: T.TOUCH.DOLLY_PAN } : { ONE: T.TOUCH.ROTATE, TWO: T.TOUCH.DOLLY_PAN };
    }
    applySettings();
    panel?.sync();
  }

  // Panel settings -> controls, camera and scene.
  function applySettings() {
    makeControls();
    const flat = mode === '2d';
    if (controls.kind === 'trackball') {
      // Trackball speeds are on another scale; these make 1x feel like the orbit's 1x.
      controls.rotateSpeed = 2.5 * settings.rotateSpeed;
      controls.panSpeed = 0.8 * settings.panSpeed;
      controls.zoomSpeed = 1.2 * settings.zoomSpeed;
      controls.staticMoving = !(settings.damping > 0);
      controls.dynamicDampingFactor = settings.damping || 0.2;
    } else {
      controls.rotateSpeed = settings.rotateSpeed;
      controls.panSpeed = settings.panSpeed;
      controls.zoomSpeed = settings.zoomSpeed;
      controls.enableDamping = settings.damping > 0;
      controls.dampingFactor = settings.damping || 0.1;
      controls.zoomToCursor = settings.zoomToCursor;
      controls.autoRotate = settings.autoRotate && !flat;
      controls.autoRotateSpeed = settings.autoRotateSpeed;
    }
    const fov = flat ? 20 : settings.fov;
    if (camera.fov !== fov) {
      // Keep the apparent size: move the camera so the target fills the same share of the view.
      const off = camera.position.clone().sub(controls.target);
      const k = Math.tan((camera.fov * Math.PI) / 360) / Math.tan((fov * Math.PI) / 360);
      camera.position.copy(controls.target).add(off.multiplyScalar(k));
      camera.fov = fov;
      camera.updateProjectionMatrix();
      controls.update();
    }
    for (const { line, opacity } of edgeLines) { line.visible = settings.edges; line.material.opacity = opacity * settings.edgeOpacity; }
    for (const it of items.values()) if (it.line) it.line.visible = settings.outlines;
    labels.hidden = !settings.labels;
    dirty = true;
  }

  const boxGeo = new T.BoxGeometry(1, 1, 1);
  const edgeGeo = new T.EdgesGeometry(boxGeo);
  const content = new T.Group();
  scene.add(content);

  let view = null;
  let mode = 'tower';
  let colorOf = () => '#cccccc';
  let marks = { cut: new Set(), removed: new Set(), match: null, selected: null, edited: new Set(), highlight: new Set() };
  let pos = new Map(); let boxes = new Map();
  const items = new Map(); // id -> { mesh, mat, outline, container }
  let pickables = [];
  let edgeLines = []; // { line, opacity }
  let panel = null;
  let labelList = [];
  let dirty = true;
  let radius = 10;

  function theme() {
    renderer.setClearColor(cssVar('--bg-alt', '#f4f5f7'));
    dirty = true;
  }
  const themeObs = new MutationObserver(() => { theme(); rebuild(false); });
  themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  const media = matchMedia('(prefers-color-scheme: dark)');
  const onMedia = () => { theme(); rebuild(false); };
  media.addEventListener?.('change', onMedia);

  function resize() {
    const w = Math.max(1, root.clientWidth); const h = Math.max(1, root.clientHeight);
    renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    controls?.handleResize?.();
    dirty = true;
  }
  const ro = new ResizeObserver(resize);
  ro.observe(root);

  function clear() {
    for (const it of items.values()) {
      it.mat.dispose();
      it.outline?.material.dispose();
    }
    for (const child of [...content.children]) {
      if (child.userData.ownGeometry) child.geometry.dispose();
      if (child.userData.ownMaterial) child.material.dispose();
    }
    content.clear();
    items.clear();
    pickables = [];
    edgeLines = [];
    labelList = [];
    labels.replaceChildren();
  }

  // Size by a metric: city buildings get taller, other layouts shrink the smaller nodes.
  const METRICS = { params: (n) => Math.sqrt(n.params || 0), bytes: (n) => Math.sqrt(n.bytes || 0), importance: (n) => n.importance?.score ?? 0 };
  function shape(s) {
    const metric = METRICS[settings.sizeBy];
    if (metric) {
      let max = 0;
      for (const id of pos.keys()) max = Math.max(max, metric(s.byId.get(id).n));
      for (const [id, p] of pos) {
        const k = max > 0 ? metric(s.byId.get(id).n) / max : 1;
        if (mode === 'city') {
          const base = p.y - p.sy / 2;
          p.sy = Math.max(0.05, p.sy * (0.15 + 1.85 * k));
          p.y = base + p.sy / 2;
        } else {
          const f = 0.25 + 0.75 * k;
          p.sx *= f; p.sy *= f; if (mode !== '2d') p.sz *= f;
        }
      }
    }
    // Explode: spread the nodes from the centre without growing them.
    const e = settings.explode;
    if (e !== 1 && pos.size) {
      const c = [0, 0, 0];
      for (const p of pos.values()) { c[0] += p.x; c[1] += p.y; c[2] += p.z; }
      c.forEach((v, i) => { c[i] = v / pos.size; });
      for (const p of pos.values()) {
        p.x = c[0] + (p.x - c[0]) * e; p.y = c[1] + (p.y - c[1]) * e;
        if (mode !== '2d') p.z = c[2] + (p.z - c[2]) * e;
      }
    }
  }

  // Nodes within `depth` connections (dataflow edges and containment) of `start`, as leaves.
  function isolateSet(s, start, depth) {
    const adj = new Map();
    const link = (a, b) => {
      if (!adj.has(a)) adj.set(a, new Set());
      if (!adj.has(b)) adj.set(b, new Set());
      adj.get(a).add(b); adj.get(b).add(a);
    };
    for (const e of view.edges) if (e.kind !== 'contains' && s.byId.has(e.from) && s.byId.has(e.to)) link(e.from, e.to);
    for (const n of view.nodes) if (n.parent && s.byId.has(n.parent)) link(n.id, n.parent);
    let frontier = [start];
    const seen = new Set(frontier);
    for (let d = 0; d < depth; d++) {
      const next = [];
      for (const x of frontier) for (const y of adj.get(x) || []) if (!seen.has(y)) { seen.add(y); next.push(y); }
      frontier = next;
    }
    // The selection's own containers are only the way to its siblings: never open them whole.
    const ancestors = new Set();
    for (let p = s.byId.get(start)?.n.parent; p && s.byId.has(p); p = s.byId.get(p).n.parent) ancestors.add(p);
    const leaves = new Set();
    const down = (id) => { if (pos.has(id)) leaves.add(id); for (const k of s.kids.get(id) || []) down(k); };
    for (const id of seen) if (!ancestors.has(id)) down(id);
    return leaves;
  }

  // The leaves to draw: isolation around the selection, then the slice plane. null = all.
  let slice = null; // { axis, lo, hi } of the full layout, for the minimap and the slider
  function visible(s) {
    let keep = null;
    if (settings.isolate && marks.selected && s.byId.has(marks.selected)) keep = isolateSet(s, marks.selected, settings.isolateDepth);
    slice = null;
    const ax = settings.sliceAxis;
    if (ax === 'x' || ax === 'y' || ax === 'z') {
      const sz = 's' + ax;
      let lo = Infinity; let hi = -Infinity;
      for (const p of pos.values()) { lo = Math.min(lo, p[ax] - p[sz] / 2); hi = Math.max(hi, p[ax] + p[sz] / 2); }
      const cut = lo + (hi - lo) * settings.slice;
      slice = { axis: ax, lo, hi, cut };
      if (settings.slice < 1) {
        const inSlice = new Set([...pos].filter(([, p]) => p[ax] - p[sz] / 2 <= cut).map(([id]) => id));
        keep = keep ? new Set([...keep].filter((x) => inSlice.has(x))) : inSlice;
      }
    }
    return keep;
  }

  let fullPos = new Map(); // the whole layout, before isolation and slicing (for the minimap)
  function rebuild(refit = true) {
    if (!view) return;
    clear();
    const s = structure(view);
    pos = LAYOUTS[mode](view, s);
    shape(s);
    fullPos = pos;
    const keep = visible(s);
    if (keep) pos = new Map([...pos].filter(([id]) => keep.has(id)));
    boxes = containerBounds(view, s, pos, mode === '2d' ? 1 : 0);
    const muted = cssVar('--muted', '#777777');
    const border = cssVar('--border', '#cccccc');

    for (const [id, p] of pos) {
      const n = s.byId.get(id).n;
      const mat = new T.MeshLambertMaterial({ color: colorOf(n) });
      const mesh = new T.Mesh(boxGeo, mat);
      mesh.position.set(p.x, p.y, p.z);
      mesh.scale.set(p.sx, p.sy, p.sz);
      mesh.userData.id = id;
      content.add(mesh);
      pickables.push(mesh);
      items.set(id, { mesh, mat, outline: null, container: false });
    }
    for (const [id, b] of boxes) {
      const mat = new T.LineBasicMaterial({ color: border, transparent: true, opacity: 0.9 });
      const line = new T.LineSegments(edgeGeo, mat);
      line.position.set(b.x, b.y, b.z);
      line.scale.set(b.sx, b.sy, b.sz);
      content.add(line);
      // An almost invisible shell makes the container clickable.
      const shellMat = new T.MeshBasicMaterial({ color: border, transparent: true, opacity: 0.05, depthWrite: false });
      const shell = new T.Mesh(boxGeo, shellMat);
      shell.position.copy(line.position);
      shell.scale.copy(line.scale);
      shell.userData.id = id;
      shell.userData.container = true;
      content.add(shell);
      pickables.push(shell);
      items.set(id, { mesh: shell, mat: shellMat, outline: null, line, container: true });
    }

    // Edges: one batch per style, colored from source (muted) to target (accent) for direction.
    const accent = new T.Color(cssVar('--accent', '#2457c5'));
    const mutedC = new T.Color(muted);
    const center = (id) => pos.get(id) || boxes.get(id);
    const batches = { main: [], faint: [], tied: [] };
    for (const e of view.edges) {
      if (e.kind === 'contains') continue;
      const a = center(e.from); const b = center(e.to);
      if (!a || !b) continue;
      const key = e.kind === 'tied' ? 'tied' : (e.kind === 'uses' || e.label === 'residual') ? 'faint' : 'main';
      batches[key].push(a, b);
    }
    const addLines = (pts, color2, opacity) => {
      if (!pts.length) return;
      const p = new Float32Array(pts.length * 3); const c = new Float32Array(pts.length * 3);
      pts.forEach((q, i) => {
        p[i * 3] = q.x; p[i * 3 + 1] = q.y; p[i * 3 + 2] = q.z;
        const col = i % 2 === 0 ? mutedC : color2;
        c[i * 3] = col.r; c[i * 3 + 1] = col.g; c[i * 3 + 2] = col.b;
      });
      const geo = new T.BufferGeometry();
      geo.setAttribute('position', new T.Float32BufferAttribute(p, 3));
      geo.setAttribute('color', new T.Float32BufferAttribute(c, 3));
      const line = new T.LineSegments(geo, new T.LineBasicMaterial({ vertexColors: true, transparent: true, opacity }));
      line.userData.ownGeometry = true; line.userData.ownMaterial = true;
      content.add(line);
      edgeLines.push({ line, opacity });
    };
    addLines(batches.main, accent, 0.85);
    addLines(batches.faint, mutedC, 0.3);
    addLines(batches.tied, new T.Color('#d9534f'), 0.7);

    // Labels: containers and roots first, then the biggest leaves.
    const sized = (id) => s.byId.get(id).n.bytes || 0;
    const ids = [...boxes.keys(), ...[...pos.keys()].sort((a, b) => sized(b) - sized(a))];
    for (const id of ids) {
      const n = s.byId.get(id).n;
      const d = document.createElement('span');
      d.textContent = n.label;
      d.className = boxes.has(id) || s.roots.includes(id) ? 'strong' : '';
      labels.append(d);
      labelList.push({ id, el: d });
    }

    let all = [...pos.values(), ...boxes.values()];
    if (!all.length) all = [...fullPos.values()];
    const c = all.reduce((acc, q) => [acc[0] + q.x, acc[1] + q.y, acc[2] + q.z], [0, 0, 0]).map((v) => v / Math.max(1, all.length));
    radius = Math.max(4, ...all.map((q) => Math.hypot(q.x - c[0], q.y - c[1], q.z - c[2]) + Math.max(q.sx, q.sy, q.sz) / 2));
    controls.maxDistance = radius * 8;
    controls.minDistance = 0.5;
    applySettings();
    applyMarks();
    if (refit) fit();
    dirty = true;
  }

  function applyMarks() {
    for (const [id, it] of items) {
      if (it.container) {
        // Containers (expanded layers / blocks) show edit and highlight marks on their outline.
        const c = marks.highlight?.has(id) ? '#8e44ad' : marks.edited?.has(id) ? '#f28e2b' : null;
        it.line?.material.color.set(c || cssVar('--border', '#cccccc'));
        continue;
      }
      const dim = marks.match && !marks.match.has(id);
      const removed = marks.removed.has(id);
      it.mat.transparent = dim || removed;
      it.mat.opacity = dim ? 0.15 : removed ? 0.35 : 1;
      const ring = marks.cut.has(id) ? '#d9534f' : (marks.selected === id || marks.match?.has(id)) ? '#2457c5'
        : marks.highlight?.has(id) ? '#8e44ad' : marks.edited?.has(id) ? '#f28e2b' : null;
      if (ring && !it.outline) {
        it.outline = new T.LineSegments(edgeGeo, new T.LineBasicMaterial({ color: ring }));
        it.outline.scale.set(1.08, 1.08, 1.08);
        it.mesh.add(it.outline);
      } else if (!ring && it.outline) {
        it.mesh.remove(it.outline);
        it.outline.material.dispose();
        it.outline = null;
      } else if (ring) {
        it.outline.material.color.set(ring);
      }
    }
    dirty = true;
  }

  // Frames the given ids (or everything) from the current direction.
  function fit(ids, keepDir = false) {
    let list = (ids?.length ? ids : [...pos.keys(), ...boxes.keys()]).map((id) => pos.get(id) || boxes.get(id)).filter(Boolean);
    if (!list.length && !ids?.length) list = [...fullPos.values()];
    if (!list.length) return;
    const lo = [Infinity, Infinity, Infinity]; const hi = [-Infinity, -Infinity, -Infinity];
    for (const q of list) {
      lo[0] = Math.min(lo[0], q.x - q.sx / 2); lo[1] = Math.min(lo[1], q.y - q.sy / 2); lo[2] = Math.min(lo[2], q.z - q.sz / 2);
      hi[0] = Math.max(hi[0], q.x + q.sx / 2); hi[1] = Math.max(hi[1], q.y + q.sy / 2); hi[2] = Math.max(hi[2], q.z + q.sz / 2);
    }
    const center = new T.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
    if (!keepDir) camera.up.set(0, 1, 0);
    const r = Math.max(1, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
    const fov = (camera.fov * Math.PI) / 180;
    const dist = (r / Math.tan(Math.min(fov, 2 * Math.atan(Math.tan(fov / 2) * camera.aspect)) / 2)) * 1.12;
    const dir = keepDir
      ? camera.position.clone().sub(controls.target).normalize()
      : (mode === '2d' ? new T.Vector3(0, 0, 1) : mode === 'city' ? new T.Vector3(0.2, 0.75, 1) : mode === 'tower' ? new T.Vector3(1, 0.45, 1.2) : new T.Vector3(1, 0.6, 1.4)).normalize();
    controls.target.copy(center);
    camera.position.copy(center).add(dir.multiplyScalar(dist));
    camera.near = Math.max(0.05, dist / 200);
    camera.far = dist * 20 + radius * 4;
    camera.updateProjectionMatrix();
    controls.stop?.();
    controls.update();
    // The sphere estimate overshoots for long, flat layouts: refine on the projected corners.
    const corner = new T.Vector3();
    for (let pass = 0; pass < 4; pass++) {
      camera.updateMatrixWorld();
      let e = 0;
      for (let i = 0; i < 8; i++) {
        corner.set(i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]).project(camera);
        e = Math.max(e, Math.abs(corner.x), Math.abs(corner.y));
      }
      if (!(e > 0) || Math.abs(e - 0.88) < 0.03) break;
      const off = camera.position.clone().sub(controls.target);
      camera.position.copy(controls.target).add(off.multiplyScalar(Math.max(0.3, Math.min(3, e / 0.88))));
    }
    controls.update();
    dirty = true;
  }

  function zoomBy(factor) {
    const off = camera.position.clone().sub(controls.target);
    const d = Math.min(controls.maxDistance, Math.max(controls.minDistance, off.length() / factor));
    camera.position.copy(controls.target).add(off.setLength(d));
    controls.update();
    dirty = true;
  }

  // Camera views: get / set, for saved views and shared links.
  const r3 = (v) => [v.x, v.y, v.z].map((x) => Math.round(x * 1000) / 1000);
  function getView() {
    return { mode, t: r3(controls.target), p: r3(camera.position), u: r3(camera.up), fov: camera.fov };
  }
  function setView(v) {
    if (!v?.t || !v?.p) return;
    if (v.u) camera.up.set(...v.u);
    controls.stop?.();
    controls.target.set(...v.t);
    camera.position.set(...v.p);
    controls.update();
    dirty = true;
  }

  // Looks at the target from a preset direction, keeping the distance.
  const VIEWS = { front: [0, 0, 1], top: [0, 1, 0.001], side: [1, 0, 0], iso: [1, 0.8, 1] };
  function lookFrom(name) {
    const d = VIEWS[name];
    if (!d || mode === '2d') return;
    const dist = camera.position.distanceTo(controls.target);
    camera.up.set(0, 1, 0);
    controls.stop?.();
    camera.position.copy(controls.target).add(new T.Vector3(...d).setLength(dist));
    controls.update();
    dirty = true;
  }

  // labels
  const v = new T.Vector3();
  function updateLabels() {
    if (!settings.labels) return;
    const w = root.clientWidth; const h = root.clientHeight;
    const placed = [];
    const maxLabels = settings.maxLabels;
    let shown = 0;
    for (const { id, el } of labelList) {
      const q = boxes.get(id) || pos.get(id);
      const isBox = boxes.has(id);
      // Containers are labelled at their top-left front corner, so the label doesn't cover the
      // label of the topmost child, which sits at the same top-center point.
      const flat = mode === '2d';
      if (isBox) v.set(q.x - q.sx / 2, q.y + q.sy / 2, q.z + q.sz / 2).project(camera);
      else v.set(q.x, flat ? q.y : q.y + q.sy / 2, q.z + (flat ? q.sz / 2 : 0)).project(camera);
      const x = (v.x + 1) / 2 * w; const y = (1 - v.y) / 2 * h;
      const lw = el.textContent.length * 6.4 + 8; const lh = 16;
      const rect = isBox
        ? (flat ? [x + 4, y + 2, x + 4 + lw, y + 2 + lh] : [x - lw, y - lh - 2, x, y - 2])
        : (flat ? [x - lw / 2, y - lh / 2, x + lw / 2, y + lh / 2] : [x - lw / 2, y - lh - 2, x + lw / 2, y - 2]);
      const visible = v.z < 1 && x > -lw && x < w + lw && y > 0 && y < h + lh && shown < maxLabels
        && !(marks.match && !marks.match.has(id) && !isBox)
        && !placed.some((r) => r[0] < rect[2] && rect[0] < r[2] && r[1] < rect[3] && rect[1] < r[3]);
      if (visible) {
        placed.push(rect); shown++;
        el.style.transform = `translate(${Math.round(rect[0])}px, ${Math.round(rect[1])}px)`;
        el.hidden = false;
      } else {
        el.hidden = true;
      }
    }
  }

  // picking
  const ray = new T.Raycaster();
  const ndc = new T.Vector2();
  function pick(ev) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObjects(pickables, false);
    // Prefer a box over the shell of the container around it.
    const hit = hits.find((x) => !x.object.userData.container) || hits[0];
    return hit?.object.userData.id ?? null;
  }
  let down = null;
  const onDown = (ev) => { down = { x: ev.clientX, y: ev.clientY }; };
  const onUp = (ev) => {
    if (!down || ev.button !== 0) return;
    const moved = Math.hypot(ev.clientX - down.x, ev.clientY - down.y);
    down = null;
    if (moved > 5) return;
    const id = pick(ev);
    if (!id) return;
    const changed = marks.selected !== id;
    marks.selected = id;
    if (settings.isolate && changed) rebuild(false); else applyMarks();
    opts.onTap?.(id);
  };
  const onDbl = (ev) => { const id = pick(ev); if (id) opts.onDblTap?.(id); };
  let hoverPending = null;
  const onMove = (ev) => {
    if (hoverPending) return;
    hoverPending = requestAnimationFrame(() => {
      hoverPending = null;
      if (ev.buttons) { tip.hidden = true; return; }
      const id = pick(ev);
      const n = id && view?.nodes.find((x) => x.id === id);
      renderer.domElement.style.cursor = n ? 'pointer' : '';
      if (!n) { tip.hidden = true; return; }
      const r = root.getBoundingClientRect();
      tip.replaceChildren(
        Object.assign(document.createElement('b'), { textContent: n.label }),
        document.createTextNode(` ${n.kind}${n.params || n.bytes ? ` · ${count(n.params)} · ${bytes(n.bytes)}` : ''}${n.removable ? ` · ${n.removable}` : ''}`));
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
  // The trackball maps pointers by the canvas's place on the page, which layout can move.
  const onRootDown = () => controls.handleResize?.();
  root.addEventListener('pointerdown', onRootDown, true);

  const minimap = createMinimap({
    root, settings,
    mode: () => mode,
    items: () => ({ pos: fullPos, shown: pos, slice }),
    camera, controls: () => controls,
    colorOf: (id) => items.get(id)?.mat.color,
    dirty: () => { dirty = true; },
  });

  panel = createControlPanel({
    root, canvas: cv, controls: () => controls, settings,
    viewsKey: opts.viewsKey,
    getView, setView,
    rebuild: () => rebuild(false),
    hasSelection: () => !!marks.selected,
    isFlat: () => mode === '2d',
    fit: () => fit(undefined, true),
    reset: () => fit(undefined, false),
    zoomBy,
    view: lookFrom,
    apply: applySettings,
    dirty: () => { dirty = true; },
  });
  applyModeControls();

  let last = performance.now();
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = (now - last) / 1000;
    const held = panel.tick(dt);
    last = now;
    // The trackball has no auto-rotate of its own: same pace as the orbit's (speed 2 = 30 s a turn).
    if (controls.kind === 'trackball' && settings.autoRotate && mode !== '2d') controls.rotateLeft((Math.PI / 30) * settings.autoRotateSpeed * Math.min(dt, 0.05));
    const moving = controls.update() || held;
    if (!moving && !dirty) return;
    dirty = false;
    renderer.render(scene, camera);
    updateLabels();
    minimap.draw();
  });

  theme();
  resize();

  return {
    backend,
    get mode() { return mode; },
    /** Full data swap: view is the /graph JSON, colorOf(node) returns a CSS colour. */
    setData(nextView, nextColorOf, refit = true) {
      view = nextView; colorOf = nextColorOf || colorOf;
      rebuild(refit);
    },
    setMode(m) { if (LAYOUTS[m] && m !== mode) { mode = m; applyModeControls(); rebuild(true); } },
    recolor(nextColorOf) {
      colorOf = nextColorOf || colorOf;
      for (const [id, it] of items) {
        if (it.container) continue;
        const n = view.nodes.find((x) => x.id === id);
        if (n) it.mat.color.set(colorOf(n));
      }
      dirty = true;
    },
    /** cut / removed: Sets of ids; match: Set of ids or null; selected: id or null. */
    setMarks(next) {
      const sel = marks.selected;
      marks = { ...marks, ...next };
      // Isolation follows the selection.
      if (settings.isolate && marks.selected !== sel) rebuild(false); else applyMarks();
    },
    getView,
    setView,
    fit: (ids) => fit(ids, true),
    zoomBy,
    resetZoom: () => fit(undefined, true),
    show(visible) { root.hidden = !visible; if (visible) resize(); },
    dispose() {
      renderer.setAnimationLoop(null);
      ro.disconnect();
      themeObs.disconnect();
      media.removeEventListener?.('change', onMedia);
      cv.removeEventListener('pointerdown', onDown);
      cv.removeEventListener('pointerup', onUp);
      cv.removeEventListener('dblclick', onDbl);
      cv.removeEventListener('pointermove', onMove);
      cv.removeEventListener('pointerleave', onLeave);
      root.removeEventListener('pointerdown', onRootDown, true);
      minimap.dispose();
      panel.dispose();
      clear();
      controls.dispose();
      boxGeo.dispose(); edgeGeo.dispose();
      renderer.dispose();
      root.remove();
    },
  };
}
