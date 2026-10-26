// Model graph renderer (wamr3d, vendored in vendor/wamr3d) over the /graph JSON: a flat 2D
// diagram, or in 3D a tower, a city skyline or a force-directed cloud.

import { count, bytes } from './ui.js';
import { loadWamr, createRenderer, cssVar } from './wamr.js';
import { structure, LAYOUTS, containerBounds } from './graph3d-layout.js';

export const MODES = { '2d': '2D', tower: '3D tower', city: '3D city', force: '3D force' };

let OrbitControls = null;
async function loadLib() {
  const w = await loadWamr();
  OrbitControls = w.OrbitControls;
  return w.T;
}

/**
 * Creates the 3D view inside `container`. Throws when wamr3d cannot start (no WebGPU and no
 * WebGL 2); the caller then keeps the 2D view.
 * opts: { onTap(id), onDblTap(id) }
 */
export async function createGraph3D(container, opts = {}) {
  const T = await loadLib();
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
  const sun = new T.DirectionalLight(0xffffff, 1.4);
  sun.position.set(30, 60, 40);
  scene.add(sun);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = true;
  // 2D: straight-on view, left drag pans, no rotation; 3D: orbit.
  function applyModeControls() {
    const flat = mode === '2d';
    controls.enableRotate = !flat;
    controls.mouseButtons = flat
      ? { LEFT: T.MOUSE.PAN, MIDDLE: T.MOUSE.DOLLY, RIGHT: T.MOUSE.PAN }
      : { LEFT: T.MOUSE.ROTATE, MIDDLE: T.MOUSE.DOLLY, RIGHT: T.MOUSE.PAN };
    controls.touches = flat ? { ONE: T.TOUCH.PAN, TWO: T.TOUCH.DOLLY_PAN } : { ONE: T.TOUCH.ROTATE, TWO: T.TOUCH.DOLLY_PAN };
    camera.fov = flat ? 20 : 45;
    camera.updateProjectionMatrix();
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
    labelList = [];
    labels.replaceChildren();
  }

  function rebuild(refit = true) {
    if (!view) return;
    clear();
    const s = structure(view);
    pos = LAYOUTS[mode](view, s);
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
    const c = all.reduce((acc, q) => [acc[0] + q.x, acc[1] + q.y, acc[2] + q.z], [0, 0, 0]).map((v) => v / Math.max(1, all.length));
    radius = Math.max(4, ...all.map((q) => Math.hypot(q.x - c[0], q.y - c[1], q.z - c[2]) + Math.max(q.sx, q.sy, q.sz) / 2));
    controls.maxDistance = radius * 8;
    controls.minDistance = 0.5;
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
    const list = (ids?.length ? ids : [...pos.keys(), ...boxes.keys()]).map((id) => pos.get(id) || boxes.get(id)).filter(Boolean);
    if (!list.length) return;
    const lo = [Infinity, Infinity, Infinity]; const hi = [-Infinity, -Infinity, -Infinity];
    for (const q of list) {
      lo[0] = Math.min(lo[0], q.x - q.sx / 2); lo[1] = Math.min(lo[1], q.y - q.sy / 2); lo[2] = Math.min(lo[2], q.z - q.sz / 2);
      hi[0] = Math.max(hi[0], q.x + q.sx / 2); hi[1] = Math.max(hi[1], q.y + q.sy / 2); hi[2] = Math.max(hi[2], q.z + q.sz / 2);
    }
    const center = new T.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
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

  // --- labels ---------------------------------------------------------------------------
  const v = new T.Vector3();
  function updateLabels() {
    const w = root.clientWidth; const h = root.clientHeight;
    const placed = [];
    const maxLabels = 160;
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

  // --- picking --------------------------------------------------------------------------
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
    if (id) { marks.selected = id; applyMarks(); opts.onTap?.(id); }
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
  controls.addEventListener('change', () => { dirty = true; });

  renderer.setAnimationLoop(() => {
    const moving = controls.update();
    if (!moving && !dirty) return;
    dirty = false;
    renderer.render(scene, camera);
    updateLabels();
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
    setMarks(next) { marks = { ...marks, ...next }; applyMarks(); },
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
      clear();
      controls.dispose();
      boxGeo.dispose(); edgeGeo.dispose();
      renderer.dispose();
      root.remove();
    },
  };
}
