// The vocabulary as a 3D point cloud: token_embd rows projected to 3D with PCA (computed by
// the server, served as a Float32Array: x, y, z, class per token), drawn with wamr3d as one
// instanced mesh. Search, class filters, click to inspect, and the prompt's tokens as a path.

import { api, apiBinary } from './api.js';
import { el, esc, errorBox } from './ui.js';
import { loadWamr, createRenderer, cssVar, noGpuMessage, flyTo } from './wamr.js';

export const CLASS_COLORS = ['#d9534f', '#9c755f', '#bab0ac', '#76b7b2', '#edc948', '#b07aa1', '#ff9da7', '#e15759', '#59a14f', '#4e79a7', '#f28e2b', '#8cd17d'];
const R = 40;

/**
 * Mounts the map in `container`. opts: { onPick(tokenId) }.
 * Returns { backend, info, search(q), showTokens(ids), focusToken(id), dispose } or throws.
 */
export async function createEmbeddingMap(container, modelId, opts = {}) {
  const status = el('p', { class: 'muted' }, 'Computing the embedding map (PCA of token_embd; the first time takes a few seconds)…');
  container.replaceChildren(status);
  const [bin, vocab] = await Promise.all([apiBinary(`/models/${modelId}/embedding-map`), api(`/models/${modelId}/vocab/texts`)]);
  const data = new Float32Array(bin.buffer);
  const stride = Number(bin.headers.get('x-neurondump-stride')) || 4;
  const n = Math.floor(data.length / stride);
  const explained = (bin.headers.get('x-neurondump-explained') || '').split(',').map(Number);
  const texts = vocab.texts;
  const classNames = vocab.classes;

  let T; let OrbitControls; let renderer; let backend;
  try {
    ({ T, OrbitControls } = await loadWamr());
    ({ renderer, backend } = await createRenderer(T));
  } catch (e) {
    container.replaceChildren(errorBox(new Error(noGpuMessage(e))));
    throw e;
  }
  const root = el('div', { class: 'graph3d embedmap' });
  const tip = el('div', { class: 'graph3d-tip', hidden: true });
  root.append(renderer.domElement, tip);
  container.replaceChildren(root);

  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(50, 1, 0.1, 2000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  camera.position.set(R * 1.4, R * 0.8, R * 1.6);
  controls.target.set(0, 0, 0);
  controls.update();

  const geo = new T.OctahedronGeometry(1, 0);
  const mesh = new T.InstancedMesh(geo, new T.MeshBasicMaterial({ color: 0xffffff }), n);
  scene.add(mesh);
  const m4 = new T.Matrix4(); const q0 = new T.Quaternion(); const c = new T.Color();
  const hidden = new Set(); // class indices
  let marked = new Set(); // searched / prompt tokens
  const P = (i) => ({ x: data[i * stride] * R, y: data[i * stride + 1] * R, z: data[i * stride + 2] * R });
  const cls = (i) => data[i * stride + 3] | 0;
  const baseSize = n > 100000 ? 0.09 : n > 40000 ? 0.12 : 0.16;
  function paint() {
    for (let i = 0; i < n; i++) {
      const k = cls(i);
      const on = !hidden.has(k);
      const big = marked.has(i);
      const s = !on ? 0 : big ? baseSize * 4.5 : classNames[k] === 'special' ? baseSize * 2.2 : baseSize;
      const p = P(i);
      m4.compose(p, q0, { x: s, y: s, z: s });
      mesh.setMatrixAt(i, m4);
      c.set(big ? cssVar('--accent', '#2457c5') : CLASS_COLORS[k % CLASS_COLORS.length]);
      if (marked.size && !big) c.lerp(new T.Color(cssVar('--bg-alt', '#f4f5f7')), 0.55);
      mesh.setColorAt(i, c);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    dirty = true;
  }

  let path = null;
  function setPath(ids) {
    if (path) { scene.remove(path); path.geometry.dispose(); path.material.dispose(); path = null; }
    if (ids.length < 2) return;
    const pts = new Float32Array((ids.length - 1) * 6);
    for (let i = 1; i < ids.length; i++) {
      const a = P(ids[i - 1]); const b = P(ids[i]);
      pts.set([a.x, a.y, a.z, b.x, b.y, b.z], (i - 1) * 6);
    }
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.Float32BufferAttribute(pts, 3));
    path = new T.LineSegments(g, new T.LineBasicMaterial({ color: cssVar('--accent', '#2457c5') }));
    scene.add(path);
    dirty = true;
  }

  let dirty = true;
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
  renderer.setClearColor(cssVar('--bg-alt', '#f4f5f7'));

  // Screen-space picking: project every visible token once per query (cheap next to a ray
  // test against 100k+ instances).
  const vp = new T.Matrix4();
  function pick(ev, maxPx = 10) {
    const r = renderer.domElement.getBoundingClientRect();
    const mx = ev.clientX - r.left; const my = ev.clientY - r.top;
    camera.updateMatrixWorld();
    vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const e = vp.elements;
    let best = -1; let bd = maxPx * maxPx;
    for (let i = 0; i < n; i++) {
      if (hidden.has(cls(i))) continue;
      const x = data[i * stride] * R; const y = data[i * stride + 1] * R; const z = data[i * stride + 2] * R;
      const w = e[3] * x + e[7] * y + e[11] * z + e[15];
      if (w <= 0) continue;
      const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w + 1) / 2 * r.width;
      const sy = (1 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w) / 2 * r.height;
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bd || (marked.has(i) && d < bd * 4)) { bd = d; best = i; }
    }
    return best;
  }
  let down = null;
  const cv = renderer.domElement;
  cv.addEventListener('pointerdown', (ev) => { down = { x: ev.clientX, y: ev.clientY }; });
  cv.addEventListener('pointerup', (ev) => {
    if (!down || Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > 5) { down = null; return; }
    down = null;
    const i = pick(ev);
    if (i >= 0) opts.onPick?.(i);
  });
  let hoverTimer = null;
  cv.addEventListener('pointermove', (ev) => {
    clearTimeout(hoverTimer);
    if (ev.buttons) { tip.hidden = true; return; }
    hoverTimer = setTimeout(() => {
      const i = pick(ev);
      if (i < 0) { tip.hidden = true; cv.style.cursor = ''; return; }
      cv.style.cursor = 'pointer';
      const r = root.getBoundingClientRect();
      tip.replaceChildren(el('b', {}, `"${esc(texts[i] ?? '')}"`), ` #${i} · ${classNames[cls(i)] || '?'}`);
      tip.style.transform = `translate(${Math.round(ev.clientX - r.left + 12)}px, ${Math.round(ev.clientY - r.top + 12)}px)`;
      tip.hidden = false;
    }, n > 60000 ? 60 : 16);
  });
  cv.addEventListener('pointerleave', () => { tip.hidden = true; });
  controls.addEventListener('change', () => { dirty = true; });
  renderer.setAnimationLoop(() => {
    const moving = controls.update();
    if (!moving && !dirty) return;
    dirty = false;
    renderer.render(scene, camera);
  });
  paint();
  resize();

  const focusOn = (ids) => {
    if (!ids.length) return;
    let cx = 0; let cy = 0; let cz = 0;
    for (const i of ids) { const p = P(i); cx += p.x; cy += p.y; cz += p.z; }
    cx /= ids.length; cy /= ids.length; cz /= ids.length;
    let rad = 2;
    for (const i of ids) { const p = P(i); rad = Math.max(rad, Math.hypot(p.x - cx, p.y - cy, p.z - cz)); }
    const target = new T.Vector3(cx, cy, cz);
    const dir = camera.position.clone().sub(controls.target).normalize();
    flyTo(camera, controls, target, target.clone().add(dir.multiplyScalar(rad * 2.4 + 6)), 600, () => { dirty = true; });
  };

  return {
    backend,
    info: { count: n, explained, classes: classNames, counts: classNames.map((_, k) => { let s = 0; for (let i = 0; i < n; i++) if (cls(i) === k) s++; return s; }) },
    /** Marks tokens whose text contains q; returns their ids (at most 5000). */
    search(q) {
      const s = q.trim().toLowerCase();
      const ids = [];
      if (s) for (let i = 0; i < n && ids.length < 5000; i++) if ((texts[i] || '').toLowerCase().includes(s)) ids.push(i);
      marked = new Set(ids);
      setPath([]);
      paint();
      focusOn(ids);
      return ids;
    },
    /** Marks a prompt's tokens and draws the path through them in order. */
    showTokens(ids) {
      marked = new Set(ids);
      paint();
      setPath(ids);
      focusOn(ids);
    },
    focusToken(i) { marked = new Set([i]); paint(); focusOn([i]); },
    toggleClass(k, on) { if (on) hidden.delete(k); else hidden.add(k); paint(); },
    text: (i) => texts[i],
    className: (i) => classNames[cls(i)],
    dispose() {
      renderer.setAnimationLoop(null);
      ro.disconnect();
      setPath([]);
      mesh.material.dispose();
      geo.dispose();
      controls.dispose();
      renderer.dispose();
      root.remove();
    },
  };
}
