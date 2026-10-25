// Shared wamr3d boot: the library (vendored in vendor/wamr3d), its orbit controls, and a
// renderer on wamr3d's automatic backend (WebGPU, else WebGL 2).

import { store } from './ui.js';

let lib = null;
let Orbit = null;

export async function loadWamr() {
  if (!lib) {
    const l = await import('./vendor/wamr3d/compat.js');
    await l.init();
    // The controls build engine values at import time, so they load after init().
    ({ OrbitControls: Orbit } = await import('./vendor/wamr3d/addons/controls/OrbitControls.js'));
    lib = l;
  }
  return { T: lib, OrbitControls: Orbit };
}

export const BACKEND_LABEL = { webgpu: 'WebGPU', 'webgpu-software': 'WebGPU (software)', webgl: 'WebGL 2' };

/** `auto` unless `?backend=webgl|webgpu` is in the page hash (or saved), for testing the fallback. */
export function requestedBackend() {
  const q = new URLSearchParams(location.hash.split('?')[1] || '').get('backend') || store.get('nd-backend');
  return q === 'webgl' || q === 'webgpu' ? q : 'auto';
}

export async function createRenderer(T) {
  const renderer = new T.WebGPURenderer({ antialias: true, backend: requestedBackend() });
  await renderer.init();
  const name = renderer.backend?.name;
  return { renderer, backend: BACKEND_LABEL[name] || name || '?' };
}

/** Why 3D is unavailable, in words. */
export function noGpuMessage(e) {
  const secure = window.isSecureContext;
  return '3D needs WebGPU or WebGL 2, and this browser provides neither'
    + (secure ? '' : ' (this page is not a secure context — plain HTTP on a non-localhost address — so WebGPU is off; WebGL 2 should still work if hardware acceleration is enabled)')
    + `. ${e?.message || e || ''}`.trimEnd();
}

export function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Ray / axis-aligned box intersection distance (or null). */
export function rayBox(o, d, b) {
  let tmin = -Infinity; let tmax = Infinity;
  const lo = [b.x - b.sx / 2, b.y - b.sy / 2, b.z - b.sz / 2];
  const hi = [b.x + b.sx / 2, b.y + b.sy / 2, b.z + b.sz / 2];
  const oo = [o.x, o.y, o.z]; const dd = [d.x, d.y, d.z];
  for (let i = 0; i < 3; i++) {
    if (Math.abs(dd[i]) < 1e-12) {
      if (oo[i] < lo[i] || oo[i] > hi[i]) return null;
      continue;
    }
    let t1 = (lo[i] - oo[i]) / dd[i]; let t2 = (hi[i] - oo[i]) / dd[i];
    if (t1 > t2) [t1, t2] = [t2, t1];
    tmin = Math.max(tmin, t1); tmax = Math.min(tmax, t2);
    if (tmin > tmax) return null;
  }
  return tmax < 0 ? null : Math.max(tmin, 0);
}

/** Smooth camera move of controls.target / camera.position to new values. */
export function flyTo(camera, controls, target, position, ms = 700, onFrame) {
  const t0 = performance.now();
  const a = controls.target.clone(); const b = camera.position.clone();
  return new Promise((resolve) => {
    const step = () => {
      const k = Math.min(1, (performance.now() - t0) / ms);
      const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
      controls.target.set(a.x + (target.x - a.x) * e, a.y + (target.y - a.y) * e, a.z + (target.z - a.z) * e);
      camera.position.set(b.x + (position.x - b.x) * e, b.y + (position.y - b.y) * e, b.z + (position.z - b.z) * e);
      controls.update();
      onFrame?.();
      if (k < 1) requestAnimationFrame(step); else resolve();
    };
    step();
  });
}
