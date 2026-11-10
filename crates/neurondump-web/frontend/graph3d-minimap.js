// Minimap of the graph view: the whole layout seen from above (front for 2D), the nodes
// shown in their colors (hidden ones faint), the slice plane, and the camera with its view
// direction. Click or drag on it to move the camera there.

import { cssVar } from './wamr.js';

/**
 * ctx: { root, settings, mode(), items() -> { pos (full layout), shown, slice }, camera,
 *        controls(), colorOf(id) -> Color, dirty() }
 */
export function createMinimap(ctx) {
  const W = 168; const H = 120; const PAD = 8;
  const cv = document.createElement('canvas');
  cv.className = 'g3-minimap';
  cv.title = 'Minimap: click or drag to move the camera';
  const dpr = Math.min(2, devicePixelRatio || 1);
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.root.append(cv);
  const g = cv.getContext('2d');
  // Projection plane: x/z from above in 3D, x/y from the front in 2D.
  const axes = () => (ctx.mode() === '2d' ? ['x', 'y', -1] : ['x', 'z', 1]);
  let frame = null; // { lo0, lo1, k, ox, oy }

  function bounds(pos) {
    const [a, b] = axes();
    let lo0 = Infinity; let hi0 = -Infinity; let lo1 = Infinity; let hi1 = -Infinity;
    for (const p of pos.values()) {
      lo0 = Math.min(lo0, p[a] - p['s' + a] / 2); hi0 = Math.max(hi0, p[a] + p['s' + a] / 2);
      lo1 = Math.min(lo1, p[b] - p['s' + b] / 2); hi1 = Math.max(hi1, p[b] + p['s' + b] / 2);
    }
    const k = Math.min((W - 2 * PAD) / Math.max(1e-6, hi0 - lo0), (H - 2 * PAD) / Math.max(1e-6, hi1 - lo1));
    return { lo0, lo1, hi0, hi1, k, ox: (W - (hi0 - lo0) * k) / 2, oy: (H - (hi1 - lo1) * k) / 2 };
  }
  // World (a, b) -> minimap pixel; `flip` turns 2D's y up.
  const toPx = (f, u, v) => {
    const [, , flip] = axes();
    return [f.ox + (u - f.lo0) * f.k, flip < 0 ? H - (f.oy + (v - f.lo1) * f.k) : f.oy + (v - f.lo1) * f.k];
  };
  const toWorld = (f, x, y) => {
    const [, , flip] = axes();
    return [f.lo0 + (x - f.ox) / f.k, f.lo1 + ((flip < 0 ? H - y : y) - f.oy) / f.k];
  };

  function draw() {
    cv.hidden = !ctx.settings.minimap;
    if (cv.hidden) return;
    const { pos, shown, slice } = ctx.items();
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.fillStyle = cssVar('--bg', '#ffffff');
    g.globalAlpha = 0.85;
    g.fillRect(0, 0, W, H);
    g.globalAlpha = 1;
    if (!pos.size) { frame = null; return; }
    const f = bounds(pos);
    frame = f;
    const [a, b] = axes();
    for (const [id, p] of pos) {
      const [x0, y0] = toPx(f, p[a] - p['s' + a] / 2, p[b] - p['s' + b] / 2);
      const [x1, y1] = toPx(f, p[a] + p['s' + a] / 2, p[b] + p['s' + b] / 2);
      const c = shown.has(id) ? ctx.colorOf(id) : null;
      g.fillStyle = c?.getStyle ? c.getStyle() : '#bbbbbb';
      g.globalAlpha = c ? 0.95 : 0.25;
      g.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.max(1, Math.abs(x1 - x0)), Math.max(1, Math.abs(y1 - y0)));
    }
    g.globalAlpha = 1;
    // Slice plane, when its axis lies in the minimap plane.
    if (slice && slice.cut < slice.hi && (slice.axis === a || slice.axis === b)) {
      g.strokeStyle = '#d9534f'; g.lineWidth = 1.5; g.setLineDash([4, 3]);
      g.beginPath();
      if (slice.axis === a) { const [x] = toPx(f, slice.cut, f.lo1); g.moveTo(x, 0); g.lineTo(x, H); } else { const [, y] = toPx(f, f.lo0, slice.cut); g.moveTo(0, y); g.lineTo(W, y); }
      g.stroke(); g.setLineDash([]);
    }
    // Camera: position, target and view direction.
    const t = ctx.controls().target; const c = ctx.camera.position;
    const [tx, ty] = toPx(f, t[a], t[b]);
    const [cx, cy] = toPx(f, c[a], c[b]);
    const accent = cssVar('--accent', '#2457c5');
    g.strokeStyle = accent; g.fillStyle = accent; g.lineWidth = 1.5;
    if (ctx.mode() !== '2d') {
      // Clamp the camera dot to the frame and draw the line of sight to the target.
      const ccx = Math.max(3, Math.min(W - 3, cx)); const ccy = Math.max(3, Math.min(H - 3, cy));
      g.beginPath(); g.moveTo(ccx, ccy); g.lineTo(tx, ty); g.stroke();
      g.beginPath(); g.arc(ccx, ccy, 3.5, 0, Math.PI * 2); g.fill();
    } else {
      // 2D: the visible area as a rectangle.
      const d = c.distanceTo(t);
      const hh = Math.tan((ctx.camera.fov * Math.PI) / 360) * d; const hw = hh * ctx.camera.aspect;
      const [x0, y0] = toPx(f, t.x - hw, t.y - hh); const [x1, y1] = toPx(f, t.x + hw, t.y + hh);
      g.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    }
    g.beginPath(); g.arc(tx, ty, 2.5, 0, Math.PI * 2); g.fill();
    g.strokeStyle = cssVar('--border', '#cccccc'); g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, W - 1, H - 1);
  }

  // Moves target and camera together so the target lands on the clicked point.
  function moveTo(ev) {
    if (!frame) return;
    const r = cv.getBoundingClientRect();
    const [u, v] = toWorld(frame, ev.clientX - r.left, ev.clientY - r.top);
    const [a, b] = axes();
    const controls = ctx.controls();
    const da = u - controls.target[a]; const db = v - controls.target[b];
    controls.target[a] += da; controls.target[b] += db;
    ctx.camera.position[a] += da; ctx.camera.position[b] += db;
    controls.update();
    ctx.dirty();
  }
  let dragging = false;
  const onDown = (e) => { e.stopPropagation(); e.preventDefault(); dragging = true; cv.setPointerCapture?.(e.pointerId); moveTo(e); };
  const onMove = (e) => { if (dragging) moveTo(e); };
  const onUp = () => { dragging = false; };
  cv.addEventListener('pointerdown', onDown);
  cv.addEventListener('pointermove', onMove);
  cv.addEventListener('pointerup', onUp);
  cv.addEventListener('pointercancel', onUp);
  cv.addEventListener('wheel', (e) => e.stopPropagation());

  return {
    draw,
    dispose() { cv.remove(); },
  };
}
