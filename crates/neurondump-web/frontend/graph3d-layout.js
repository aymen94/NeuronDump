// Layouts of the graph view for the wamr3d renderer (graph3d.js): flat 2D, tower, city and
// force-directed. Pure functions of the /graph JSON: positions and sizes, no drawing.

// --- layout -----------------------------------------------------------------------------
// Every layout places leaves (nodes with no visible children) as boxes; containers are drawn
// as outlines around their descendants. Positions are box centers, sizes full extents.

export function structure(view) {
  const byId = new Map(view.nodes.map((n, i) => [n.id, { n, i }]));
  const kids = new Map();
  for (const n of view.nodes) {
    if (n.parent && byId.has(n.parent)) {
      if (!kids.has(n.parent)) kids.set(n.parent, []);
      kids.get(n.parent).push(n.id);
    }
  }
  const rootOf = (id) => {
    let cur = byId.get(id)?.n;
    while (cur?.parent && byId.has(cur.parent)) cur = byId.get(cur.parent).n;
    return cur?.id ?? id;
  };
  const flow = view.edges.filter((e) => e.kind === 'dataflow' && byId.has(e.from) && byId.has(e.to));
  // Order ids topologically along dataflow edges (mapped through `map`), ties by server order.
  const order = (ids, map = (x) => x) => {
    const set = new Set(ids);
    const indeg = new Map(ids.map((x) => [x, 0]));
    const out = new Map(ids.map((x) => [x, new Set()]));
    for (const e of flow) {
      const a = map(e.from); const b = map(e.to);
      if (a === b || !set.has(a) || !set.has(b) || out.get(a).has(b)) continue;
      out.get(a).add(b); indeg.set(b, indeg.get(b) + 1);
    }
    const idx = (x) => byId.get(x)?.i ?? 0;
    const ready = ids.filter((x) => indeg.get(x) === 0).sort((a, b) => idx(a) - idx(b));
    const res = [];
    const seen = new Set();
    while (res.length < ids.length) {
      if (!ready.length) { // cycle: take the earliest remaining node
        const next = ids.filter((x) => !seen.has(x)).sort((a, b) => idx(a) - idx(b))[0];
        ready.push(next);
      }
      const x = ready.shift();
      if (seen.has(x)) continue;
      seen.add(x); res.push(x);
      for (const y of out.get(x)) {
        indeg.set(y, indeg.get(y) - 1);
        if (indeg.get(y) === 0 && !seen.has(y)) { ready.push(y); ready.sort((a, b) => idx(a) - idx(b)); }
      }
    }
    return res;
  };
  const roots = order(view.nodes.filter((n) => !n.parent || !byId.has(n.parent)).map((n) => n.id), rootOf);
  const leaves = view.nodes.filter((n) => !kids.has(n.id));
  const maxB = Math.max(1, ...leaves.map((n) => n.bytes || n.params || 0));
  const weight = (n) => Math.cbrt((n.bytes || n.params || 0) / maxB); // 0..1
  return { byId, kids, roots, order, weight, flow };
}

const COARSE = new Set(['model', 'section', 'layer', 'group']);

// Nested grid on the XZ plane; each root becomes one floor, floors stacked upward in flow order.
function layoutTower(view, s) {
  const pos = new Map();
  const GAP = 0.6; const PAD = 0.5; const LIFT = 0.3; const FLOOR = 0.5;
  const sizes = new Map();
  // Footprint (w × d) and height of a subtree, with its children's grid.
  const measure = (id) => {
    if (sizes.has(id)) return sizes.get(id);
    const n = s.byId.get(id).n;
    const children = s.kids.get(id);
    let m;
    if (!children) {
      const w = s.weight(n);
      const coarse = COARSE.has(n.kind);
      const side = coarse ? 2 + 6 * w : 0.6 + 2.2 * w;
      m = { w: side, d: side, h: coarse ? 0.6 : 0.4 + 0.8 * w };
    } else {
      const ordered = s.order(children);
      // A run of transformer layers becomes floors stacked upward; anything else a grid.
      const layers = ordered.filter((c) => s.byId.get(c).n.kind === 'layer').length;
      if (ordered.length > 1 && layers / ordered.length >= 0.7) {
        const ms = ordered.map(measure);
        const w = Math.max(...ms.map((x) => x.w)); const d = Math.max(...ms.map((x) => x.d));
        const h = ms.reduce((a, x) => a + x.h, 0) + FLOOR * (ms.length - 1);
        m = { w: w + 2 * PAD, d: d + 2 * PAD, h: h + 2 * LIFT, ordered, stack: ms };
        sizes.set(id, m);
        return m;
      }
      const cols = Math.ceil(Math.sqrt(ordered.length));
      const colW = []; const rowD = []; let h = 0;
      ordered.forEach((c, i) => {
        const cm = measure(c);
        const col = i % cols; const row = Math.floor(i / cols);
        colW[col] = Math.max(colW[col] || 0, cm.w); rowD[row] = Math.max(rowD[row] || 0, cm.d);
        h = Math.max(h, cm.h);
      });
      const sum = (xs) => xs.reduce((a, b) => a + b, 0);
      m = { w: sum(colW) + GAP * (colW.length - 1) + 2 * PAD, d: sum(rowD) + GAP * (rowD.length - 1) + 2 * PAD, h: h + 2 * LIFT, ordered, cols, colW, rowD };
    }
    sizes.set(id, m);
    return m;
  };
  const place = (id, x, y, z) => {
    const m = measure(id);
    if (!m.ordered) { pos.set(id, { x, y: y + m.h / 2, z, sx: m.w, sy: m.h, sz: m.d }); return; }
    if (m.stack) {
      let fy = y + LIFT;
      m.ordered.forEach((c, i) => { place(c, x, fy, z); fy += m.stack[i].h + FLOOR; });
      return;
    }
    const x0 = x - m.w / 2 + PAD; const z0 = z - m.d / 2 + PAD;
    m.ordered.forEach((c, i) => {
      const col = i % m.cols; const row = Math.floor(i / m.cols);
      const cx = x0 + m.colW.slice(0, col).reduce((a, b) => a + b, 0) + GAP * col + m.colW[col] / 2;
      const cz = z0 + m.rowD.slice(0, row).reduce((a, b) => a + b, 0) + GAP * row + m.rowD[row] / 2;
      place(c, cx, y + LIFT, cz);
    });
  };
  let y = 0;
  for (const r of s.roots) {
    place(r, 0, y, 0);
    y += measure(r).h + 1.4;
  }
  return pos;
}

// Flat diagram on the XY plane, read top to bottom: roots in flow order, a run of layers
// stacked, other children in a grid. Seen straight on, without rotation.
function layout2d(view, s) {
  const pos = new Map();
  const GAP = 0.5; const PAD = 0.45; const HEAD = 0.9; const DEPTH = 0.2;
  const sizes = new Map();
  const measure = (id) => {
    if (sizes.has(id)) return sizes.get(id);
    const n = s.byId.get(id).n;
    const children = s.kids.get(id);
    let m;
    if (!children) {
      const w = s.weight(n);
      const coarse = COARSE.has(n.kind);
      m = { w: coarse ? 4 + 6 * w : 1.6 + 3 * w, h: coarse ? 1.2 : 0.9 };
    } else {
      const ordered = s.order(children);
      const ms = ordered.map(measure);
      const layers = ordered.filter((c) => s.byId.get(c).n.kind === 'layer').length;
      // Collapsed layers stack top to bottom; expanded layers sit side by side as columns, and
      // a layer's own blocks run top to bottom in flow order. Anything else is a grid.
      const layerRun = ordered.length > 1 && layers / ordered.length >= 0.7;
      const cols = layerRun ? (ordered.some((c) => s.kids.has(c)) ? ordered.length : 1)
        : n.kind === 'layer' ? 1 : Math.ceil(Math.sqrt(ordered.length));
      const colW = []; const rowH = [];
      ms.forEach((cm, i) => {
        const col = i % cols; const row = Math.floor(i / cols);
        colW[col] = Math.max(colW[col] || 0, cm.w); rowH[row] = Math.max(rowH[row] || 0, cm.h);
      });
      const sum = (xs) => xs.reduce((a, b) => a + b, 0);
      m = { w: sum(colW) + GAP * (colW.length - 1) + 2 * PAD, h: sum(rowH) + GAP * (rowH.length - 1) + PAD + HEAD, ordered, cols, colW, rowH };
    }
    sizes.set(id, m);
    return m;
  };
  // (x, top) is the top-center of the subtree.
  const place = (id, x, top) => {
    const m = measure(id);
    if (!m.ordered) { pos.set(id, { x, y: top - m.h / 2, z: 0, sx: m.w, sy: m.h, sz: DEPTH }); return; }
    const x0 = x - m.w / 2 + PAD;
    m.ordered.forEach((c, i) => {
      const col = i % m.cols; const row = Math.floor(i / m.cols);
      const cx = x0 + m.colW.slice(0, col).reduce((a, b) => a + b, 0) + GAP * col + m.colW[col] / 2;
      const ct = top - HEAD - m.rowH.slice(0, row).reduce((a, b) => a + b, 0) - GAP * row;
      place(c, cx, ct);
    });
  };
  let top = 0;
  for (const r of s.roots) {
    place(r, 0, top);
    top -= measure(r).h + 1.2;
  }
  return pos;
}

// Roots side by side along X in flow order; leaves of a root in a grid, height = size.
function layoutCity(view, s) {
  const pos = new Map();
  const cell = 1.6;
  let x = 0;
  for (const r of s.roots) {
    const leaves = [];
    const walk = (id) => { const k = s.kids.get(id); if (!k) leaves.push(id); else s.order(k).forEach(walk); };
    walk(r);
    const rows = Math.max(1, Math.ceil(Math.sqrt(leaves.length / 2)));
    const cols = Math.ceil(leaves.length / rows);
    leaves.forEach((id, i) => {
      const n = s.byId.get(id).n;
      const h = 0.3 + 9 * s.weight(n) ** 1.5;
      const col = Math.floor(i / rows); const row = i % rows;
      pos.set(id, { x: x + col * cell, y: h / 2, z: (row - (rows - 1) / 2) * cell, sx: 1.1, sy: h, sz: 1.1 });
    });
    x += cols * cell + 2.2;
  }
  // Center on X.
  for (const p of pos.values()) p.x -= (x - 2.2 - cell) / 2;
  return pos;
}

// Force-directed in 3D, seeded from the tower so runs are stable.
function layoutForce(view, s) {
  const seed = layoutTower(view, s);
  const ids = [...seed.keys()];
  const n = ids.length;
  const index = new Map(ids.map((id, i) => [id, i]));
  const p = new Float64Array(n * 3);
  ids.forEach((id, i) => { const q = seed.get(id); p[i * 3] = q.x * 0.6; p[i * 3 + 1] = q.y * 0.6; p[i * 3 + 2] = q.z * 0.6; });
  // Springs: dataflow between leaves, plus each leaf to the next leaf of the same root.
  const springs = [];
  for (const e of s.flow) {
    const a = index.get(e.from); const b = index.get(e.to);
    if (a !== undefined && b !== undefined && a !== b) springs.push([a, b, 3]);
  }
  const rootOf = new Map();
  const walk = (id, root) => { const k = s.kids.get(id); if (!k) rootOf.set(id, root); else k.forEach((c) => walk(c, root)); };
  s.roots.forEach((r) => walk(r, r));
  const byRoot = new Map();
  for (const id of ids) { const r = rootOf.get(id); if (!byRoot.has(r)) byRoot.set(r, []); byRoot.get(r).push(index.get(id)); }
  for (const group of byRoot.values()) for (let i = 1; i < group.length; i++) springs.push([group[i - 1], group[i], 1.5]);
  const iters = Math.max(40, Math.min(300, Math.floor(3e7 / Math.max(1, n * n))));
  const f = new Float64Array(n * 3);
  for (let it = 0; it < iters; it++) {
    const t = 1 - it / iters;
    f.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = p[i * 3] - p[j * 3]; let dy = p[i * 3 + 1] - p[j * 3 + 1]; let dz = p[i * 3 + 2] - p[j * 3 + 2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; dz = Math.random() - 0.5; d2 = 0.01; }
        const k = 6 / d2;
        f[i * 3] += dx * k; f[i * 3 + 1] += dy * k; f[i * 3 + 2] += dz * k;
        f[j * 3] -= dx * k; f[j * 3 + 1] -= dy * k; f[j * 3 + 2] -= dz * k;
      }
    }
    for (const [a, b, len] of springs) {
      const dx = p[b * 3] - p[a * 3]; const dy = p[b * 3 + 1] - p[a * 3 + 1]; const dz = p[b * 3 + 2] - p[a * 3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.01;
      const k = (d - len) * 0.08 / d;
      f[a * 3] += dx * k; f[a * 3 + 1] += dy * k; f[a * 3 + 2] += dz * k;
      f[b * 3] -= dx * k; f[b * 3 + 1] -= dy * k; f[b * 3 + 2] -= dz * k;
    }
    for (let i = 0; i < n * 3; i++) {
      const v = f[i] - p[i] * 0.002; // weak pull to the center
      p[i] += Math.max(-1, Math.min(1, v)) * t;
    }
  }
  // Rescale so the median spring is ~3.5 units long, whatever the repulsion did to the spread.
  const lens = springs.map(([a, b]) => Math.hypot(p[a * 3] - p[b * 3], p[a * 3 + 1] - p[b * 3 + 1], p[a * 3 + 2] - p[b * 3 + 2])).sort((a, b) => a - b);
  const scale = lens.length ? 3.5 / Math.max(0.1, lens[Math.floor(lens.length / 2)]) : 1;
  const pos = new Map();
  ids.forEach((id, i) => {
    const q = seed.get(id);
    const side = Math.min(2.4, Math.max(1, Math.max(q.sx, q.sz) * 0.5));
    pos.set(id, { x: p[i * 3] * scale, y: p[i * 3 + 1] * scale, z: p[i * 3 + 2] * scale, sx: side, sy: side, sz: side });
  });
  return pos;
}

export const LAYOUTS = { '2d': layout2d, tower: layoutTower, city: layoutCity, force: layoutForce };

// Container boxes: bounds of their descendants plus padding, bottom-up.
export function containerBounds(view, s, pos, head = 0) {
  const out = new Map();
  const bounds = (id) => {
    if (pos.has(id) && !s.kids.has(id)) {
      const p = pos.get(id);
      return [p.x - p.sx / 2, p.y - p.sy / 2, p.z - p.sz / 2, p.x + p.sx / 2, p.y + p.sy / 2, p.z + p.sz / 2];
    }
    const k = s.kids.get(id);
    if (!k) return null;
    const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (const c of k) {
      const cb = bounds(c);
      if (!cb) continue;
      for (let i = 0; i < 3; i++) { b[i] = Math.min(b[i], cb[i]); b[i + 3] = Math.max(b[i + 3], cb[i + 3]); }
    }
    const pad = 0.35;
    for (let i = 0; i < 3; i++) { b[i] -= pad; b[i + 3] += pad; }
    b[4] += head; // room for the label above the children (2D)
    out.set(id, { x: (b[0] + b[3]) / 2, y: (b[1] + b[4]) / 2, z: (b[2] + b[5]) / 2, sx: b[3] - b[0], sy: b[4] - b[1], sz: b[5] - b[2] });
    return b;
  };
  s.roots.forEach(bounds);
  return out;
}

// --- renderer ---------------------------------------------------------------------------
