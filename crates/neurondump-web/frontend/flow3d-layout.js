// Layout of the 3D dataflow view (flow3d.js): layers stacked along the dataflow, blocks around
// their layer, fine nodes on rings or grids, side sections beside the column. Pure function of
// the /graph JSON.

const GAP = 4.2; // between layers
const SIDE = 16; // x of the side sections
export const SPHERE_KINDS = new Set(['head', 'kv_group', 'neuron_bucket', 'expert', 'token']);
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
