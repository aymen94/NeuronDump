// Model graph view: interactive structure, node details, a cut selection with live static
// impact, simulated ablation, sweeps, live edits and plan → slim. Two renderers, both on
// wamr3d: the flat 2D diagram (graph3d.js, which also draws the tower / city / force
// layouts) and the 3D dataflow view (flow3d.js) with the token flow animation. Node ids,
// selection, cuts, edits and highlights are shared between them.

import { api, qs, runJob } from './api.js';
import { el, bytes, count, esc, table, errorBox, toast, store } from './ui.js';
import { jobWithProgress, planView } from './views.js';
import { createGraph3D } from './graph3d.js';
import { createFlow3D } from './flow3d.js';
import { noGpuMessage } from './wamr.js';
import { session, editedNodes, loadEdits, pushEdit } from './session.js';

const LEVELS = ['section', 'layer', 'block', 'tensor'];
const LAYOUTS3D = { flow: 'dataflow', tower: 'tower', city: 'city', force: 'force' };
const GROUPS = ['section', 'layer', 'block', 'quant', 'kind', 'importance'];
const COLORS = ['removable', 'section', 'kind', 'quant', 'importance'];
const PALETTE = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'];
const REMOVABLE = { safe: '#59a14f', risky: '#e9b03c', breaks_model: '#d9534f' };
const SAMPLE_CORPUS = 'The quick brown fox jumps over the lazy dog. It was a sunny day.\n\nParis is the capital of France. Berlin is the capital of Germany.\n\nRust is a systems programming language focused on safety and speed.';

function selectEl(options, value, label) {
  return el('label', {}, label, ' ', el('select', {}, options.map((o) => el('option', { value: o, selected: o === value }, o))));
}

function heat(score) {
  // 0 → pale, 1 → strong red (readable in both themes).
  const s = Math.max(0, Math.min(1, score ?? 0));
  const r = Math.round(240 - 30 * s); const g = Math.round(240 - 190 * s); const b = Math.round(230 - 190 * s);
  return `rgb(${r},${g},${b})`;
}

export async function graph(id, params) {
  const savedView = params?.get('view') || store.get('nd-graph-view') || '3d';
  const state = {
    level: params?.get('level') || store.get('nd-graph-level') || 'layer',
    groupBy: params?.get('group_by') || 'section',
    color: store.get('nd-graph-color') || 'removable',
    expand: new Set((params?.get('expand') || '').split(',').filter(Boolean)),
    selection: [],
    view: null,
    damage: null, // node id -> damage (sweep overlay)
    removed: new Set(),
    force: false,
    // '2d', or a 3D layout: flow (dataflow, default) / tower / city / force.
    dim: savedView === '2d' ? '2d' : '3d',
    layout3d: LAYOUTS3D[savedView] ? savedView : (LAYOUTS3D[store.get('nd-graph-layout3d')] ? store.get('nd-graph-layout3d') : 'flow'),
  };
  const s = session(id);
  if (params?.get('highlight') === '1') {
    // Coming from a token journey: open the layers that hold the contributing nodes.
    for (const h of s.highlight) { const m = /^layer\.(\d+)/.exec(h); if (m) state.expand.add(`layer.${m[1]}`); }
  }
  const levelSel = selectEl(LEVELS, state.level, 'level');
  const groupSel = selectEl(GROUPS, state.groupBy, 'group by');
  const colorSel = selectEl([...COLORS, 'damage', 'flow'], state.color, 'color by');
  const search = el('input', { type: 'search', placeholder: 'search nodes / tensors (Enter flies to it)', size: 26 });
  const info = el('span', { class: 'muted' });
  const backendBadge = el('span', { class: 'badge-backend', title: 'wamr3d renderer backend (auto: WebGPU, else WebGL 2)' }, '…');
  const dimBtns = { '2d': el('button', { 'aria-checked': String(state.dim === '2d'), onclick: () => setDim('2d') }, '2D'), '3d': el('button', { 'aria-checked': String(state.dim === '3d'), onclick: () => setDim('3d') }, '3D') };
  const layoutSel = el('select', {}, Object.entries(LAYOUTS3D).map(([v, label]) => el('option', { value: v, selected: v === state.layout3d }, label)));
  const layoutLbl = el('label', { hidden: state.dim === '2d' }, '3D layout ', layoutSel);
  const canvas2 = el('div', { class: 'graph-canvas graph-canvas3d' });
  const canvasF = el('div', { class: 'graph-canvas graph-canvas3d' });
  const side = el('div', { class: 'graph-side' }, el('p', { class: 'muted' }, 'Click a node to see what it contains. Double-click to expand or collapse it.'));
  const selBox = el('div', { class: 'graph-selection' });
  const results = el('div', {});
  const flowBtn = el('button', { title: 'Animate the last token journey (or run one) along the dataflow; layers glow with their push on the prediction' }, '▶ token flow');
  let g2 = null; let gf = null; let failed2 = false; let failedF = false;

  const useFlow = () => state.dim === '3d' && state.layout3d === 'flow';
  const cur = () => (useFlow() ? gf : g2);

  // --- colors ---------------------------------------------------------------------------
  const sectionOf = (nid, byId) => {
    let n = byId.get(nid);
    while (n && n.parent && byId.has(n.parent)) n = byId.get(n.parent);
    return n ? n.id : nid;
  };
  const categorical = new Map();
  const catColor = (key) => {
    if (!categorical.has(key)) categorical.set(key, PALETTE[categorical.size % PALETTE.length]);
    return categorical.get(key);
  };
  const flowValues = () => {
    const j = s.journey;
    const out = new Map();
    if (!j) return out;
    if (j.decision) {
      const max = Math.max(1e-6, ...j.decision.layers.map((l) => Math.abs(l.attn) + Math.abs(l.ffn)));
      const maxP = Math.max(1e-6, ...j.decision.layers.map((l) => Math.max(Math.abs(l.attn), Math.abs(l.ffn))));
      for (const l of j.decision.layers) out.set(l.layer, { total: (Math.abs(l.attn) + Math.abs(l.ffn)) / max, attn: Math.abs(l.attn) / maxP, ffn: Math.abs(l.ffn) / maxP });
    } else {
      // Embedding models: growth of the residual stream per layer.
      const d = j.layers.map((l, i) => Math.abs(l.residual_norm - (i ? j.layers[i - 1].residual_norm : 0)));
      const max = Math.max(1e-6, ...d);
      d.forEach((v, i) => out.set(i, { total: v / max, attn: v / max, ffn: v / max }));
    }
    return out;
  };
  function colorOf(n, byId) {
    switch (state.color) {
      case 'section': return catColor('s:' + sectionOf(n.id, byId));
      case 'kind': return catColor('k:' + (n.layer_kind || n.block || n.kind));
      case 'quant': return catColor('q:' + (n.quant.join('+') || 'none'));
      case 'importance': return heat(n.importance?.score);
      case 'damage': {
        if (!state.damage) return '#dddddd';
        const d = state.damage.get(n.id) ?? (n.layer != null ? state.damage.get(`layer.${n.layer}`) : undefined);
        return d === undefined ? '#dddddd' : heat(d);
      }
      case 'flow': {
        const v = n.layer != null ? flowValues().get(n.layer) : undefined;
        if (!v) return '#dddddd';
        return heat(n.block === 'attention' || n.block === 'ssm' ? v.attn : n.block === 'ffn' || n.block === 'moe' ? v.ffn : v.total);
      }
      default: return REMOVABLE[n.removable] || '#cccccc';
    }
  }
  const colorFn = () => {
    const byId = new Map(state.view.nodes.map((n) => [n.id, n]));
    categorical.clear();
    return (n) => colorOf(n, byId);
  };

  // Hidden ids (fine nodes of collapsed blocks, edits on tensors) map to the closest visible
  // ancestor, so a mark is never lost.
  const visible = (ids) => {
    const have = new Set((state.view?.nodes || []).map((n) => n.id));
    const out = new Set();
    for (const x of ids) {
      let c = x;
      while (c && !have.has(c)) { const i = c.lastIndexOf('.'); c = i < 0 ? null : c.slice(0, i); }
      if (c) out.add(c);
    }
    return out;
  };
  const marks = () => ({ cut: new Set(state.selection), removed: state.removed, edited: visible(editedNodes(id)), highlight: visible(s.highlight) });

  // --- renderers --------------------------------------------------------------------------
  async function ensure() {
    const opts = { onTap: (nid) => showNode(nid), onDblTap: (nid) => toggleExpand(nid) };
    if (useFlow()) {
      if (!gf && !failedF) {
        try { gf = await createFlow3D(canvasF, opts); } catch (e) { failedF = true; canvasF.replaceChildren(errorBox(new Error(noGpuMessage(e)))); }
      }
    } else if (!g2 && !failed2) {
      try { g2 = await createGraph3D(canvas2, opts); } catch (e) { failed2 = true; canvas2.replaceChildren(errorBox(new Error(noGpuMessage(e)))); }
    }
    canvasF.hidden = !useFlow();
    canvas2.hidden = useFlow();
    const r = cur();
    backendBadge.textContent = r ? r.backend : 'no WebGPU / WebGL 2';
    backendBadge.classList.toggle('error', !r);
    flowBtn.disabled = !useFlow();
    return r;
  }

  async function render(refit = true) {
    const v = state.view;
    if (!v) return;
    const r = await ensure();
    if (!r) return;
    if (!useFlow()) r.setMode(state.dim === '2d' ? '2d' : state.layout3d);
    r.setData(v, colorFn(), refit);
    r.setMarks(marks());
  }

  function recolor() {
    const r = cur();
    if (!r || !state.view) return;
    r.recolor(colorFn());
    r.setMarks(marks());
  }

  async function reload() {
    info.textContent = 'loading…';
    try {
      const v = await api(`/models/${id}/graph${qs({ level: state.level, group_by: state.groupBy, expand: [...state.expand].join(',') })}`);
      state.view = v;
      await render();
      info.textContent = `${v.nodes.length} nodes · ${v.edges.filter((e) => e.kind !== 'contains').length} connections${state.expand.size ? ` · expanded: ${[...state.expand].join(', ')}` : ''}`;
      applySearch(false);
    } catch (e) {
      info.replaceChildren(errorBox(e));
    }
  }

  async function setDim(d) {
    state.dim = d;
    store.set('nd-graph-view', d === '2d' ? '2d' : state.layout3d);
    dimBtns['2d'].setAttribute('aria-checked', String(d === '2d'));
    dimBtns['3d'].setAttribute('aria-checked', String(d === '3d'));
    layoutLbl.hidden = d === '2d';
    if (!useFlow()) gf?.stopFlow();
    await render();
  }

  function toggleExpand(nid) {
    const n = state.view?.nodes.find((x) => x.id === nid);
    const hasVisibleChildren = state.view?.nodes.some((x) => x.parent === nid);
    if (state.expand.has(nid)) {
      state.expand.delete(nid);
      // Collapsing also collapses everything inside.
      for (const e of [...state.expand]) if (e.startsWith(nid + '.')) state.expand.delete(e);
    } else if (n && (n.lazy || !hasVisibleChildren) && (n.contains?.length || n.lazy)) {
      state.expand.add(nid);
    } else {
      return;
    }
    reload();
  }

  // --- search ------------------------------------------------------------------------------
  let tensorIndex = null;
  async function applySearch(fly = true) {
    const r = cur();
    if (!r) return;
    const q = search.value.trim().toLowerCase();
    if (!q) { r.setMarks({ match: null }); return; }
    const ids = (state.view?.nodes || []).filter((n) => n.id.toLowerCase().includes(q) || n.label.toLowerCase().includes(q)).map((n) => n.id);
    if (ids.length) {
      r.setMarks({ match: new Set(ids) });
      if (fly) { if (ids.length === 1 && r.focus) r.focus(ids[0]); else r.fit(ids); }
      return;
    }
    r.setMarks({ match: null });
    if (fly) return expandToFind(q);
  }
  async function expandToFind(q) {
    // Not visible at this level: look the name up at tensor level and expand its parents.
    if (!tensorIndex) tensorIndex = await api(`/models/${id}/graph${qs({ level: 'tensor' })}`).catch(() => null);
    const found = (tensorIndex?.nodes || []).filter((n) => n.id.toLowerCase().includes(q) || n.label.toLowerCase().includes(q)).slice(0, 12);
    if (!found.length) { toast('no match'); return; }
    let added = false;
    for (const n of found) if (n.parent && !state.expand.has(n.parent)) { state.expand.add(n.parent); added = true; }
    if (added) reload(); else toast(`${found.length} match(es) are already visible`);
  }
  search.addEventListener('keydown', (e) => { if (e.key === 'Enter') applySearch(true); });

  // --- token flow -----------------------------------------------------------------------------
  flowBtn.addEventListener('click', async () => {
    if (!gf) return;
    if (gf.flowing) { gf.stopFlow(); flowBtn.textContent = '▶ token flow'; return; }
    if (!s.journey) {
      flowBtn.disabled = true; flowBtn.textContent = 'tracing…';
      try {
        const j = await runJob('journey', id, { prompt: s.prompt, chat: s.chat, token: -1 });
        if (j.status !== 'done') throw new Error(j.error || j.status);
        s.journey = await api(`/jobs/${j.id}/result`);
      } catch (e) { toast(`token flow needs a trace: ${e.message}`); flowBtn.disabled = false; flowBtn.textContent = '▶ token flow'; return; }
      flowBtn.disabled = false;
    }
    const tok = s.journey.tokens[s.journey.position];
    gf.playFlow(flowValues(), tok?.text);
    flowBtn.textContent = '■ stop flow';
    toast(`token flow: "${esc(tok?.text ?? '')}" (from the last journey; change it on the journey page)`);
  });

  // --- node detail ---------------------------------------------------------------------------
  const edit = (op, label) => el('button', { class: 'small', onclick: async () => { try { await pushEdit(id, op); toast(`edit added: ${label}`); } catch (e) { toast(e.message); } } }, label);
  async function showNode(nid) {
    side.replaceChildren(el('p', { class: 'muted' }, 'Loading…'));
    try {
      const d = await api(`/models/${id}/nodes/${encodeURIComponent(nid)}`);
      const n = d.node;
      const cut = state.selection.includes(n.id);
      const scalable = ['layer', 'block', 'kv_group', 'head', 'neuron_bucket', 'expert'].includes(n.kind);
      const stats = n.stats ? Object.entries(n.stats).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => `${k} ${typeof v === 'number' ? v.toPrecision(4) : v}`).join(', ') : '';
      const factor = el('input', { type: 'number', step: 'any', value: 0.5, style: 'width:4.5em' });
      side.replaceChildren(
        el('h3', {}, n.label),
        el('div', { class: 'muted' }, `${n.kind} · `, el('code', {}, n.id), editedNodes(id).has(n.id) ? el('span', { class: 'badge-edited' }, 'edited') : ''),
        el('p', {}, d.ancestors.slice().reverse().map((a) => el('a', { href: '#', onclick: (e) => { e.preventDefault(); showNode(a.id); } }, a.label)).flatMap((x) => [x, ' › ']), n.label),
        el('table', { class: 'kv' }, el('tbody', {},
          el('tr', {}, el('th', {}, 'size'), el('td', {}, `${count(n.params)} params · ${bytes(n.bytes)}`)),
          n.quant.length ? el('tr', {}, el('th', {}, 'quant'), el('td', {}, n.quant.join(', '))) : '',
          n.layer_kind ? el('tr', {}, el('th', {}, 'layer kind'), el('td', {}, n.layer_kind)) : '',
          el('tr', {}, el('th', {}, 'removable'), el('td', {}, el('span', { class: `rem rem-${n.removable}` }, n.removable), ' ', n.removable_reason)),
          el('tr', {}, el('th', {}, 'importance'), el('td', {}, `${n.importance.score.toFixed(2)} — ${n.importance.source}`)),
          stats ? el('tr', {}, el('th', {}, 'weights'), el('td', {}, stats)) : '',
          n.activation ? el('tr', {}, el('th', {}, 'activation'), el('td', {}, el('code', {}, JSON.stringify(n.activation)))) : '',
          n.info ? el('tr', {}, el('th', {}, 'info'), el('td', {}, el('code', {}, JSON.stringify(n.info)))) : '')),
        el('div', { class: 'row' },
          el('button', { class: cut ? '' : 'primary', onclick: () => { toggleCut(n.id); showNode(n.id); } }, cut ? 'Uncut' : 'Cut this'),
          (n.lazy || d.children.length) ? el('button', { onclick: () => toggleExpand(n.id) }, state.expand.has(n.id) ? 'Collapse' : 'Expand') : '',
          el('button', { title: 'Fly the camera to this node', onclick: () => { const r = cur(); if (r?.focus) r.focus(n.id); else r?.fit([n.id]); } }, 'Focus'),
          (n.kind === 'block' && n.block === 'attention') ? el('button', { onclick: () => runSweep('head', n.layer) }, `Sweep heads of layer ${n.layer}`) : ''),
        el('div', { class: 'row edit-quick' }, el('strong', {}, 'Edit:'),
          edit({ op: 'zero', target: n.id }, 'zero'),
          scalable ? [edit({ op: 'scale', target: n.id, factor: 0.5 }, '×0.5'), edit({ op: 'scale', target: n.id, factor: 2 }, '×2'),
            factor, el('button', { class: 'small', onclick: () => pushEdit(id, { op: 'scale', target: n.id, factor: Number(factor.value) }).then(() => toast('edit added')).catch((e) => toast(e.message)) }, 'scale')] : '',
          n.kind === 'layer' ? edit({ op: 'skip', layer: n.layer }, 'skip layer') : ''),
        d.children.length ? el('div', {}, el('h4', {}, `Contains (${d.children.length})`), table([
          { label: 'child', get: (c) => el('a', { href: '#', onclick: (e) => { e.preventDefault(); showNode(c.id); } }, c.label) },
          { label: 'kind', get: (c) => c.kind },
          { label: 'size', get: (c) => bytes(c.bytes), num: true, sort: (c) => c.bytes },
          { label: 'removable', get: (c) => c.removable },
          { label: 'imp', get: (c) => c.importance.score.toFixed(2), num: true },
          { label: '', get: (c) => el('button', { onclick: () => { toggleCut(c.id); showNode(n.id); } }, state.selection.includes(c.id) ? 'uncut' : 'cut') },
        ], d.children.slice(0, 300))) : '',
        (d.incoming.length || d.outgoing.length) ? el('div', {}, el('h4', {}, 'Connections'),
          el('ul', { class: 'plain' },
            d.incoming.map((e) => el('li', {}, '← ', el('a', { href: '#', onclick: (ev) => { ev.preventDefault(); showNode(e.from); } }, e.from), ` (${e.kind}${e.label ? ': ' + e.label : ''})`)),
            d.outgoing.map((e) => el('li', {}, '→ ', el('a', { href: '#', onclick: (ev) => { ev.preventDefault(); showNode(e.to); } }, e.to), ` (${e.kind}${e.label ? ': ' + e.label : ''})`)))) : '');
      cur()?.setMarks({ selected: [...visible([n.id])][0] ?? n.id });
    } catch (e) {
      side.replaceChildren(errorBox(e));
    }
  }

  // --- selection + static impact -------------------------------------------------------
  const promptIn = el('input', { size: 40, value: s.prompt });
  const tokensIn = el('input', { type: 'number', min: 0, max: 64, value: 12, style: 'width:5em' });
  const corpusIn = el('textarea', { rows: 4, placeholder: 'Corpus for perplexity and sweeps (texts separated by blank lines)' }, SAMPLE_CORPUS);
  const forceIn = el('input', { type: 'checkbox', onchange: () => { state.force = forceIn.checked; renderSelection(); } });
  const topIn = el('input', { type: 'number', min: 1, value: 2, style: 'width:4em' });
  let impactSeq = 0;

  function toggleCut(nid) {
    const i = state.selection.indexOf(nid);
    if (i >= 0) state.selection.splice(i, 1); else state.selection.push(nid);
    state.removed.clear();
    recolor();
    renderSelection();
  }

  async function renderSelection() {
    const seq = ++impactSeq;
    const list = el('ul', { class: 'plain' }, state.selection.map((x) => el('li', {}, el('code', {}, x), ' ', el('button', { onclick: () => toggleCut(x) }, '×'))));
    const impactBox = el('div', {});
    selBox.replaceChildren(
      el('h3', {}, `Cut selection (${state.selection.length})`),
      state.selection.length ? list : el('p', { class: 'muted' }, 'Nothing selected. Click a node, then "Cut this". Vocabulary ranges: type ', el('code', {}, 'vocab:100-200'), ' below.'),
      el('div', { class: 'row' }, el('input', { size: 18, placeholder: 'add id (vocab:10-20)', onkeydown: (e) => { if (e.key === 'Enter' && e.target.value.trim()) { toggleCut(e.target.value.trim()); } } }),
        state.selection.length ? el('button', { onclick: () => { state.selection = []; state.removed.clear(); recolor(); renderSelection(); } }, 'clear') : ''),
      impactBox);
    if (!state.selection.length) return;
    impactBox.append(el('p', { class: 'muted' }, 'computing impact…'));
    try {
      const i = await api(`/models/${id}/impact`, { method: 'POST', body: { cut: state.selection, force: state.force } });
      if (seq !== impactSeq) return;
      impactBox.replaceChildren(
        el('p', {}, el('span', { class: `rem rem-${i.verdict}` }, i.verdict), i.refused ? el('span', { class: 'error' }, ' refused without force') : '',
          ` · ${count(i.params_removed)} removed + ${count(i.params_zeroed)} zeroed (${i.percent_of_model.toFixed(2)}%) · ${bytes(i.original_size)} → ${bytes(i.new_size)}`,
          i.other_files_removed ? ` (+${bytes(i.other_files_removed)} projector)` : ''),
        el('ul', { class: 'plain small' },
          i.reasons.map((r) => el('li', {}, r)),
          i.dependencies.map((d) => el('li', {}, 'depends: ' + d)),
          i.disconnected.length ? el('li', {}, `cut off: ${i.disconnected.slice(0, 8).join(', ')}${i.disconnected.length > 8 ? ` … +${i.disconnected.length - 8}` : ''}`) : '',
          el('li', {}, `llama.cpp: ${i.llama_cpp_loads ? 'loads the slimmed file' : 'would NOT load the slimmed file'}`),
          i.file_ops.map((f) => el('li', { class: 'muted' }, 'slim would ' + f)),
          i.unknown.map((u) => el('li', { class: 'error' }, 'unknown id ' + u)),
          i.simulation_notes.map((n) => el('li', { class: 'muted' }, n))),
        el('label', {}, forceIn, ' force (allow breaks_model)'),
        el('div', { class: 'row' }, 'prompt', promptIn, 'tokens', tokensIn),
        corpusIn,
        el('div', { class: 'row' },
          el('button', { class: 'primary', disabled: !i.simulation_supported || i.refused, title: i.simulation_supported ? '' : i.simulation_notes.join('; '), onclick: simulate }, 'Simulate'),
          el('button', { title: 'Put this cut on the edit stack (undo / toggle / restore like any edit)', onclick: () => pushEdit(id, { op: 'ablate', cut: state.selection }).then(() => toast('cut added to the edit stack')).catch((e) => toast(e.message)) }, 'Apply as edit'),
          el('button', { onclick: () => runSweep('layer') }, 'Sweep this level (layers)'),
          el('button', { disabled: i.refused, onclick: savePlan }, 'Save as plan → Slim')));
    } catch (e) {
      if (seq === impactSeq) impactBox.replaceChildren(errorBox(e));
    }
  }

  const corpusTexts = () => corpusIn.value.split(/\n\s*\n/).map((x) => x.trim()).filter(Boolean);

  async function simulate() {
    results.replaceChildren();
    try {
      const r = await jobWithProgress(results, 'ablate', id, {
        cut: state.selection, force: state.force, prompt: promptIn.value || null, corpus: corpusTexts(), tokens: Number(tokensIn.value) || 0,
      });
      state.removed = new Set(r.impact.cut.map((c) => c.id));
      recolor();
      results.replaceChildren(ablationView(r));
    } catch (e) { results.append(errorBox(e)); }
  }

  async function runSweep(level, layer) {
    results.replaceChildren();
    const corpus = corpusTexts();
    if (!corpus.length) { toast('the sweep needs a corpus (texts separated by blank lines)'); return; }
    try {
      let jobId = null;
      const r = await jobWithProgress(results, 'sweep', id, { level, layer, corpus }, (j) => { jobId = j.id; });
      state.damage = new Map(r.entries.map((e) => [e.id, r.entries.length ? e.mean_kl / Math.max(...r.entries.map((x) => x.mean_kl), 1e-12) : 0]));
      state.color = 'damage';
      colorSel.querySelector('select').value = 'damage';
      if (level === 'head' && layer != null) { state.expand.add(`layer.${layer}`); state.expand.add(`layer.${layer}.attn`); for (let k = 0; k < 64; k++) state.expand.add(`layer.${layer}.attn.kv.${k}`); await reload(); } else recolor();
      results.replaceChildren(sweepView(r, jobId));
    } catch (e) { results.append(errorBox(e)); }
  }

  async function savePlan(_, body) {
    const out = el('div', {});
    results.replaceChildren(out);
    try {
      const r = await api(`/models/${id}/plan-from-ablation`, { method: 'POST', body: body || { cut: state.selection, force: state.force } });
      out.replaceChildren(el('h3', {}, 'Plan from ablation'), planView(id, r.plan));
      toast('plan ready: pick an output name and press Slim');
    } catch (e) { out.replaceChildren(errorBox(e)); }
  }

  function ablationView(r) {
    const sim = r.simulation;
    if (!sim) return el('p', {}, 'static impact only');
    const tk = (v) => v.map((t) => `"${esc(t.text)}" ${t.prob.toFixed(3)}`).join('  ');
    return el('div', {},
      el('h3', {}, `Simulated: ${sim.cut.join(', ')} (${sim.elapsed_ms} ms, nothing written)`),
      sim.prompts.map((p) => el('div', {},
        el('p', {}, el('strong', {}, `"${esc(p.prompt)}"`), ` · ${p.n_tokens} tokens · KL ${p.kl?.toFixed(4) ?? '-'} nats · top-1 ${p.top1_changed ? 'CHANGED' : 'unchanged'}`, p.embedding_cosine != null ? ` · embedding cosine ${p.embedding_cosine.toFixed(4)}` : ''),
        table([{ label: '', get: (x) => x[0] }, { label: 'baseline', get: (x) => x[1] }, { label: 'ablated', get: (x) => x[2] }], [
          ['top-k', tk(p.baseline.top_k), tk(p.ablated.top_k)],
          ['generated', `"${esc(p.baseline.generated)}"`, `"${esc(p.ablated.generated)}"`],
          ['tokens/s', p.baseline.tokens_per_s.toFixed(2), p.ablated.tokens_per_s.toFixed(2)],
        ], { sortable: false }),
        p.lens.length ? el('details', {}, el('summary', {}, 'logit lens per layer'), table([
          { label: 'layer', get: (l) => l.layer + (l.skipped ? ' (cut)' : ''), num: true, sort: (l) => l.layer },
          { label: 'baseline', get: (l) => l.baseline ? `"${esc(l.baseline.text)}" ${l.baseline.prob.toFixed(2)}` : '' },
          { label: 'ablated', get: (l) => l.ablated ? `"${esc(l.ablated.text)}" ${l.ablated.prob.toFixed(2)}` : '' },
          { label: 'P(target) base → ablated', get: (l) => `${l.baseline_target_prob.toFixed(3)} → ${l.ablated_target_prob.toFixed(3)}` },
        ], p.lens)) : '')),
      sim.corpus ? el('p', {}, `corpus: ${sim.corpus.samples} sample(s), ${sim.corpus.tokens} positions · perplexity ${sim.corpus.baseline_ppl?.toFixed(3) ?? '-'} → ${sim.corpus.ablated_ppl?.toFixed(3) ?? '-'} · mean KL ${sim.corpus.mean_kl?.toFixed(4) ?? '-'} · top-1 agreement ${(sim.corpus.top1_agreement * 100).toFixed(1)}%`) : '',
      el('p', {}, `weights in use: ${bytes(sim.weights_bytes_baseline)} → ${bytes(sim.weights_bytes_ablated)}`),
      sim.warnings.map((w) => el('p', { class: 'muted' }, w)),
      el('button', { onclick: () => savePlan() }, 'Save this cut as plan → Slim'));
  }

  function sweepView(r, jobId) {
    const rows = r.entries.slice().sort((a, b) => a.rank - b.rank);
    const max = Math.max(...rows.map((e) => e.mean_kl), 1e-12);
    return el('div', {},
      el('h3', {}, `Sweep by ${r.level}${r.layer != null ? ` (layer ${r.layer})` : ''}: ${r.samples} sample(s), baseline perplexity ${r.baseline_ppl?.toFixed(3) ?? '-'} (${(r.elapsed_ms / 1000).toFixed(1)} s)`),
      el('p', { class: 'muted' }, 'Graph colors now show damage when each part is removed (red = essential).'),
      table([
        { label: 'rank', get: (e) => e.rank, num: true },
        { label: 'removed', get: (e) => el('a', { href: '#', onclick: (ev) => { ev.preventDefault(); showNode(e.id); } }, e.label) },
        { label: 'mean KL', get: (e) => e.mean_kl.toFixed(4), num: true },
        { label: 'perplexity', get: (e) => e.ppl?.toFixed(3) ?? '-', num: true },
        { label: 'ppl +%', get: (e) => e.ppl_increase != null ? (e.ppl_increase * 100).toFixed(1) + '%' : '-', num: true, sort: (e) => e.ppl_increase ?? 0 },
        { label: 'top-1 same', get: (e) => (e.top1_agreement * 100).toFixed(1) + '%', num: true, sort: (e) => e.top1_agreement },
        { label: 'damage', get: (e) => el('span', { class: 'bar', style: `width:${(100 * e.mean_kl / max).toFixed(0)}px` }) },
        { label: '', get: (e) => el('button', { onclick: () => toggleCut(e.id) }, state.selection.includes(e.id) ? 'uncut' : 'cut') },
      ], rows),
      el('div', { class: 'row' }, 'save the', topIn, 'least damaging as a plan',
        el('button', { disabled: !jobId, onclick: () => savePlan(null, { sweep_job: jobId, top: Number(topIn.value) || 1, force: state.force }) }, 'Save as plan → Slim')));
  }

  // --- controls ------------------------------------------------------------------------
  levelSel.querySelector('select').addEventListener('change', (e) => { state.level = e.target.value; store.set('nd-graph-level', state.level); reload(); });
  groupSel.querySelector('select').addEventListener('change', (e) => { state.groupBy = e.target.value; reload(); });
  layoutSel.addEventListener('change', () => { state.layout3d = layoutSel.value; store.set('nd-graph-layout3d', state.layout3d); setDim('3d'); });
  colorSel.querySelector('select').addEventListener('change', (e) => { state.color = e.target.value; store.set('nd-graph-color', state.color); recolor(); });
  const zoomCtl = el('span', { class: 'zoom-ctl' },
    el('button', { title: 'Zoom out', onclick: () => cur()?.zoomBy(1 / 1.25) }, '−'),
    el('button', { title: 'Zoom in', onclick: () => cur()?.zoomBy(1.25) }, '+'),
    el('button', { title: 'Reset zoom to the whole model', onclick: () => cur()?.resetZoom() }, '100%'),
    el('button', { title: 'Fit graph to view', onclick: () => cur()?.fit() }, 'Fit'));
  const legend = el('span', { class: 'muted small' }, 'removable: ',
    ...Object.entries(REMOVABLE).flatMap(([k, c]) => [el('span', { class: 'swatch', style: `background:${c}` }), k + ' ']),
    el('span', { class: 'swatch', style: 'background:#f28e2b' }), 'edited ', el('span', { class: 'swatch', style: 'background:#8e44ad' }), 'token path');
  const clearHl = el('button', { title: 'Remove the token-journey highlight', onclick: () => { s.highlight = new Set(); cur()?.setMarks(marks()); } }, 'clear path');

  const onEdits = (e) => { if (e.detail.model === id) cur()?.setMarks(marks()); };
  const onHl = (e) => { if (e.detail.model === id) cur()?.setMarks(marks()); };
  window.addEventListener('nd-edits', onEdits);
  window.addEventListener('nd-highlight', onHl);

  const node = el('section', { class: 'graph-view' },
    el('div', { class: 'row' }, el('h2', {}, 'Graph'), el('span', { class: 'segmented', role: 'radiogroup' }, dimBtns['2d'], dimBtns['3d']), layoutLbl, backendBadge,
      levelSel, groupSel, colorSel, search, zoomCtl, flowBtn,
      el('button', { onclick: () => { state.expand.clear(); reload(); } }, 'Collapse all'), clearHl, legend),
    info,
    el('div', { class: 'graph-main' }, el('div', { class: 'graph-canvases' }, canvas2, canvasF), side),
    selBox,
    results);
  node.cleanup = () => {
    window.removeEventListener('nd-edits', onEdits);
    window.removeEventListener('nd-highlight', onHl);
    g2?.dispose(); gf?.dispose(); g2 = null; gf = null;
  };
  renderSelection();
  loadEdits(id).catch(() => {});
  // Render after the node is attached so the canvas can measure its container.
  setTimeout(reload, 0);
  return node;
}
