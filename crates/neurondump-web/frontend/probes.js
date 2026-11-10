// Probes page: causal tracing, attention head roles, layer similarity (feature `trace`), and
// the static probes that read the weights only: weight spectrum, neuron explorer, glitch
// tokens and the memory estimate. Head roles and the spectrum can color the graph.

import { api, qs } from './api.js';
import { el, bytes, table, errorBox, toast } from './ui.js';
import { session, pushEdit } from './session.js';
import { vis } from './inspector.js';
import { heatmap, traceSupport, unsupported, job } from './journey.js';

const SAMPLE_CORPUS = 'The quick brown fox jumps over the lazy dog. It was a sunny day in the park.\n\nParis is the capital of France. Berlin is the capital of Germany and Rome is the capital of Italy.\n\nRust is a systems programming language focused on safety, speed and concurrency.\n\nThe mitochondria is the powerhouse of the cell, producing energy for the organism.';

export const ROLE_COLORS = {
  induction: '#e15759', 'previous-token': '#4e79a7', 'duplicate-token': '#f28e2b', self: '#76b7b2', sink: '#bab0ac', mixed: '#dddddd',
};

const card = (title, intro, ...body) => el('section', { class: 'probe' }, el('h3', {}, title), el('p', { class: 'muted small' }, intro), ...body);
const quickEdit = (id, op, label) => el('button', { class: 'small', onclick: async () => { try { await pushEdit(id, op); toast(`edit added: ${label}`); } catch (e) { toast(e.message); } } }, label);
const num = (v, d = 3) => (v == null ? '-' : Number(v).toFixed(d));

function causal(id, t, s) {
  const clean = el('input', { size: 46, value: 'The Eiffel Tower is located in the city of' });
  const corrupt = el('input', { size: 46, value: 'The Colosseum is located in the city of', placeholder: 'corrupted prompt' });
  const target = el('input', { size: 10, placeholder: 'clean top-1' });
  const comp = el('select', {}, ['residual', 'attn', 'ffn'].map((c) => el('option', { value: c }, c)));
  const chat = el('input', { type: 'checkbox', checked: s.chat });
  const out = el('div', {});
  const render = (r) => {
    const rows = r.effect.map((row) => row.map((v) => Math.max(0, Math.min(1, v ?? 0))));
    const tok = (i) => `#${i} "${vis(r.corrupt_tokens[i]?.text ?? '')}"`;
    out.replaceChildren(
      el('p', {}, `target "${vis(r.target.text)}": clean ${r.clean_prob.toFixed(3)} → corrupted ${r.corrupt_prob.toFixed(3)} · ${r.component} · ${(r.elapsed_ms / 1000).toFixed(1)} s`),
      el('p', { class: 'small' }, 'corrupted run predicts ', r.corrupt_top.map((x) => `"${vis(x.text)}" ${x.prob.toFixed(3)}`).join('  ')),
      el('div', { class: 'tokens-row small' }, r.corrupt_tokens.map((x, i) => el('code', { title: `#${i}` }, `${i}:${vis(x.text)}`))),
      heatmap(rows, (l) => `layer ${l}`, tok, { scale: 'abs', cell: 12 }),
      el('p', { class: 'muted small' }, 'Rows: layers. Columns: tokens of the corrupted prompt. Dark = restoring the clean state there brings the clean answer back (share of the gap, clamped to 0..1).'),
      table([
        { label: 'layer', get: (p) => p.layer, num: true },
        { label: 'token', get: (p) => tok(p.position) },
        { label: 'effect', get: (p) => p.effect.toFixed(3), num: true },
        { label: '', get: (p) => el('a', { href: `#/model/${id}/graph?expand=layer.${p.layer}` }, 'graph →') },
      ], r.peaks),
      ...r.warnings.map((w) => el('p', { class: 'muted small' }, w)));
  };
  const go = async () => {
    try {
      render(await job(out, 'causal', id, { prompt: clean.value, corrupt: corrupt.value, target: target.value || null, component: comp.value, chat: chat.checked }));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  return card('Causal tracing',
    'Runs the clean and the corrupted prompt, then restores the clean state at one layer and one token inside the corrupted run, for every layer and token, and measures how much of the clean answer comes back. Shows where a fact is held. Costs one forward pass per cell (layers × tokens).',
    t.ok ? '' : unsupported(t),
    el('div', { class: 'row' }, 'clean', clean), el('div', { class: 'row' }, 'corrupt', corrupt),
    el('div', { class: 'row' }, 'target', target, 'restore', comp, el('label', {}, chat, ' chat'), el('button', { class: 'primary', disabled: !t.ok, onclick: go }, 'Trace')),
    out);
}

function roleGrid(r) {
  const layers = Math.max(0, ...r.heads.map((h) => h.layer)) + 1;
  const heads = Math.max(0, ...r.heads.map((h) => h.head)) + 1;
  const cw = Math.max(6, Math.min(18, Math.floor(640 / heads))); const ch = 12; const left = 56;
  const cv = el('canvas', { width: left + heads * cw, height: layers * ch + 2, class: 'heatmap' });
  const g = cv.getContext('2d');
  g.font = '11px system-ui, sans-serif';
  const by = new Map(r.heads.map((h) => [`${h.layer}.${h.head}`, h]));
  for (let l = 0; l < layers; l++) {
    g.fillStyle = getComputedStyle(document.body).color;
    g.fillText(`layer ${l}`, 2, l * ch + ch - 2);
    for (let h = 0; h < heads; h++) {
      const x = by.get(`${l}.${h}`);
      if (!x) continue;
      g.fillStyle = ROLE_COLORS[x.role] || '#ccc';
      g.fillRect(left + h * cw, l * ch + 1, cw - 1, ch - 1);
    }
  }
  cv.addEventListener('mousemove', (e) => {
    const b = cv.getBoundingClientRect();
    const l = Math.floor((e.clientY - b.top) / ch); const h = Math.floor((e.clientX - b.left - left) / cw);
    const x = by.get(`${l}.${h}`);
    cv.title = x ? `L${l} H${h}: ${x.role} · induction ${x.induction.toFixed(2)} · prev ${x.prev_token.toFixed(2)} · dup ${x.duplicate.toFixed(2)} · sink ${x.sink.toFixed(2)} · self ${x.self_attn.toFixed(2)}` : '';
  });
  return cv;
}

function heads(id, t, s) {
  const samples = el('input', { type: 'number', value: 4, min: 1, max: 32, style: 'width:4em' });
  const length = el('input', { type: 'number', value: 24, min: 4, max: 256, style: 'width:4em' });
  const out = el('div', {});
  const render = (r) => {
    s.heads = r;
    const special = r.heads.filter((h) => !['mixed', 'sink'].includes(h.role));
    out.replaceChildren(
      el('p', {}, r.counts.map(([k, n]) => el('span', { class: 'chip' }, el('span', { class: 'swatch', style: `background:${ROLE_COLORS[k] || '#ccc'}` }), `${k} ${n}`))),
      roleGrid(r),
      el('p', { class: 'muted small' }, 'Rows: layers, columns: heads. Hover a cell for its scores.'),
      el('div', { class: 'row' }, el('a', { class: 'button', href: `#/model/${id}/graph?color=role&level=block` }, 'Color the graph by head role →')),
      table([
        { label: 'head', get: (h) => `L${h.layer} H${h.head}`, sort: (h) => h.layer * 1000 + h.head },
        { label: 'role', get: (h) => el('span', {}, el('span', { class: 'swatch', style: `background:${ROLE_COLORS[h.role]}` }), h.role) },
        { label: 'induction', get: (h) => num(h.induction, 2), num: true, sort: (h) => h.induction },
        { label: 'prev token', get: (h) => num(h.prev_token, 2), num: true, sort: (h) => h.prev_token },
        { label: 'duplicate', get: (h) => num(h.duplicate, 2), num: true, sort: (h) => h.duplicate },
        { label: 'sink', get: (h) => num(h.sink, 2), num: true, sort: (h) => h.sink },
        { label: '', get: (h) => quickEdit(id, { op: 'zero', target: `layer.${h.layer}.attn.head.${h.head}` }, 'zero') },
      ], special.sort((a, b) => Math.max(b.induction, b.prev_token, b.duplicate) - Math.max(a.induction, a.prev_token, a.duplicate))),
      ...r.warnings.map((w) => el('p', { class: 'muted small' }, w)));
  };
  const go = async () => {
    try { render(await job(out, 'heads', id, { samples: Number(samples.value), length: Number(length.value) })); } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  if (s.heads) setTimeout(() => render(s.heads), 0);
  return card('Attention head roles',
    'Feeds random tokens repeated twice and scores what every head attends to: induction heads (the token after the earlier copy, the in-context copying circuit), previous-token, duplicate-token, self and attention-sink heads.',
    t.ok ? '' : unsupported(t),
    el('div', { class: 'row' }, 'sequences', samples, 'length', length, el('button', { class: 'primary', disabled: !t.ok, onclick: go }, 'Score heads')),
    out);
}

function similarity(id, t) {
  const corpus = el('textarea', { rows: 4, cols: 80 }, SAMPLE_CORPUS);
  const positions = el('input', { type: 'number', value: 256, min: 8, max: 1024, style: 'width:5em' });
  const out = el('div', {});
  const render = (r) => {
    const maxU = Math.max(1e-6, ...r.layers.map((c) => c.rel_update));
    const lo = Math.min(...r.cka.flat());
    out.replaceChildren(
      el('p', {}, `${r.samples} sample(s), ${r.positions} positions · ${(r.elapsed_ms / 1000).toFixed(1)} s`),
      heatmap(r.cka.map((row) => row.map((v) => (v - lo) / Math.max(1e-6, 1 - lo))), (i) => r.labels[i], (j) => `${r.labels[j]}`, { scale: 'abs', cell: 10 }),
      el('p', { class: 'muted small' }, `Linear CKA between the residual stream at every pair of points, colors stretched over ${lo.toFixed(2)} (pale) … 1 (dark). Blocks along the diagonal are groups of layers that compute alike.`),
      r.redundant.length
        ? el('p', {}, 'Most redundant (CKA in→out ≥ 0.99): ', r.redundant.map((l) => el('span', { class: 'chip' }, `layer ${l} `, quickEdit(id, { op: 'skip', layer: l }, 'skip'))))
        : el('p', {}, 'No layer leaves the representation almost unchanged.'),
      table([
        { label: 'layer', get: (c) => c.layer, num: true },
        { label: 'CKA in→out', get: (c) => c.cka_in_out.toFixed(4), num: true, sort: (c) => c.cka_in_out },
        { label: 'cosine in→out', get: (c) => c.cos_in_out.toFixed(4), num: true, sort: (c) => c.cos_in_out },
        { label: 'update / input', get: (c) => el('span', {}, el('span', { class: 'bar', style: `width:${Math.round(80 * c.rel_update / maxU)}px` }), ` ${c.rel_update.toFixed(3)}`), sort: (c) => c.rel_update },
        { label: '', get: (c) => quickEdit(id, { op: 'skip', layer: c.layer }, 'skip') },
      ], r.layers),
      el('p', { class: 'muted small' }, 'Check a skip with Edits → compare, or rank layers by measured damage with a sweep on the graph page.'),
      ...r.warnings.map((w) => el('p', { class: 'muted small' }, w)));
  };
  const go = async () => {
    const texts = corpus.value.split(/\n\s*\n/).map((x) => x.trim()).filter(Boolean);
    try { render(await job(out, 'similarity', id, { corpus: texts, max_positions: Number(positions.value) })); } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  return card('Layer similarity',
    'Runs a corpus and compares the residual stream after every layer with linear CKA. Layers that barely change it are candidates to skip or remove.',
    t.ok ? '' : unsupported(t), corpus,
    el('div', { class: 'row' }, 'positions', positions, el('button', { class: 'primary', disabled: !t.ok, onclick: go }, 'Compare layers')),
    out);
}

function spectrum(id, s) {
  const layers = el('input', { size: 12, placeholder: 'all, or 0,3,10-12' });
  const emb = el('input', { type: 'checkbox' });
  const out = el('div', {});
  const parse = (v) => {
    if (!v.trim()) return null;
    return v.split(',').flatMap((p) => { const [a, b] = p.split('-').map((x) => Number(x.trim())); return b >= a ? Array.from({ length: b - a + 1 }, (_, i) => a + i) : [a]; }).filter((x) => Number.isInteger(x));
  };
  const render = (r) => {
    s.spectrum = r;
    out.replaceChildren(
      el('p', {}, `${r.tensors.length} matrices · ${(r.elapsed_ms / 1000).toFixed(1)} s${r.skipped.length ? ` · ${r.skipped.length} skipped (expert stacks / unknown types)` : ''}`),
      el('div', { class: 'row' }, el('a', { class: 'button', href: `#/model/${id}/graph?color=rank&level=block` }, 'Color the graph by stable rank →')),
      table([
        { label: 'tensor', get: (x) => x.name },
        { label: 'shape', get: (x) => `${x.rows}×${x.cols}` },
        { label: 'quant', get: (x) => x.quant },
        { label: 'σ max', get: (x) => num(x.sigma_max), num: true, sort: (x) => x.sigma_max },
        { label: 'stable rank', get: (x) => num(x.stable_rank, 1), num: true, sort: (x) => x.stable_rank },
        { label: 'rank %', get: (x) => `${(x.rank_ratio * 100).toFixed(2)}%`, num: true, sort: (x) => x.rank_ratio },
        { label: 'outlier ×', get: (x) => num(x.outlier_ratio, 1), num: true, sort: (x) => x.outlier_ratio },
        { label: 'kurtosis', get: (x) => num(x.kurtosis, 1), num: true, sort: (x) => x.kurtosis },
      ], r.tensors.slice().sort((a, b) => a.rank_ratio - b.rank_ratio)),
      el('p', { class: 'muted small' }, 'Stable rank = ‖W‖²F / σ²max: how many directions carry the weight energy. A small share of the full rank means a low-rank approximation keeps most of the matrix. High outlier ratios and kurtosis make a matrix sensitive to quantization.'));
  };
  const go = async () => {
    try { render(await job(out, 'spectrum', id, { layers: parse(layers.value), embeddings: emb.checked })); } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  if (s.spectrum) setTimeout(() => render(s.spectrum), 0);
  return card('Weight spectrum',
    'Reads every weight matrix: largest singular value (power iteration), stable rank and outlier weights. No forward pass, works for every architecture.',
    el('div', { class: 'row' }, 'layers', layers, el('label', {}, emb, ' include token_embd / output (large)'), el('button', { class: 'primary', onclick: go }, 'Compute spectrum')),
    out);
}

function neuron(id, params) {
  const layer = el('input', { type: 'number', min: 0, value: params?.get('layer') ?? 0, style: 'width:5em' });
  const idx = el('input', { type: 'number', min: 0, value: params?.get('neuron') ?? 0, style: 'width:6em' });
  const out = el('div', {});
  const list = (title, xs) => el('div', {}, el('h4', {}, title), el('ol', { class: 'small' }, xs.map((x) => el('li', {}, el('code', {}, `"${vis(x.text)}"`), ` ${x.score >= 0 ? '+' : ''}${x.score.toFixed(3)}`))));
  const render = (r) => out.replaceChildren(
    el('p', {}, `layer ${r.layer}, neuron ${r.neuron} of ${r.n_ff} · input row (${r.read_from}) norm ${r.read_norm.toFixed(3)} · output column norm ${r.write_norm.toFixed(3)} `,
      quickEdit(id, { op: 'zero', target: `layer.${r.layer}.ffn.neuron.${r.neuron}` }, 'zero this neuron')),
    el('div', { class: 'grid3' }, list('Reads (embeddings along its input row)', r.reads), list('Promotes (through the unembedding)', r.promotes), list('Suppresses', r.suppresses)),
    ...r.warnings.map((w) => el('p', { class: 'muted small' }, w)));
  const go = async () => {
    try { render(await job(out, 'neuron', id, { layer: Number(layer.value), neuron: Number(idx.value), top_k: 15 })); } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  if (params?.get('neuron')) setTimeout(go, 0);
  return card('Neuron explorer',
    'What one FFN neuron reads and writes in token space, from the weights: tokens whose embedding points along its input row, and tokens its output column raises or lowers through the unembedding. A first look on the direct path only; the journey measures neurons on real text.',
    el('div', { class: 'row' }, 'layer', layer, 'neuron', idx, el('button', { class: 'primary', onclick: go }, 'Explore')),
    out);
}

function glitch(id) {
  const z = el('input', { type: 'number', step: 0.5, value: -3, style: 'width:4.5em' });
  const out = el('div', {});
  const render = (r) => out.replaceChildren(
    el('p', {}, `${r.flagged} of ${r.checked} tokens below z ${z.value} · embedding norms mean ${r.embed.mean.toFixed(3)} ± ${r.embed.std.toFixed(3)}${r.unembed ? ` · output norms ${r.unembed.mean.toFixed(3)} ± ${r.unembed.std.toFixed(3)}` : ' · tied output'}`),
    table([
      { label: 'id', get: (c) => c.id, num: true },
      { label: 'token', get: (c) => el('a', { href: `#/model/${id}/vocab?q=${encodeURIComponent(c.text)}` }, el('code', {}, `"${vis(c.text)}"`)) },
      { label: 'type', get: (c) => c.token_type },
      { label: 'embed norm', get: (c) => c.embed_norm.toFixed(3), num: true, sort: (c) => c.embed_norm },
      { label: 'embed z', get: (c) => c.embed_z.toFixed(2), num: true, sort: (c) => c.embed_z },
      { label: 'output z', get: (c) => num(c.unembed_z, 2), num: true, sort: (c) => c.unembed_z ?? 0 },
      { label: 'reachable', get: (c) => (c.reachable ? 'yes' : 'no'), sort: (c) => c.reachable },
      { label: '', get: (c) => quickEdit(id, { op: 'ban', token: c.id }, 'ban') },
    ], r.candidates),
    ...r.warnings.map((w) => el('p', { class: 'muted small' }, w)));
  const go = async () => {
    try { render(await job(out, 'glitch', id, { top: 200, z: Number(z.value) })); } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  return card('Glitch tokens',
    'Under-trained token candidates: embedding or output rows much smaller than the rest, typical of tokens the model (almost) never saw in training. Prompts containing them can make it behave erratically. "Reachable" = the tokenizer can produce the token from its own text.',
    el('div', { class: 'row' }, 'z below', z, el('button', { class: 'primary', onclick: go }, 'Find candidates')),
    out);
}

function memory(id) {
  const extra = el('input', { size: 14, placeholder: 'e.g. 65536,200000' });
  const out = el('div', {});
  const load = async () => {
    try {
      const r = await api(`/models/${id}/memory${qs({ ctx: extra.value })}`);
      out.replaceChildren(
        el('p', {}, `weights ${bytes(r.weights_bytes)} · ${r.kv_layers} of ${r.layers} layers cache K/V · ${r.values_per_token.toLocaleString()} values per token (${r.kv_types.map((k, i) => `${k} ${bytes(r.kv_bytes_per_token[i])}`).join(', ')} per token)`),
        table([
          { label: 'context', get: (x) => x.context.toLocaleString() + (x.context === r.train_context ? ' (trained)' : ''), num: true, sort: (x) => x.context },
          ...r.kv_types.map((k, i) => ({ label: `KV ${k}`, get: (x) => bytes(x.kv_bytes[i]), num: true, sort: (x) => x.kv_bytes[i] })),
          ...r.kv_types.map((k, i) => ({ label: `total (${k})`, get: (x) => bytes(x.total_bytes[i]), num: true, sort: (x) => x.total_bytes[i] })),
        ], r.rows),
        ...r.notes.map((n) => el('p', { class: 'muted small' }, n)));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  setTimeout(load, 0);
  return card('Memory per context length',
    'Weights plus the KV cache for the cache types llama.cpp and Ollama offer (OLLAMA_KV_CACHE_TYPE / --cache-type-k). Read from the metadata, instant.',
    el('div', { class: 'row' }, 'extra lengths', extra, el('button', { onclick: load }, 'Update')),
    out);
}

export async function probes(id, params) {
  const s = session(id);
  const t = await traceSupport(id);
  const open = params?.get('probe');
  const sections = {
    memory: memory(id),
    causal: causal(id, t, s),
    heads: heads(id, t, s),
    similarity: similarity(id, t),
    spectrum: spectrum(id, s),
    neuron: neuron(id, params),
    glitch: glitch(id),
  };
  const nav = el('nav', { class: 'probe-nav' }, Object.keys(sections).map((k) => el('a', { href: '#', onclick: (e) => { e.preventDefault(); sections[k].scrollIntoView({ behavior: 'smooth' }); } }, sections[k].querySelector('h3').textContent)));
  for (const [k, n] of Object.entries(sections)) n.id = `probe-${k}`;
  if (open && sections[open]) setTimeout(() => sections[open].scrollIntoView(), 50);
  return el('section', {}, el('h2', {}, 'Probes'), nav, ...Object.values(sections));
}
