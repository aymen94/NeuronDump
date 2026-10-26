// Model pages: list, overview, metadata, vocabulary, tensors, layers, chat template.

import { api, qs, downloadUrl, isLocal } from '../api.js';
import { el, bytes, count, esc, table, kv, pager, copyable, highlight, cards } from '../ui.js';
import { select } from '../views.js';

const TOKEN_CLASSES = ['', 'special', 'reserved', 'byte', 'whitespace', 'digit', 'punctuation', 'emoji', 'cjk', 'code', 'word', 'subword', 'mixed'];
const CODE_LANGS = ['', 'rust', 'python', 'javascript', 'c_cpp', 'java', 'go', 'sql', 'shell', 'symbol'];

export async function models() {
  const data = await api('/models');
  const rescan = el('button', { onclick: async () => { await api('/models/rescan', { method: 'POST' }); location.reload(); } }, 'Rescan');
  const filter = el('input', { type: 'search', placeholder: 'filter by name, architecture or quantization', size: 40 });
  const total = data.models.reduce((a, m) => a + (m.size || 0), 0);
  const cols = [
    { label: 'name', get: (m) => el('a', { href: `#/model/${m.id}/overview` }, highlight(m.name, filter.value)), sort: (m) => m.name },
    { label: 'source', get: (m) => m.source },
    { label: 'arch', get: (m) => (m.architecture || (m.error ? 'error' : '-')) + (m.has_projector ? ' +vision' : '') },
    { label: 'params', get: (m) => count(m.params), num: true, sort: (m) => m.params || 0 },
    { label: 'quant', get: (m) => m.quantization || '-' },
    { label: 'ctx', get: (m) => count(m.context_length), num: true, sort: (m) => m.context_length || 0 },
    { label: 'size', get: (m) => bytes(m.size), num: true, sort: (m) => m.size || 0 },
    { label: 'id', get: (m) => copyable(m.id), sort: (m) => m.id },
  ];
  const box = el('div', {});
  const render = () => {
    const f = filter.value.trim().toLowerCase();
    const rows = data.models.filter((m) => !f || [m.name, m.architecture, m.quantization, m.source].some((x) => (x || '').toLowerCase().includes(f)));
    box.replaceChildren(table(cols, rows), rows.length ? '' : el('p', { class: 'muted' }, 'No model matches.'));
  };
  filter.addEventListener('input', render);
  render();
  return el('section', {},
    el('div', { class: 'row' }, el('h2', {}, `Models (${data.models.length})`), el('span', { class: 'muted' }, `${bytes(total)} on disk`), rescan),
    el('div', { class: 'row' }, filter, el('span', { class: 'muted' }, 'Click a column header to sort.')),
    box,
    data.warnings.map((w) => el('p', { class: 'muted' }, 'warning: ' + w)));
}

// --- Overview ---------------------------------------------------------------------

export async function overview(id, params) {
  const r = await api(`/models/${id}`);
  const a = r.architecture;
  const hp = a.hyperparams;
  const metaKey = params?.get('meta') || '';
  const exportLinks = isLocal() ? '' : el('div', { class: 'row' }, 'Export:',
    el('a', { href: downloadUrl(`/export/${id}?format=json`) }, 'report.json'),
    ...['tensors', 'vocab', 'metadata', 'layers'].flatMap((t) => [
      el('a', { href: downloadUrl(`/export/${id}?format=csv&table=${t}`) }, `${t}.csv`),
      el('a', { href: downloadUrl(`/export/${id}?format=ndjson&table=${t}`) }, `${t}.ndjson`),
    ]));
  return el('section', {},
    el('h2', {}, r.model.name, ' ', copyable(r.model.id)),
    cards([
      ['architecture', `${a.name} · ${a.kind}`], ['parameters', count(a.total_params)], ['quantization', r.model.quantization || '?'],
      ['file size', bytes(r.file.file_size)], ['layers', String(Object.values(a.layer_kinds || {}).reduce((n, l) => n + l.length, 0))],
      ['vocabulary', r.tokenizer ? count(r.tokenizer.size) : '-'], ['context', count(hp.context_length)],
      ['heads (q/kv)', hp.head_count ? `${hp.head_count} / ${hp.head_count_kv ?? hp.head_count}` : '-'],
    ]),
    el('p', { class: 'muted' }, `GGUF v${r.file.version} · ${r.file.n_tensors} tensors · ${r.file.n_kv} metadata keys · `, copyable(r.model.path)),
    exportLinks,
    el('div', { class: 'grid2' },
      el('div', {}, el('h3', {}, 'Hyperparameters'), kv(a.hyperparams)),
      el('div', {},
        el('h3', {}, 'Components'),
        table([{ label: 'component', get: (r) => r[0] }, { label: 'params', get: (r) => count(r[1]), num: true }, { label: 'bytes', get: (r) => bytes(a.bytes_by_component[r[0]]), num: true }], Object.entries(a.params_by_component)),
        el('h3', {}, 'Tensor types'),
        table([{ label: 'type', get: (t) => t.ggml_type }, { label: 'tensors', get: (t) => t.tensors, num: true }, { label: 'bytes', get: (t) => bytes(t.bytes), num: true }, { label: 'bits/w', get: (t) => t.bits_per_weight?.toFixed(2) ?? '?', num: true }], a.types),
        el('h3', {}, 'Layer kinds'),
        table([{ label: 'kind', get: (r) => r[0] }, { label: 'layers', get: (r) => r[1].join(', ') }], Object.entries(a.layer_kinds)),
        a.tied_embeddings ? el('p', {}, 'Embeddings are tied (no output.weight).') : ''),
      r.tokenizer ? el('div', {},
        el('h3', {}, `Tokenizer: ${r.tokenizer.model} (${r.tokenizer.pre || '-'}), ${r.tokenizer.size} tokens`),
        kv(r.tokenizer.class_counts),
        el('h4', {}, 'Special tokens'),
        table([{ label: 'role', get: (s) => s.role }, { label: 'id', get: (s) => s.id, num: true }, { label: 'text', get: (s) => esc(s.text) }], r.tokenizer.specials)) : '',
      r.ollama ? el('div', {}, el('h3', {}, 'Ollama layers'),
        r.ollama.system ? el('p', {}, 'SYSTEM: ', el('code', {}, r.ollama.system)) : '',
        r.ollama.params ? kv(r.ollama.params) : '',
        (r.ollama.licenses || []).map((l) => el('p', { class: 'muted' }, 'License: ' + l))) : '',
      r.projector ? el('div', {}, el('h3', {}, 'Vision projector'), kv(r.projector)) : ''),
    el('h3', {}, `Warnings (${r.warnings.length})`),
    el('ul', {}, r.warnings.map((w) => el('li', {}, w))),
    el('details', {}, el('summary', {}, 'Architecture keys'), kv(a.keys)),
    metadataDetails(id, metaKey));
}

/** Collapsible metadata table with a key filter; opens itself when `focus` is given. */
function metadataDetails(id, focus) {
  const filter = el('input', { type: 'search', placeholder: 'filter keys', value: focus, size: 30 });
  const box = el('div', {});
  let meta = null;
  const render = () => {
    const f = filter.value.trim().toLowerCase();
    const rows = meta.filter((m) => !f || m.key.toLowerCase().includes(f));
    box.replaceChildren(table([
      { label: 'key', get: (m) => highlight(m.key, filter.value), sort: (m) => m.key },
      { label: 'type', get: (m) => m.type },
      { label: 'value', get: (m) => JSON.stringify(m.value).slice(0, 200) },
    ], rows));
  };
  filter.addEventListener('input', () => meta && render());
  const d = el('details', { open: focus ? true : null, id: 'metadata', ontoggle: async (e) => {
    if (e.target.open && !meta) {
      meta = await api(`/models/${id}/metadata`);
      render();
    }
  } }, el('summary', {}, 'All metadata'), el('div', { class: 'row' }, filter), box);
  if (focus) setTimeout(() => d.scrollIntoView({ block: 'start' }), 300);
  return d;
}

// --- Vocabulary -------------------------------------------------------------------

export function vocab(id, params) {
  const q = el('input', { type: 'search', placeholder: 'search text, raw or id', size: 30, value: params?.get('q') || '' });
  const regex = el('input', { type: 'checkbox' });
  const cls = select(TOKEN_CLASSES, params?.get('class') || '');
  const lang = select(CODE_LANGS, '');
  const script = el('input', { placeholder: 'script (latin, han, …)', size: 14 });
  const summary = el('p', { class: 'muted' });
  const cols = [
    { label: 'id', get: (t) => t.id, num: true },
    { label: 'text', get: (t) => el('code', {}, '"', highlight(esc(t.text), regex.checked ? '' : q.value), '"') },
    { label: 'raw', get: (t) => el('code', {}, esc(t.raw)) },
    { label: 'type', get: (t) => t.token_type },
    { label: 'class', get: (t) => t.class },
    { label: 'script/lang', get: (t) => (t.code_langs || []).join(',') || t.script || '' },
    { label: 'score', get: (t) => t.score?.toFixed(3) ?? '', num: true },
  ];
  const results = pager(cols, async (offset, limit) => {
    const d = await api(`/models/${id}/vocab` + qs({ q: q.value, regex: regex.checked ? 'true' : '', class: cls.value, lang: lang.value, script: script.value, offset, limit }));
    summary.textContent = `${d.summary.model} tokenizer, ${d.summary.size} tokens, ${d.summary.merges} merges`;
    return { total: d.page.total, rows: d.page.tokens };
  }, 100);
  const go = () => results.reload();
  q.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
  cls.addEventListener('change', go);
  lang.addEventListener('change', go);
  return el('section', {}, el('h2', {}, 'Vocabulary'), summary,
    el('div', { class: 'row' }, q, el('label', {}, regex, ' regex'), 'class', cls, 'lang', lang, script, el('button', { onclick: go }, 'Search')),
    results);
}

// --- Tensors ----------------------------------------------------------------------

export function tensors(id, params) {
  const layer = el('input', { placeholder: 'layer', size: 5, value: params.get('layer') || '' });
  const role = el('input', { placeholder: 'role (ffn_down…)', size: 16, value: params.get('role') || '' });
  const name = el('input', { type: 'search', placeholder: 'name contains', size: 16, value: params.get('q') || '' });
  const sort = select(['offset', 'size', 'type', 'layer', 'name'], 'offset');
  const cols = [
    { label: 'name', get: (t) => el('a', { href: `#/model/${id}/stats?tensor=${encodeURIComponent(t.name)}` }, highlight(t.name, name.value)) },
    { label: 'layer', get: (t) => t.layer ?? '', num: true },
    { label: 'role', get: (t) => t.role },
    { label: 'component', get: (t) => t.component },
    { label: 'shape', get: (t) => t.dims.join('×') },
    { label: 'type', get: (t) => t.ggml_type },
    { label: 'params', get: (t) => count(t.n_elements), num: true },
    { label: 'bytes', get: (t) => bytes(t.n_bytes), num: true },
    { label: 'offset', get: (t) => '0x' + t.offset.toString(16), num: true },
  ];
  const results = pager(cols, async (offset, limit) => {
    const d = await api(`/models/${id}/tensors` + qs({ layer: layer.value, role: role.value, q: name.value, sort: sort.value, offset, limit }));
    return { total: d.total, rows: d.tensors };
  }, 200);
  return el('section', {}, el('h2', {}, 'Tensors'),
    el('div', { class: 'row' }, layer, role, name, 'sort', sort, el('button', { onclick: () => results.reload() }, 'Filter')),
    results);
}

// --- Layers -----------------------------------------------------------------------

export async function layers(id) {
  const d = await api(`/models/${id}/layers`);
  const map = el('div', { class: 'layermap' }, d.layers.map((l) =>
    el('a', { class: `cell ${l.mixer} ${l.ffn}`, title: `layer ${l.index}: ${l.mixer} + ${l.ffn}, ${bytes(l.bytes)}`, href: `#/model/${id}/tensors?layer=${l.index}` }, l.index)));
  return el('section', {}, el('h2', {}, `Layers (${d.layers.length})`),
    el('div', { class: 'legend' }, el('span', { class: 'full_attention' }, 'full attention'), el('span', { class: 'sliding_attention' }, 'sliding window'),
      el('span', { class: 'linear_attention' }, 'linear attention / SSM'), el('span', { class: 'moe' }, 'mixture of experts'), 'Click a layer for its tensors.'),
    map,
    table([
      { label: 'layer', get: (l) => l.index, num: true }, { label: 'mixer', get: (l) => l.mixer }, { label: 'ffn', get: (l) => l.ffn },
      { label: 'heads', get: (l) => l.head_count ?? '-', num: true }, { label: 'kv', get: (l) => l.head_count_kv ?? '-', num: true },
      { label: 'ffn len', get: (l) => l.ffn_length ?? '-', num: true }, { label: 'params', get: (l) => count(l.params), num: true },
      { label: 'bytes', get: (l) => bytes(l.bytes), num: true }, { label: 'roles', get: (l) => l.roles.join(' ') },
    ], d.layers));
}

// --- Template ---------------------------------------------------------------------

function treeNode(n) {
  let label;
  if (n.kind === 'branch') label = [el('span', { class: 'kind' }, n.branch), ' ', el('code', {}, n.condition || '')];
  else if (n.kind === 'emit') label = [el('span', { class: 'kind' }, 'emit '), el('code', {}, esc(n.text || '').slice(0, 120))];
  else label = [el('span', { class: 'kind' }, n.kind), ' ', el('code', {}, n.condition || '')];
  const extra = [];
  if (n.vars?.length && n.kind !== 'emit') extra.push(el('span', { class: 'muted' }, ` reads ${n.vars.join(', ')}`));
  if (n.emits?.length) extra.push(el('span', { class: 'emits' }, ` → ${n.emits.join(' ')}`));
  return el('li', {}, ...label, ...extra, n.children?.length ? el('ul', {}, n.children.map(treeNode)) : '');
}

export async function template(id) {
  const d = await api(`/models/${id}/template`);
  return el('section', {}, el('h2', {}, 'Chat templates'),
    d.templates.length ? '' : el('p', {}, 'No chat template in this model.'),
    d.templates.map((t) => el('div', { class: 'tree' },
      el('h3', {}, `${t.name} (${t.syntax}) — ${t.decisions} decisions`),
      el('p', {}, 'variables: ', el('code', {}, t.variables.join(', '))),
      el('p', {}, 'special tokens: ', el('code', {}, t.special_tokens.join(' '))),
      t.errors.map((e) => el('p', { class: 'error' }, e)),
      el('ul', {}, treeNode(t.root)),
      el('details', {}, el('summary', {}, 'source'), el('pre', {}, t.source)))),
    d.ollama_system ? el('p', {}, 'SYSTEM: ', el('code', {}, d.ollama_system)) : '',
    d.ollama_params ? kv(d.ollama_params) : '');
}

// --- Stats ------------------------------------------------------------------------
