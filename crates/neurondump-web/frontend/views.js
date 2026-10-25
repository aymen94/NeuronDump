// One function per view. Each returns a Node (or a Promise of one).

import { api, qs, runJob, watchJob, downloadUrl } from './api.js';
import { el, bytes, count, esc, table, kv, pager, progressBar, errorBox, toast, copyable, highlight, cards } from './ui.js';

const TOKEN_CLASSES = ['', 'special', 'reserved', 'byte', 'whitespace', 'digit', 'punctuation', 'emoji', 'cjk', 'code', 'word', 'subword', 'mixed'];
const CODE_LANGS = ['', 'rust', 'python', 'javascript', 'c_cpp', 'java', 'go', 'sql', 'shell', 'symbol'];

function select(options, value) {
  return el('select', {}, options.map((o) => el('option', { value: o, selected: o === value }, o || '(any)')));
}

function jobStatus(j) {
  return el('span', {}, progressBar(j.progress), ' ', el('span', { class: `status-${j.status}` }, j.status),
    ` ${j.progress.toFixed(0)}% ${j.step || ''}`, j.cached ? el('span', { class: 'muted' }, ' (cached)') : '',
    j.error ? el('span', { class: 'error' }, ' ' + j.error) : '');
}

/** Runs a job with an inline progress line; resolves to the parsed result or throws. */
export async function jobWithProgress(container, kind, model, params, onJob) {
  const line = el('div', { class: 'row' }, 'starting…');
  container.replaceChildren(line);
  let cancelBtn = null;
  const j = await runJob(kind, model, params, (j) => {
    if (onJob) onJob(j);
    if (!cancelBtn) cancelBtn = el('button', { onclick: () => api(`/jobs/${j.id}`, { method: 'DELETE' }) }, 'cancel');
    line.replaceChildren(jobStatus(j), j.status === 'running' || j.status === 'queued' ? cancelBtn : '');
  });
  if (j.status !== 'done') throw new Error(j.error || j.status);
  return api(`/jobs/${j.id}/result`);
}

// --- Models -----------------------------------------------------------------------

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
  const exportLinks = el('div', { class: 'row' }, 'Export:',
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

function histogram(h) {
  const max = Math.max(...h.bins, 1);
  return el('div', {}, el('div', { class: 'hist', title: `${h.min} … ${h.max}` }, h.bins.map((b) => el('div', { style: `height:${(Math.log1p(b) / Math.log1p(max)) * 100}%`, title: String(b) }))),
    el('div', { class: 'muted' }, `${h.min.toFixed(4)} … ${h.max.toFixed(4)} (log scale)`));
}

function tensorStatsView(s) {
  return el('div', {}, el('h3', {}, s.name),
    kv({ type: s.ggml_type, shape: s.dims.join('×'), n: s.n, mean: s.mean, std: s.std, min: s.min, max: s.max, l2: s.l2, sparsity: s.sparsity, outliers: s.outliers, outlier_fraction: s.outlier_fraction, nan_or_inf: s.nan_or_inf }),
    histogram(s.histogram));
}

export function stats(id, params) {
  const name = el('input', { placeholder: 'tensor name (e.g. blk.0.attn_q.weight)', size: 40, value: params.get('tensor') || '' });
  const one = el('div', {});
  const full = el('div', {});
  const loadOne = async () => {
    one.replaceChildren(el('p', { class: 'muted' }, 'computing…'));
    try { one.replaceChildren(tensorStatsView(await api(`/models/${id}/tensors/${encodeURIComponent(name.value)}/stats`))); } catch (e) { one.replaceChildren(errorBox(e)); }
  };
  if (name.value) loadOne();
  const runFull = async () => {
    try {
      const r = await jobWithProgress(full, 'stats', id, {});
      full.replaceChildren(
        el('p', {}, `${r.tensors.length} tensors in ${(r.elapsed_ms / 1000).toFixed(1)} s`),
        table([
          { label: 'tensor', get: (t) => t.name }, { label: 'type', get: (t) => t.ggml_type }, { label: 'mean', get: (t) => t.mean.toExponential(2), num: true },
          { label: 'std', get: (t) => t.std.toExponential(3), num: true }, { label: 'min', get: (t) => t.min.toFixed(3), num: true }, { label: 'max', get: (t) => t.max.toFixed(3), num: true },
          { label: 'sparsity', get: (t) => (t.sparsity * 100).toFixed(2) + '%', num: true }, { label: 'outliers', get: (t) => t.outliers, num: true },
        ], r.tensors),
        r.heads.length ? el('h3', {}, 'Per-head |Q| norms') : '',
        r.heads.length ? table([{ label: 'layer', get: (h) => h.layer }, { label: '|Q| per head', get: (h) => h.q.map((x) => x.toFixed(1)).join(' ') }, { label: '|O| per head', get: (h) => h.o.map((x) => x.toFixed(1)).join(' ') }], r.heads) : '');
    } catch (e) { full.append(errorBox(e)); }
  };
  return el('section', {}, el('h2', {}, 'Weight statistics'),
    el('div', { class: 'row' }, name, el('button', { onclick: loadOne }, 'Tensor stats')), one,
    el('h3', {}, 'Whole model'), el('button', { onclick: runFull }, 'Compute all statistics (job)'), full);
}

// --- Analyze + slim -----------------------------------------------------------------

export function analyze(id) {
  const corpus = el('textarea', { rows: 6, cols: 80, placeholder: 'Optional corpus text for vocabulary trimming' });
  const withStats = el('input', { type: 'checkbox' });
  const fast = el('input', { type: 'checkbox' });
  const out = el('div', {});
  const run = async () => {
    try {
      const plan = await jobWithProgress(out, 'analyze', id, { corpus_text: corpus.value || null, compute_stats: withStats.checked, skip_layer_similarity: fast.checked });
      out.replaceChildren(planView(id, plan));
    } catch (e) { out.append(errorBox(e)); }
  };
  return el('section', {}, el('h2', {}, 'Analyze — useless parts'), corpus,
    el('div', { class: 'row' }, el('label', {}, withStats, ' compute weight stats (slow)'), el('label', {}, fast, ' skip layer similarity'), el('button', { onclick: run }, 'Analyze')),
    out);
}

export function planView(id, plan) {
  const boxes = new Map();
  const outName = el('input', { value: 'slim.gguf', size: 24 });
  const result = el('div', {});
  const total = el('strong', {});
  const recompute = () => {
    total.textContent = bytes(plan.candidates.filter((c) => boxes.get(c.id)?.checked).reduce((a, c) => a + c.bytes_saved, 0));
  };
  const rows = plan.candidates.map((c) => {
    const b = el('input', { type: 'checkbox', checked: c.selected && c.applicable, disabled: !c.applicable, onchange: recompute });
    boxes.set(c.id, b);
    return c;
  });
  const slim = async () => {
    const edited = { ...plan, candidates: plan.candidates.map((c) => ({ ...c, selected: !!boxes.get(c.id)?.checked })) };
    try {
      const r = await jobWithProgress(result, 'slim', id, { plan: edited, only: [], out: outName.value });
      result.replaceChildren(el('h3', {}, 'Slim result'), kv({
        output: r.output, modelfile: r.modelfile, original_size: bytes(r.original_size), new_size: bytes(r.new_size),
        applied: r.applied.join(', '), refused: r.refused.map((x) => x.join(': ')).join('; '), removed_tokens: r.removed_tokens,
        removed_layers: r.removed_layers.join(', '), requantized: r.requantized.length, validation: r.validation.ok ? 'OK' : r.validation.problems.join('; '),
      }));
      toast('slim done');
    } catch (e) { result.append(errorBox(e)); }
  };
  const view = el('div', {},
    el('h3', {}, `Prune plan — original ${bytes(plan.original_size)}`),
    table([
      { label: 'apply', get: (c) => boxes.get(c.id) }, { label: 'id', get: (c) => c.id }, { label: 'risk', get: (c) => c.risk },
      { label: 'saves', get: (c) => bytes(c.bytes_saved), num: true }, { label: 'title', get: (c) => c.title },
      { label: 'detail', get: (c) => c.detail + (c.reason ? ` — not applicable: ${c.reason}` : '') },
    ], rows),
    el('ul', {}, plan.notes.map((n) => el('li', { class: 'muted' }, n))),
    el('div', { class: 'row' }, 'selected saves', total, 'output file', outName, el('button', { onclick: slim }, 'Slim')),
    el('details', {}, el('summary', {}, 'plan JSON'), el('pre', {}, JSON.stringify(plan, null, 2).slice(0, 20000))),
    result);
  recompute();
  return view;
}

// --- Trace ------------------------------------------------------------------------

export async function trace(id) {
  const info = await api('/info');
  const prompt = el('input', { size: 60, value: 'The name of the star that Earth orbits is' });
  const corpus = el('textarea', { rows: 4, cols: 80, placeholder: 'Corpus mode: texts separated by blank lines (aggregated)' });
  const out = el('div', {});
  const run = async () => {
    const params = corpus.value.trim() ? { corpus: corpus.value.split(/\n\s*\n/).filter((s) => s.trim()) } : { prompt: prompt.value };
    try {
      const r = await jobWithProgress(out, 'trace', id, params);
      out.replaceChildren(
        el('p', {}, `${r.architecture} via ${r.backend}, ${r.samples} sample(s), ${r.elapsed_ms} ms`),
        el('p', {}, 'tokens: ', el('code', {}, r.tokens.map(([i, t]) => `${i}:"${esc(t)}"`).join(' '))),
        r.prediction.length ? el('p', {}, 'next token: ', el('code', {}, r.prediction.map((t) => `"${esc(t.text)}" ${t.prob.toFixed(3)}`).join('  '))) : '',
        table([
          { label: 'layer', get: (l) => l.index, num: true }, { label: '|h|', get: (l) => l.hidden_norm.toFixed(1), num: true },
          { label: 'cos(in,out)', get: (l) => l.cos_prev?.toFixed(4) ?? '-', num: true },
          { label: 'attn entropy (mean)', get: (l) => l.head_entropy.length ? (l.head_entropy.reduce((a, b) => a + b, 0) / l.head_entropy.length).toFixed(2) : '-', num: true },
          { label: 'top neurons', get: (l) => l.top_neurons.slice(0, 5).map(([i, v]) => `${i}:${v.toFixed(2)}`).join(' ') },
          { label: 'logit lens', get: (l) => l.logit_lens.map((t) => `"${esc(t.text)}" ${t.prob.toFixed(2)}`).join('  ') },
        ], r.layers),
        r.embedding_contributions?.length ? table([{ label: 'token', get: (t) => esc(t.text) }, { label: 'contribution', get: (t) => t.contribution.toFixed(3), num: true }], r.embedding_contributions) : '',
        ...r.warnings.map((w) => el('p', { class: 'muted' }, w)));
    } catch (e) { out.append(errorBox(e)); }
  };
  return el('section', {}, el('h2', {}, 'Trace / logit lens'),
    el('p', { class: 'muted' }, info.trace_architectures.length ? `Supported architectures: ${info.trace_architectures.join(', ')}` : 'This build has no trace support (build with --features trace).'),
    el('div', { class: 'row' }, prompt), corpus, el('div', {}, el('button', { onclick: run }, 'Trace')), out);
}

// --- Compare ----------------------------------------------------------------------

export async function compare() {
  const data = await api('/models');
  const boxes = data.models.map((m) => [m, el('input', { type: 'checkbox' })]);
  const out = el('div', {});
  const run = async () => {
    const ids = boxes.filter(([, b]) => b.checked).map(([m]) => m.id);
    if (ids.length < 2) { toast('pick at least two models'); return; }
    out.replaceChildren(el('p', { class: 'muted' }, 'comparing…'));
    try {
      const r = await api('/compare' + qs({ models: ids.join(',') }));
      const props = ['architecture', 'kind', 'params', 'quantization', 'file_size', 'layers', 'vocab_size', 'tokenizer', 'context_length', 'embedding_length', 'head_count', 'head_count_kv', 'feed_forward_length', 'expert_count', 'layer_kinds'];
      out.replaceChildren(
        table([{ label: 'property', get: (p) => p }, ...r.models.map((m) => ({ label: m.name, get: (p) => p === 'file_size' ? bytes(m[p]) : p === 'params' ? count(m[p]) : Array.isArray(m[p]) ? m[p].join('; ') : (m[p] ?? '-') }))], props),
        el('h3', {}, 'Vocabulary overlap (Jaccard)'),
        table([{ label: '', get: (row) => r.models[row.i].name }, ...r.models.map((m, j) => ({ label: m.name, get: (row) => row.v[j].toFixed(3), num: true }))], r.vocab_overlap.map((v, i) => ({ v, i }))),
        el('h3', {}, `Differing metadata (${r.metadata_diff.length})`),
        table([{ label: 'key', get: (d) => d.key }, ...r.models.map((m, j) => ({ label: m.name, get: (d) => d.values[j] ?? '∅' }))], r.metadata_diff),
        el('p', {}, `${r.shared_tensor_names} shared tensor names, ${r.tensor_diffs.length} differ in shape/type`));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  return el('section', {}, el('h2', {}, 'Compare'),
    el('div', {}, boxes.map(([m, b]) => el('label', { style: 'display:block' }, b, ' ', m.name))),
    el('button', { onclick: run }, 'Compare'), out);
}

// --- Jobs -------------------------------------------------------------------------

export async function jobs() {
  const list = await api('/jobs');
  const body = el('tbody', {});
  const closers = [];
  for (const j of list.slice().reverse()) {
    const status = el('td', {}, jobStatus(j));
    const actions = el('td', {});
    const render = (j) => {
      status.replaceChildren(jobStatus(j));
      actions.replaceChildren(
        ['running', 'queued'].includes(j.status) ? el('button', { onclick: () => api(`/jobs/${j.id}`, { method: 'DELETE' }) }, 'cancel') : '',
        j.status === 'done' ? el('a', { href: downloadUrl(`/jobs/${j.id}/result`), target: '_blank' }, 'result') : '');
    };
    render(j);
    if (!['done', 'failed', 'cancelled'].includes(j.status)) closers.push(watchJob(j.id, render));
    body.append(el('tr', {}, el('td', {}, el('code', {}, j.id)), el('td', {}, j.kind), el('td', {}, j.model), status, actions));
  }
  const section = el('section', {}, el('h2', {}, `Jobs (${list.length})`),
    el('table', {}, el('thead', {}, el('tr', {}, ['id', 'kind', 'model', 'status', ''].map((h) => el('th', {}, h)))), body));
  section.cleanup = () => closers.forEach((c) => c());
  return section;
}

// --- Search -----------------------------------------------------------------------

const HIT_GROUPS = [['model', 'Models', 0], ['token', 'Tokens', 1], ['tensor', 'Tensors', 2], ['metadata', 'Metadata keys', 3]];

function hitLink(h, q) {
  const id = encodeURIComponent(h.model_id);
  const t = encodeURIComponent(h.target);
  switch (h.kind) {
    case 'model': return `#/model/${id}/overview`;
    case 'token': return `#/model/${id}/vocab?q=${encodeURIComponent(q)}`;
    case 'tensor': return `#/model/${id}/stats?tensor=${t}`;
    default: return `#/model/${id}/overview?meta=${t}`;
  }
}

/** Results page for the header search box. `model` (optional) also searches inside that model. */
export async function search(_, params) {
  const q = params.get('q') || '';
  const modelId = params.get('model') || '';
  const list = await api('/models');
  const scope = el('select', {}, el('option', { value: '' }, 'model names only'),
    list.models.map((m) => el('option', { value: m.id, selected: m.id === modelId }, `inside ${m.name}`)));
  scope.addEventListener('change', () => { location.hash = `#/search?q=${encodeURIComponent(q)}&model=${scope.value}`; });
  const head = el('div', { class: 'row' }, el('h2', {}, q ? `Search: "${q}"` : 'Search'), 'in', scope);
  if (!q.trim()) return el('section', {}, head, el('p', { class: 'muted' }, 'Type in the search box at the top (press / to focus it).'));
  const r = await api('/search' + qs({ q, model: modelId, limit: 25 }));
  const groups = HIT_GROUPS.map(([kind, title, i]) => {
    const hits = r.hits.filter((h) => h.kind === kind);
    if (!r.totals[i]) return null;
    const more = r.totals[i] > hits.length ? el('span', { class: 'muted' }, ` showing ${hits.length}`) : '';
    return el('div', {},
      el('h3', {}, title, ' ', el('span', { class: 'count' }, `(${r.totals[i]})`), more),
      table([
        { label: 'match', get: (h) => el('a', { href: hitLink(h, q) }, highlight(h.label, q)) },
        { label: 'detail', get: (h) => h.detail },
        { label: 'model', get: (h) => h.model_name },
      ], hits),
      kind === 'token' && r.totals[i] > hits.length ? el('a', { href: hitLink(hits[0], q) }, `all ${r.totals[i]} tokens →`) : '');
  }).filter(Boolean);
  return el('section', { class: 'hits' }, head,
    groups.length ? groups : el('p', { class: 'muted' }, 'No results.'),
    !modelId ? el('p', { class: 'muted' }, 'Tip: open a model first, or choose "inside …" above, to also search its tokens, tensors and metadata.') : '');
}

export { graph } from './graph.js';
