// Pages not tied to one model: compare, jobs, search.

import { api, qs, watchJob, downloadUrl } from '../api.js';
import { el, bytes, count, table, errorBox, toast, highlight } from '../ui.js';
import { select, jobStatus } from '../views.js';

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
