// Live edits: the edit stack of the current model (toggles, undo / redo / restore,
// snapshots, import / export, bake) and the before / after comparison. Shown as a drawer on
// every model page and as the full `edits` page.

import { api, downloadUrl, runJob } from './api.js';
import { el, esc, bytes, table, errorBox, toast, progressBar } from './ui.js';
import { session, setEdits, loadEdits, pushEdit, editAction } from './session.js';

const OPS = {
  scale: { label: 'scale a node', fields: [['target', 'node id (layer.7, layer.7.attn.head.3, layer.7.ffn.neuron.42…)'], ['factor', 'factor', 'number', '0.5']] },
  zero: { label: 'zero a node', fields: [['target', 'node id (any graph id, vocab:10-20)']] },
  skip: { label: 'skip a layer', fields: [['layer', 'layer', 'number']] },
  ablate: { label: 'ablation cut', fields: [['cut', 'ids, comma-separated']] },
  ban: { label: 'ban a token', fields: [['token', 'token text or #id']] },
  boost: { label: 'boost a token', fields: [['token', 'token text or #id'], ['bias', 'logit bias', 'number', '5']] },
  embed: { label: 'edit an embedding row', fields: [['token', 'token text or #id'], ['scale', 'scale (optional)', 'number'], ['replace_with', 'use the row of (optional)']] },
  steer: { label: 'steering vector', fields: [['layer', 'layer', 'number'], ['a', 'prompt A', 'text', 'Dear Sir or Madam, I am writing to formally request'], ['b', 'prompt B', 'text', 'hey dude, just wanna ask'], ['strength', 'strength', 'number', '4']] },
  patch: { label: 'activation patching', fields: [['layer', 'layer', 'number'], ['head', 'head (empty = whole layer)', 'number'], ['source', 'source prompt', 'text', 'The capital of France is'], ['all_positions', 'all positions', 'checkbox']] },
  sampling: { label: 'sampling', fields: [['temperature', 'temperature', 'number', '0.8'], ['top_k', 'top-k', 'number'], ['top_p', 'top-p', 'number'], ['seed', 'seed', 'number', '1']] },
  force: { label: 'force a token', fields: [['step', 'generation step', 'number', '0'], ['token', 'token text or #id']] },
  system: { label: 'system prompt', fields: [['text', 'system prompt', 'textarea', 'You are a pirate. Answer like one.']] },
  template: { label: 'chat template', fields: [['text', 'Go template ({{ .System }}, {{ .Prompt }})', 'textarea']] },
};

// Edit types as shown in the panel, grouped so the right one is one click away.
const GROUPS = [
  ['Nodes', ['scale', 'zero', 'skip', 'ablate']],
  ['Tokens', ['ban', 'boost', 'embed', 'force']],
  ['Activations', ['steer', 'patch']],
  ['Generation', ['sampling', 'system', 'template']],
];

/** `#123`, `123` or a text that tokenizes to one token -> id. */
export async function resolveToken(id, s) {
  const t = String(s ?? '').trim();
  if (/^#?\d+$/.test(t)) return Number(t.replace('#', ''));
  const r = await api(`/models/${id}/tokenize`, { method: 'POST', body: { text: s, neighbors: 0 } });
  const toks = r.tokens.filter((x) => !x.added);
  if (toks.length !== 1) throw new Error(`"${s}" is ${toks.length} tokens (${toks.map((x) => `"${x.text}"`).join(' ')}); give one token or #id`);
  return toks[0].id;
}

async function buildOp(id, op, vals) {
  const num = (k) => (vals[k] === '' || vals[k] === undefined ? undefined : Number(vals[k]));
  const tok = (k) => resolveToken(id, vals[k]);
  switch (op) {
    case 'scale': return { op, target: vals.target.trim(), factor: num('factor') ?? 1 };
    case 'zero': return { op, target: vals.target.trim() };
    case 'skip': return { op, layer: num('layer') };
    case 'ablate': return { op, cut: vals.cut.split(',').map((x) => x.trim()).filter(Boolean) };
    case 'ban': return { op, token: await tok('token') };
    case 'boost': return { op, token: await tok('token'), bias: num('bias') ?? 5 };
    case 'embed': return { op, token: await tok('token'), scale: num('scale'), replace_with: vals.replace_with ? await tok('replace_with') : undefined };
    case 'steer': return { op, layer: num('layer'), a: vals.a, b: vals.b, strength: num('strength') ?? 4 };
    case 'patch': return { op, layer: num('layer'), head: num('head'), source: vals.source, all_positions: !!vals.all_positions };
    case 'sampling': return { op, temperature: num('temperature') ?? 0, top_k: num('top_k'), top_p: num('top_p'), seed: num('seed') };
    case 'force': return { op, step: num('step') ?? 0, token: await tok('token') };
    default: return { op, text: vals.text };
  }
}

function addForm(id, prefill) {
  let op = OPS[prefill?.op] ? prefill.op : 'scale';
  let inputs = {};
  const fieldsBox = el('div', { class: 'edit-fields' });
  const apply = el('button', { class: 'primary' });
  const buttons = {};
  const render = () => {
    inputs = {};
    for (const [k, b] of Object.entries(buttons)) b.setAttribute('aria-checked', String(k === op));
    apply.textContent = `Apply: ${OPS[op].label}`;
    fieldsBox.replaceChildren(...OPS[op].fields.map(([k, label, type, def]) => {
      const v = prefill?.[k] ?? def ?? '';
      const inp = type === 'textarea' ? el('textarea', { rows: 3, placeholder: label }, v)
        : type === 'checkbox' ? el('input', { type: 'checkbox', checked: !!v })
          : el('input', { type: type === 'number' ? 'number' : 'text', step: 'any', placeholder: label, value: v, size: type === 'number' ? 6 : 28 });
      inputs[k] = inp;
      return type === 'checkbox'
        ? el('label', { class: 'edit-field edit-check' }, inp, ' ', label)
        : el('label', { class: `edit-field${type === 'textarea' ? ' wide' : ''}` }, el('span', {}, label), inp);
    }));
  };
  const pick = (k) => { op = k; prefill = null; render(); };
  const groups = GROUPS.map(([name, ops]) => el('div', { class: 'edit-group' },
    el('span', { class: 'muted small' }, name),
    el('span', { class: 'segmented', role: 'radiogroup' }, ops.map((k) => (buttons[k] = el('button', { role: 'radio', title: OPS[k].label, onclick: () => pick(k) }, k))))));
  apply.addEventListener('click', async () => {
    const vals = Object.fromEntries(Object.entries(inputs).map(([k, i]) => [k, i.type === 'checkbox' ? i.checked : i.value]));
    try { await pushEdit(id, await buildOp(id, op, vals)); toast('edit added'); } catch (e) { toast(e.message); }
  });
  render();
  return el('div', { class: 'edit-section' }, el('h4', {}, 'New edit'), el('div', { class: 'edit-groups' }, groups), fieldsBox, el('div', { class: 'row' }, apply));
}

function stackView(id, v) {
  if (!v) return el('p', { class: 'muted' }, 'loading…');
  const list = v.edits.length ? el('ul', { class: 'plain edit-list' }, v.edits.map((e) => el('li', { class: e.enabled ? '' : 'off' },
    el('label', {}, el('input', { type: 'checkbox', checked: e.enabled, title: 'on / off', onchange: () => editAction(id, 'POST', `/${e.id}/toggle`).catch((x) => toast(x.message)) }),
      el('span', { class: 'muted small' }, `#${e.id}`), el('span', {}, e.label)),
    el('button', { class: 'small', title: 'remove', onclick: () => editAction(id, 'DELETE', `/${e.id}`).catch((x) => toast(x.message)) }, '×'))))
    : el('p', { class: 'muted' }, 'No edits: this is the original model.');
  return el('div', { class: 'edit-section' },
    el('div', { class: 'edit-head' },
      el('h4', {}, `Edits (${v.active} on)`),
      el('span', { class: 'row' },
        el('button', { class: 'small', disabled: !v.can_undo, onclick: () => editAction(id, 'POST', '/undo') }, '↶ undo'),
        el('button', { class: 'small', disabled: !v.can_redo, onclick: () => editAction(id, 'POST', '/redo') }, '↷ redo'),
        el('button', { class: 'small', disabled: !v.edits.length, title: 'Back to the original model (undoable)', onclick: () => editAction(id, 'POST', '/restore') }, 'restore all'))),
    list,
    v.error ? el('p', { class: 'error' }, v.error) : '',
    v.compiled?.touched?.length ? el('p', { class: 'muted small' }, 'changed nodes: ', v.compiled.touched.join(', ')) : '',
    (v.compiled?.notes || []).map((n) => el('p', { class: 'muted small' }, n)));
}

function snapshotsView(id, v) {
  const name = el('input', { placeholder: 'snapshot name', size: 14 });
  const pick = el('select', {}, (v?.snapshots || []).map((s) => el('option', { value: s }, s)));
  const file = el('input', { type: 'file', accept: '.json,application/json', onchange: async () => {
    const f = file.files[0];
    if (!f) return;
    try {
      const j = JSON.parse(await f.text());
      const body = Array.isArray(j) ? { schema_version: 1, edits: j } : j;
      const r = await api(`/models/${id}/edits`, { method: 'POST', body });
      setEdits(id, r);
      (r.warnings || []).forEach((w) => toast(w));
      toast(`imported ${r.edits.length} edit(s)`);
    } catch (e) { toast(e.message); }
  } });
  return el('div', { class: 'edit-section' }, el('h4', {}, 'Snapshots and files'),
    el('div', { class: 'row small' },
      name, el('button', { onclick: async () => { try { setEdits(id, await api(`/models/${id}/snapshots`, { method: 'POST', body: { name: name.value } })); toast('saved'); } catch (e) { toast(e.message); } } }, 'save'),
      v?.snapshots?.length ? [pick, el('button', { onclick: async () => { try { setEdits(id, await api(`/models/${id}/snapshots`, { method: 'POST', body: { name: pick.value, action: 'load' } })); toast('loaded'); } catch (e) { toast(e.message); } } }, 'load')]
        : el('span', { class: 'muted' }, 'no snapshots yet')),
    el('div', { class: 'row small' },
      el('a', { href: downloadUrl(`/models/${id}/edits/export`) }, 'export edits.json'),
      el('label', {}, 'import ', file)));
}

function bakeView(id) {
  const name = el('input', { value: 'edited.gguf', size: 18 });
  const out = el('div', {});
  const go = async () => {
    out.replaceChildren(el('p', { class: 'muted' }, 'baking…'));
    try {
      const { id: jid } = await api(`/models/${id}/bake`, { method: 'POST', body: { out: name.value } });
      const j = await new Promise((resolve) => {
        const poll = async () => {
          const s = await api(`/jobs/${jid}`);
          out.replaceChildren(el('span', {}, progressBar(s.progress), ` ${s.status} ${s.step}`));
          if (['done', 'failed', 'cancelled'].includes(s.status)) resolve(s); else setTimeout(poll, 500);
        };
        poll();
      });
      if (j.status !== 'done') throw new Error(j.error || j.status);
      const r = await api(`/jobs/${jid}/result`);
      out.replaceChildren(el('div', { class: 'small' },
        el('p', {}, `wrote ${r.output} (${bytes(r.new_size)}), validate: ${r.validation.ok ? 'OK' : r.validation.problems.join('; ')}`),
        el('p', {}, 'Ollama: ', el('code', {}, `ollama create edited -f "${r.modelfile}"`)),
        r.skipped.map(([e, why]) => el('p', { class: 'muted' }, `not baked: ${e} — ${why}`)),
        r.notes.map((n) => el('p', { class: 'muted' }, n))));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  return el('div', { class: 'edit-section' }, el('h4', {}, 'Bake into a new GGUF'),
    el('div', { class: 'row small' }, name, el('button', { onclick: go }, 'Bake')), out);
}

const fmtTok = (v) => v.map((t) => `"${esc(t.text)}" ${t.prob.toFixed(3)}`).join('  ');

function decisionDiff(b, a) {
  if (!b || !a) return '';
  const rows = b.layers.map((lb, i) => ({ layer: lb.layer, base: lb.attn + lb.ffn, ed: (a.layers[i]?.attn || 0) + (a.layers[i]?.ffn || 0) })).filter((r) => Math.abs(r.ed - r.base) > 1e-3);
  const max = Math.max(1e-6, ...rows.map((r) => Math.abs(r.ed - r.base)));
  return el('div', {},
    el('p', {}, `decision: baseline "${esc(b.chosen.text)}" vs "${esc(b.runner_up.text)}" (Δlogit ${b.logit_diff.toFixed(2)}) → edited "${esc(a.chosen.text)}" vs "${esc(a.runner_up.text)}" (Δlogit ${a.logit_diff.toFixed(2)})`),
    rows.length ? table([
      { label: 'layer', get: (r) => r.layer, num: true },
      { label: 'baseline push', get: (r) => r.base.toFixed(3), num: true },
      { label: 'edited push', get: (r) => r.ed.toFixed(3), num: true },
      { label: 'change', get: (r) => el('span', { class: `bar ${r.ed < r.base ? 'neg' : ''}`, style: `width:${(60 * Math.abs(r.ed - r.base) / max).toFixed(0)}px`, title: (r.ed - r.base).toFixed(3) }) },
    ], rows) : el('p', { class: 'muted small' }, 'per-layer pushes unchanged (same chosen and runner-up comparison)'));
}

export function compareView(r) {
  const s = r.simulation;
  return el('div', { class: 'compare-result' },
    el('p', { class: 'muted' }, `${r.compiled.active} edit(s) · ${s.elapsed_ms} ms · nothing written`),
    s.prompts.map((p) => el('div', {},
      el('p', {}, el('strong', {}, `"${esc(p.prompt.length > 120 ? p.prompt.slice(-120) : p.prompt)}"`), ` · KL ${p.kl?.toFixed(4) ?? '-'} nats · top-1 ${p.top1_changed ? 'CHANGED' : 'unchanged'}`,
        p.embedding_cosine != null ? ` · embedding cosine ${p.embedding_cosine.toFixed(4)}` : ''),
      p.edited_prompt ? el('details', {}, el('summary', {}, 'edited prompt text'), el('pre', {}, p.edited_prompt)) : '',
      el('div', { class: 'grid2 side-by-side' },
        el('div', {}, el('h4', {}, 'baseline'), el('pre', { class: 'gen' }, p.baseline.generated || '(no generation)'), el('p', { class: 'small' }, fmtTok(p.baseline.top_k))),
        el('div', {}, el('h4', {}, 'edited'), el('pre', { class: 'gen' }, p.ablated.generated || '(no generation)'), el('p', { class: 'small' }, fmtTok(p.ablated.top_k)))),
      decisionDiff(p.decision_baseline, p.decision_edited),
      p.lens.length ? el('details', {}, el('summary', {}, 'logit lens diff'), table([
        { label: 'layer', get: (l) => l.layer + (l.skipped ? ' (skip)' : ''), num: true, sort: (l) => l.layer },
        { label: 'baseline', get: (l) => l.baseline ? `"${esc(l.baseline.text)}" ${l.baseline.prob.toFixed(2)}` : '' },
        { label: 'edited', get: (l) => l.ablated ? `"${esc(l.ablated.text)}" ${l.ablated.prob.toFixed(2)}` : '' },
        { label: 'P(target) base → edited', get: (l) => `${l.baseline_target_prob.toFixed(3)} → ${l.ablated_target_prob.toFixed(3)}` },
      ], p.lens)) : '')),
    s.corpus ? el('p', {}, `corpus: ${s.corpus.samples} sample(s) · perplexity ${s.corpus.baseline_ppl?.toFixed(3) ?? '-'} → ${s.corpus.ablated_ppl?.toFixed(3) ?? '-'} · mean KL ${s.corpus.mean_kl?.toFixed(4) ?? '-'} · top-1 agreement ${(s.corpus.top1_agreement * 100).toFixed(1)}%`) : '',
    s.warnings.map((w) => el('p', { class: 'muted small' }, w)));
}

function compareBox(id) {
  const s = session(id);
  const prompt = el('input', { size: 36, value: s.prompt });
  const tokens = el('input', { type: 'number', min: 0, max: 64, value: 12, style: 'width:4em' });
  const chat = el('input', { type: 'checkbox', checked: s.chat });
  const corpus = el('textarea', { rows: 3, placeholder: 'optional corpus for perplexity (texts separated by blank lines)' });
  const out = el('div', {});
  const run = async () => {
    s.prompt = prompt.value; s.chat = chat.checked;
    const line = el('div', { class: 'row' }, 'starting…');
    out.replaceChildren(line);
    try {
      const j = await runJob('compare', id, { prompt: prompt.value, chat: chat.checked, tokens: Number(tokens.value) || 0, corpus: corpus.value.split(/\n\s*\n/).map((x) => x.trim()).filter(Boolean) },
        (j) => line.replaceChildren(progressBar(j.progress), ` ${j.status} ${j.progress.toFixed(0)}% ${j.step}`, j.cached ? ' (cached)' : ''));
      if (j.status !== 'done') throw new Error(j.error || j.status);
      out.replaceChildren(compareView(await api(`/jobs/${j.id}/result`)));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  return el('div', { class: 'edit-section' },
    el('h4', {}, 'Before / after'),
    el('div', { class: 'row' }, prompt, 'tokens', tokens, el('label', {}, chat, ' chat'), el('button', { class: 'primary', onclick: run }, 'Compare')),
    el('details', {}, el('summary', {}, 'corpus'), corpus),
    out);
}

/** The edit panel for a model. opts: { prefill: {op, ...} }. */
export function editPanel(id, opts = {}) {
  const box = el('div', { class: 'edit-panel' });
  const stack = el('div', {});
  const snaps = el('div', {});
  const render = (v) => { stack.replaceChildren(stackView(id, v)); snaps.replaceChildren(snapshotsView(id, v)); };
  const onEdits = (e) => { if (e.detail.model === id) render(e.detail.view); };
  window.addEventListener('nd-edits', onEdits);
  box.cleanup = () => window.removeEventListener('nd-edits', onEdits);
  render(session(id).edits);
  loadEdits(id).catch((e) => stack.replaceChildren(errorBox(e)));
  box.append(stack, addForm(id, opts.prefill), snaps, compareBox(id), bakeView(id));
  return box;
}

/** The full `edits` page. */
export function edits(id) {
  const panel = editPanel(id);
  const sec = el('section', { class: 'edits-page' },
    el('h2', {}, 'Live edits'),
    el('p', { class: 'muted' }, 'Edits change the model in memory only (the GGUF is never modified). Toggle, undo or restore them at any time; "Bake" writes a new GGUF. Graph nodes changed by edits are outlined in orange in 2D and 3D.'),
    panel);
  sec.cleanup = () => panel.cleanup();
  return sec;
}

/** Header drawer shown on every model page. */
export function mountDrawer(getModel) {
  const btn = document.getElementById('edits-toggle');
  const drawer = document.getElementById('edit-drawer');
  let panel = null; let current = null;
  const refresh = () => {
    const id = getModel();
    btn.hidden = !id;
    if (!id) { drawer.hidden = true; return; }
    const v = session(id).edits;
    btn.textContent = `Edits${v?.active ? ` (${v.active})` : ''}`;
    btn.classList.toggle('active', !!v?.active);
    if (!drawer.hidden && current !== id) open();
  };
  const open = () => {
    const id = getModel();
    panel?.cleanup?.();
    current = id;
    panel = editPanel(id);
    drawer.replaceChildren(el('div', { class: 'row' }, el('h3', {}, 'Live edits'), el('button', { onclick: () => { drawer.hidden = true; } }, 'close')), panel);
    drawer.hidden = false;
  };
  btn.addEventListener('click', () => (drawer.hidden ? open() : (drawer.hidden = true)));
  window.addEventListener('nd-edits', refresh);
  return { refresh, open, prefill(op) { open(); panel.replaceWith(panel = editPanel(getModel(), { prefill: op })); } };
}
