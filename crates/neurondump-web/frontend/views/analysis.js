// Analysis pages: weight statistics, analyze → plan → slim, trace.

import { api, isLocal } from '../api.js';
import { el, bytes, esc, table, kv, errorBox, toast } from '../ui.js';
import { jobWithProgress } from '../views.js';

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
    isLocal() ? el('p', { class: 'muted' }, 'Whole-model statistics need the desktop app (the demo reads one tensor at a time).')
      : [el('h3', {}, 'Whole model'), el('button', { onclick: runFull }, 'Compute all statistics (job)'), full]);
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
