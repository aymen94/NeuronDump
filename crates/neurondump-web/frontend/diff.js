// Model diff: weights (relative L2, cosine, max |diff| per tensor / layer / block / expert),
// metadata, vocabulary and chat template of model B against model A. The per-node numbers
// color model A's graph ("color by diff") through the shared session.

import { api } from './api.js';
import { el, bytes, esc, table, errorBox, toast } from './ui.js';
import { jobWithProgress } from './views.js';
import { setDiff } from './session.js';

export const fmt = (x) => (x === 0 ? '0' : Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toFixed(4));

/** Rows of key / value for one node's diff numbers (used by the graph side panel). */
export function diffRows(g) {
  return [
    ['relative L2', fmt(g.rel_l2)],
    ['cosine', g.cosine.toFixed(6)],
    ['max |diff|', fmt(g.max_abs)],
    ['values', g.n.toLocaleString()],
  ];
}

function bar(v, max) {
  return el('span', { class: 'bar', style: `width:${Math.round(100 * v / Math.max(max, 1e-12))}px` });
}

function names(title, list) {
  if (!list.length) return '';
  return el('details', {}, el('summary', {}, `${title} (${list.length})`), el('pre', {}, list.join('\n')));
}

function report(r, aId) {
  const t = r.total;
  const layers = r.groups.filter((g) => g.kind === 'layer');
  const blocks = r.groups.filter((g) => g.kind === 'block');
  const experts = r.groups.filter((g) => g.kind === 'expert').sort((x, y) => y.rel_l2 - x.rel_l2);
  const blk = (l, kinds) => blocks.find((b) => b.layer === l && kinds.includes(b.block));
  const maxL = Math.max(0, ...layers.map((l) => l.rel_l2));
  const side = (s) => `${s.name} — ${s.architecture}, ${s.quantization ?? '?'}, ${bytes(s.file_size)}, ${s.tensors} tensors${s.removed_layers.length ? `, slimmed (layers ${s.removed_layers.join(', ')} removed)` : ''}`;
  const show = el('button', { class: 'primary', disabled: !r.groups.length, onclick: () => { setDiff(aId, r); location.hash = `#/model/${aId}/graph?color=diff`; } }, 'Color the graph of A by this diff');
  return el('div', {},
    el('p', {}, el('strong', {}, 'A '), side(r.a), el('br'), el('strong', {}, 'B '), side(r.b)),
    (r.a.removed_layers.length || r.b.removed_layers.length) ? el('p', { class: 'muted' }, 'Layers were realigned to the original numbering before matching.') : '',
    el('p', {}, t
      ? `${r.compared} tensors compared (${r.identical} identical): relative L2 ${fmt(t.rel_l2)}, cosine ${t.cosine.toFixed(6)}, max |diff| ${fmt(t.max_abs)} · ${(r.elapsed_ms / 1000).toFixed(1)} s`
      : 'Weights not compared.'),
    el('div', { class: 'row' }, show),
    names('Only in A', r.only_in_a), names('Only in B', r.only_in_b),
    names('Shape mismatch', r.shape_mismatch.map((m) => `${m.name}  [${m.dims_a}] → [${m.dims_b}]`)),
    names('Skipped', r.skipped.map(([n, why]) => `${n}  (${why})`)),
    layers.length ? el('div', {}, el('h3', {}, 'Per layer'), table([
      { label: 'layer', get: (g) => g.layer, num: true },
      { label: 'relative L2', get: (g) => fmt(g.rel_l2), num: true, sort: (g) => g.rel_l2 },
      { label: '', get: (g) => bar(g.rel_l2, maxL), sort: (g) => g.rel_l2 },
      { label: 'cosine', get: (g) => g.cosine.toFixed(6), num: true },
      { label: 'max |diff|', get: (g) => fmt(g.max_abs), num: true, sort: (g) => g.max_abs },
      { label: 'attn / ssm', get: (g) => { const b = blk(g.layer, ['attention', 'ssm']); return b ? fmt(b.rel_l2) : '-'; }, num: true, sort: (g) => blk(g.layer, ['attention', 'ssm'])?.rel_l2 ?? -1 },
      { label: 'ffn', get: (g) => { const b = blk(g.layer, ['ffn']); return b ? fmt(b.rel_l2) : '-'; }, num: true, sort: (g) => blk(g.layer, ['ffn'])?.rel_l2 ?? -1 },
      { label: 'moe', get: (g) => { const b = blk(g.layer, ['moe']); return b ? fmt(b.rel_l2) : '-'; }, num: true, sort: (g) => blk(g.layer, ['moe'])?.rel_l2 ?? -1 },
    ], layers)) : '',
    experts.length ? el('details', {}, el('summary', {}, `Experts (${experts.length}, most changed first)`), table([
      { label: 'expert', get: (g) => g.id },
      { label: 'relative L2', get: (g) => fmt(g.rel_l2), num: true, sort: (g) => g.rel_l2 },
      { label: 'cosine', get: (g) => g.cosine.toFixed(6), num: true },
      { label: 'max |diff|', get: (g) => fmt(g.max_abs), num: true, sort: (g) => g.max_abs },
    ], experts.slice(0, 200))) : '',
    r.tensors.length ? el('div', {}, el('h3', {}, 'Most changed tensors'), table([
      { label: 'tensor', get: (d) => d.name },
      { label: 'types', get: (d) => (d.type_a === d.type_b ? d.type_a : `${d.type_a} → ${d.type_b}`) },
      { label: 'relative L2', get: (d) => fmt(d.rel_l2), num: true, sort: (d) => d.rel_l2 },
      { label: 'cosine', get: (d) => d.cosine.toFixed(6), num: true },
      { label: 'max |diff|', get: (d) => fmt(d.max_abs), num: true, sort: (d) => d.max_abs },
    ], r.tensors.slice(0, 100))) : '',
    el('h3', {}, `Metadata (${r.metadata.length} differing keys)`),
    r.metadata.length ? table([{ label: 'key', get: (d) => d.key }, { label: 'A', get: (d) => d.a ?? '∅' }, { label: 'B', get: (d) => d.b ?? '∅' }], r.metadata) : el('p', { class: 'muted' }, 'identical'),
    el('h3', {}, 'Vocabulary'),
    el('p', {}, `${r.vocab.size_a} → ${r.vocab.size_b} tokens, ${r.vocab.changed} changed`),
    r.vocab.examples.length ? table([{ label: 'id', get: (e) => e[0], num: true }, { label: 'A', get: (e) => (e[1] == null ? '∅' : esc(e[1])) }, { label: 'B', get: (e) => (e[2] == null ? '∅' : esc(e[2])) }], r.vocab.examples) : '',
    el('h3', {}, 'Chat template'),
    r.chat_template.changed
      ? el('div', {}, el('p', {}, `changed (${r.chat_template.chars_a} → ${r.chat_template.chars_b} chars)`),
        r.chat_template.first_difference ? el('pre', {}, `line ${r.chat_template.first_difference[0]}\nA: ${r.chat_template.first_difference[1]}\nB: ${r.chat_template.first_difference[2]}`) : '')
      : el('p', { class: 'muted' }, 'identical'));
}

export async function diff(_, params) {
  const data = await api('/models');
  const pick = (sel) => el('select', {}, data.models.map((m) => el('option', { value: m.id, selected: m.id === sel }, `${m.name} (${m.quantization ?? '?'})`)));
  const a = pick(params?.get('a') || data.models[0]?.id);
  const b = pick(params?.get('b') || data.models[1]?.id);
  const metaOnly = el('input', { type: 'checkbox' });
  const out = el('div', {});
  const run = async () => {
    if (a.value === b.value) { toast('pick two different models'); return; }
    history.replaceState(null, '', `#/diff?a=${a.value}&b=${b.value}`);
    try {
      const r = await jobWithProgress(out, 'diff', a.value, { other: b.value, metadata_only: metaOnly.checked });
      out.replaceChildren(report(r, a.value));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  return el('section', {}, el('h2', {}, 'Diff'),
    el('p', { class: 'muted' }, 'Base vs fine-tune, Q4 vs Q8 of the same model, or original vs a slimmed model. Tensors are matched by name and compared block by block, so any quantization pair works.'),
    el('div', { class: 'row' }, el('label', {}, 'A ', a), el('label', {}, 'B ', b),
      el('label', {}, metaOnly, ' metadata only'), el('button', { class: 'primary', onclick: run }, 'Diff')),
    out);
}
