// Token inspector: tokenization (static, any model) with the 3D embedding map, the token
// journey through the layers, and the generation timeline (feature `trace`).

import { api, runJob } from './api.js';
import { el, esc, table, errorBox, toast, progressBar } from './ui.js';
import { session, setHighlight, pushEdit, loadEdits } from './session.js';
import { createEmbeddingMap, CLASS_COLORS } from './embedmap.js';

const CLASSES = ['special', 'reserved', 'byte', 'whitespace', 'digit', 'punctuation', 'emoji', 'cjk', 'code', 'word', 'subword', 'mixed'];
const classColor = (c) => CLASS_COLORS[Math.max(0, CLASSES.indexOf(c)) % CLASS_COLORS.length];
const vis = (t) => esc(t).replace(/ /g, '·') || '∅';

async function traceSupport(id) {
  const [info, m] = await Promise.all([api('/info'), api(`/models/${id}`)]);
  const arch = m.architecture.name;
  return { ok: info.trace_architectures.includes(arch), arch, list: info.trace_architectures };
}

function unsupported(t) {
  return el('div', { class: 'notice' },
    t.list.length
      ? `Runtime tracing does not support the "${t.arch}" architecture (supported: ${t.list.join(', ')}). The static token information on the tokens page still works.`
      : 'This build has no trace support (rebuild with `cargo build --release --features trace`). The static token information on the tokens page still works.');
}

/** Runs a job with a progress line in `box`; resolves to the result. */
async function job(box, kind, id, params) {
  const line = el('div', { class: 'row' }, 'starting…');
  box.replaceChildren(line);
  let jid = null;
  const j = await runJob(kind, id, params, (j) => {
    jid = j.id;
    line.replaceChildren(progressBar(j.progress), ` ${j.status} ${j.progress.toFixed(0)}% ${j.step}`, j.cached ? ' (cached)' : '',
      ['running', 'queued'].includes(j.status) ? el('button', { onclick: () => api(`/jobs/${jid}`, { method: 'DELETE' }) }, 'cancel') : '');
  });
  if (j.status !== 'done') throw new Error(j.error || j.status);
  return api(`/jobs/${j.id}/result`);
}

// --- shared views -----------------------------------------------------------------------

/** rows: number[][] (values 0..1 after normalizing per row max), drawn as a canvas. */
export function heatmap(rows, rowLabel, colLabel, opts = {}) {
  const cols = Math.max(1, ...rows.map((r) => r.length));
  const cw = Math.max(6, Math.min(28, Math.floor(720 / cols))); const ch = opts.cell || 14; const left = 64;
  const cv = el('canvas', { width: left + cols * cw, height: rows.length * ch + 2, class: 'heatmap' });
  const g = cv.getContext('2d');
  const dark = matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.getAttribute('data-theme') !== 'light' || document.documentElement.getAttribute('data-theme') === 'dark';
  g.font = '11px system-ui, sans-serif';
  rows.forEach((r, i) => {
    const max = Math.max(1e-9, ...r);
    g.fillStyle = dark ? '#ccc' : '#333';
    g.fillText(rowLabel(i), 2, i * ch + ch - 3);
    r.forEach((v, j) => {
      const s = Math.max(0, v / max);
      g.fillStyle = `rgba(${dark ? '255,140,60' : '200,60,20'},${(0.06 + 0.94 * s).toFixed(3)})`;
      g.fillRect(left + j * cw, i * ch + 1, cw - 1, ch - 1);
    });
    if (opts.mark?.has(i)) {
      g.strokeStyle = '#8e44ad'; g.lineWidth = 2;
      g.strokeRect(left - 1, i * ch + 1, cols * cw, ch - 1);
    }
  });
  cv.addEventListener('mousemove', (e) => {
    const r = cv.getBoundingClientRect();
    const i = Math.floor((e.clientY - r.top) / ch); const j = Math.floor((e.clientX - r.left - left) / cw);
    cv.title = rows[i] && j >= 0 && j < rows[i].length ? `${rowLabel(i)} → ${colLabel(j)}: ${rows[i][j].toFixed(3)}` : '';
  });
  if (opts.onRow) cv.addEventListener('click', (e) => { const i = Math.floor((e.clientY - cv.getBoundingClientRect().top) / ch); if (rows[i]) opts.onRow(i); });
  return cv;
}

function bar(v, max, neg) {
  return el('span', { class: `bar ${neg ? 'neg' : ''}`, style: `width:${Math.round(80 * Math.min(1, Math.abs(v) / Math.max(1e-9, max)))}px`, title: v.toFixed(4) });
}

/** Decision breakdown: candidates, per-layer pushes, top heads and neurons, with quick edits. */
export function decisionView(id, d) {
  if (!d) return el('p', { class: 'muted' }, 'No decision breakdown (embedding model or no output head).');
  const maxL = Math.max(1e-6, ...d.layers.map((l) => Math.max(Math.abs(l.attn), Math.abs(l.ffn))));
  const maxH = Math.max(1e-6, ...d.top_heads.map((h) => Math.abs(h.value)));
  const quick = (op, label) => el('button', { class: 'small', onclick: async () => { try { await pushEdit(id, op); toast(`edit added: ${label}`); } catch (e) { toast(e.message); } } }, label);
  return el('div', { class: 'decision' },
    el('p', {}, el('strong', {}, `"${vis(d.chosen.text)}" ${d.chosen.prob.toFixed(3)}`), ` beats "${vis(d.runner_up.text)}" ${d.runner_up.prob.toFixed(3)} by ${d.logit_diff.toFixed(3)} logits (attributed ${d.attributed.toFixed(3)}; embedding ${d.embedding.toFixed(3)})`),
    el('div', { class: 'cands' }, d.candidates.map((c) => el('div', {}, el('code', {}, `"${vis(c.text)}"`), ' ', bar(c.prob, 1), ` ${c.prob.toFixed(3)}`))),
    el('div', { class: 'grid2' },
      el('div', {}, el('h4', {}, 'Push per layer (chosen − runner-up)'), table([
        { label: 'layer', get: (l) => l.layer, num: true },
        { label: 'attention', get: (l) => el('span', {}, bar(l.attn, maxL, l.attn < 0), ` ${l.attn.toFixed(2)}`), sort: (l) => l.attn },
        { label: 'FFN', get: (l) => el('span', {}, bar(l.ffn, maxL, l.ffn < 0), ` ${l.ffn.toFixed(2)}`), sort: (l) => l.ffn },
      ], d.layers)),
      el('div', {},
        el('h4', {}, 'Top heads'),
        table([
          { label: 'head', get: (h) => `L${h.layer} H${h.head}` },
          { label: 'push', get: (h) => el('span', {}, bar(h.value, maxH, h.value < 0), ` ${h.value.toFixed(3)}`), sort: (h) => h.value },
          { label: '', get: (h) => quick({ op: 'zero', target: `layer.${h.layer}.attn.head.${h.head}` }, 'zero') },
        ], d.top_heads),
        el('h4', {}, 'Top neurons'),
        table([
          { label: 'neuron', get: (n) => `L${n.layer} N${n.neuron}` },
          { label: 'activation', get: (n) => n.activation.toFixed(2), num: true },
          { label: 'push', get: (n) => n.value.toFixed(3), num: true },
          { label: '', get: (n) => quick({ op: 'zero', target: `layer.${n.layer}.ffn.neuron.${n.neuron}` }, 'zero') },
        ], d.top_neurons))));
}

function tokenChips(tokens, selected, onPick, cls) {
  return el('div', { class: 'tokchips' }, tokens.map((t, i) => el('span', {
    class: `tokchip ${t.special ? 'special' : ''} ${t.added ? 'added' : ''} ${i === selected ? 'sel' : ''} ${cls?.(t, i) || ''}`,
    style: t.class ? `border-color:${classColor(t.class)}` : '',
    title: `#${i} id ${t.id}${t.class ? ` · ${t.class}` : ''}${t.added ? ' · added by the template' : ''}`,
    onclick: () => onPick(i),
  }, vis(t.text))));
}

// --- tokens page ------------------------------------------------------------------------

export function tokens(id, params) {
  const s = session(id);
  const text = el('textarea', { rows: 3, cols: 70 }, params?.get('text') || s.prompt);
  const chat = el('input', { type: 'checkbox', checked: params?.get('chat') === '1' || s.chat });
  const system = el('input', { size: 30, placeholder: 'system prompt (default: the model\'s)' });
  const k = el('input', { type: 'number', min: 0, max: 32, value: 8, style: 'width:4em' });
  const out = el('div', {});
  const detail = el('div', { class: 'tok-detail' });
  const mapBox = el('div', { class: 'graph-canvas embedmap-box' });
  const mapCtl = el('div', { class: 'row' });
  const mapInfo = el('div', { class: 'muted small' });
  let map = null; let last = null;

  const showDetail = (t) => {
    const q = encodeURIComponent(s.prompt);
    detail.replaceChildren(
      el('h4', {}, `#${t.index} "${vis(t.text)}" (id ${t.id})`),
      el('table', { class: 'kv' }, el('tbody', {},
        el('tr', {}, el('th', {}, 'raw'), el('td', {}, el('code', {}, esc(t.raw)))),
        el('tr', {}, el('th', {}, 'class / type'), el('td', {}, `${t.class} / ${t.token_type}${t.special ? ' (special)' : ''}`)),
        el('tr', {}, el('th', {}, 'source'), el('td', {}, t.added ? 'added by the chat template / tokenizer' : 'from your text')),
        el('tr', {}, el('th', {}, '|embedding row|'), el('td', {}, t.embedding_norm?.toFixed(4) ?? '-')),
        el('tr', {}, el('th', {}, '|output row|'), el('td', {}, t.output_norm?.toFixed(4) ?? '-')),
        el('tr', {}, el('th', {}, 'predictable'), el('td', {}, t.banned ? 'no (banned by an edit)' : t.in_output ? 'yes (has an output head row)' : 'no')))),
      t.neighbors.length ? el('div', {}, el('h4', {}, 'Nearest tokens (cosine of token_embd rows)'),
        el('ul', { class: 'plain' }, t.neighbors.map((nb) => el('li', {}, el('a', { href: '#', onclick: (e) => { e.preventDefault(); map?.focusToken(nb.id); } }, `"${vis(nb.text)}"`), ` #${nb.id} ${nb.cosine.toFixed(3)}`)))) : '',
      el('div', { class: 'row' },
        el('a', { href: `#/model/${id}/journey?token=${t.index}&prompt=${q}${chat.checked ? '&chat=1' : ''}` }, 'journey of this token →'),
        el('button', { onclick: () => map ? map.focusToken(t.id) : toast('load the embedding map first') }, 'show on map'),
        el('button', { onclick: () => pushEdit(id, { op: 'ban', token: t.id }).then(() => toast('banned')).catch((e) => toast(e.message)) }, 'ban'),
        el('button', { onclick: () => pushEdit(id, { op: 'boost', token: t.id, bias: 5 }).then(() => toast('boosted')).catch((e) => toast(e.message)) }, 'boost +5'),
        el('button', { onclick: () => pushEdit(id, { op: 'embed', token: t.id, scale: 0 }).then(() => toast('embedding zeroed')).catch((e) => toast(e.message)) }, 'embedding ×0')));
  };

  const run = async () => {
    s.prompt = text.value; s.chat = chat.checked;
    out.replaceChildren(el('p', { class: 'muted' }, 'tokenizing…'));
    try {
      const r = await api(`/models/${id}/tokenize`, { method: 'POST', body: { text: text.value, chat: chat.checked, system: system.value || null, neighbors: Number(k.value) || 0 } });
      last = r;
      let full = '';
      if (r.chat) {
        const [a, b] = r.chat.prompt_span || [r.full_text.length, r.full_text.length];
        full = el('div', {}, el('h4', {}, `Full templated prompt (${r.chat.source})`),
          el('pre', { class: 'templated' }, r.full_text.slice(0, a), el('mark', {}, r.full_text.slice(a, b)), r.full_text.slice(b)));
      }
      out.replaceChildren(
        el('p', {}, `${r.tokens.length} tokens · ${r.tokenizer} tokenizer · ${r.tied_embeddings ? 'embeddings tied to the output head' : 'separate output head'} · ${r.elapsed_ms} ms`),
        full,
        tokenChips(r.tokens, -1, (i) => showDetail(r.tokens[i])),
        el('div', { class: 'legend small' }, CLASSES.map((c) => el('span', { class: 'swatch', style: `background:${classColor(c)}` })).flatMap((sw, i) => [sw, CLASSES[i] + ' ']), ' · dimmed = added by the template, bold = special'),
        table([
          { label: '#', get: (t) => t.index, num: true },
          { label: 'id', get: (t) => t.id, num: true },
          { label: 'text', get: (t) => el('code', {}, `"${vis(t.text)}"`) },
          { label: 'class', get: (t) => t.class },
          { label: 'added', get: (t) => (t.added ? 'template' : '') },
          { label: '|emb|', get: (t) => t.embedding_norm?.toFixed(3) ?? '-', num: true },
          { label: 'predictable', get: (t) => (t.banned ? 'banned' : t.in_output ? 'yes' : 'no') },
          { label: 'nearest', get: (t) => t.neighbors.slice(0, 4).map((n) => `"${vis(n.text)}" ${n.cosine.toFixed(2)}`).join('  ') },
        ], r.tokens),
        ...r.warnings.map((w) => el('p', { class: 'muted' }, w)));
      map?.showTokens(r.tokens.map((t) => t.id));
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };

  const loadMap = async () => {
    mapCtl.replaceChildren(el('span', { class: 'muted' }, 'loading…'));
    try {
      map = await createEmbeddingMap(mapBox, id, { onPick: (i) => {
        mapInfo.replaceChildren(`#${i} "${vis(map.text(i) ?? '')}" · ${map.className(i)} `,
          el('button', { class: 'small', onclick: () => { text.value += map.text(i) ?? ''; } }, 'append to text'));
      } });
      const search = el('input', { type: 'search', placeholder: 'search tokens on the map', size: 22 });
      const found = el('span', { class: 'muted' });
      search.addEventListener('keydown', (e) => { if (e.key === 'Enter') found.textContent = `${map.search(search.value).length} match(es)`; });
      mapCtl.replaceChildren(
        el('span', { class: 'badge-backend' }, map.backend),
        `${map.info.count} tokens · PCA explains ${(map.info.explained.reduce((a, b) => a + (b || 0), 0) * 100).toFixed(1)}% of the variance`,
        search, found,
        el('span', { class: 'small' }, map.info.classes.map((c, kk) => map.info.counts[kk] ? el('label', { class: 'cls-toggle' },
          el('input', { type: 'checkbox', checked: true, onchange: (e) => map.toggleClass(kk, e.target.checked) }),
          el('span', { class: 'swatch', style: `background:${CLASS_COLORS[kk % CLASS_COLORS.length]}` }), `${c} ${map.info.counts[kk]}`) : '')));
      if (last) map.showTokens(last.tokens.map((t) => t.id));
    } catch (e) {
      mapCtl.replaceChildren(errorBox(e));
    }
  };

  const sec = el('section', { class: 'tokens-page' },
    el('h2', {}, 'Tokens'),
    el('div', { class: 'row' }, text),
    el('div', { class: 'row' }, el('label', {}, chat, ' chat template'), system, 'neighbours', k, el('button', { class: 'primary', onclick: run }, 'Tokenize')),
    out, detail,
    el('h3', {}, 'Embedding map'),
    el('p', { class: 'muted small' }, 'token_embd rows projected to 3D with PCA (computed on the server). Drag to orbit, scroll to zoom, click a point to inspect it. Your tokens are enlarged and joined in order.'),
    mapCtl, mapInfo, mapBox);
  mapCtl.replaceChildren(el('button', { onclick: loadMap }, 'Load embedding map'));
  sec.cleanup = () => map?.dispose();
  if (params?.get('map') === '1') setTimeout(loadMap, 0);
  setTimeout(run, 0);
  return sec;
}

// --- journey page -----------------------------------------------------------------------

export async function journey(id, params) {
  const s = session(id);
  const t = await traceSupport(id);
  const prompt = el('input', { size: 50, value: params?.get('prompt') || s.prompt });
  const chat = el('input', { type: 'checkbox', checked: params?.get('chat') === '1' || s.chat });
  const idx = el('input', { type: 'number', value: params?.get('token') ?? -1, style: 'width:5em' });
  const baseline = el('input', { type: 'checkbox' });
  const out = el('div', {});
  let headLayer = null;

  const render = (r) => {
    const tok = r.tokens[r.position];
    const maxNorm = Math.max(1e-6, ...r.layers.map((l) => l.residual_norm));
    const topHeads = new Map();
    (r.decision?.top_heads || []).slice(0, 6).forEach((h) => { if (!topHeads.has(h.layer)) topHeads.set(h.layer, new Set()); topHeads.get(h.layer).add(h.head); });
    const keys = r.layers[0]?.attention[0]?.length || 0;
    const headBox = el('div', {});
    const showHeads = (li) => {
      headLayer = li;
      const L = r.layers[li];
      if (!L?.attention.length) { headBox.replaceChildren(el('p', { class: 'muted' }, `layer ${li}: no attention recorded (skipped or zeroed)`)); return; }
      headBox.replaceChildren(el('h4', {}, `Layer ${li}: attention of each head (query = token ${r.position})`),
        heatmap(L.attention, (h) => `head ${h}`, (k) => `#${k} "${vis(r.tokens[k]?.text ?? '')}"`, { mark: topHeads.get(li) }),
        el('p', { class: 'muted small' }, 'purple outline = heads with the largest push on the decision'));
    };
    const meanRows = r.layers.map((l) => (l.attention.length ? Array.from({ length: keys }, (_, k) => l.attention.reduce((a, h) => a + (h[k] || 0), 0) / l.attention.length) : []));
    out.replaceChildren(
      el('p', {}, `${r.architecture} via ${r.backend} · ${r.elapsed_ms} ms`, r.edited ? el('span', { class: 'badge-edited' }, 'edited') : ''),
      tokenChips(r.tokens, r.position, (i) => { idx.value = i; go(); }),
      el('p', {}, `token #${r.position} "${vis(tok.text)}"`, r.next ? ` → next in the sequence "${vis(r.next.text)}"` : '',
        r.prediction.length ? ` · final prediction ${r.prediction.slice(0, 5).map((p) => `"${vis(p.text)}" ${p.prob.toFixed(3)}`).join('  ')}` : ''),
      el('p', {}, `target first top-1 at layer ${r.first_top1 ?? '-'}, first in the top 5 at layer ${r.first_top5 ?? '-'}`),
      el('div', { class: 'row' },
        el('button', { class: 'primary', disabled: !r.highlight.length, onclick: () => { setHighlight(id, r.highlight, r); location.hash = `#/model/${id}/graph?highlight=1`; } }, 'Show path in the graph (2D / 3D)'),
        el('a', { href: `#/model/${id}/timeline?prompt=${encodeURIComponent(prompt.value)}` }, 'generation timeline →')),
      el('h3', {}, 'Layer by layer'),
      table([
        { label: 'layer', get: (l) => l.index + (l.skipped ? ' (skip)' : ''), num: true, sort: (l) => l.index },
        { label: '|residual|', get: (l) => el('span', {}, bar(l.residual_norm, maxNorm), ` ${l.residual_norm.toFixed(1)}`), sort: (l) => l.residual_norm },
        { label: 'logit lens', get: (l) => l.lens.slice(0, 3).map((x) => `"${vis(x.text)}" ${x.prob.toFixed(2)}`).join('  ') },
        { label: 'P(target)', get: (l) => el('span', {}, bar(l.target_prob, 1), ` ${l.target_prob.toFixed(3)}`), sort: (l) => l.target_prob },
        { label: 'rank', get: (l) => l.target_rank, num: true },
        { label: 'attends to', get: (l) => l.attended.slice(0, 3).map(([k, w]) => `#${k}"${vis(r.tokens[k]?.text ?? '')}" ${w.toFixed(2)}`).join(' ') },
        { label: 'top FFN buckets', get: (l) => l.ffn_buckets.slice(0, 2).map(([b, v]) => `${b.split('.n.')[1]} ${v.toFixed(2)}`).join('  ') },
      ], r.layers),
      el('h3', {}, 'Attention'),
      el('p', { class: 'muted small' }, 'Mean over heads, one row per layer, one column per earlier token. Click a row for its heads.'),
      heatmap(meanRows, (i) => `layer ${i}`, (k) => `#${k} "${vis(r.tokens[k]?.text ?? '')}"`, { onRow: showHeads }),
      headBox,
      el('h3', {}, 'Decision breakdown'),
      decisionView(id, r.decision),
      ...r.warnings.map((w) => el('p', { class: 'muted' }, w)),
      el('p', { class: 'muted small' }, 'MoE routing and SSM state contributions are not measurable with this forward pass (dense llama-family and BERT-family architectures only).'));
    const best = headLayer ?? (r.decision?.top_heads?.[0]?.layer ?? null);
    if (best !== null) showHeads(best);
  };

  const go = async () => {
    s.prompt = prompt.value; s.chat = chat.checked;
    try {
      const r = await job(out, 'journey', id, { prompt: prompt.value, chat: chat.checked, token: Number(idx.value), baseline: baseline.checked });
      s.journey = r;
      render(r);
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  const sec = el('section', {},
    el('h2', {}, 'Token journey'),
    t.ok ? '' : unsupported(t),
    el('div', { class: 'row' }, prompt, el('label', {}, chat, ' chat'), 'token #', idx, el('label', { title: 'ignore the edit stack' }, baseline, ' baseline'),
      el('button', { class: 'primary', disabled: !t.ok, onclick: go }, 'Follow token')),
    el('p', { class: 'muted small' }, 'Token index: 0 = first; negative counts from the end (-1 = last). Runs with the current edits unless "baseline" is checked.'),
    out);
  if (t.ok) {
    loadEdits(id).catch(() => {});
    setTimeout(go, 0);
  }
  return sec;
}

// --- generation timeline ---------------------------------------------------------------

export async function timeline(id, params) {
  const s = session(id);
  const t = await traceSupport(id);
  const prompt = el('input', { size: 50, value: params?.get('prompt') || s.prompt });
  const chat = el('input', { type: 'checkbox', checked: s.chat });
  const steps = el('input', { type: 'number', min: 1, max: 128, value: 12, style: 'width:4em' });
  const explain = el('input', { type: 'checkbox' });
  const temp = el('input', { type: 'number', step: 'any', placeholder: 'greedy', style: 'width:5em' });
  const topk = el('input', { type: 'number', placeholder: '-', style: 'width:4em' });
  const topp = el('input', { type: 'number', step: 'any', placeholder: '-', style: 'width:4em' });
  const seed = el('input', { type: 'number', placeholder: '-', style: 'width:5em' });
  const baseline = el('input', { type: 'checkbox' });
  const out = el('div', {});
  const stepBox = el('div', {});
  const num = (i) => (i.value === '' ? null : Number(i.value));

  const openStep = async (r, st) => {
    stepBox.replaceChildren(el('h3', {}, `Step ${st.index}: "${vis(st.token.text)}"`));
    if (st.decision) { stepBox.append(decisionView(id, st.decision)); return; }
    const box = el('div', {});
    stepBox.append(box);
    try {
      // The decision for step k is made at the token before it: append the generated tokens
      // up to and including k and inspect the second to last.
      const j = await job(box, 'journey', id, { prompt: prompt.value, chat: chat.checked, append: r.generated_ids.slice(0, st.index + 1), token: -2, baseline: baseline.checked });
      box.replaceChildren(decisionView(id, j.decision),
        el('a', { href: '#', onclick: (e) => { e.preventDefault(); setHighlight(id, j.highlight, j); location.hash = `#/model/${id}/graph?highlight=1`; } }, 'show this decision in the graph →'));
    } catch (e) { box.replaceChildren(errorBox(e)); }
  };

  const render = (r) => {
    const maxE = Math.max(1e-6, ...r.steps.map((x) => x.entropy));
    out.replaceChildren(
      el('p', {}, `${r.steps.length} step(s) · ${r.tokens_per_s.toFixed(2)} tokens/s · sampling ${r.sampling.temperature > 0 ? `temperature ${r.sampling.temperature}` : 'greedy'}`, r.edited ? el('span', { class: 'badge-edited' }, 'edited') : ''),
      el('pre', { class: 'gen' }, r.generated),
      el('div', { class: 'timeline' }, r.steps.map((st) => el('div', { class: `step ${st.forced ? 'forced' : ''}`, onclick: () => openStep(r, st), title: 'click for the decision breakdown' },
        el('div', { class: 'tok' }, `"${vis(st.token.text)}"`),
        el('div', { class: 'meter' }, el('div', { style: `width:${(st.token.prob * 100).toFixed(0)}%` })),
        el('div', { class: 'small' }, `p ${st.token.prob.toFixed(2)}`),
        el('div', { class: 'meter ent' }, el('div', { style: `width:${(100 * st.entropy / maxE).toFixed(0)}%` })),
        el('div', { class: 'small' }, `H ${st.entropy.toFixed(2)}`),
        st.alternatives[0] ? el('div', { class: 'small muted' }, `alt "${vis(st.alternatives[0].text)}" ${st.alternatives[0].prob.toFixed(2)}`) : ''))),
      table([
        { label: 'step', get: (x) => x.index + (x.forced ? ' (forced)' : ''), num: true, sort: (x) => x.index },
        { label: 'token', get: (x) => el('code', {}, `"${vis(x.token.text)}"`) },
        { label: 'probability', get: (x) => x.token.prob.toFixed(3), num: true },
        { label: 'entropy', get: (x) => x.entropy.toFixed(3), num: true },
        { label: 'top alternative', get: (x) => (x.alternatives[0] ? `"${vis(x.alternatives[0].text)}" ${x.alternatives[0].prob.toFixed(3)}` : '') },
      ], r.steps),
      ...r.warnings.map((w) => el('p', { class: 'muted small' }, w)));
  };

  const go = async () => {
    s.prompt = prompt.value; s.chat = chat.checked;
    stepBox.replaceChildren();
    try {
      const r = await job(out, 'generate', id, {
        prompt: prompt.value, chat: chat.checked, steps: Number(steps.value) || 12, explain: explain.checked,
        temperature: num(temp), sample_top_k: num(topk), top_p: num(topp), seed: num(seed), baseline: baseline.checked,
      });
      render(r);
    } catch (e) { out.replaceChildren(errorBox(e)); }
  };
  return el('section', {},
    el('h2', {}, 'Generation timeline'),
    t.ok ? '' : unsupported(t),
    el('div', { class: 'row' }, prompt, el('label', {}, chat, ' chat'), 'steps', steps, el('label', { title: 'decision breakdown for every step (slower)' }, explain, ' explain'),
      el('button', { class: 'primary', disabled: !t.ok, onclick: go }, 'Generate')),
    el('div', { class: 'row small' }, 'temperature', temp, 'top-k', topk, 'top-p', topp, 'seed', seed, el('label', { title: 'ignore the edit stack' }, baseline, ' baseline')),
    out, stepBox);
}
