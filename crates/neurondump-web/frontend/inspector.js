// Token inspector: tokenization (static, any model) with the 3D embedding map, the token
// journey through the layers, and the generation timeline (feature `trace`).

import { api } from './api.js';
import { el, esc, table, errorBox, toast } from './ui.js';
import { session, pushEdit } from './session.js';
import { createEmbeddingMap, CLASS_COLORS } from './embedmap.js';

const CLASSES = ['special', 'reserved', 'byte', 'whitespace', 'digit', 'punctuation', 'emoji', 'cjk', 'code', 'word', 'subword', 'mixed'];
const classColor = (c) => CLASS_COLORS[Math.max(0, CLASSES.indexOf(c)) % CLASS_COLORS.length];
export const vis = (t) => esc(t).replace(/ /g, '·') || '∅';

export function tokenChips(tokens, selected, onPick, cls) {
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

export { journey, timeline, heatmap, decisionView } from './journey.js';
