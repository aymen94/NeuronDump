// Tiny DOM helpers (no framework).

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function bytes(n) {
  if (n === null || n === undefined) return '-';
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0; let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return i === 0 ? `${n} B` : `${v.toFixed(2)} ${u[i]}`;
}

export function count(n) {
  if (n === null || n === undefined) return '-';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function esc(s) { return JSON.stringify(s ?? '').slice(1, -1); }

/** Inline code that copies its text on click. */
export function copyable(text, label) {
  return el('code', {
    class: 'copy',
    title: 'Click to copy',
    onclick: async (e) => {
      e.preventDefault();
      try { await navigator.clipboard.writeText(text); toast(`Copied ${text.length > 40 ? text.slice(0, 40) + '…' : text}`); } catch { toast('Copy failed'); }
    },
  }, label ?? text);
}

/** Text with every case-insensitive occurrence of `needle` wrapped in <mark>. */
export function highlight(text, needle) {
  const s = String(text ?? '');
  if (!needle) return document.createTextNode(s);
  const out = el('span', {});
  const lower = s.toLowerCase(); const n = needle.toLowerCase();
  let i = 0;
  for (;;) {
    const j = lower.indexOf(n, i);
    if (j < 0) { out.append(s.slice(i)); break; }
    out.append(s.slice(i, j), el('mark', {}, s.slice(j, j + n.length)));
    i = j + n.length;
  }
  return out;
}

/** Inline info tooltip (?) with popup explanation. */
export function infoTip(text) {
  return el('span', { class: 'info-tip', tabindex: '0', title: text }, '?',
    el('span', { class: 'tip-content' }, text));
}

/** Row of summary cards: [[label, value, tip?], ...] */
export function cards(items) {
  return el('div', { class: 'cards' }, items.filter(([, v]) => v !== undefined && v !== null && v !== '-')
    .map(([k, v, tip]) => el('div', { class: 'card' },
      el('div', { class: 'k' }, k, tip ? infoTip(tip) : ''),
      el('div', { class: 'v' }, v))));
}

function cellText(v) { return v instanceof Node ? v.textContent : String(v ?? ''); }

/**
 * columns: [{label, get(row) -> string|Node, tip?, num?, sort?(row) -> number|string}]
 * Click a header to sort (client-side) unless opts.sortable === false.
 */
export function table(columns, rows, opts = {}) {
  const sortable = opts.sortable !== false;
  let order = { i: -1, dir: 1 };
  const tbody = el('tbody', {});
  const ths = columns.map((c, i) => {
    const headerChildren = [c.label];
    if (c.tip) {
      headerChildren.push(infoTip(c.tip));
    }
    headerChildren.push(el('span', { class: 'arrow' }));
    return el('th', {
      class: sortable ? 'sortable' : '',
      onclick: sortable ? (e) => {
        if (!e.target.closest('.info-tip')) sortBy(i);
      } : null,
    }, ...headerChildren);
  });
  const render = (list) => tbody.replaceChildren(...list.map((r) => el('tr', {}, columns.map((c) => {
    const v = c.get(r);
    return el('td', { class: c.num ? 'num' : '' }, v instanceof Node ? v : String(v ?? ''));
  }))));
  function key(c, r) {
    if (c.sort) return c.sort(r);
    const t = cellText(c.get(r));
    const n = Number(t.replace(/[,%]/g, ''));
    return t !== '' && !Number.isNaN(n) ? n : t.toLowerCase();
  }
  function sortBy(i) {
    order = { i, dir: order.i === i ? -order.dir : 1 };
    const c = columns[i];
    const sorted = rows.slice().sort((a, b) => {
      const x = key(c, a); const y = key(c, b);
      return (x < y ? -1 : x > y ? 1 : 0) * order.dir;
    });
    ths.forEach((th, j) => { th.lastChild.textContent = j === i ? (order.dir > 0 ? ' ▲' : ' ▼') : ''; });
    render(sorted);
  }
  render(rows);
  return el('div', { class: 'tablewrap' }, el('table', {}, el('thead', {}, el('tr', {}, ths)), tbody));
}

export function kv(obj) {
  const rows = Object.entries(obj || {}).filter(([, v]) => v !== null && v !== undefined);
  return table([{ label: 'key', get: (r) => r[0] }, { label: 'value', get: (r) => typeof r[1] === 'object' ? JSON.stringify(r[1]) : String(r[1]) }], rows, { sortable: false });
}

export function progressBar(pct) {
  return el('span', { class: 'progress' }, el('div', { style: `width:${Math.max(0, Math.min(100, pct))}%` }));
}

/** Paginated table: fetchPage(offset, limit) -> {total, rows}. Sorting is done server-side. */
export function pager(columns, fetchPage, limit = 100) {
  const box = el('div', {});
  let offset = 0;
  async function load() {
    box.replaceChildren(el('p', { class: 'muted' }, 'Loading…'));
    try {
      const { total, rows } = await fetchPage(offset, limit);
      const nav = el('div', { class: 'row' },
        el('button', { disabled: offset === 0, onclick: () => { offset = Math.max(0, offset - limit); load(); } }, '◀ prev'),
        el('span', { class: 'muted' }, `${total === 0 ? 0 : offset + 1}–${Math.min(offset + limit, total)} of ${total}`),
        el('button', { disabled: offset + limit >= total, onclick: () => { offset += limit; load(); } }, 'next ▶'));
      box.replaceChildren(nav, table(columns, rows, { sortable: false }));
    } catch (e) {
      box.replaceChildren(errorBox(e));
    }
  }
  box.reload = () => { offset = 0; load(); };
  load();
  return box;
}

export function errorBox(e) {
  return el('p', { class: 'error' }, `Error${e.status ? ' ' + e.status : ''}: ${e.message || e}`);
}

export function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => { t.hidden = true; }, 3000);
}

/** localStorage that never throws (private windows, blocked storage). */
export const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};
