// Hash router: #/models, #/model/<id>/<view>[?query], #/search?q=…, #/compare, #/diff, #/jobs.

import { api, getToken, isLocal } from './api.js';
import { el, errorBox, store } from './ui.js';
import * as views from './views.js';
import * as inspector from './inspector.js';
import { edits as editsPage, mountDrawer } from './edits.js';
import { loadEdits } from './session.js';
import { diff as diffPage } from './diff.js';

const MODEL_VIEWS = isLocal()
  ? ['overview', 'vocab', 'tensors', 'layers', 'graph', 'template', 'stats'] // browser demo: static inspection only
  : ['overview', 'vocab', 'tensors', 'layers', 'graph', 'tokens', 'journey', 'timeline', 'edits', 'template', 'stats', 'analyze', 'trace'];
const PAGES = { ...views, tokens: inspector.tokens, journey: inspector.journey, timeline: inspector.timeline, edits: editsPage };
const drawer = mountDrawer(() => currentModel);
let current = null;
let currentModel = '';
const names = new Map();

async function modelName(id) {
  if (!names.has(id)) {
    try {
      const d = await api('/models');
      d.models.forEach((m) => names.set(m.id, m.name));
    } catch { /* ignore */ }
  }
  return names.get(id) || id;
}

function parseHash() {
  const [path, query] = (location.hash.slice(1) || '/models').split('?');
  return { parts: path.split('/').filter(Boolean), params: new URLSearchParams(query || '') };
}

async function route() {
  const view = document.getElementById('view');
  const modelnav = document.getElementById('modelnav');
  const label = document.getElementById('model-label');
  if (current?.cleanup) current.cleanup();
  if (!getToken() && !isLocal()) {
    view.replaceChildren(errorBox({ message: 'No session token. Open the URL printed by `neurondump web` (it contains ?token=…).' }));
    return;
  }
  const { parts, params } = parseHash();
  document.querySelectorAll('#mainnav a').forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#/' + parts[0]));
  view.replaceChildren(el('p', { class: 'muted' }, 'Loading…'));
  try {
    let node;
    if (parts[0] === 'model' && parts[1]) {
      const id = parts[1];
      currentModel = id;
      const v = MODEL_VIEWS.includes(parts[2]) ? parts[2] : 'overview';
      modelnav.hidden = false;
      modelnav.replaceChildren(...MODEL_VIEWS.map((m) => el('a', { href: `#/model/${id}/${m}`, class: m === v ? 'active' : '' }, m)));
      label.textContent = await modelName(id);
      document.title = `${label.textContent} · ${v} · NeuronDump`;
      loadEdits(id).catch(() => {});
      drawer.refresh();
      node = await PAGES[v](id, params);
    } else {
      if (parts[0] !== 'search') currentModel = '';
      drawer.refresh();
      modelnav.hidden = true;
      label.textContent = '';
      document.title = 'NeuronDump';
      if (parts[0] === 'search') {
        search.value = params.get('q') || '';
        node = await views.search(null, params);
      } else {
        node = parts[0] === 'compare' ? await views.compare() : parts[0] === 'diff' ? await diffPage(null, params) : parts[0] === 'jobs' ? await views.jobs() : await views.models();
      }
    }
    current = node;
    view.replaceChildren(node);
  } catch (e) {
    view.replaceChildren(errorBox(e));
  }
}

// --- Header search ------------------------------------------------------------------

const search = document.getElementById('search');
let searchTimer = null;
function runSearch(replace) {
  const q = search.value.trim();
  if (!q) return;
  const { parts, params } = parseHash();
  const model = parts[0] === 'search' ? (params.get('model') || '') : currentModel;
  const hash = `#/search?q=${encodeURIComponent(q)}${model ? `&model=${model}` : ''}`;
  if (replace) history.replaceState(null, '', hash); else history.pushState(null, '', hash);
  route();
}
search.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { clearTimeout(searchTimer); runSearch(parseHash().parts[0] === 'search'); }
  if (e.key === 'Escape') search.blur();
});
search.addEventListener('input', () => {
  // Live results once on the search page; elsewhere wait for Enter.
  if (parseHash().parts[0] !== 'search') return;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => runSearch(true), 350);
});
document.addEventListener('keydown', (e) => {
  const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
  if (e.key === '/' && !typing) { e.preventDefault(); search.focus(); search.select(); }
});

// --- Theme: System (default) / Light / Dark ------------------------------------------
// "system" sets no attribute, so the CSS media query follows the OS, including live changes.

function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  else document.documentElement.removeAttribute('data-theme');
  document.querySelectorAll('[data-theme-choice]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.themeChoice === t)));
}
const savedTheme = store.get('nd-theme');
applyTheme(savedTheme === 'light' || savedTheme === 'dark' ? savedTheme : 'system');
document.querySelectorAll('[data-theme-choice]').forEach((b) => b.addEventListener('click', () => {
  store.set('nd-theme', b.dataset.themeChoice);
  applyTheme(b.dataset.themeChoice);
}));

// --- Running jobs badge ---------------------------------------------------------------

async function pollJobs() {
  const badge = document.getElementById('jobs-badge');
  try {
    const jobs = await api('/jobs');
    const running = jobs.filter((j) => j.status === 'running' || j.status === 'queued').length;
    badge.textContent = running;
    badge.hidden = running === 0;
  } catch { badge.hidden = true; }
}
if (!isLocal()) {
  if (getToken()) setInterval(pollJobs, 3000);
  pollJobs();
}

window.addEventListener('hashchange', route);
route();
