// In-browser NeuronDump demo. The regular web UI (app.js and its views) runs unchanged; the
// HTTP API is replaced by a local backend backed by the WebAssembly build of the core. The
// GGUF file is read lazily with Blob.slice: the header first, then one tensor's bytes only
// when its statistics are requested. Nothing leaves the browser.

import init, { Session } from './pkg/neurondump_wasm.js';
import { ApiError, setLocalBackend } from './api.js';
import { el, errorBox } from './ui.js';

const RELEASES = 'https://github.com/aymen94/NeuronDump/releases/latest';
const FIRST_READ = 8 << 20; // header prefix; grown ×4 while the header does not fit
const MAX_TENSOR_READ = 512 << 20; // per-tensor statistics limit in the browser

let file = null;
let session = null;

const slice = async (start, end) => new Uint8Array(await file.slice(start, end).arrayBuffer());

async function openFile(f) {
  file = f;
  let n = Math.min(f.size, FIRST_READ);
  for (;;) {
    const prefix = await slice(0, n);
    try {
      return new Session(prefix, f.size, f.name);
    } catch (e) {
      if (!String(e?.message).startsWith('NEED_MORE') || n >= f.size) throw e;
      n = Math.min(f.size, n * 4);
    }
  }
}

const backend = {
  async api(path, opts = {}) {
    if (opts.method && opts.method !== 'GET') throw new ApiError(501, 'not available in the browser demo');
    const [p, q = ''] = path.split('?');
    if (!session) {
      if (p === '/models') return { models: [], warnings: [] };
      throw new ApiError(404, 'drop a .gguf file first');
    }
    try {
      const stats = /^\/models\/[^/]+\/tensors\/([^/]+)\/stats$/.exec(p);
      if (stats) {
        const name = decodeURIComponent(stats[1]);
        const [start, end] = session.tensor_range(name);
        if (end - start > MAX_TENSOR_READ) throw new Error(`${name} is ${((end - start) / 2 ** 20).toFixed(0)} MiB; the browser demo reads up to ${MAX_TENSOR_READ / 2 ** 20} MiB per tensor`);
        return JSON.parse(session.tensor_stats(name, await slice(start, end)));
      }
      return JSON.parse(session.api(p.split('/').map(decodeURIComponent).join('/'), q));
    } catch (e) {
      throw e instanceof ApiError ? e : new ApiError(400, e?.message || String(e));
    }
  },
};

function banner() {
  const input = el('input', { type: 'file', accept: '.gguf' });
  const status = el('span', { class: 'muted' }, 'or drop a .gguf file anywhere on this page');
  const load = async (f) => {
    if (!f) return;
    status.replaceChildren(`reading ${f.name}…`);
    try {
      session?.free();
      session = null;
      session = await openFile(f);
      status.replaceChildren(`${f.name} (${(f.size / 2 ** 30).toFixed(2)} GiB)`);
      const target = `#/model/${session.id()}/overview`;
      if (location.hash === target) window.dispatchEvent(new HashChangeEvent('hashchange'));
      else location.hash = target;
    } catch (e) {
      status.replaceChildren(errorBox(e));
    }
  };
  input.addEventListener('change', () => load(input.files[0]));
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => { e.preventDefault(); load(e.dataTransfer.files[0]); });
  return el('div', { class: 'demo-banner' },
    el('div', { class: 'row' }, el('strong', {}, 'Browser demo · inspection only'), input, status),
    el('p', { class: 'muted small' },
      'Runs 100% locally in your browser: the file is never uploaded, only the parts you look at are read. ',
      'Traces, token journeys, edits, analysis, slimming and diff need the desktop app: ',
      el('a', { href: RELEASES }, 'download NeuronDump'), '.'));
}

document.head.append(el('style', {}, `
  #mainnav, #edits-toggle { display: none !important; }
  .demo-banner { padding: .5rem 1rem; border-bottom: 1px solid var(--border); background: var(--bg-alt); }
  .demo-banner p { margin: .25rem 0 0; }
`));
document.getElementById('top').after(banner());

try {
  await init();
  setLocalBackend(backend);
  await import('./app.js');
} catch (e) {
  document.getElementById('view').replaceChildren(errorBox(new Error(`Could not start the WebAssembly module: ${e.message}`)));
}
