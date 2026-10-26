// API client: every call carries the session token from the startup URL.

const params = new URLSearchParams(location.search);
let token = params.get('token') || sessionStorage.getItem('nd-token') || '';
if (params.get('token')) {
  sessionStorage.setItem('nd-token', token);
  // Keep the token out of the visible URL / history.
  history.replaceState(null, '', location.pathname + location.hash);
}

export function getToken() { return token; }

// In-browser demo: a local backend ({ api(path, opts) }, backed by WebAssembly) answers the
// read-only endpoints instead of the HTTP server.
let local = null;
export function setLocalBackend(b) { local = b; }
export function isLocal() { return local !== null; }

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export async function api(path, opts = {}) {
  if (local) return local.api(path, opts);
  const headers = { 'x-neurondump-token': token, ...(opts.headers || {}) };
  if (opts.body && typeof opts.body !== 'string') {
    headers['content-type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }
  const r = await fetch('/api' + path, { ...opts, headers });
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!r.ok) throw new ApiError(r.status, (data && data.error) || r.statusText);
  return data;
}

/** Binary GET: resolves to { buffer, headers }. */
export async function apiBinary(path) {
  const r = await fetch('/api' + path, { headers: { 'x-neurondump-token': token } });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch { /* not JSON */ }
    throw new ApiError(r.status, msg);
  }
  return { buffer: await r.arrayBuffer(), headers: r.headers };
}

export function qs(obj) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(obj)) if (v !== undefined && v !== null && v !== '') p.set(k, v);
  const s = p.toString();
  return s ? '?' + s : '';
}

/** Live job updates over Server-Sent Events. Returns a close function. */
export function watchJob(id, onUpdate) {
  const es = new EventSource(`/api/jobs/${encodeURIComponent(id)}/events?token=${encodeURIComponent(token)}`);
  const handler = (e) => onUpdate(JSON.parse(e.data));
  es.addEventListener('progress', handler);
  es.addEventListener('done', (e) => { handler(e); es.close(); });
  es.onerror = () => es.close();
  return () => es.close();
}

export function downloadUrl(path) {
  return `/api${path}${path.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`;
}

export async function runJob(kind, model, params, onUpdate) {
  const { id } = await api('/jobs', { method: 'POST', body: { kind, model, params } });
  return new Promise((resolve) => {
    watchJob(id, (j) => {
      onUpdate && onUpdate(j);
      if (['done', 'failed', 'cancelled'].includes(j.status)) resolve(j);
    });
  });
}
