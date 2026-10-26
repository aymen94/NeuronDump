// State shared between pages of one model during the session: the last prompt, the last
// token journey (for the graph's token flow), nodes to highlight in the graph, and the
// edit stack (kept in sync with the server and announced with an `nd-edits` event), and the
// last diff against another model (for the graph's "color by diff").

import { api } from './api.js';

const state = new Map(); // model id -> { prompt, chat, journey, highlight:Set, edits }

export function session(id) {
  if (!state.has(id)) state.set(id, { prompt: 'The name of the star that Earth orbits is', chat: false, journey: null, highlight: new Set(), edits: null, diff: null });
  return state.get(id);
}

/** Highlights these graph node ids (and remembers the journey) for the graph pages. */
export function setHighlight(id, ids, journey) {
  const s = session(id);
  s.highlight = new Set(ids || []);
  if (journey) s.journey = journey;
  window.dispatchEvent(new CustomEvent('nd-highlight', { detail: { model: id } }));
}

/** Remembers a diff report (model `id` is its A side) for the graph. */
export function setDiff(id, report) {
  session(id).diff = report;
}

/** Replaces the cached edit view and tells every page. */
export function setEdits(id, view) {
  session(id).edits = view;
  window.dispatchEvent(new CustomEvent('nd-edits', { detail: { model: id, view } }));
  return view;
}

export async function loadEdits(id) {
  return setEdits(id, await api(`/models/${id}/edits`));
}

/** Pushes one edit op ({op: 'scale', target, factor}, ...). */
export async function pushEdit(id, op) {
  return setEdits(id, await api(`/models/${id}/edits`, { method: 'POST', body: op }));
}

export async function editAction(id, method, path) {
  return setEdits(id, await api(`/models/${id}/edits${path}`, { method }));
}

/** Graph node ids touched by the enabled edits. */
export function editedNodes(id) {
  return new Set(session(id).edits?.compiled?.touched || []);
}
