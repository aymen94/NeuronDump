// One function per view. Each returns a Node (or a Promise of one).

import { api, runJob } from './api.js';
import { el, progressBar } from './ui.js';

export function select(options, value) {
  return el('select', {}, options.map((o) => el('option', { value: o, selected: o === value }, o || '(any)')));
}

export function jobStatus(j) {
  return el('span', {}, progressBar(j.progress), ' ', el('span', { class: `status-${j.status}` }, j.status),
    ` ${j.progress.toFixed(0)}% ${j.step || ''}`, j.cached ? el('span', { class: 'muted' }, ' (cached)') : '',
    j.error ? el('span', { class: 'error' }, ' ' + j.error) : '');
}

/** Runs a job with an inline progress line; resolves to the parsed result or throws. */
export async function jobWithProgress(container, kind, model, params, onJob) {
  const line = el('div', { class: 'row' }, 'starting…');
  container.replaceChildren(line);
  let cancelBtn = null;
  const j = await runJob(kind, model, params, (j) => {
    if (onJob) onJob(j);
    if (!cancelBtn) cancelBtn = el('button', { onclick: () => api(`/jobs/${j.id}`, { method: 'DELETE' }) }, 'cancel');
    line.replaceChildren(jobStatus(j), j.status === 'running' || j.status === 'queued' ? cancelBtn : '');
  });
  if (j.status !== 'done') throw new Error(j.error || j.status);
  return api(`/jobs/${j.id}/result`);
}

// --- Models -----------------------------------------------------------------------

export * from './views/model.js';
export * from './views/analysis.js';
export * from './views/global.js';
export { graph } from './graph.js';
