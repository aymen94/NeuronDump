// Navigation / display panel over the graph canvas: press-and-hold orbit, pan and zoom pads,
// camera presets, auto-rotate, control speeds, saved views and share links, layout tools
// (size by, explode, slice, isolate) and display options. Keyboard on the focused canvas:
// WASD / arrows pan, Q/E orbit, R/F tilt, +/- zoom, Space or 0 fits.

import { el, store } from './ui.js';

const KEY = 'nd-graph3d-panel';

export const DISPLAY_DEFAULTS = {
  labels: true, maxLabels: 160, edges: true, edgeOpacity: 1, outlines: true, fov: 45,
  nav: 'orbit', rotateSpeed: 1, panSpeed: 1, zoomSpeed: 1, damping: 0.12, zoomToCursor: true,
  autoRotate: false, autoRotateSpeed: 1.5, open: true,
  sizeBy: 'layout', explode: 1, sliceAxis: 'off', slice: 1, isolate: false, isolateDepth: 1, minimap: true,
};

// Settings that change the layout itself (the scene is rebuilt).
const LAYOUT_KEYS = new Set(['sizeBy', 'explode', 'sliceAxis', 'slice', 'isolate', 'isolateDepth']);

export function loadSettings() {
  try { return { ...DISPLAY_DEFAULTS, ...JSON.parse(store.get(KEY) || '{}') }; } catch { return { ...DISPLAY_DEFAULTS }; }
}

const KEYMAP = {
  KeyW: 'panUp', ArrowUp: 'panUp', KeyS: 'panDown', ArrowDown: 'panDown',
  KeyA: 'panLeft', ArrowLeft: 'panLeft', KeyD: 'panRight', ArrowRight: 'panRight',
  KeyQ: 'orbitLeft', KeyE: 'orbitRight', KeyR: 'tiltUp', KeyF: 'tiltDown',
  Equal: 'zoomIn', NumpadAdd: 'zoomIn', Minus: 'zoomOut', NumpadSubtract: 'zoomOut',
};

/**
 * ctx: { root, canvas, controls() (the current controls), settings, isFlat(), fit(), reset(), zoomBy(f), view(name),
 *        apply() (settings changed), rebuild() (layout settings changed), dirty(),
 *        getView(), setView(v), viewsKey (localStorage key of saved views), hasSelection() }
 * Returns { tick(dt), sync(), dispose() }; tick runs the held actions once per frame.
 */
export function createControlPanel(ctx) {
  const { root, canvas, settings } = ctx;
  const held = new Set();
  const save = () => store.set(KEY, JSON.stringify(settings));
  // Layout changes rebuild the scene at most once per frame while a slider moves.
  let rebuildQueued = false;
  const set = (k, v) => {
    settings[k] = v;
    save();
    if (!LAYOUT_KEYS.has(k)) { ctx.apply(); return; }
    if (rebuildQueued) return;
    rebuildQueued = true;
    requestAnimationFrame(() => { rebuildQueued = false; ctx.rebuild(); });
  };

  // Press-and-hold buttons: the action runs every frame while the pointer is down.
  const hold = (action, label, title) => {
    const b = el('button', { type: 'button', class: 'g3p-btn', title }, label);
    const stop = () => { held.delete(action); };
    b.addEventListener('pointerdown', (e) => { e.preventDefault(); b.setPointerCapture?.(e.pointerId); held.add(action); ctx.dirty(); });
    b.addEventListener('pointerup', stop);
    b.addEventListener('pointercancel', stop);
    b.addEventListener('lostpointercapture', stop);
    return b;
  };
  const btn = (label, title, onclick) => el('button', { type: 'button', class: 'g3p-btn', title, onclick }, label);
  const pad = (title, up, left, right, down, mid) => el('div', { class: 'g3p-pad', title },
    el('span'), up, el('span'), left, mid || el('span'), right, el('span'), down, el('span'));

  const range = (key, label, min, max, step, fmt = (v) => v) => {
    const out = el('output', {}, fmt(settings[key]));
    const input = el('input', { type: 'range', min, max, step, value: settings[key] });
    input.addEventListener('input', () => { set(key, Number(input.value)); out.textContent = fmt(settings[key]); });
    return { row: el('label', { class: 'g3p-range' }, el('span', {}, label), input, out), input, out, key, fmt };
  };
  const check = (key, label) => {
    const input = el('input', { type: 'checkbox', checked: settings[key] });
    input.addEventListener('change', () => set(key, input.checked));
    return { row: el('label', { class: 'g3p-check' }, input, label), input, key };
  };
  const choice = (key, label, options) => {
    const input = el('select', {}, Object.entries(options).map(([v, l]) => el('option', { value: v, selected: settings[key] === v }, l)));
    input.addEventListener('change', () => set(key, input.value));
    return { row: el('label', { class: 'g3p-range g3p-choice' }, el('span', {}, label), input, el('span')), input, key };
  };

  const orbitPad = pad('Orbit (Q/E, R/F)',
    hold('tiltUp', '▲', 'Tilt up (R)'), hold('orbitLeft', '◀', 'Orbit left (Q)'),
    hold('orbitRight', '▶', 'Orbit right (E)'), hold('tiltDown', '▼', 'Tilt down (F)'),
    el('span', { class: 'g3p-pad-mid' }, '⟳'));
  const panPad = pad('Pan (WASD / arrows)',
    hold('panUp', '▲', 'Pan up (W)'), hold('panLeft', '◀', 'Pan left (A)'),
    hold('panRight', '▶', 'Pan right (D)'), hold('panDown', '▼', 'Pan down (S)'),
    el('span', { class: 'g3p-pad-mid' }, '✥'));
  const zoomCol = el('div', { class: 'g3p-zoom' },
    hold('zoomIn', '+', 'Zoom in (+)'), hold('zoomOut', '−', 'Zoom out (−)'),
    btn('Fit', 'Fit everything (Space)', () => ctx.fit()));

  const views = el('div', { class: 'g3p-views' },
    ...[['front', 'Front'], ['top', 'Top'], ['side', 'Side'], ['iso', 'Iso']]
      .map(([v, l]) => btn(l, `Look from the ${l.toLowerCase()} (keeps the zoom)`, () => ctx.view(v))),
    btn('Reset', 'Default camera for this layout', () => ctx.reset()));

  const NAVS = {
    orbit: ['Orbit', 'Left drag orbits around the target; the vertical stays up'],
    map: ['Map', 'Left drag slides over the model like a map; right drag orbits'],
    trackball: ['Trackball', 'Free rotation in every direction, over the top and under the bottom'],
  };
  const navBtns = Object.entries(NAVS).map(([v, [label, title]]) => el('button', {
    type: 'button', 'aria-checked': String(settings.nav === v), title,
    onclick: () => { for (const [i, b] of navBtns.entries()) b.setAttribute('aria-checked', String(Object.keys(NAVS)[i] === v)); set('nav', v); syncNav(); },
  }, label));
  const navRow = el('div', { class: 'g3p-navmode' }, el('span', { class: 'segmented', role: 'radiogroup', title: 'How the mouse moves the camera' }, ...navBtns));

  const fields = [
    check('autoRotate', 'Auto-rotate'),
    range('autoRotateSpeed', 'spin', 0.2, 8, 0.1, (v) => v.toFixed(1)),
    check('zoomToCursor', 'Wheel zooms toward cursor'),
    range('rotateSpeed', 'orbit speed', 0.2, 3, 0.1, (v) => `${v.toFixed(1)}×`),
    range('panSpeed', 'pan speed', 0.2, 3, 0.1, (v) => `${v.toFixed(1)}×`),
    range('zoomSpeed', 'zoom speed', 0.2, 3, 0.1, (v) => `${v.toFixed(1)}×`),
    range('damping', 'inertia', 0, 0.5, 0.01, (v) => (v === 0 ? 'off' : v.toFixed(2))),
    range('fov', 'field of view', 15, 90, 1, (v) => `${v}°`),
  ];
  const display = [
    check('labels', 'Labels'),
    range('maxLabels', 'max labels', 20, 600, 10),
    check('edges', 'Connections'),
    range('edgeOpacity', 'line strength', 0.1, 1, 0.05, (v) => `${Math.round(v * 100)}%`),
    check('outlines', 'Container outlines'),
  ];
  display.push(check('minimap', 'Minimap'));
  const layout = [
    choice('sizeBy', 'size by', { layout: 'layout', params: 'parameters', bytes: 'bytes on disk', importance: 'importance' }),
    range('explode', 'explode', 1, 4, 0.05, (v) => `${v.toFixed(2)}×`),
    choice('sliceAxis', 'slice along', { off: 'off', x: 'x', y: 'y (height)', z: 'z (depth)' }),
    range('slice', 'slice at', 0, 1, 0.01, (v) => `${Math.round(v * 100)}%`),
    check('isolate', 'Isolate the selection'),
    range('isolateDepth', 'neighbours', 1, 4, 1, (v) => `${v} hop${v > 1 ? 's' : ''}`),
  ];
  const isolateHint = el('p', { class: 'g3p-keys' }, 'Click a node to isolate it with its neighbours.');
  const syncLayout = () => {
    layout[3].row.hidden = settings.sliceAxis === 'off';
    layout[5].row.hidden = !settings.isolate;
    isolateHint.hidden = !settings.isolate || ctx.hasSelection();
  };
  layout[2].input.addEventListener('change', syncLayout);
  layout[4].input.addEventListener('change', syncLayout);

  // Saved camera views (per model) and share links.
  const viewsKey = ctx.viewsKey || 'nd-graph3d-views';
  const loadViews = () => { try { return JSON.parse(store.get(viewsKey) || '[]'); } catch { return []; } };
  const viewName = el('input', { type: 'text', placeholder: 'name this view', size: 10, class: 'g3p-name' });
  const viewList = el('div', { class: 'g3p-saved' });
  const renderViews = () => {
    const list = loadViews();
    viewList.replaceChildren(...(list.length ? list.map((v, i) => el('span', { class: 'g3p-chip' },
      el('button', { type: 'button', class: 'g3p-btn', title: `Go to "${v.name}"${v.mode !== undefined ? ` (${v.mode})` : ''}`, onclick: () => ctx.setView(v) }, v.name),
      el('button', { type: 'button', class: 'g3p-btn g3p-x', title: 'Delete this view', onclick: () => { const l = loadViews(); l.splice(i, 1); store.set(viewsKey, JSON.stringify(l)); renderViews(); } }, '×')))
      : [el('span', { class: 'g3p-keys' }, 'No saved views yet.')]));
  };
  const saveView = () => {
    const list = loadViews();
    const name = viewName.value.trim() || `view ${list.length + 1}`;
    list.push({ name, ...ctx.getView() });
    store.set(viewsKey, JSON.stringify(list.slice(-20)));
    viewName.value = '';
    renderViews();
  };
  viewName.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') saveView(); });
  const copyLink = async () => {
    const v = ctx.getView();
    const [path, query] = location.hash.split('?');
    const p = new URLSearchParams(query || '');
    p.set('view', v.mode);
    p.set('cam', [...v.t, ...v.p, ...v.u].join(','));
    const url = `${location.origin}${location.pathname}${path}?${p}`;
    history.replaceState(null, '', `${path}?${p}`);
    try { await navigator.clipboard.writeText(url); linkBtn.textContent = 'Link copied'; } catch { linkBtn.textContent = 'Link in the address bar'; }
    setTimeout(() => { linkBtn.textContent = 'Copy link'; }, 1600);
  };
  const linkBtn = btn('Copy link', 'Copy a link to this camera view (the session token is not included)', copyLink);
  renderViews();

  const only3d = [navRow, orbitPad, views, fields[0].row, fields[1].row, fields[3].row, fields[7].row];
  // The trackball always zooms toward the centre.
  const syncNav = () => { fields[2].row.hidden = !ctx.isFlat() && settings.nav === 'trackball'; };

  const body = el('div', { class: 'g3p-body' },
    navRow,
    el('div', { class: 'g3p-nav' }, orbitPad, panPad, zoomCol),
    views,
    el('details', { class: 'g3p-sec', open: true }, el('summary', {}, 'Camera'), ...fields.map((f) => f.row)),
    el('details', { class: 'g3p-sec' }, el('summary', {}, 'Saved views'),
      el('div', { class: 'g3p-row' }, viewName, btn('Save', 'Save the current camera view', saveView), linkBtn), viewList),
    el('details', { class: 'g3p-sec' }, el('summary', {}, 'Layout'), ...layout.map((f) => f.row), isolateHint),
    el('details', { class: 'g3p-sec' }, el('summary', {}, 'Display'), ...display.map((f) => f.row)),
    el('p', { class: 'g3p-keys' }, 'Click the canvas, then: WASD/arrows pan · Q/E orbit · R/F tilt · +/− zoom · Space fit'));
  const toggle = el('button', { type: 'button', class: 'g3p-toggle', title: 'Show / hide the navigation panel' });
  const panel = el('div', { class: 'g3p' }, toggle, body);
  const showOpen = () => {
    body.hidden = !settings.open;
    toggle.textContent = settings.open ? '✕ Controls' : '☰ Controls';
    panel.classList.toggle('closed', !settings.open);
  };
  toggle.addEventListener('click', () => { settings.open = !settings.open; save(); showOpen(); });
  // Events in the panel must not reach the canvas (orbit drag, picking).
  for (const t of ['pointerdown', 'wheel', 'dblclick']) panel.addEventListener(t, (e) => e.stopPropagation());
  root.append(panel);
  showOpen();

  // Keyboard on the canvas.
  canvas.tabIndex = 0;
  const onKey = (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code === 'Space' || e.code === 'Digit0' || e.code === 'Numpad0') { e.preventDefault(); ctx.fit(); return; }
    const a = KEYMAP[e.code];
    if (!a) return;
    e.preventDefault();
    if (ctx.isFlat() && /^(orbit|tilt)/.test(a)) return;
    held.add(a); ctx.dirty();
  };
  const onKeyUp = (e) => { const a = KEYMAP[e.code]; if (a) held.delete(a); };
  const onBlur = () => held.clear();
  const onCanvasDown = () => canvas.focus({ preventScroll: true });
  canvas.addEventListener('keydown', onKey);
  canvas.addEventListener('keyup', onKeyUp);
  canvas.addEventListener('blur', onBlur);
  canvas.addEventListener('pointerdown', onCanvasDown);

  function tick(dt) {
    if (!held.size) return false;
    dt = Math.min(dt, 0.05);
    const rot = 1.4 * settings.rotateSpeed * dt;
    const pan = 520 * settings.panSpeed * dt;
    const zoom = Math.pow(2.2, settings.zoomSpeed * dt);
    const flat = ctx.isFlat();
    const controls = ctx.controls();
    for (const a of held) {
      switch (a) {
        case 'panUp': controls.pan(0, pan); break;
        case 'panDown': controls.pan(0, -pan); break;
        case 'panLeft': controls.pan(pan, 0); break;
        case 'panRight': controls.pan(-pan, 0); break;
        case 'orbitLeft': if (!flat) controls.rotateLeft(rot); break;
        case 'orbitRight': if (!flat) controls.rotateLeft(-rot); break;
        case 'tiltUp': if (!flat) controls.rotateUp(rot); break;
        case 'tiltDown': if (!flat) controls.rotateUp(-rot); break;
        case 'zoomIn': ctx.zoomBy(zoom); break;
        case 'zoomOut': ctx.zoomBy(1 / zoom); break;
      }
    }
    return true;
  }

  return {
    tick,
    /** Re-applies mode visibility (2D hides orbit-only controls). */
    sync() { const flat = ctx.isFlat(); for (const n of only3d) n.hidden = flat; syncNav(); syncLayout(); },
    dispose() {
      canvas.removeEventListener('keydown', onKey);
      canvas.removeEventListener('keyup', onKeyUp);
      canvas.removeEventListener('blur', onBlur);
      canvas.removeEventListener('pointerdown', onCanvasDown);
      panel.remove();
    },
  };
}
