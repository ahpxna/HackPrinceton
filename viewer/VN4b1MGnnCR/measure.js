/**
 * Metric Z-up viewer extension. Required ctx: { THREE, camera, renderer,
 * collision: Object3D, getDepthMeshes: () => Object3D[] }. Optional: document,
 * window, floorAt(Vector3): number|undefined, parent, roomDimensions, url,
 * acceptDepthHit(hit): boolean (optional reconstructed-sample filter).
 * getDepthMeshes must return the CURRENT real-photo depth meshes, not the splat.
 * Hook (after photoLayer exists; render() goes AFTER photoLayer.render(camera)):
 *   import { installMeasure } from './measure.js';
 *   const measure = installMeasure({THREE,camera,renderer,collision,getDepthMeshes:()=>photoLayer.depthMeshes()});
 *   measure.render();
 */
export const ACCURACY_NOTE = 'on reconstructed geometry; checkpoints within 2 cm (W9)';

export function formatDistance(metres) {
  const inches = Math.round(metres / .0254);
  return `${metres.toFixed(3)} m · ${Math.floor(inches / 12)} ft ${inches % 12} in`;
}

export function snapMeasurementPoint(THREE, hit, collisionHits = [], floorAt) {
  const point = hit.point.clone();
  const normal = hit.face?.normal?.clone();
  if (normal && hit.object?.matrixWorld) normal.transformDirection(hit.object.matrixWorld);
  let snap = 'surface';
  // Prefer the nearby authoritative plane, never an arbitrary world grid.
  const plane = collisionHits.find(p => p.face && p.point.distanceTo(point) <= .08 &&
    (Math.abs(p.face.normal.clone().transformDirection(p.object.matrixWorld).z) > .9 ||
      Math.abs(p.face.normal.clone().transformDirection(p.object.matrixWorld).z) < .15));
  if (plane) {
    const n = plane.face.normal.clone().transformDirection(plane.object.matrixWorld).normalize();
    point.addScaledVector(n, plane.point.clone().sub(point).dot(n));
    snap = Math.abs(n.z) > .9 ? 'floor' : 'wall';
  } else if (normal && Math.abs(normal.z) > .9) {
    const floor = floorAt?.(point);
    if (Number.isFinite(floor) && Math.abs(floor - point.z) <= .08) point.z = floor;
    snap = 'floor';
  } else if (normal && Math.abs(normal.z) < .15) snap = 'wall';
  return { point, snap };
}

export function installMeasure(ctx) {
  const { THREE: T, camera, renderer, collision } = ctx;
  if (!T || !camera || !renderer || !collision || !ctx.getDepthMeshes)
    throw Error('installMeasure requires THREE, camera, renderer, collision, getDepthMeshes');
  const doc = ctx.document ?? globalThis.document, win = ctx.window ?? globalThis.window;
  const canvas = renderer.domElement, ray = new T.Raycaster(), ndc = new T.Vector2();
  const overlay = new T.Scene(), measurements = [], listeners = [], disposables = [];
  const state = { enabled: false, pending: null, measurements, error: null, accuracy: ACCURACY_NOTE,
    roomDimensions: ctx.roomDimensions ?? null, lastSource: null };
  const panel = doc.createElement('section');
  panel.setAttribute('aria-label', 'Measure reconstructed geometry');
  panel.style.cssText = 'position:fixed;left:12px;bottom:12px;z-index:40;background:#fff;color:#171717;padding:10px;border:1px solid #aaa;border-radius:6px;font:13px system-ui;max-width:330px;max-height:45vh;overflow:auto';
  const toggle = doc.createElement('button'), clearButton = doc.createElement('button');
  const note = doc.createElement('p'), help = doc.createElement('p'), list = doc.createElement('ol');
  toggle.type = clearButton.type = 'button'; clearButton.textContent = 'Clear';
  note.textContent = ACCURACY_NOTE; help.textContent = 'Click two surfaces. Esc cancels the current pair.';
  panel.append(toggle, clearButton, help, note, list);
  if (ctx.roomDimensions) {
    const comparison = doc.createElement('p');
    comparison.textContent = `Matterport room dimensions (separate source): ${typeof ctx.roomDimensions === 'string' ? ctx.roomDimensions : JSON.stringify(ctx.roomDimensions)}`;
    panel.append(comparison);
  }
  (ctx.parent ?? doc.body).append(panel);
  function on(target, name, fn, options) { target.addEventListener(name, fn, options); listeners.push(() => target.removeEventListener(name, fn, options)); }
  function dot(point, color = 0x222222) {
    const geometry = new T.BufferGeometry().setFromPoints([new T.Vector3()]);
    const material = new T.PointsMaterial({ color: 0xffd400, size: 9, sizeAttenuation: false, depthTest: false });
    const mesh = new T.Points(geometry, material); mesh.position.copy(point); overlay.add(mesh);
    disposables.push(geometry, material); return mesh;
  }
  function cancel() { if (state.pending) overlay.remove(state.pending.marker); state.pending = null; help.textContent = 'Click two surfaces. Esc cancels the current pair.'; }
  function setEnabled(value) {
    state.enabled = !!value; toggle.textContent = value ? 'Measure: on' : 'Measure: off';
    toggle.setAttribute('aria-pressed', String(state.enabled)); if (!value) cancel();
    if (value) win.dispatchEvent(new (win.CustomEvent ?? CustomEvent)('viewer-tool', { detail: 'measure' }));
    return state.enabled;
  }
  function clear() {
    cancel(); overlay.clear(); measurements.splice(0); list.replaceChildren();
    disposables.splice(0).forEach(x => x.dispose());
  }
  function pick(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    ndc.set((clientX - rect.left) / rect.width * 2 - 1, 1 - (clientY - rect.top) / rect.height * 2);
    camera.updateMatrixWorld(true); ray.setFromCamera(ndc, camera); ray.near = .001; ray.far = camera.far;
    const meshes = ctx.getDepthMeshes() ?? [];
    meshes.forEach(mesh => mesh.updateMatrixWorld(true)); collision.updateMatrixWorld(true);
    // Ignore surfaces within nearIgnore m of the eye (camera capsule / near proxy faces).
    const near = ctx.nearIgnore ?? .35;
    const coarse = ray.intersectObject(collision, true).filter(hit => hit.distance > near);
    const depth = ray.intersectObjects(meshes, true).filter(hit => hit.distance > near && !hit.object.userData?.measurementIgnore);
    // Reject synthetic zero-range sky closure and geometry behind the first wall.
    const hit = depth.find(h => (!ctx.acceptDepthHit || ctx.acceptDepthHit(h)) &&
      h.distance < Math.min(camera.far * .85, (coarse[0]?.distance ?? Infinity) + .08)) ?? coarse[0];
    if (!hit) { state.error = 'No reconstructed surface at this point'; help.textContent = state.error; return null; }
    state.lastSource = depth.includes(hit) ? 'photo-depth' : 'collision.obj'; state.error = null;
    return { ...snapMeasurementPoint(T, hit, coarse, ctx.floorAt), source: state.lastSource };
  }
  function addPoint(value) {
    const point = value.point ?? value;
    if (!state.pending) {
      state.pending = { point: point.clone(), marker: dot(point), snap: value.snap, source: value.source };
      help.textContent = 'Choose the second point. Esc cancels.'; return null;
    }
    const first = state.pending, distance = first.point.distanceTo(point);
    if (distance < .001) return null;
    dot(point);
    const geometry = new T.BufferGeometry().setFromPoints([first.point, point]);
    const material = new T.LineBasicMaterial({ color: 0x222222, depthTest: false });
    const line = new T.Line(geometry, material); overlay.add(line); disposables.push(geometry, material);
    const label = doc.createElement('span'); label.textContent = formatDistance(distance);
    label.style.cssText = 'position:fixed;pointer-events:none;background:white;color:black;padding:2px 5px;border:1px solid #aaa;z-index:39;font:12px system-ui';
    (ctx.parent ?? doc.body).append(label);
    const row = doc.createElement('li'); row.textContent = label.textContent; list.append(row);
    const measurement = { a: first.point.toArray(), b: point.toArray(), metres: distance,
      text: label.textContent, sources: [first.source, value.source], snaps: [first.snap, value.snap], label };
    measurements.push(measurement); state.pending = null; help.textContent = 'Click two more surfaces to add another measurement.';
    return measurement;
  }
  // Labels live outside the renderer and must also be removed by Clear/dispose.
  const clearAll = () => { measurements.forEach(m => m.label.remove()); clear(); };
  let down;
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel']) on(canvas, type, e => {
    if (!state.enabled || (e.button !== undefined && e.button !== 0)) return;
    e.preventDefault(); e.stopImmediatePropagation();
    if (type === 'pointerdown') down = { id: e.pointerId, x: e.clientX, y: e.clientY };
    if (type === 'pointerup' && down?.id === e.pointerId) {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6) { const hit = pick(e.clientX, e.clientY); if (hit) addPoint(hit); }
      down = null;
    }
    if (type === 'pointercancel') down = null;
  }, true);
  on(win, 'keydown', e => {
    if (e.key === 'Escape' && state.enabled) { e.stopImmediatePropagation(); cancel(); }
  }, true);
  on(win, 'viewer-tool', e => { if (e.detail !== 'measure') setEnabled(false); });
  on(toggle, 'click', () => setEnabled(!state.enabled)); on(clearButton, 'click', clearAll);
  const api = { state, setEnabled, pick, addPoint, clear: clearAll, cancel,
    render() {
      if (!measurements.length && !overlay.children.length) return;
      const rect = canvas.getBoundingClientRect();
      for (const item of measurements) {
        const p = new T.Vector3().fromArray(item.a).add(new T.Vector3().fromArray(item.b)).multiplyScalar(.5).project(camera);
        item.label.hidden = p.z < -1 || p.z > 1 || Math.abs(p.x) > 1 || Math.abs(p.y) > 1;
        item.label.style.left = `${rect.left + (p.x + 1) * rect.width / 2}px`;
        item.label.style.top = `${rect.top + (1 - p.y) * rect.height / 2}px`;
      }
      if (overlay.children.length) {
        const auto = renderer.autoClear; renderer.autoClear = false;
        try { renderer.render(overlay, camera); } finally { renderer.autoClear = auto; }
      }
    },
    dispose() { clearAll(); listeners.forEach(off => off()); panel.remove(); if (win.demoReview?.measure === api) delete win.demoReview.measure; }
  };
  win.demoReview ??= {}; win.demoReview.measure = api;
  setEnabled(new URL(ctx.url ?? win.location.href).searchParams.get('mode') === 'measure');
  return api;
}
