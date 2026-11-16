/**
 * W14 Phase B extension; all coordinates are metric Z-up. No global imports.
 * Required ctx: THREE, scene, camera, renderer, collision, walkMesh,
 * splat (loaded main SplatMesh), SplatMesh and PackedSplats constructors;
 * getPhotoEntries(): iterable of {capture:{id}, meshScene, fullMesh?:{meshScene}};
 * suspendWalk(boolean): clears input and pauses walk physics/look while true;
 * setCameraPose(position:Vector3,target:Vector3): sets camera AND walk yaw/pitch.
 * Optional ctx: obstacleGrid/contactGrid (the live W13 Maps), document, window,
 * baseUrl ('./'), fetch, loadTexture(url), parent, storage, autoFrame (default true).
 * Hook (after photoLayer and W13 grids exist):
 *   import { installFurniture } from './furniture_edit.js';
 *   const furniture = installFurniture({THREE,scene,camera,renderer,collision,walkMesh,splat,SplatMesh,PackedSplats,getPhotoEntries:()=>photoLayer.entries(),contactGrid,obstacleGrid,suspendWalk,setCameraPose});
 *   furniture.render(); // after photoLayer.render(camera); entries() exposes cache.values().
 * See W19 HOOKS.md for the camera/input adapter and exact render order.
 */
export function decodeVoxels(object) {
  const { origin, shape, runs, encoding } = object.voxels;
  if (encoding !== 'rle_xyz_z_fastest') throw Error(`Unsupported voxel encoding: ${encoding}`);
  const out = [];
  for (let r = 0; r < runs.length; r += 2) for (let n = runs[r]; n < runs[r] + runs[r + 1]; n++)
    out.push([origin[0] + Math.floor(n / (shape[1] * shape[2])), origin[1] + Math.floor(n / shape[2]) % shape[1], origin[2] + n % shape[2]]);
  return out;
}

export function createVoxelLookup(data, dilate = 1) {
  // Exact 48-bit integer keys avoid millions of temporary string allocations.
  const map = new Map(), key = (x, y, z) => (x + 32768) * 4294967296 + (y + 32768) * 65536 + z + 32768;
  // Exact ownership wins over a neighbouring object's dilation.
  const decoded = data.objects.map(o => [o.id, decodeVoxels(o)]);
  for (const [id, cells] of decoded) for (const p of cells) map.set(key(...p), id);
  if (dilate) for (const [id, cells] of decoded) for (const p of cells)
    for (let x = -dilate; x <= dilate; x++) for (let y = -dilate; y <= dilate; y++) for (let z = -dilate; z <= dilate; z++) {
      const k = key(p[0] + x, p[1] + y, p[2] + z); if (!map.has(k)) map.set(k, id);
    }
  const size = data.voxel_size, origin = data.voxel_origin ?? [0, 0, 0];
  return point => {
    const x = Math.floor(((point.x ?? point[0]) - origin[0]) / size), y = Math.floor(((point.y ?? point[1]) - origin[1]) / size), z = Math.floor(((point.z ?? point[2]) - origin[2]) / size);
    if (x < -32768 || x > 32767 || y < -32768 || y > 32767 || z < -32768 || z > 32767) return 0;
    return map.get(key(x, y, z)) ?? 0;
  };
}

export function footprint(object) {
  const { centre: p, size, yaw } = object.obb, c = Math.cos(yaw), s = Math.sin(yaw);
  return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) =>
    [p[0] + x * size[0] / 2 * c - y * size[1] / 2 * s, p[1] + x * size[0] / 2 * s + y * size[1] / 2 * c]);
}

function polygonsOverlap(a, b, clearance = .001) {
  for (const poly of [a, b]) for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length], dx = q[0] - p[0], dy = q[1] - p[1];
    const length = Math.hypot(dx, dy); if (length < 1e-8) continue;
    const project = v => (-dy * v[0] + dx * v[1]) / length;
    const av = a.map(project), bv = b.map(project);
    if (Math.max(...av) <= Math.min(...bv) + clearance || Math.max(...bv) <= Math.min(...av) + clearance) return false;
  }
  return true;
}

export function obbOverlap(a, b) {
  if (a.hidden || b.hidden) return false;
  if (Math.abs(a.obb.centre[2] - b.obb.centre[2]) >= (a.obb.size[2] + b.obb.size[2]) / 2 - .001) return false;
  return polygonsOverlap(footprint(a), footprint(b));
}

export function placementValidity(object, objects, { floorAt, walls = [] } = {}) {
  if (objects.some(other => other.id !== object.id && obbOverlap(object, other))) return { valid: false, reason: 'Chạm đồ khác' };
  const corners = footprint(object), floor = object.floor_height;
  if (!floorAt) return { valid: false, reason: 'Chưa có dữ liệu sàn' };
  // Sample the whole footprint, not only its centre: catches narrow ledges/holes.
  const nx = Math.max(1, Math.ceil(object.obb.size[0] / .1)), ny = Math.max(1, Math.ceil(object.obb.size[1] / .1));
  for (let x = 0; x <= nx; x++) for (let y = 0; y <= ny; y++) {
    const p = [corners[0][0] + (corners[1][0] - corners[0][0]) * x / nx + (corners[3][0] - corners[0][0]) * y / ny,
      corners[0][1] + (corners[1][1] - corners[0][1]) * x / nx + (corners[3][1] - corners[0][1]) * y / ny, floor];
    const z = floorAt(p, object.storey);
    if (!Number.isFinite(z) || Math.abs(z - floor) > .12) return { valid: false, reason: 'Ra khỏi sàn tầng này' };
  }
  const low = object.obb.centre[2] - object.obb.size[2] / 2, high = low + object.obb.size[2];
  for (const wall of walls) {
    if (wall.owner && objects.some(o => (o.sourceId ?? o.id) === wall.owner) || high <= wall.low + .01 || low >= wall.high - .01) continue;
    // A thin rectangle around the real triangle's longest XY edge permits SAT
    // against both axis-aligned and oblique vertical walls, with finite extents.
    if (polygonsOverlap(corners, wall.polygon, 0)) return { valid: false, reason: 'Chạm tường' };
  }
  return { valid: true, reason: '' };
}

export function backprojectSample(capture, face, x, y, range, size = 256) {
  const [centre, right, down] = capture.face_bases_center_right_down[face];
  const u = 2 * (x + .5) / size - 1, v = 2 * (y + .5) / size - 1;
  const d = centre.map((c, i) => c + u * right[i] + v * down[i]), scale = range / Math.hypot(...d);
  return d.map((c, i) => capture.position[i] + c * scale);
}

/** Real mask/depth shell used by duplicates; no thumbnail planes or invented back. */
export function buildObjectShell(THREE, capture, face, depth, mask, id, size = 256) {
  if (depth.byteLength !== size * size * 2 || mask.length !== size * size) throw Error('Mask/depth size mismatch');
  const view = new DataView(depth.buffer ?? depth, depth.byteOffset ?? 0, depth.byteLength);
  const positions = [], uvs = [], indices = [], vertex = new Int32Array(size * size).fill(-1);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const p = y * size + x, mm = view.getUint16(p * 2, true);
    if (mask[p] !== id || !mm) continue;
    vertex[p] = positions.length / 3; positions.push(...backprojectSample(capture, face, x, y, mm / 1000, size));
    uvs.push((x + .5) / size, 1 - (y + .5) / size);
  }
  function tri(a, b, c) {
    if ([a, b, c].some(i => vertex[i] < 0)) return;
    const r = [a, b, c].map(i => view.getUint16(i * 2, true));
    if (Math.max(...r) - Math.min(...r) <= .08 * Math.min(...r)) indices.push(vertex[a], vertex[c], vertex[b]);
  }
  for (let y = 0; y < size - 1; y++) for (let x = 0; x < size - 1; x++) {
    const a = y * size + x; tri(a, a + 1, a + size); tri(a + 1, a + size + 1, a + size);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2)); geometry.setIndex(indices);
  return geometry;
}

export function installFurniture(ctx) {
  const { THREE: T, scene, camera, renderer, collision, walkMesh } = ctx;
  for (const name of ['THREE', 'scene', 'camera', 'renderer', 'collision', 'walkMesh', 'splat', 'SplatMesh', 'PackedSplats', 'getPhotoEntries', 'suspendWalk', 'setCameraPose'])
    if (!ctx[name]) throw Error(`installFurniture requires ctx.${name}`);
  const win = ctx.window ?? globalThis.window, doc = ctx.document ?? globalThis.document;
  const fetcher = ctx.fetch ?? globalThis.fetch, base = new URL(ctx.baseUrl ?? './', win.location.href);
  const url = path => new URL(path, base).href;
  const storage = () => ctx.storage ?? win.localStorage;
  const listeners = [], resources = [], objects = [], original = new Map(), groups = new Map(), shells = new Map();
  const patched = new Map(), meshRecords = new Map(), maskCache = new Map(), collisionRecords = [], gridRecords = [];
  const root = new T.Group(), overlayScene = new T.Scene(); scene.add(root);
  const ray = new T.Raycaster(), ndc = new T.Vector2(), floorRay = new T.Raycaster();
  let heavyActivated = false, heavyActivating = null;
  const state = { objects, selected: null, mode: false, valid: true, reason: '', ready: false, heavyActivated: false,
    error: null, splatAssignmentMs: null, assignedSplats: 0, photoMeshes: 0, maskFaces: 0, cameraOwned: false };
  win.furnitureState = state;
  const transforms = new Float32Array(5 * 1024 * 4);
  const texture = new T.DataTexture(transforms, 5, 1024, T.RGBAFormat, T.FloatType); texture.needsUpdate = true; resources.push(texture);
  const identity = new T.Matrix4();
  for (let i = 0; i < 1024; i++) { transforms.set(identity.elements, i * 20); transforms.set([1, 1, 0, 0], i * 20 + 16); }
  let lookup, walls = [], data, captures, savedCamera, tween, disposed = false, frameId, drag, rotationTimer, mainAlpha;
  let lastSelection, snapshot, hovered, nextId = 256;
  const panel = doc.createElement('section'), title = doc.createElement('strong'), edit = doc.createElement('button');
  const restore = doc.createElement('button'), rows = doc.createElement('div'), message = doc.createElement('p');
  panel.style.cssText = 'position:fixed;right:12px;top:100px;z-index:42;background:#fff;color:#181818;border:1px solid #aaa;border-radius:6px;padding:10px;font:13px system-ui;width:245px;max-height:65vh;overflow:auto';
  title.textContent = 'Đồ trong nhà'; edit.textContent = 'Sửa nội thất'; restore.textContent = 'Khôi phục tất cả';
  panel.append(title, edit, restore, message, rows); (ctx.parent ?? doc.body).append(panel);
  const toolbar = doc.createElement('div');
  toolbar.style.cssText = 'position:fixed;z-index:44;background:white;border:1px solid #aaa;padding:5px;border-radius:5px;white-space:nowrap';
  toolbar.hidden = true; (ctx.parent ?? doc.body).append(toolbar);
  const buttons = {};
  function on(target, name, fn, options) { target.addEventListener(name, fn, options); listeners.push(() => target.removeEventListener(name, fn, options)); }
  function button(label, fn, parent = toolbar) {
    const b = doc.createElement('button'); b.type = 'button'; b.textContent = label; on(b, 'click', fn); parent.append(b); return b;
  }
  const footprintGeometry = new T.BufferGeometry(), footprintMaterial = new T.LineBasicMaterial({ color: 0x16803c, depthTest: false });
  const outline = new T.LineLoop(footprintGeometry, footprintMaterial); outline.visible = false; overlayScene.add(outline); resources.push(footprintGeometry, footprintMaterial);
  function cloneLayout(o) { return { obb: { centre: [...o.obb.centre], size: [...o.obb.size], yaw: o.obb.yaw }, hidden: !!o.hidden }; }
  function loadJSON(path) { return fetcher(url(path)).then(r => { if (!r.ok) throw Error(`${path}: HTTP ${r.status}`); return r.json(); }); }
  function loadBinary(path) { return fetcher(url(path)).then(r => { if (!r.ok) throw Error(`${path}: HTTP ${r.status}`); return r.arrayBuffer(); }); }
  function mask(id, face) {
    const key = `${id}/${face}`;
    if (!maskCache.has(key)) maskCache.set(key, loadBinary(`objmask/${encodeURIComponent(id)}/face${face}.u8`).then(b => {
      if (b.byteLength !== 65536) throw Error('Invalid objmask byte length'); state.maskFaces++; return new Uint8Array(b);
    }));
    return maskCache.get(key);
  }
  function save() {
    try { storage().setItem(`furniture-layout:${data.model_id}`, JSON.stringify({ version: 1,
      objects: objects.map(o => ({ id: o.id, sourceId: o.sourceId, ...cloneLayout(o) })) })); } catch { /* Storage can be disabled. */ }
  }
  function matrixFor(o, lifted = false) {
    const source = original.get(o.sourceId ?? o.id), m = new T.Matrix4().makeTranslation(...o.obb.centre);
    if (lifted) m.elements[14] += .03;
    return m.multiply(new T.Matrix4().makeRotationZ(o.obb.yaw - source.obb.yaw))
      .multiply(new T.Matrix4().makeTranslation(...source.obb.centre.map(v => -v)));
  }
  function floorAt(p) {
    floorRay.set(new T.Vector3(p[0], p[1], p[2] + .15), new T.Vector3(0, 0, -1)); floorRay.near = 0; floorRay.far = .3;
    const hit = floorRay.intersectObject(walkMesh, true).find(h => h.face &&
      Math.abs(h.face.normal.clone().transformDirection(h.object.matrixWorld).z) > .65);
    return hit?.point.z;
  }
  function validate(o) { return placementValidity(o, objects, { floorAt, walls }); }
  function refresh(o) {
    if (o.id >= 1024) throw Error('Furniture limit reached');
    const selected = state.selected === o.id, m = matrixFor(o, selected);
    transforms.set(m.elements, o.id * 20);
    transforms.set([o.hidden ? 0 : 1, selected ? (state.valid ? .6 : .45) : 1, selected && !state.valid ? 1 : 0, hovered === o.id ? .18 : 0], o.id * 20 + 16);
    texture.needsUpdate = true;
    const group = groups.get(o.id);
    if (group) {
      group.matrix.copy(m); group.matrixAutoUpdate = false; group.visible = !o.hidden;
      group.traverse(child => {
        if (child.isSplatMesh || child.recolor) { child.opacity = selected ? (state.valid ? .6 : .45) : 1; child.recolor?.set(selected && !state.valid ? 0xff3333 : 0xffffff); }
        if (child.material?.isMeshBasicMaterial) { child.material.opacity = selected ? (state.valid ? .6 : .45) : 1; child.material.color.set(selected && !state.valid ? 0xff3333 : 0xffffff); }
      });
    }
    if (selected) {
      outline.geometry.setFromPoints(footprint(o).map(([x, y]) => new T.Vector3(x, y, o.floor_height + .025)));
      outline.material.color.set(state.valid ? 0x16803c : 0xdd2222); outline.visible = !o.hidden;
      message.textContent = state.reason || 'Kéo để di chuyển · lưới 5 cm'; buttons.confirm.disabled = !state.valid;
    }
  }
  function updateSelected() {
    const o = objects.find(o => o.id === state.selected); if (!o) return;
    const check = validate(o); Object.assign(state, check); refresh(o);
  }
  function patchMaterial(material) {
    if (!material?.isShaderMaterial || patched.has(material) || !/vec4\(position,\s*1\.0\)/.test(material.vertexShader)) return;
    patched.set(material, { vertexShader: material.vertexShader, fragmentShader: material.fragmentShader });
    material.uniforms.furnitureState = { value: texture };
    material.vertexShader = `attribute float furnitureId; uniform sampler2D furnitureState; varying vec4 furnitureStyle;
      vec3 furniturePosition(vec3 p) { if(furnitureId < .5) { furnitureStyle=vec4(1.,1.,0.,0.); return p; } float y=(furnitureId+.5)/1024.;
        mat4 m=mat4(texture2D(furnitureState,vec2(.1,y)),texture2D(furnitureState,vec2(.3,y)),texture2D(furnitureState,vec2(.5,y)),texture2D(furnitureState,vec2(.7,y)));
        furnitureStyle=texture2D(furnitureState,vec2(.9,y)); return (m*vec4(p,1.)).xyz; }\n` +
      material.vertexShader.replace(/vec4\(position,\s*1\.0\)/g, 'vec4(furniturePosition(position), 1.0)');
    material.fragmentShader = 'varying vec4 furnitureStyle;\n' + material.fragmentShader
      .replace(/void main\(\)\s*\{/, 'void main() { if(furnitureStyle.x < .5) discard;')
      .replace(/gl_FragColor\s*=\s*([^;]+);/, '$& gl_FragColor.rgb=mix(gl_FragColor.rgb,vec3(1.,.08,.08),furnitureStyle.z*.75); gl_FragColor.rgb=mix(gl_FragColor.rgb,vec3(1.),furnitureStyle.w); gl_FragColor.a*=furnitureStyle.y;');
    material.needsUpdate = true;
  }
  function attachPhotoMesh(mesh, capture) {
    if (!mesh.isMesh || meshRecords.has(mesh)) return;
    const previous = mesh.geometry;
    if (!previous.attributes.position || !previous.attributes.uv || !previous.attributes.photoFace) return;
    // One transform per triangle avoids stretched bridges at object boundaries.
    const geometry = previous.index ? previous.toNonIndexed() : previous.clone();
    mesh.geometry = geometry;
    const pos = geometry.attributes.position, uv = geometry.attributes.uv, faces = geometry.attributes.photoFace;
    if (!pos || !uv || !faces) return;
    const ids = new Float32Array(pos.count), attr = new T.BufferAttribute(ids, 1);
    geometry.setAttribute('furnitureId', attr);
    const prior = mesh.onBeforeRender;
    const pickGeometry = geometry.clone(), pickMesh = new T.Mesh(pickGeometry, mesh.material);
    const record = { mesh, prior, previous, geometry, pickGeometry, oldRaycast: mesh.raycast, oldCull: mesh.frustumCulled };
    meshRecords.set(mesh, record); mesh.frustumCulled = false;
    mesh.onBeforeRender = function (...args) { prior?.apply(this, args); patchMaterial(args[4]); };
    // Shader transforms do not affect THREE's CPU raycaster. Mirror them only
    // during a pick so measurement sees placed objects and ignores hidden ones.
    mesh.raycast = function (raycaster, intersections) {
      const target = pickGeometry.attributes.position, p = new T.Vector3(), m = new T.Matrix4();
      for (let i = 0; i < pos.count; i += 3) {
        const id = ids[i], visible = transforms[id * 20 + 16] > .5;
        m.fromArray(transforms, id * 20);
        for (let j = i; j < Math.min(i + 3, pos.count); j++) {
          p.fromBufferAttribute(pos, j).applyMatrix4(m); if (!visible) p.set(0, 0, -100000);
          target.setXYZ(j, p.x, p.y, p.z);
        }
      }
      pickGeometry.computeBoundingSphere(); pickGeometry.computeBoundingBox();
      pickMesh.matrixWorld.copy(mesh.matrixWorld); const hits = [];
      T.Mesh.prototype.raycast.call(pickMesh, raycaster, hits);
      for (const hit of hits) { hit.object = mesh; intersections.push(hit); }
    };
    patchMaterial(mesh.material); state.photoMeshes++;
    const world = new T.Vector3();
    const samples = new Float32Array(pos.count);
    function triangleIds() {
      for (let i = 0; i < samples.length; i += 3) {
        const a = samples[i], b = samples[i + 1], c = samples[i + 2];
        const id = a === b || a === c ? a : b === c ? b : 0;
        ids.fill(id, i, Math.min(i + 3, ids.length));
      }
      attr.needsUpdate = true;
    }
    for (let i = 0; i < pos.count; i++) samples[i] = lookup(world.fromBufferAttribute(pos, i));
    triangleIds();
    // Exact 1:1 Phase A labels replace the voxel fallback as each mask arrives.
    for (let face = 0; face < 6; face++) mask(capture.id, face).then(values => {
      if (disposed || !meshRecords.has(mesh)) return;
      for (let i = 0; i < pos.count; i++) if (Math.round(faces.getX(i)) === face)
        samples[i] = values[Math.max(0, Math.min(255, Math.floor(uv.getY(i) * 256))) * 256 + Math.max(0, Math.min(255, Math.floor(uv.getX(i) * 256)))];
      triangleIds();
    }).catch(e => { state.error = e.message; });
  }
  function syncPhotoMeshes() {
    for (const entry of ctx.getPhotoEntries() ?? []) for (const target of [entry, entry.fullMesh])
      target?.meshScene?.traverse(mesh => attachPhotoMesh(mesh, entry.capture));
    for (const [mesh, rec] of meshRecords) if (!mesh.parent) {
      mesh.onBeforeRender = rec.prior; mesh.frustumCulled = rec.oldCull; mesh.raycast = rec.oldRaycast;
      mesh.geometry = rec.previous; rec.geometry.dispose(); rec.pickGeometry.dispose(); meshRecords.delete(mesh);
    }
  }
  function collectColliders() {
    collision.updateMatrixWorld(true); walkMesh.updateMatrixWorld(true);
    collision.traverse(mesh => {
      if (!mesh.isMesh) return;
      const previous = mesh.geometry, geometry = previous.index ? previous.toNonIndexed() : previous.clone();
      mesh.geometry = geometry; resources.push(geometry);
      const pos = geometry.attributes.position, basePositions = pos.array.slice(), owners = new Uint16Array(pos.count / 3);
      const inverse = mesh.matrixWorld.clone().invert();
      for (let i = 0; i < pos.count; i += 3) {
        const points = [0, 1, 2].map(j => new T.Vector3().fromBufferAttribute(pos, i + j).applyMatrix4(mesh.matrixWorld));
        const centre = points[0].clone().add(points[1]).add(points[2]).multiplyScalar(1 / 3);
        const owner = lookup(centre); owners[i / 3] = owner;
        const normal = points[1].clone().sub(points[0]).cross(points[2].clone().sub(points[0])).normalize();
        if (Math.abs(normal.z) > .3) continue;
        const pairs = [[points[0], points[1]], [points[1], points[2]], [points[2], points[0]]].sort((a, b) => b[0].clone().sub(b[1]).setZ(0).lengthSq() - a[0].clone().sub(a[1]).setZ(0).lengthSq());
        const [a, b] = pairs[0], length = Math.hypot(b.x - a.x, b.y - a.y); if (length < .03) continue;
        const dx = -(b.y - a.y) / length * .006, dy = (b.x - a.x) / length * .006;
        walls.push({ owner, low: Math.min(...points.map(p => p.z)), high: Math.max(...points.map(p => p.z)), polygon: [[a.x + dx, a.y + dy], [b.x + dx, b.y + dy], [b.x - dx, b.y - dy], [a.x - dx, a.y - dy]] });
      }
      collisionRecords.push({ mesh, previous, pos, basePositions, owners, inverse });
    });
    for (const [kind, grid] of [['contact', ctx.contactGrid], ['obstacle', ctx.obstacleGrid]]) if (grid) {
      const seen = new Map();
      for (const [key, items] of grid) for (const item of items) {
        if (seen.has(item)) { seen.get(item)?.keys.push(key); continue; }
        let p;
        if (kind === 'contact' && item.points) p = item.points.reduce((a, p) => a.add(p), new T.Vector3()).multiplyScalar(1 / item.points.length);
        else if (kind === 'obstacle') { const [x, y] = key.split(',').map(Number); p = new T.Vector3((x + .5) * .1, (y + .5) * .1, (item.low + item.high) / 2); }
        const owner = p ? lookup(p) : 0;
        const record = owner ? { kind, grid, item, owner, key, keys: [key], point: p, points: item.points?.map(p => p.clone()) } : null;
        seen.set(item, record); if (record) gridRecords.push(record);
      }
    }
  }
  function updateColliders() {
    const byId = new Map(objects.map(o => [o.id, o]));
    for (const rec of collisionRecords) {
      const { pos, basePositions, owners, mesh, inverse } = rec;
      for (let i = 0; i < owners.length; i++) {
        const o = byId.get(owners[i]); if (!o) continue;
        const m = matrixFor(o), p = new T.Vector3();
        for (let j = 0; j < 3; j++) {
          const offset = (i * 3 + j) * 3;
          p.fromArray(basePositions, offset).applyMatrix4(mesh.matrixWorld).applyMatrix4(m).applyMatrix4(inverse);
          if (o.hidden) p.set(0, 0, -100000); pos.setXYZ(i * 3 + j, p.x, p.y, p.z);
        }
      }
      pos.needsUpdate = true; mesh.geometry.computeBoundingSphere(); mesh.geometry.computeBoundingBox();
    }
    // Remove only tagged original furniture contacts; preserve all structural
    // contacts and floor-boundary contours from W13.
    const tagged = new Set(gridRecords.map(r => r.item));
    for (const grid of [ctx.contactGrid, ctx.obstacleGrid]) if (grid) for (const [key, items] of grid) {
      const keep = items.filter(i => !tagged.has(i) && !i.furnitureW19); if (keep.length) grid.set(key, keep); else grid.delete(key);
    }
    const insert = (grid, key, item) => { if (!grid.has(key)) grid.set(key, []); grid.get(key).push(item); };
    for (const o of objects) {
      if (o.hidden) continue;
      const sourceId = o.sourceId ?? o.id, m = matrixFor(o);
      for (const rec of gridRecords.filter(r => r.owner === sourceId)) {
        const base = original.get(sourceId);
        if (!o.sourceId && o.obb.yaw === base.obb.yaw && o.obb.centre.every((v, i) => v === base.obb.centre[i])) {
          for (const key of rec.keys) insert(rec.grid, key, rec.item);
          continue;
        }
        if (rec.kind === 'contact') {
          const points = rec.points.map(p => p.clone().applyMatrix4(m)), xs = points.map(p => p.x), ys = points.map(p => p.y);
          const item = { ...rec.item, furnitureW19: true, points, minZ: Math.min(...points.map(p => p.z)), maxZ: Math.max(...points.map(p => p.z)),
            minX: Math.min(...xs) - .25, maxX: Math.max(...xs) + .25, minY: Math.min(...ys) - .25, maxY: Math.max(...ys) + .25 };
          for (let x = Math.floor(item.minX); x <= Math.floor(item.maxX); x++) for (let y = Math.floor(item.minY); y <= Math.floor(item.maxY); y++) insert(rec.grid, `${x},${y}`, item);
        } else {
          const p = rec.point.clone().applyMatrix4(m), dz = p.z - rec.point.z;
          insert(rec.grid, `${Math.floor(p.x / .1)},${Math.floor(p.y / .1)}`, { ...rec.item, furnitureW19: true, low: rec.item.low + dz, high: rec.item.high + dz, floor: rec.item.floor + dz });
        }
      }
    }
    // Copies may have no original W13 contact; add conservative OBB sides.
    if (ctx.contactGrid) for (const o of objects.filter(o => !o.hidden && (o.sourceId ||
      o.obb.yaw !== original.get(o.id).obb.yaw || o.obb.centre.some((v, i) => v !== original.get(o.id).obb.centre[i])))) {
      const corners = footprint(o), low = o.obb.centre[2] - o.obb.size[2] / 2;
      for (let i = 0; i < 4; i++) {
        const a = corners[i], b = corners[(i + 1) % 4], points = [new T.Vector3(...a, low), new T.Vector3(...b, low), new T.Vector3(...b, low + o.obb.size[2])];
        const item = { points, minZ: low, maxZ: low + o.obb.size[2], coarse: true, furnitureW19: true,
          minX: Math.min(a[0], b[0]) - .25, maxX: Math.max(a[0], b[0]) + .25, minY: Math.min(a[1], b[1]) - .25, maxY: Math.max(a[1], b[1]) + .25 };
        for (let x = Math.floor(item.minX); x <= Math.floor(item.maxX); x++) for (let y = Math.floor(item.minY); y <= Math.floor(item.maxY); y++) insert(ctx.contactGrid, `${x},${y}`, item);
      }
    }
  }
  async function assignSplats() {
    await ctx.splat.initialized; if (disposed) return;
    const start = performance.now(), bins = new Map(), packedMain = ctx.splat.packedSplats;
    if (!packedMain?.packedArray || !Number.isInteger(packedMain.numSplats)) throw Error('Furniture requires resident PackedSplats');
    mainAlpha = new Uint8Array(packedMain.numSplats);
    ctx.splat.forEachSplat((index, centre, scales, quaternion, opacity, color) => {
      const id = lookup(centre); if (!id || opacity <= 0) return;
      let packed = bins.get(id); if (!packed) {
        packed = new ctx.PackedSplats({ splatEncoding: packedMain.splatEncoding }); bins.set(id, packed); resources.push(packed);
      }
      const n = packed.numSplats;
      packed.ensureSplats(n + 1).set(packedMain.packedArray.subarray(index * 4, index * 4 + 4), n * 4);
      for (const [key, level, stride] of [['sh1', 1, 2], ['sh2', 2, 4], ['sh3', 3, 4]]) if (packedMain.extra?.[key])
        packed.ensureSplatsSh(level, n + 1).set(packedMain.extra[key].subarray(index * stride, index * stride + stride), n * stride);
      packed.numSplats++; mainAlpha[index] = packedMain.packedArray[index * 4] >>> 24; state.assignedSplats++;
    });
    for (const [id, packed] of bins) {
      const mesh = new ctx.SplatMesh({ packedSplats: packed }); groups.get(id).add(mesh);
    }
    for (let i = 0; i < mainAlpha.length; i++) if (mainAlpha[i]) packedMain.packedArray[i * 4] &= 0x00ffffff;
    ctx.splat.packedSplats.needsUpdate = true; ctx.splat.generatorDirty = true;
    state.splatAssignmentMs = performance.now() - start;
  }
  async function sourceShell(id) {
    if (shells.has(id)) return shells.get(id);
    const promise = (async () => {
      const o = original.get(id), match = /captures2k\/([^/]+)\/face(\d)\.jpg$/.exec(o.photo.path);
      if (!match) throw Error('Object photo path is not a Phase A capture');
      const capture = captures.find(c => c.id === match[1]), face = Number(match[2]);
      const [depth, labels, map] = await Promise.all([loadBinary(`capdepth/${capture.id}/face${face}.u16`), mask(capture.id, face),
        ctx.loadTexture ? ctx.loadTexture(url(o.photo.path)) : new T.TextureLoader().loadAsync(url(o.photo.path))]);
      if (disposed) { map.dispose(); return null; }
      const geometry = buildObjectShell(T, capture, face, depth, labels, id);
      const material = new T.MeshBasicMaterial({ map, side: T.DoubleSide, transparent: true, depthWrite: true });
      resources.push(geometry, material, map); return new T.Mesh(geometry, material);
    })(); shells.set(id, promise); return promise;
  }
  function renderRows() {
    rows.replaceChildren();
    for (const o of objects) {
      const row = doc.createElement('div'), image = doc.createElement('img');
      row.style.cssText = 'display:flex;align-items:center;gap:5px;margin-top:6px'; image.src = url(o.thumbnail); image.width = 42; image.height = 34; image.alt = `Đồ ${o.id}`; image.loading = 'lazy';
      row.append(image); button(`Đồ ${o.id}`, () => select(o.id), row);
      const eye = button(o.hidden ? 'Hiện' : 'Ẩn', () => { hide(o.id, !o.hidden); }, row); eye.setAttribute('aria-pressed', String(!o.hidden)); rows.append(row);
    }
  }
  function pose(position, target) { ctx.setCameraPose(position, target); }
  function easeCamera(position, target) {
    tween = { start: performance.now(), from: camera.position.clone(), to: position.clone(), qFrom: camera.quaternion.clone(), target: target.clone() };
    const dummy = camera.clone(); dummy.position.copy(position); dummy.lookAt(target); tween.qTo = dummy.quaternion.clone();
  }
  function restoreCamera() { if (savedCamera) { easeCamera(savedCamera.position, savedCamera.target); savedCamera = null; } }
  async function ensureHeavyActivated() {
    if (heavyActivated) return;
    if (heavyActivating) return heavyActivating;
    heavyActivating = (async () => {
      collectColliders(); await assignSplats(); if (disposed) return;
      objects.forEach(refresh); syncPhotoMeshes(); updateColliders();
      heavyActivated = true; state.heavyActivated = true;
    })();
    return heavyActivating;
  }
  function deactivateHeavy() {
    if (!heavyActivated) return;
    if (state.selected !== null) cancel();
    if (state.mode) { state.mode = false; edit.setAttribute('aria-pressed', 'false'); edit.textContent = 'Sửa nội thất'; }
    for (const [mesh, rec] of meshRecords) {
      mesh.onBeforeRender = rec.prior; mesh.frustumCulled = rec.oldCull; mesh.raycast = rec.oldRaycast;
      mesh.geometry = rec.previous; rec.geometry.dispose(); rec.pickGeometry.dispose();
    }
    meshRecords.clear();
    for (const [material, source] of patched) { Object.assign(material, source); delete material.uniforms.furnitureState; material.needsUpdate = true; }
    patched.clear();
    for (const rec of collisionRecords) rec.mesh.geometry = rec.previous;
    collisionRecords.length = 0; walls.length = 0;
    for (const grid of [ctx.contactGrid, ctx.obstacleGrid]) if (grid) for (const [key, items] of grid) {
      const keep = items.filter(i => !i.furnitureW19); if (keep.length) grid.set(key, keep); else grid.delete(key);
    }
    for (const rec of gridRecords) for (const key of rec.keys) { if (!rec.grid.has(key)) rec.grid.set(key, []); if (!rec.grid.get(key).includes(rec.item)) rec.grid.get(key).push(rec.item); }
    gridRecords.length = 0;
    for (const g of groups.values()) for (const child of [...g.children]) if (child.isSplatMesh || child.packedSplats) g.remove(child);
    const main = ctx.splat.packedSplats;
    if (mainAlpha && main?.packedArray) {
      for (let i = 0; i < mainAlpha.length; i++) if (mainAlpha[i]) main.packedArray[i * 4] = (main.packedArray[i * 4] & 0x00ffffff) | (mainAlpha[i] << 24);
      main.needsUpdate = true; ctx.splat.generatorDirty = true;
    }
    mainAlpha = null; state.assignedSplats = 0; state.photoMeshes = 0;
    heavyActivated = false; heavyActivating = null; state.heavyActivated = false;
  }
  function setMode(value) {
    if (!state.ready && value) return false;
    if (value && !heavyActivated) { void ensureHeavyActivated().then(() => setMode(true)); return true; }
    if (!value && state.selected !== null) cancel();
    state.mode = !!value; edit.setAttribute('aria-pressed', String(state.mode));
    edit.textContent = state.mode ? 'Xong nội thất' : 'Sửa nội thất';
    if (value) win.dispatchEvent(new (win.CustomEvent ?? CustomEvent)('viewer-tool', { detail: 'furniture' }));
    return state.mode;
  }
  function select(id) {
    if (!state.ready) return false;
    if (!heavyActivated) { void ensureHeavyActivated().then(() => select(id)); return true; }
    const o = objects.find(o => o.id === id); if (!o || o.hidden) return false;
    if (state.selected !== null) cancel(false);
    setMode(true); snapshot = cloneLayout(o); lastSelection = id; state.selected = id;
    if (!savedCamera) savedCamera = { position: camera.position.clone(), target: camera.position.clone().add(camera.getWorldDirection(new T.Vector3())) };
    state.cameraOwned = true; ctx.suspendWalk(true);
    const p = new T.Vector3().fromArray(o.obb.centre), distance = Math.max(2.2, Math.max(...o.obb.size) * 1.4);
    const direction = camera.position.clone().sub(p); direction.z = 0;
    if (direction.lengthSq() < .001) direction.set(1, -1, 0); direction.normalize().multiplyScalar(distance);
    easeCamera(p.clone().add(direction).setZ(o.floor_height + 2.2), p);
    toolbar.hidden = false; updateSelected(); return true;
  }
  function finish(restoreView = true) {
    const o = objects.find(o => o.id === state.selected); state.selected = null; snapshot = null;
    toolbar.hidden = true; outline.visible = false; if (o) refresh(o);
    if (restoreView) restoreCamera(); else tween = null;
    updateColliders(); renderRows();
  }
  function confirm() {
    updateSelected(); if (!state.valid || state.selected === null) return false;
    objects.find(o => o.id === state.selected).isNew = false;
    finish(); save(); return true;
  }
  function cancel(restoreView = true) {
    const o = objects.find(o => o.id === state.selected);
    if (o && o.isNew) { root.remove(groups.get(o.id)); groups.delete(o.id); objects.splice(objects.indexOf(o), 1); }
    else if (o && snapshot) Object.assign(o, snapshot);
    finish(restoreView); state.valid = true; state.reason = ''; return true;
  }
  function hide(id, hidden = true) {
    const o = objects.find(o => o.id === id); if (!o) return false;
    if (!heavyActivated) { void ensureHeavyActivated().then(() => hide(id, hidden)); return true; }
    if (state.selected === id) { cancel(); }
    o.hidden = hidden; refresh(o); updateColliders(); save(); renderRows(); return true;
  }
  function moveTo(x, y) {
    const o = objects.find(o => o.id === state.selected); if (!o) return false;
    o.obb.centre[0] = Math.round(x / .05) * .05; o.obb.centre[1] = Math.round(y / .05) * .05;
    updateSelected(); return state.valid;
  }
  function rotate(direction = 1) {
    const o = objects.find(o => o.id === state.selected); if (!o) return;
    o.obb.yaw += Math.sign(direction) * Math.PI / 12; updateSelected();
  }
  async function duplicate(id = state.selected ?? lastSelection, persisted) {
    const source = objects.find(o => o.id === id); if (!source || nextId >= 1024) return null;
    if (state.selected !== null) cancel(false);
    const sourceId = source.sourceId ?? source.id, clone = { ...source, ...cloneLayout(source), id: persisted?.id ?? nextId++, sourceId, isNew: true, hidden: false };
    const group = new T.Group(); groups.set(clone.id, group); root.add(group);
    const sourceGroup = groups.get(sourceId);
    for (const child of sourceGroup.children) if (child.packedSplats) group.add(new ctx.SplatMesh({ packedSplats: child.packedSplats }));
    let shell;
    try { shell = await sourceShell(sourceId); }
    catch (e) { root.remove(group); groups.delete(clone.id); throw e; }
    if (disposed) return null;
    if (shell) { const mesh = shell.clone(); mesh.material = shell.material.clone(); resources.push(mesh.material); group.add(mesh); }
    objects.push(clone); nextId = Math.max(nextId, clone.id + 1);
    if (persisted) {
      Object.assign(clone, { obb: { ...clone.obb, centre: [...persisted.obb.centre], yaw: persisted.obb.yaw }, hidden: !!persisted.hidden, isNew: false });
      if (!clone.hidden && !validate(clone).valid) { root.remove(group); groups.delete(clone.id); objects.splice(objects.indexOf(clone), 1); return null; }
      refresh(clone);
    }
    else {
      select(clone.id); const forward = camera.getWorldDirection(new T.Vector3()); forward.z = 0; forward.normalize();
      moveTo(camera.position.x + forward.x * 1.5, camera.position.y + forward.y * 1.5);
    }
    renderRows(); return clone;
  }
  buttons.left = button('⟲', () => rotate(-1)); buttons.right = button('⟳', () => rotate(1));
  for (const [b, direction] of [[buttons.left, -1], [buttons.right, 1]]) {
    on(b, 'pointerdown', () => { clearInterval(rotationTimer); rotationTimer = setInterval(() => rotate(direction), 160); });
    for (const event of ['pointerup', 'pointerleave', 'pointercancel']) on(b, event, () => clearInterval(rotationTimer));
  }
  button('👁', () => hide(state.selected)); buttons.confirm = button('✓', confirm); button('✕', cancel); button('Nhân bản', () => duplicate().catch(e => { state.error = e.message; message.textContent = e.message; }));
  on(edit, 'click', () => setMode(!state.mode));
  on(restore, 'click', () => {
    if (state.selected !== null) cancel();
    for (const o of [...objects]) {
      if (o.sourceId) { root.remove(groups.get(o.id)); groups.delete(o.id); objects.splice(objects.indexOf(o), 1); }
      else { Object.assign(o, cloneLayout(original.get(o.id))); refresh(o); }
    }
    updateColliders(); save(); renderRows();
  });
  on(win, 'viewer-tool', e => { if (e.detail !== 'furniture') setMode(false); });
  on(win, 'keydown', e => {
    if (!state.mode || /INPUT|TEXTAREA|SELECT/.test(e.target?.tagName)) return;
    if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); cancel(); }
    if (/^(Key[WASD]|Arrow)/.test(e.code)) { e.preventDefault(); e.stopImmediatePropagation(); }
  }, true);
  function eventRay(e) {
    const rect = renderer.domElement.getBoundingClientRect(); ndc.set((e.clientX - rect.left) / rect.width * 2 - 1, 1 - (e.clientY - rect.top) / rect.height * 2);
    camera.updateMatrixWorld(true); ray.setFromCamera(ndc, camera);
  }
  function pickObject(e) {
    eventRay(e); let best, nearest = Infinity;
    for (const o of objects.filter(o => !o.hidden)) {
      const m = new T.Matrix4().makeTranslation(...o.obb.centre).multiply(new T.Matrix4().makeRotationZ(o.obb.yaw));
      const localRay = ray.ray.clone().applyMatrix4(m.clone().invert());
      const half = new T.Vector3().fromArray(o.obb.size).multiplyScalar(.5), hit = localRay.intersectBox(new T.Box3(half.clone().negate(), half), new T.Vector3());
      if (!hit) continue; hit.applyMatrix4(m); const d = hit.distanceTo(ray.ray.origin);
      if (d < nearest) { nearest = d; best = o; }
    }
    return best;
  }
  for (const type of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel']) on(renderer.domElement, type, e => {
    if (!state.mode || (e.button !== undefined && e.button !== 0)) return;
    e.preventDefault(); e.stopImmediatePropagation();
    if (type === 'pointerdown') {
      const o = pickObject(e); if (o && state.selected !== o.id) select(o.id);
      if (state.selected !== null) {
        const selected = objects.find(o => o.id === state.selected), p = new T.Vector3(); eventRay(e);
        if (ray.ray.intersectPlane(new T.Plane(new T.Vector3(0, 0, 1), -selected.floor_height), p))
          drag = { id: e.pointerId, dx: selected.obb.centre[0] - p.x, dy: selected.obb.centre[1] - p.y };
        renderer.domElement.setPointerCapture?.(e.pointerId);
      }
    } else if (type === 'pointermove' && drag?.id === e.pointerId) {
      const selected = objects.find(o => o.id === state.selected); if (!selected) return;
      eventRay(e); const p = new T.Vector3();
      if (ray.ray.intersectPlane(new T.Plane(new T.Vector3(0, 0, 1), -selected.floor_height), p)) moveTo(p.x + drag.dx, p.y + drag.dy);
    } else if (type === 'pointermove') {
      const old = hovered; hovered = pickObject(e)?.id;
      for (const id of [old, hovered]) { const o = objects.find(o => o.id === id); if (o) refresh(o); }
    } else if (type === 'pointerup' || type === 'pointercancel') drag = null;
  }, true);
  function update(now = performance.now()) {
    if (!state.ready || disposed) return;
    if (heavyActivated) syncPhotoMeshes();
    if (tween) {
      const t = Math.min(1, (now - tween.start) / 400), ease = t * t * (3 - 2 * t);
      const position = tween.from.clone().lerp(tween.to, ease), q = tween.qFrom.clone().slerp(tween.qTo, ease);
      pose(position, position.clone().add(new T.Vector3(0, 0, -1).applyQuaternion(q)));
      if (t === 1) { tween = null; if (state.selected === null) { state.cameraOwned = false; ctx.suspendWalk(false); } }
    }
    const o = objects.find(o => o.id === state.selected);
    if (o) {
      const p = new T.Vector3().fromArray(o.obb.centre).project(camera), rect = renderer.domElement.getBoundingClientRect();
      toolbar.hidden = p.z < -1 || p.z > 1;
      toolbar.style.left = `${Math.max(5, Math.min(rect.width - 255, rect.left + (p.x + 1) * rect.width / 2))}px`;
      toolbar.style.top = `${Math.max(5, rect.top + (1 - p.y) * rect.height / 2 - 45)}px`;
    }
  }
  const api = { state, texture, setMode, select, moveTo, rotate, confirm, cancel, hide, duplicate, update, syncPhotoMeshes, ensureHeavyActivated, deactivateHeavy,
    render() { if (outline.visible) { const auto = renderer.autoClear; renderer.autoClear = false; try { renderer.render(overlayScene, camera); } finally { renderer.autoClear = auto; } } },
    dispose() {
      disposed = true; win.cancelAnimationFrame?.(frameId); clearInterval(rotationTimer); listeners.forEach(off => off());
      ctx.suspendWalk(false); panel.remove(); toolbar.remove(); scene.remove(root);
      for (const [mesh, rec] of meshRecords) { mesh.onBeforeRender = rec.prior; mesh.frustumCulled = rec.oldCull; mesh.raycast = rec.oldRaycast; mesh.geometry = rec.previous; rec.geometry.dispose(); rec.pickGeometry.dispose(); }
      for (const [material, source] of patched) { Object.assign(material, source); delete material.uniforms.furnitureState; material.needsUpdate = true; }
      for (const rec of collisionRecords) rec.mesh.geometry = rec.previous;
      for (const grid of [ctx.contactGrid, ctx.obstacleGrid]) if (grid) for (const [key, items] of grid) {
        const keep = items.filter(i => !i.furnitureW19); if (keep.length) grid.set(key, keep); else grid.delete(key);
      }
      for (const rec of gridRecords) for (const key of rec.keys) { if (!rec.grid.has(key)) rec.grid.set(key, []); if (!rec.grid.get(key).includes(rec.item)) rec.grid.get(key).push(rec.item); }
      const main = ctx.splat.packedSplats;
      if (mainAlpha && main?.packedArray) {
        for (let i = 0; i < mainAlpha.length; i++) if (mainAlpha[i]) main.packedArray[i * 4] = (main.packedArray[i * 4] & 0x00ffffff) | (mainAlpha[i] << 24);
        main.needsUpdate = true; ctx.splat.generatorDirty = true;
      }
      resources.forEach(r => r.dispose()); if (win.demoReview?.furniture === api) delete win.demoReview.furniture;
    }
  };
  win.demoReview ??= {}; win.demoReview.furniture = api;
  api.ready = (async () => {
    [data, captures] = await Promise.all([loadJSON('objects.json'), loadJSON('captures.json').then(d => d.captures)]);
    if (data.up !== 'Z' || data.units !== 'metres' || data.mask?.size !== 256) throw Error('Expected metric Z-up Phase A data');
    if (disposed) return api;
    lookup = createVoxelLookup(data); objects.push(...data.objects.map(o => ({ ...o, ...cloneLayout(o) })));
    for (const o of objects) { original.set(o.id, { ...o, ...cloneLayout(o) }); const g = new T.Group(); groups.set(o.id, g); root.add(g); }
    if (!ctx.lazy) {
      await ensureHeavyActivated(); if (disposed) return api;
      let saved; try { saved = JSON.parse(storage().getItem(`furniture-layout:${data.model_id}`)); } catch { /* No saved layout. */ }
      for (const row of saved?.version === 1 ? saved.objects ?? [] : []) {
        if (!Number.isInteger(row.id) || !row.obb || row.obb.centre?.length !== 3 || !row.obb.centre.every(Number.isFinite) || !Number.isFinite(row.obb.yaw)) continue;
        if (row.sourceId && (row.id < 256 || row.id >= 1024 || objects.some(o => o.id === row.id))) continue;
        if (row.sourceId && original.has(row.sourceId)) await duplicate(row.sourceId, row);
        else { const o = objects.find(o => o.id === row.id); if (o) { const old = cloneLayout(o); Object.assign(o, { obb: { ...o.obb, centre: row.obb.centre, yaw: row.obb.yaw }, hidden: !!row.hidden }); if (!o.hidden && !validate(o).valid) Object.assign(o, old); } }
      }
      objects.forEach(refresh); syncPhotoMeshes(); updateColliders();
    }
    state.ready = true; message.textContent = `${objects.length} đồ từ hình học tái dựng`; renderRows();
    if (ctx.autoFrame !== false && win.requestAnimationFrame) {
      const frame = now => { if (disposed) return; update(now); frameId = win.requestAnimationFrame(frame); }; frameId = win.requestAnimationFrame(frame);
    }
    return api;
  })().catch(e => { state.error = e.message; message.textContent = `Không tải được đồ: ${e.message}`; throw e; });
  return api;
}
