import * as THREE from 'three';
import { SplatMesh, SparkRenderer, PackedSplats, dyno } from '@sparkjsdev/spark';
import { OBJLoader } from './vendor/OBJLoader.js';
import { installMeasure } from './measure.js?v=w0200';
import { addNeighbourhood } from './neighbourhood.js?v=w0200';
import { installFurniture } from './furniture_edit.js?v=sharp-v3';

const status = document.querySelector('#status');

async function loadCaptures() {
  try {
    const response = await fetch('./captures.json');
    if (response.ok === false) return [];
    const data = await response.json();
    if (data.up !== 'Z' || data.units !== 'metres' || !Array.isArray(data.captures)) return [];
    return data.captures.map(c => ({ ...c, point: new THREE.Vector3().fromArray(c.position),
      basisVectors: c.face_bases_center_right_down.flat().map(v => new THREE.Vector3().fromArray(v)) }));
  } catch { return []; }
}

// captures.json already contains metric positions and world-rotated face bases.
function depthPhotoVertex(capture, face, px, py, size, range) {
  const [centre, right, down] = capture.face_bases_center_right_down[face];
  const u = 2 * (px + .5) / size - 1, v = 2 * (py + .5) / size - 1;
  const ray = centre.map((c, k) => c + u * right[k] + v * down[k]);
  const scale = range / Math.hypot(...ray);
  return ray.map((d, k) => capture.position[k] + d * scale);
}

function depthPhotoGrid(capture, face, buffer, size, step = 2, cameraFar = 150) {
  if (buffer.byteLength !== size * size * 2) throw Error('Invalid capdepth byte length');
  const data = new DataView(buffer), pixels = [];
  for (let p = 0; p < size; p += step) pixels.push(p);
  // Include the last pixel centre so adjacent cube faces leave no 2-pixel seam.
  if (pixels.at(-1) !== size - 1) pixels.push(size - 1);
  const n = pixels.length, positions = new Float32Array(n * n * 3), uvs = new Float32Array(n * n * 2);
  const ranges = new Float64Array(n * n), indices = new Uint32Array((n - 1) ** 2 * 6);
  const photoFaces = new Float32Array(n * n).fill(face);
  const far = new Uint8Array(n * n), farRange = Math.min(500, .9 * cameraFar);
  let vertex = 0, count = 0, farTriangles = 0;
  for (const py of pixels) for (const px of pixels) {
    let range = data.getUint16(2 * (py * size + px), true) / 1000;
    if (!range && depthPhotoVertex(capture, face, px, py, size, 1)[2] - capture.position[2] > -.15) {
      range = farRange; far[vertex] = 1;
    }
    ranges[vertex] = range;
    positions.set(depthPhotoVertex(capture, face, px, py, size, range), vertex * 3);
    uvs.set([(px + .5) / size, (py + .5) / size], vertex * 2); vertex++;
  }
  function triangle(a, b, c) {
    const near = Math.min(ranges[a], ranges[b], ranges[c]);
    const allFar = far[a] && far[b] && far[c];
    if (allFar || (near && Math.max(ranges[a], ranges[b], ranges[c]) - near <= .08 * near)) {
      indices[count++] = a; indices[count++] = b; indices[count++] = c;
      if (allFar) farTriangles++;
    }
  }
  for (let y = 0; y < n - 1; y++) for (let x = 0; x < n - 1; x++) {
    const a = y * n + x, b = a + 1, c = a + n, d = c + 1;
    triangle(a, c, b); triangle(b, c, d);
  }
  return { positions, uvs, photoFaces, indices: indices.subarray(0, count), grid: n, farTriangles };
}

function createPhotoLayer(renderer, collision, captures, initialClassic = true, stableWalk = false) {
  let classic0200 = initialClassic;
  const state = window.photoState = { captures: captures.length, cached: [], active: [], opacity: 0,
    gpuMs: null, cpuMs: 0, depthBuildMs: 0, depthBuilds: 0, hitches: 0, error: null,
    mode: 'collision', profile: classic0200 ? 'view-motion' : 'w21', meshBuildMs: 0, meshAttachMs: 0, triangleCount: 0, hiRes: [],
    stretch: { tornTris: 0, grazingDeg: null, fadeDeg: null, activeMax: 1 },
    handoff: { switches: 0, history: [] },
    cost: stableWalk ? 'opaque photo depth first; complementary capture handoff; one grid pass' :
      'one primary draw; two retained full grids; one upload or geometry face/frame' };
  if (!captures.length) return { state, entries: () => [], depthMeshes: () => [], setRoof() {}, update() {}, render() {} };
  const roofLimit = { value: 1e6 };
  const roofPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 1e6);
  const anisotropy = Math.min(16, renderer.capabilities.getMaxAnisotropy());
  const vertexShader = `varying vec3 worldPoint;
    void main() { vec4 p = modelMatrix * vec4(position, 1.0); worldPoint = p.xyz;
      gl_Position = projectionMatrix * viewMatrix * p; }`;
  const slots = Array(6).fill(null);
  const uniforms = { photo: { value: null }, depth: { value: null },
    pos: { value: new THREE.Vector3() }, bases: { value: [] },
    opacity: { value: 0 }, photoWeight: { value: 1 },
    stableCoverage: { value: stableWalk ? 1 : 0 }, handoffSide: { value: 0 }, handoffProgress: { value: 0 }, far: { value: 150 } };
  const photoAlphaShader = `uniform float photoWeight, stableCoverage, handoffSide, handoffProgress;
    float photoAlpha(vec3 point) {
      if (stableCoverage > .5) {
        float threshold = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(.06711056, .00583715))));
        if (handoffSide < -.5 && threshold < handoffProgress) discard;
        if (handoffSide > .5 && threshold >= handoffProgress) discard;
        return photoWeight;
      }
      return opacity * photoWeight;
    }`;
  // Six original faces are kept as texture-array layers: no cubemap reorientation
  // or resynthesis. Both face selection and UV use the exported world-space bases.
  const stencil = { stencilWrite: true, stencilFunc: THREE.EqualStencilFunc,
    stencilRef: 0, stencilZPass: THREE.IncrementWrapStencilOp };
  const material = new THREE.ShaderMaterial({ uniforms, vertexShader, ...stencil,
    side: THREE.DoubleSide, transparent: true, depthWrite: false, forceSinglePass: true,
    fragmentShader: `precision highp sampler2DArray;
      uniform sampler2DArray photo;
      uniform samplerCube depth; uniform vec3 pos, bases[18];
      uniform float opacity, far;
      varying vec3 worldPoint;
      ${photoAlphaShader}
      #include <packing>
      vec3 faceUV(vec3 direction, vec3 bases[18]) {
        int face = 0; float best = -2.0;
        for (int i = 0; i < 6; i++) {
          float facing = dot(direction, bases[i * 3]);
          if (facing > best) { best = facing; face = i; }
        }
        vec2 uv = vec2(dot(direction, bases[face * 3 + 1]),
                       dot(direction, bases[face * 3 + 2])) / best * 0.5 + 0.5;
        return vec3(clamp(uv, 0.0, 1.0), float(face));
      }
      float visibleFrom(samplerCube depths, vec3 delta) {
        float distance = length(delta);
        float seen = unpackRGBAToDepth(textureCube(depths, normalize(delta))) * far;
        // Raster/depth quantization tolerance in metres, not a wall-sized bias.
        return step(distance, seen + 0.04 + distance * 0.006);
      }
      void main() {
        float alpha = photoAlpha(worldPoint);
        if (alpha < .002) discard;
        vec3 delta = worldPoint - pos;
        if (visibleFrom(depth, delta) < .5) discard;
        gl_FragColor = vec4(texture(photo, faceUV(normalize(delta), bases)).rgb, alpha);
        #include <colorspace_fragment>
      }` });
  const photoScene = new THREE.Scene(); photoScene.add(collision.clone(true));
  photoScene.overrideMaterial = material;
  const meshUniforms = { photo: uniforms.photo, opacity: uniforms.opacity,
    pos: uniforms.pos, photoWeight: uniforms.photoWeight, stableCoverage: uniforms.stableCoverage,
    handoffSide: uniforms.handoffSide, handoffProgress: uniforms.handoffProgress };
  const meshMaterial = new THREE.ShaderMaterial({ uniforms: meshUniforms, ...stencil,
    side: THREE.DoubleSide, transparent: true, depthWrite: false, forceSinglePass: true,
    vertexShader: `invariant gl_Position;
      attribute float photoFace; varying vec3 photoUV, worldPoint;
      void main() { photoUV = vec3(uv, photoFace);
        worldPoint = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `precision highp sampler2DArray; uniform sampler2DArray photo;
      uniform float opacity; uniform vec3 pos; varying vec3 photoUV, worldPoint;
      ${photoAlphaShader}
      void main() { float alpha = photoAlpha(worldPoint); if (alpha < .002) discard;
        gl_FragColor = vec4(texture(photo, photoUV).rgb, alpha);
        #include <colorspace_fragment>
      }` });
  if (stableWalk) for (const shader of [material, meshMaterial]) {
    shader.transparent = false; shader.depthWrite = true;
    shader.stencilFunc = THREE.AlwaysStencilFunc; shader.stencilRef = 1;
    shader.stencilZPass = THREE.ReplaceStencilOp;
  }
  const prepass = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, colorWrite: false });
  // Identical clip-space arithmetic keeps the distant photo and its depth pass coincident.
  const meshPrepass = new THREE.ShaderMaterial({ side: THREE.DoubleSide, colorWrite: false,
    vertexShader: meshMaterial.vertexShader,
    fragmentShader: 'void main() { gl_FragColor = vec4(0.0); }' });
  const depthMaterial = new THREE.ShaderMaterial({ vertexShader, side: THREE.DoubleSide,
    uniforms: { centre: { value: new THREE.Vector3() }, far: uniforms.far },
    fragmentShader: `uniform vec3 centre; uniform float far; varying vec3 worldPoint;
      #include <packing>
      void main() { gl_FragColor = packDepthToRGBA(min(length(worldPoint - centre) / far, 0.999999)); }` });
  const depthScene = new THREE.Scene(); depthScene.add(collision.clone(true));
  depthScene.overrideMaterial = depthMaterial;
  for (const shader of [material, meshMaterial, meshPrepass]) {
    shader.uniforms.roofLimit = roofLimit;
    shader.vertexShader = 'varying float roofWorldZ;\n' + shader.vertexShader.replace(
      /void main\(\)\s*\{/, 'void main() { roofWorldZ = (modelMatrix * vec4(position, 1.0)).z;');
    shader.fragmentShader = 'uniform float roofLimit; varying float roofWorldZ;\n' +
      shader.fragmentShader.replace(/void main\(\)\s*\{/, 'void main() { if (roofWorldZ > roofLimit) discard;');
  }
  prepass.clippingPlanes = [roofPlane];
  const cache = new Map();
  const hiRes = new Map();
  function tearPhotoGrid(grid) {
    grid.tornTris = 0;
    return grid;
  }
  let worker, workerJob, workerRequest = 0, workerDisabled = typeof Worker === 'undefined';
  function resetWorker() {
    const previousWorker = worker;
    worker = null;
    previousWorker?.terminate();
    if (workerJob) workerJob.target.building = false;
    workerJob = null;
  }
  function failWorker() {
    workerDisabled = true;
    resetWorker();
  }
  function getWorker() {
    if (workerDisabled) return null;
    if (!worker) {
      let url;
      try {
        const script = `${depthPhotoVertex.toString()}\n${depthPhotoGrid.toString()}\n${tearPhotoGrid.toString()}\n
          onmessage = ({ data }) => {
            try {
              const start = performance.now();
              const raw = depthPhotoGrid(data.capture, data.face, data.buffer, data.size, data.step, data.cameraFar);
              const grid = tearPhotoGrid(raw, data.capture);
              postMessage({ requestId: data.requestId, grid, ms: performance.now() - start },
                [grid.positions.buffer, grid.uvs.buffer, grid.photoFaces.buffer, grid.indices.buffer]);
            } catch { postMessage({ requestId: data.requestId, error: true }); }
          };`;
        url = URL.createObjectURL(new Blob([script], { type: 'text/javascript' }));
        const activeWorker = worker = new Worker(url);
        activeWorker.onmessage = ({ data }) => {
          const job = workerJob;
          if (worker !== activeWorker || !job || data.requestId !== job.requestId) return;
          workerJob = null;
          job.target.building = false;
          if (data.error) { failWorker(); return; }
          if (!job.target.cancelled) job.target.result = data;
        };
        activeWorker.onerror = event => {
          event.preventDefault?.();
          if (worker === activeWorker) failWorker();
        };
        activeWorker.onmessageerror = () => { if (worker === activeWorker) failWorker(); };
      } catch { failWorker(); }
      finally { if (url) URL.revokeObjectURL(url); }
    }
    return worker;
  }
  function disposeMesh(target) {
    if (!target) return;
    target.cancelled = true; target.result = null;
    target.meshScene?.traverse(mesh => mesh.geometry?.dispose());
  }
  function releaseHiRes(id, entry) {
    entry.abort.abort(); entry.images.forEach(image => image?.close());
    entry.texture?.dispose(); hiRes.delete(id);
  }
  async function loadHiRes(entry) {
    try {
      for (let face = 0; face < 6; face++) {
        const url = `./captures2k/${encodeURIComponent(entry.capture.id)}/face${face}.jpg`;
        const response = await fetch(url, { signal: entry.abort.signal });
        if (!response.ok) throw Error('Missing high-resolution face');
        const image = await createImageBitmap(await response.blob(), { imageOrientation: 'none' });
        if (entry.abort.signal.aborted) { image.close(); return; }
        if (image.width !== 2048 || image.height !== 2048) {
          image.close(); throw Error('Expected 2048px face');
        }
        entry.images.push(image);
      }
    } catch {
      entry.failed = true; entry.images.forEach(image => image?.close()); entry.images = [];
      entry.texture?.dispose(); entry.texture = null;
    }
  }
  let depthIndex;
  async function loadRanges(entry) {
    try {
      depthIndex ??= fetch('./capdepth/index.json').then(r => r.ok ? r.json() : null).catch(() => null);
      const index = await depthIndex;
      if (!index) { entry.depthStatus = 'missing'; return; }
      if (index.size !== 256) throw Error('Expected 256px capdepth');
      const buffers = await Promise.all(Array.from({ length: 6 }, async (_, face) => {
        const url = `./capdepth/${encodeURIComponent(entry.capture.id)}/face${face}.u16`;
        const response = await fetch(url, { signal: entry.abort.signal });
        if (!response.ok) throw Error(`Capdepth face ${response.status}: ${url}`);
        const buffer = await response.arrayBuffer();
        if (buffer.byteLength !== index.size * index.size * 2) throw Error('Invalid capdepth byte length');
        return buffer;
      }));
      if (entry.abort.signal.aborted) return;
      entry.ranges = buffers; entry.depthSize = index.size; entry.depthStatus = 'loaded';
      entry.meshScene = new THREE.Scene(); entry.meshFaces = 0; entry.triangles = 0; entry.farTriangles = 0; entry.meshBuildMs = 0; entry.grid = 129;
    } catch (error) {
      if (!entry.abort.signal.aborted) {
        entry.depthStatus = 'missing'; state.depthError = error.message;
      }
    }
  }
  function buildMeshFace(entry, target = entry, cameraFar = 150) {
    const face = target.meshFaces, activeWorker = getWorker();
    if (!target.result && activeWorker) {
      if (!workerJob) {
        const buffer = entry.ranges[face].slice(0);
        workerJob = { target, requestId: ++workerRequest }; target.building = true;
        try { activeWorker.postMessage({ requestId: workerJob.requestId, capture: { position: entry.capture.position,
          face_bases_center_right_down: entry.capture.face_bases_center_right_down },
          face, buffer, size: entry.depthSize, step: target.grid === 129 ? 2 : 1, cameraFar, classic0200 }, [buffer]); }
        catch { failWorker(); }
      }
      return;
    }
    if (target !== entry && !activeWorker) return;
    const start = performance.now();
    const { grid, ms } = target.result || (() => { const raw = depthPhotoGrid(entry.capture, face, entry.ranges[face], entry.depthSize, 2, cameraFar); return { grid: tearPhotoGrid(raw, entry.capture), ms: 0 }; })();
    target.result = null;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(grid.positions, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(grid.uvs, 2));
    geometry.setAttribute('photoFace', new THREE.BufferAttribute(grid.photoFaces, 1));
    geometry.setIndex(new THREE.BufferAttribute(grid.indices, 1));
    target.meshScene.add(new THREE.Mesh(geometry, meshMaterial));
    target.triangles += grid.indices.length / 3; target.grid = grid.grid;
    target.tornTris = (target.tornTris || 0) + grid.tornTris;
    target.farTriangles += grid.farTriangles;
    target.meshFaces++; target.meshBuildMs += ms || performance.now() - start;
    state.meshAttachMs = performance.now() - start;
    if (target.meshFaces === 6) {
      target.ready = true;
      if (target === entry) entry.mode = 'depthmesh';
      state.depthBuilds++; state.meshBuildMs = target.meshBuildMs;
    }
  }
  async function load(entry) {
    try {
      if (entry.capture.faces.length !== 6) throw Error('Expected six photo faces');
      for (const face of entry.capture.faces) {
        const response = await fetch(face, { signal: entry.abort.signal });
        if (!response.ok) throw Error(`Photo face ${response.status}: ${face}`);
        const image = await createImageBitmap(await response.blob(), { imageOrientation: 'none' });
        if (entry.abort.signal.aborted) { image.close(); return; }
        if (image.width !== image.height || (entry.size && image.width !== entry.size)) {
          image.close(); throw Error('Non-square or unequal cube faces');
        }
        entry.size = image.width; entry.images.push(image);
      }
    } catch (error) {
      if (!entry.abort.signal.aborted) {
        entry.failed = true; entry.images.forEach(img => img?.close()); entry.images = [];
        state.error = error.message; console.warn('Photo projection:', error);
      }
    }
  }
  function uploadFace(entry) {
    if (!entry.texture) {
      const texture = entry.texture = new THREE.DataArrayTexture(null, entry.size, entry.size, 6);
      texture.colorSpace = THREE.SRGBColorSpace; texture.flipY = false;
      texture.magFilter = THREE.LinearFilter; texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.anisotropy = anisotropy;
      texture.generateMipmaps = true; texture.source.dataReady = false; texture.needsUpdate = true;
      renderer.initTexture(texture);
    }
    const face = entry.uploaded, image = entry.images[face];
    // Copy the decoded bitmap straight to its array layer; no canvas/readback.
    const source = new THREE.Texture(image);
    source.anisotropy = anisotropy;
    entry.texture.generateMipmaps = face === 5;
    try {
      renderer.copyTextureToTexture(source, entry.texture, null, new THREE.Vector3(0, 0, face));
      entry.uploaded++;
    } finally { source.dispose(); image.close(); entry.images[face] = null; }
  }
  function buildDepth(entry) {
    const start = performance.now();
    if (!entry.depthTarget) {
      entry.depthTarget = new THREE.WebGLCubeRenderTarget(256, { minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter, generateMipmaps: false });
      const cube = entry.depthCamera = new THREE.CubeCamera(.03, uniforms.far.value, entry.depthTarget);
      cube.coordinateSystem = renderer.coordinateSystem; cube.updateCoordinateSystem();
      cube.position.copy(entry.capture.point); cube.updateMatrixWorld(true);
    }
    const cube = entry.depthCamera;
    depthMaterial.uniforms.centre.value.copy(cube.position);
    const clear = renderer.getClearColor(new THREE.Color()), alpha = renderer.getClearAlpha();
    const autoClear = renderer.autoClear, xr = renderer.xr.enabled;
    const target = renderer.getRenderTarget(), face = renderer.getActiveCubeFace(), mip = renderer.getActiveMipmapLevel();
    try {
      renderer.xr.enabled = false; renderer.autoClear = true; renderer.setClearColor(0xffffff, 1);
      renderer.setRenderTarget(entry.depthTarget, entry.depthFaces);
      renderer.render(depthScene, cube.children[entry.depthFaces++]);
      if (entry.depthFaces === 6) {
        entry.depth = entry.depthTarget; entry.ready = true; entry.mode = 'collision'; state.depthBuilds++;
      }
      state.depthBuildMs = performance.now() - start;
    } finally {
      renderer.setRenderTarget(target, face, mip); renderer.xr.enabled = xr;
      renderer.setClearColor(clear, alpha); renderer.autoClear = autoClear;
    }
  }
  const gl = renderer.getContext(), timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  const queries = []; let tick = 0, draw = false, nearestEntry, transition;
  let lastTime = performance.now(), lastPosition, challenger, challengerSince = 0, switchedAt = 0;
  const direction = new THREE.Vector3(), look = new THREE.Vector3(), hitchTimes = [];
  let wasMoving = false;
  function pollTimers() {
    if (!timer) return;
    const disjoint = gl.getParameter(timer.GPU_DISJOINT_EXT);
    while (queries.length && gl.getQueryParameter(queries[0], gl.QUERY_RESULT_AVAILABLE)) {
      const query = queries.shift();
      if (!disjoint) state.gpuMs = gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6;
      gl.deleteQuery(query);
    }
    if (disjoint) { queries.splice(0).forEach(query => gl.deleteQuery(query)); state.gpuMs = null; }
  }
  return { state,
    update(camera, enabled) {
      pollTimers();
      const now = performance.now(), elapsed = now - lastTime;
      if (transition && (!enabled || now - transition.start >= 420)) transition = null;
      const ease = 1 - Math.exp(-Math.max(0, elapsed) / 250);
      const motion = lastPosition ? camera.position.clone().sub(lastPosition) : new THREE.Vector3();
      if (!stableWalk) motion.z = 0;
      const moving = window.walkState?.moving ?? motion.lengthSq() > 1e-8;
      if (elapsed > 33 && (moving || wasMoving)) hitchTimes.push(now);
      while (hitchTimes.length && hitchTimes[0] < now - 10000) hitchTimes.shift();
      state.hitches = hitchTimes.length; wasMoving = moving; lastTime = now;
      camera.getWorldDirection(look); if (!stableWalk) look.z = 0;
      if (look.lengthSq() > 1e-8) look.normalize();
      if (motion.lengthSq() > 1e-8) direction.copy(motion).normalize();
      else direction.copy(look);
      lastPosition = camera.position.clone();
      const nearest = captures.map(capture => {
        const delta = capture.point.clone().sub(camera.position), distance = delta.length();
        const alignment = distance > .001 ? delta.dot(look) / distance : 1;
        return { capture, distance, score: distance + .18 * (1 - alignment * alignment) };
      }).sort((a, b) => a.score - b.score);
      const ahead = camera.position.clone().addScaledVector(direction, .9).addScaledVector(look, .35);
      const ordered = [nearest.find(item => item.capture.id === transition?.from.capture.id),
        nearest.find(item => item.capture.id === nearestEntry?.capture.id),
        ...nearest.slice(0, 2), ...nearest.slice().sort((a, b) =>
          ahead.distanceToSquared(a.capture.point) - ahead.distanceToSquared(b.capture.point))].filter(Boolean);
      const preload = [...new Map(ordered.map(item => [item.capture.id, item])).values()].slice(0, 6);
      const wanted = new Set(preload.map(item => item.capture.id));
      for (const [id, entry] of cache) if (!wanted.has(id) && !entry.weight) {
        entry.abort.abort(); entry.images.forEach(img => img?.close());
        entry.texture?.dispose(); entry.depthTarget?.dispose();
        disposeMesh(entry); disposeMesh(entry.fullMesh); cache.delete(id);
      }
      for (const { capture } of preload) if (!cache.has(capture.id)) {
        const entry = { capture, abort: new AbortController(), images: [], uploaded: 0, depthFaces: 0,
          depthStatus: 'loading', weight: 0 };
        cache.set(capture.id, entry); void load(entry); void loadRanges(entry);
      }
      const targets = new Map();
      const readyNearest = nearest.filter(({ capture }) => {
        const entry = cache.get(capture.id);
        return entry?.ready && !entry.failed;
      }).slice(0, 2);
      if (enabled) for (const { capture, distance } of readyNearest) {
        const entry = cache.get(capture.id);
        if (entry.failed || !entry.ready) continue;
        if (!slots.includes(entry)) {
          const empty = slots.indexOf(null);
          if (empty < 0) continue;
          slots[empty] = entry;
        }
        targets.set(entry, 1 / Math.max(.001, distance));
      }
      const sum = [...targets.values()].reduce((a, b) => a + b, 0);
      slots.forEach((entry, i) => {
        if (!entry) return;
        const target = (targets.get(entry) || 0) / (sum || 1);
        entry.weight += (target - entry.weight) * ease;
        if (!target && entry.weight < .001) entry.weight = 0;
        if (!entry.weight) slots[i] = null;
      });
      const totalWeight = slots.reduce((total, entry) => total + (entry?.weight || 0), 0);
      for (const entry of slots) if (entry) {
        if (enabled && sum && totalWeight) entry.weight /= totalWeight;
        entry.drawOpacity = THREE.MathUtils.smoothstep(entry.weight, .02, .12);
      }
      const best = readyNearest[0] && cache.get(readyNearest[0].capture.id);
      if (!nearestEntry || nearestEntry.failed) { nearestEntry = best; switchedAt = now; }
      const candidate = best !== nearestEntry ? best :
        readyNearest.map(item => cache.get(item.capture.id)).find(entry => entry !== nearestEntry);
      const sharp = enabled ? [...new Set([nearestEntry, transition?.from || candidate].filter(Boolean))] : [];
      const sharpIds = new Set(sharp.map(entry => entry.capture.id));
      for (const [id, entry] of hiRes) if (!sharpIds.has(id)) releaseHiRes(id, entry);
      for (const base of sharp) if (!hiRes.has(base.capture.id)) {
        const entry = { capture: base.capture, abort: new AbortController(), images: [], uploaded: 0, size: 2048 };
        hiRes.set(base.capture.id, entry); void loadHiRes(entry);
      }
      for (const entry of cache.values()) if (entry.fullMesh &&
        (!sharpIds.has(entry.capture.id) || workerDisabled)) {
        disposeMesh(entry.fullMesh); entry.fullMesh = null;
      }
      for (const entry of sharp) if (entry.mode === 'depthmesh' && getWorker()) {
        entry.fullMesh ??= { meshScene: new THREE.Scene(), meshFaces: 0,
          triangles: 0, farTriangles: 0, meshBuildMs: 0, grid: 256 };
      }
      const fullPending = sharp.find(entry => entry.fullMesh && !entry.fullMesh.ready &&
        (entry.fullMesh.result || (!entry.fullMesh.building && !workerJob)));
      const sharpPending = [...hiRes.values()].find(entry => !entry.failed && entry.images[entry.uploaded]);
      const pending = preload.map(item => cache.get(item.capture.id)).find(entry => !entry.failed &&
        ((entry.uploaded < 6 && entry.images[entry.uploaded]) ||
          (entry.uploaded === 6 && !entry.ready && entry.depthStatus !== 'loading' &&
            (entry.depthStatus !== 'loaded' || entry.result || (!entry.building && !workerJob)))));
      let operation;
      try {
        if (fullPending?.fullMesh.result) {
          operation = fullPending; buildMeshFace(fullPending, fullPending.fullMesh, camera.far);
        } else if (sharpPending) {
          operation = sharpPending; uploadFace(sharpPending);
        } else if (fullPending) {
          operation = fullPending; buildMeshFace(fullPending, fullPending.fullMesh, camera.far);
        }
        else if (pending) {
          operation = pending;
          if (pending.uploaded < 6) uploadFace(pending);
          else if (pending.depthStatus === 'loaded') buildMeshFace(pending, pending, camera.far);
          else buildDepth(pending);
        }
      } catch (error) {
        state.error = error.message; console.warn('Photo projection:', error);
        if (operation === sharpPending) {
          sharpPending.failed = true;
          sharpPending.images.forEach(image => image?.close()); sharpPending.images = [];
          sharpPending.texture?.dispose(); sharpPending.texture = null;
        } else if (operation) operation.failed = true;
      }
      const currentDistance = nearestEntry ? camera.position.distanceTo(nearestEntry.capture.point) : Infinity;
      const currentScore = nearest.find(item => item.capture.id === nearestEntry?.capture.id)?.score ?? Infinity;
      const bestScore = nearest.find(item => item.capture.id === best?.capture.id)?.score ?? Infinity;
      const gain = currentScore - bestScore;
      const hasAdvantage = enabled && best && best !== nearestEntry && gain > Math.max(.28, currentDistance * .16);
      if (!hasAdvantage) { challenger = null; challengerSince = now; }
      else if (challenger !== best) { challenger = best; challengerSince = now; }
      const nextSharp = best && hiRes.get(best.capture.id);
      const geometryReady = best?.mode !== 'depthmesh' || workerDisabled || best.fullMesh?.ready;
      const textureReady = nextSharp?.uploaded === 6 || nextSharp?.failed;
      if (!transition && hasAdvantage && now - challengerSince >= 300 && now - switchedAt >= 750 && geometryReady && textureReady) {
        state.handoff.history.push({ from: nearestEntry.capture.id, to: best.capture.id, time: now,
          distance: camera.position.distanceTo(best.capture.point), grid: best.fullMesh?.ready ? 256 : best.grid });
        if (state.handoff.history.length > 12) state.handoff.history.shift();
        if (stableWalk) transition = { from: nearestEntry, to: best, start: now };
        state.handoff.switches++; nearestEntry = best; switchedAt = now; challenger = null;
      }
      state.handoff.progress = transition ? THREE.MathUtils.smoothstep((now - transition.start) / 420, 0, 1) : 1;
      state.handoff.outgoing = transition?.from.capture.id || null;
      state.handoff.primary = nearestEntry?.capture.id;
      state.handoff.candidate = challenger?.capture.id || null;
      state.handoff.waitingFor = hasAdvantage ? (!geometryReady ? 'geometry' : !textureReady ? 'texture' : 'hold') : null;
      state.handoff.fullGrids = sharp.filter(entry => entry.fullMesh?.ready).map(entry => entry.capture.id);
      state.hiRes = [...hiRes].filter(([, entry]) => entry.uploaded === 6 && !entry.failed).map(([id]) => id);
      state.mode = nearestEntry?.mode || 'collision';
      const opacity = stableWalk ? (enabled && nearestEntry ? 1 : 0) : classic0200
        ? (enabled && sum ? 1 - THREE.MathUtils.smoothstep(nearestEntry ? camera.position.distanceTo(nearestEntry.capture.point) : Infinity,
            state.mode === 'depthmesh' ? 1 : .25, state.mode === 'depthmesh' ? 2.2 : .9) : 0)
        : (enabled && sum ? 1 : 0);
      if ((stableWalk || !classic0200) && opacity) state.opacity = 1;
      else state.opacity += (opacity - state.opacity) * ease;
      if (!opacity && state.opacity < .001) state.opacity = 0;
      const fallback = slots.find(Boolean);
      uniforms.opacity.value = state.opacity;
      state.cached = [...cache.keys()]; state.active = slots.filter(Boolean).map(entry => entry.capture.id);
      state.ready = [...cache.values()].filter(entry => !entry.failed && entry.ready)
        .map(entry => entry.capture.id);
      state.blends = slots.filter(Boolean).map(entry => {
        const mesh = entry.fullMesh?.ready ? entry.fullMesh : entry;
        return { id: entry.capture.id, weight: entry.weight, grid: entry.mode === 'depthmesh' ? mesh.grid : 0,
          mode: entry.mode, meshBuildMs: mesh.meshBuildMs || 0, triangleCount: mesh.triangles || 0,
          farTriangles: mesh.farTriangles || 0 };
      });
      state.triangleCount = state.blends.reduce((sum, entry) => sum + entry.triangleCount, 0);
      state.stretch.tornTris = [...cache.values()].reduce((sum, entry) =>
        sum + (entry.tornTris || 0) + (entry.fullMesh?.tornTris || 0), 0);
      draw = !!fallback && state.opacity > 0;
    },
    entries: () => cache.values(),
    setRoof(z) { roofLimit.value = z; roofPlane.constant = z; },
    setClassic0200(value) {
      const next = !!value;
      if (next === classic0200) return;
      classic0200 = next; state.profile = classic0200 ? 'view-motion' : 'w21';
      resetWorker();
      for (const entry of cache.values()) {
        disposeMesh(entry.fullMesh); entry.fullMesh = null;
        if (entry.depthStatus === 'loaded') {
          disposeMesh(entry);
          entry.cancelled = false; entry.meshScene = new THREE.Scene();
          entry.meshFaces = 0; entry.triangles = 0; entry.farTriangles = 0;
          entry.tornTris = 0; entry.meshBuildMs = 0; entry.grid = 129; entry.ready = false;
        }
      }
    },
    depthMeshes: () => [...cache.values()].flatMap(e =>
      [...(e.meshScene?.children ?? []), ...(e.fullMesh?.meshScene?.children ?? [])]),
    render(camera) {
      if (!draw) return;
      const start = performance.now();
      const query = timer && ++tick % 30 === 0 && queries.length < 4 ? gl.createQuery() : null;
      if (query) gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
      const autoClear = renderer.autoClear;
      try {
        renderer.autoClear = false; renderer.clear(false, true, true);
        // Prepass the nearest capture only; torn/invalid triangles leave splat
        // pixels available. Stencil reserves each pixel for its highest weight.
        const meshScene = entry => (stableWalk || classic0200 ? entry.fullMesh?.ready : (entry.fullMesh?.ready && !wasMoving))
          ? entry.fullMesh.meshScene : entry.meshScene;
        const nearestScene = nearestEntry?.mode === 'depthmesh' ? meshScene(nearestEntry) : photoScene;
        if (!stableWalk) {
          nearestScene.overrideMaterial = nearestEntry?.mode === 'depthmesh' ? meshPrepass : prepass;
          renderer.render(nearestScene, camera);
          nearestScene.overrideMaterial = null;
        }
        const active = transition ? [transition.from, transition.to] : nearestEntry ? [nearestEntry] : [];
        uniforms.handoffProgress.value = state.handoff.progress;
        state.drawPasses = active.length;
        for (const [rank, entry] of active.entries()) {
          uniforms.handoffSide.value = transition ? (rank === 0 ? -1 : 1) : 0;
          const sharp = hiRes.get(entry.capture.id);
          uniforms.photo.value = sharp?.uploaded === 6 && !sharp.failed ? sharp.texture : entry.texture;
          uniforms.pos.value.copy(entry.capture.point);
          uniforms.photoWeight.value = stableWalk || classic0200 ? 1 : (rank === 0 ? 1 : entry.drawOpacity);
          const targetScene = entry.mode === 'depthmesh' ? meshScene(entry) : photoScene;
          // A secondary capture can fill a hole with several overlapping
          // surfaces. Resolve its own nearest surface before stencil claims it.
          if (!stableWalk && targetScene !== nearestScene) {
            targetScene.overrideMaterial = entry.mode === 'depthmesh' ? meshPrepass : prepass;
            renderer.render(targetScene, camera);
          }
          if (entry.mode === 'collision') {
            uniforms.depth.value = entry.depth.texture;
            uniforms.pos.value.copy(entry.capture.point); uniforms.bases.value = entry.capture.basisVectors;
          }
          targetScene.overrideMaterial = entry.mode === 'depthmesh' ? meshMaterial : material;
          targetScene.overrideMaterial.uniformsNeedUpdate = true;
          renderer.render(targetScene, camera);
          targetScene.overrideMaterial = null;
        }
      } finally {
        renderer.autoClear = autoClear;
        if (query) { gl.endQuery(timer.TIME_ELAPSED_EXT); queries.push(query); }
        state.cpuMs = performance.now() - start;
      }
    }
  };
}

function governPixelRatio(renderer) {
  const maximum = Math.max(1, Math.min(devicePixelRatio, 2));
  const requested = typeof URLSearchParams === 'undefined' ? NaN
    : Number(new URLSearchParams(globalThis.location?.search ?? '').get('pr'));
  const fixed = requested >= 1 && requested <= 2;
  let ratio = fixed ? requested : maximum, previous, elapsed = 0, frames = 0, fastMs = 0;
  let raisedAt = -Infinity, recoveryAfter = 0;
  renderer.setPixelRatio(ratio);
  const setLoop = renderer.setAnimationLoop.bind(renderer);
  renderer.setAnimationLoop = callback => setLoop(callback && (time => {
    if (previous !== undefined) {
      const interval = Math.max(0, time - previous);
      if (interval > 250) { elapsed = 0; frames = 0; fastMs = 0; }
      else { elapsed += interval; frames++; }
    }
    previous = time;
    if (!fixed && elapsed >= 1000) {
      const average = elapsed / frames;
      fastMs = average < 18 && time >= recoveryAfter ? fastMs + elapsed : 0;
      const next = average > 22 ? Math.max(1, ratio - .25)
        : fastMs >= 3000 ? Math.min(maximum, ratio + .25) : ratio;
      if (next !== ratio) {
        if (next > ratio) raisedAt = time;
        else if (time - raisedAt < 4000) recoveryAfter = time + 15000;
        ratio = next; renderer.setPixelRatio(ratio); fastMs = 0;
      }
      elapsed = 0; frames = 0;
    }
    if (window.walkState) window.walkState.pixelRatio = ratio;
    callback(time);
  }));
}

try {
  const config = await (await fetch('./scene.json')).json();
  const captures = await loadCaptures();
  const params = typeof URLSearchParams === 'undefined' ? null : new URLSearchParams(globalThis.location?.search ?? '');
  const startCapture = captures.find(c => c.id === params?.get('capture'));
  if (startCapture) {
    config.position = startCapture.position.slice();
    config.forward = startCapture.face_bases_center_right_down[1][0].slice();
  }
  const scene = new THREE.Scene();
  const renderer = new THREE.WebGLRenderer({ antialias: false, stencil: true });
  renderer.debug.onShaderError = (gl, program, vertex, fragment) => {
    window.previewError = ['GPU shader', gl.getProgramInfoLog(program),
      gl.getShaderInfoLog(vertex), gl.getShaderInfoLog(fragment)].filter(Boolean).join(': ').slice(0, 2000);
    console.error(window.previewError);
  };
  renderer.localClippingEnabled = true;
  governPixelRatio(renderer);
  renderer.setSize(innerWidth, innerHeight);
  document.body.append(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, .05, 150);
  camera.up.set(0, 0, 1);
  const stableHF = location.pathname.includes('/HFwcH424n9w/');
  const refineR5 = location.pathname.includes('/HFwcH424n9w/') && params?.get('refine') === '1';
  const spark = new SparkRenderer({ renderer, sortRadial: !stableHF });
  if (stableHF) {
    spark.material.stencilFunc = THREE.EqualStencilFunc; spark.material.stencilRef = 0;
    spark.material.stencilWriteMask = 0; spark.material.stencilZPass = THREE.KeepStencilOp;
  } scene.add(spark);
  if (params?.get('raw') !== '1' && spark.material?.vertexShader) {
    spark.material.vertexShader = spark.material.vertexShader.replace('vRgba = rgba;', `
      if (!enableCovSplats) {
        float sMax = max(scales.x, max(scales.y, scales.z));
        float sMid = max(min(scales.x, scales.y), min(max(scales.x, scales.y), scales.z));
        if (${!stableHF ? 'true' : 'false'} && sMax > 0.04 && sMid > 0.0 && sMax / sMid > 6.0) return;
        if (${refineR5 ? 'true' : 'false'}) {
          float sMin = min(scales.x, min(scales.y, scales.z));
          if (sMax > 0.025 && sMid > 0.0 && sMax / sMid > 3.2) return;
          if (sMid > 0.04 && sMin > 0.015 && rgba.a < 0.45) return;
        }
      }
      rgba.a *= smoothstep(0.18, 0.65, length(viewCenter));
      vRgba = rgba;`);
    spark.material.needsUpdate = true;
  }
  // splat-transform/SPZ loader preserves the source frame. No Y-up conversion:
  // source Gaussians, camera and collision OBJ all use the same metric Z-up frame.
  const splat = new SplatMesh({ url: refineR5 ? './scene-refined.spz' : './scene.spz' }); scene.add(splat);
  if (params?.get('diagnostics') === '1') splat.initialized.then(() => {
    const proof = document.createElement('pre'); proof.id = 'splat-decode-proof';
    proof.hidden = true;
    proof.textContent = JSON.stringify([0, 1, 1000].map(i => JSON.parse(JSON.stringify(splat.packedSplats.getSplat(i)))));
    document.body.append(proof);
  });
  const collision = await new OBJLoader().loadAsync('./collision.obj');
  collision.traverse(o => { if (o.isMesh) o.material = new THREE.MeshBasicMaterial({
    color: 0xaaaaaa, side: THREE.DoubleSide, wireframe: true, transparent: true, opacity: .35 }); });
  collision.updateMatrixWorld(true);
  const overlay = collision.clone(true); overlay.visible = false; scene.add(overlay);
  const coverage = new THREE.Group(); coverage.visible = false; scene.add(coverage);
  const unseenGeometry = await new OBJLoader().loadAsync('./unseen.obj');
  unseenGeometry.traverse(o => { if (o.isMesh) o.material = new THREE.MeshBasicMaterial({
    color: 0xdc2dbe, side: THREE.DoubleSide, transparent: true, opacity: .7,
    depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }); });
  coverage.add(unseenGeometry);
  const holes = await (await fetch('./holes.json')).json();
  const floors = [];
  const floorGrid = new Map();
  const textures = new THREE.TextureLoader();
  for (const plane of holes.planes) {
    const origin = new THREE.Vector3().fromArray(plane.origin);
    const u = new THREE.Vector3().fromArray(plane.u), v = new THREE.Vector3().fromArray(plane.v);
    const normal = new THREE.Vector3().fromArray(plane.normal);
    const w = plane.width * plane.step, h = plane.height * plane.step;
    const corners = [origin, origin.clone().addScaledVector(u, w),
      origin.clone().addScaledVector(u, w).addScaledVector(v, h), origin.clone().addScaledVector(v, h)];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(corners.flatMap(p => p.toArray()), 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0,0,1,0,1,1,0,1], 2));
    geometry.setIndex([0,1,2,0,2,3]); geometry.computeVertexNormals();
    let coverageTexture;
    for (const [asset, parent, order] of [[plane.patch, coverage, 1], [plane.overlay, coverage, 2]]) {
      const texture = await textures.loadAsync(asset); texture.flipY = false; texture.colorSpace = THREE.SRGBColorSpace;
      if (order === 2) coverageTexture = texture;
      texture.magFilter = THREE.NearestFilter; texture.minFilter = THREE.NearestFilter;
      const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, alphaTest: .01,
        side: THREE.DoubleSide, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -order, polygonOffsetUnits: -order });
      const mesh = new THREE.Mesh(geometry, material); mesh.renderOrder = order; parent.add(mesh);
    }
    // A slab labelled ceiling from downstairs is also the floor upstairs.
    if (Math.abs(plane.normal[2]) > .85 && plane.surface !== 'other') {
      const texture = await textures.loadAsync(plane.support);
      const canvas = document.createElement('canvas'); canvas.width = plane.width; canvas.height = plane.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true }); ctx.drawImage(texture.image, 0, 0);
      const pixels = ctx.getImageData(0, 0, plane.width, plane.height).data;
      // Seen cells plus labelled unseen cells are the rasterized mesh footprint.
      // Visibility is a rendering label, not a hole in the physical floor.
      ctx.clearRect(0, 0, plane.width, plane.height); ctx.drawImage(coverageTexture.image, 0, 0);
      const unseen = ctx.getImageData(0, 0, plane.width, plane.height).data;
      for (let i = 0; i < pixels.length; i += 4) if (unseen[i + 3]) pixels[i] = 255;
      const floor = { ...plane, origin, u, v, normal, pixels };
      floors.push(floor);
      for (let x = Math.floor(Math.min(...corners.map(p => p.x))); x <= Math.floor(Math.max(...corners.map(p => p.x))); x++) {
        for (let y = Math.floor(Math.min(...corners.map(p => p.y))); y <= Math.floor(Math.max(...corners.map(p => p.y))); y++) {
          const key = `${x},${y}`;
          if (!floorGrid.has(key)) floorGrid.set(key, []);
          floorGrid.get(key).push(floor);
        }
      }
      texture.dispose();
    }
  }
  document.querySelector('#coverage').onchange = e => { coverage.visible = e.target.checked; };
  function supportedFloor(position, z) {
    if (meshFloorAt(position, z, .025) !== undefined) return true;
    const point = position.clone(); point.z = z;
    // Fail closed outside the mesh footprint; keep unseen floor walkable inside it.
    if ((floorGrid.get(`${Math.floor(point.x)},${Math.floor(point.y)}`) ?? []).some(p => supportedFloorOn(p, point, z))) return true;
    point.z = z + eyeHeight;
    const bridged = floorAt(point);
    const tolerance = bridged !== undefined && meshFloorAt(point, bridged, .025) !== undefined ? .35 : .15;
    return bridged !== undefined && Math.abs(bridged - z) < tolerance;
  }
  function supportedFloorOn(p, position, z) {
    const dx = position.x - p.origin.x, dy = position.y - p.origin.y, dz = z - p.origin.z;
    if (Math.abs(dx * p.normal.x + dy * p.normal.y + dz * p.normal.z) > .15) return false;
    const x = Math.floor((dx * p.u.x + dy * p.u.y + dz * p.u.z) / p.step);
    const y = Math.floor((dx * p.v.x + dy * p.v.y + dz * p.v.z) / p.step);
    return x >= 0 && y >= 0 && x < p.width && y < p.height && p.pixels[(y * p.width + x) * 4] > 127;
  }
  await splat.initialized;
  const bodyRadius = .25, maxStep = .20;
  // Optional measured obstacle cells; old/fixture exports have no such asset.
  const obstacleData = await fetch('./obstacles.json', { cache: 'no-cache' }).then(r => r.json()).catch(() => null);
  const obstacleGrid = new Map();
  if (obstacleData?.version === 1 && obstacleData.step === .1) {
    for (const [x, y, floor, low, high] of obstacleData.cells) {
      const key = `${x},${y}`;
      if (!obstacleGrid.has(key)) obstacleGrid.set(key, []);
      obstacleGrid.get(key).push({ floor, low, high });
    }
  }
  // The detailed local mesh retains treads omitted by the simple-plane export.
  const walkMesh = await new OBJLoader().loadAsync('./walkmesh.obj').catch(() => collision);
  walkMesh.updateMatrixWorld(true);
  const walkGrid = new Map();
  walkMesh.traverse(o => {
    if (!o.isMesh) return;
    const positions = o.geometry.attributes.position, indices = o.geometry.index;
    for (let i = 0; i < (indices ? indices.count : positions.count); i += 3) {
      const points = [0, 1, 2].map(j => new THREE.Vector3().fromBufferAttribute(positions,
        indices ? indices.getX(i + j) : i + j).applyMatrix4(o.matrixWorld));
      const normal = points[1].clone().sub(points[0]).cross(points[2].clone().sub(points[0]));
      if (normal.lengthSq() < 1e-10 || normal.z <= .6 * normal.length()) continue;
      const [a, b, c] = points;
      const triangle = { a, b, c, normal };
      for (let x = Math.floor(Math.min(a.x, b.x, c.x)); x <= Math.floor(Math.max(a.x, b.x, c.x)); x++) {
        for (let y = Math.floor(Math.min(a.y, b.y, c.y)); y <= Math.floor(Math.max(a.y, b.y, c.y)); y++) {
          const key = `${x},${y}`;
          if (!walkGrid.has(key)) walkGrid.set(key, []);
          walkGrid.get(key).push(triangle);
        }
      }
    }
  });
  // Index wall triangles and mask contours in metre cells. Contacts use the
  // actual wall tangent, including oblique walls, rather than world-axis sliding.
  const contactGrid = new Map();
  function indexContact(item, points) {
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    Object.assign(item, { minX: Math.min(...xs) - bodyRadius, maxX: Math.max(...xs) + bodyRadius,
      minY: Math.min(...ys) - bodyRadius, maxY: Math.max(...ys) + bodyRadius });
    for (let x = Math.floor(item.minX); x <= Math.floor(item.maxX); x++) {
      for (let y = Math.floor(item.minY); y <= Math.floor(item.maxY); y++) {
        const key = `${x},${y}`;
        if (!contactGrid.has(key)) contactGrid.set(key, []);
        contactGrid.get(key).push(item);
      }
    }
  }
  // holes.py exports all simple-plane faces before its fused-depth faces.
  let coarsePrefix = true;
  collision.traverse(o => {
    if (!o.isMesh) return;
    const geometry = o.geometry, positions = geometry.attributes.position, indices = geometry.index;
    const count = indices ? indices.count : positions.count;
    for (let i = 0; i < count; i += 3) {
      const points = [0, 1, 2].map(j => new THREE.Vector3().fromBufferAttribute(positions,
        indices ? indices.getX(i + j) : i + j).applyMatrix4(o.matrixWorld));
      const normal = points[1].clone().sub(points[0]).cross(points[2].clone().sub(points[0]));
      if (normal.lengthSq() < 1e-10 || Math.abs(normal.z) > .7 * normal.length()) continue;
      const minZ = Math.min(...points.map(p => p.z)), maxZ = Math.max(...points.map(p => p.z));
      // Coarse planes are authoritative. Short, tilted monocular-depth sheets
      // around cameras are not solid objects; retain upright furniture faces
      // only when they have at least half a metre of vertical evidence.
      const coarse = coarsePrefix && holes.planes.some(p => Math.abs(p.normal[2]) <= .7 && points.every(q =>
        Math.abs((q.x - p.origin[0]) * p.normal[0] + (q.y - p.origin[1]) * p.normal[1] +
          (q.z - p.origin[2]) * p.normal[2]) < .005));
      if (!coarse) coarsePrefix = false;
      if (!coarse && (Math.abs(normal.z) > .3 * normal.length() || maxZ - minZ < .5)) continue;
      indexContact({ points, minZ, maxZ, coarse }, points);
    }
  });
  for (const p of floors) {
    const filled = (x, y) => x >= 0 && y >= 0 && x < p.width && y < p.height && p.pixels[(y * p.width + x) * 4] > 127;
    const point = (x, y) => p.origin.clone().addScaledVector(p.u, x * p.step).addScaledVector(p.v, y * p.step);
    for (let y = 0; y < p.height; y++) for (let x = 0; x < p.width; x++) {
      if (!filled(x, y)) continue;
      for (const [dx, dy, ax, ay, bx, by] of [
        [-1, 0, x, y, x, y + 1], [1, 0, x + 1, y, x + 1, y + 1],
        [0, -1, x, y, x + 1, y], [0, 1, x, y + 1, x + 1, y + 1]]) {
        if (filled(x + dx, y + dy)) continue;
        const a = point(ax, ay), b = point(bx, by);
        const normal = p.u.clone().multiplyScalar(-dx).addScaledVector(p.v, -dy); normal.z = 0; normal.normalize();
        indexContact({ boundary: true, a, b, normal }, [a, b]);
      }
    }
  }
  const keys = new Set(); let yaw = 0, pitch = 0;
  const velocity = new THREE.Vector3();
  let turnVelocity = 0, moveGesture = null, turnGesture = null;
  let step = null, turnStep = null, clickTarget = null, ringFade = 0, moveCarry = null, turnCarry = 0;
  let pendingInput = null, lastInputToMoveMs = null, actualSpeed = 0, stopRate = 12;
  const moveCodes = new Set(['KeyW', 'KeyS', 'KeyA', 'KeyD', 'ArrowUp', 'ArrowDown']);
  const turnCodes = new Set(['ArrowLeft', 'ArrowRight']);
  const joystick = document.querySelector('#joystick');
  const stick = document.querySelector('#stick');
  const joy = new THREE.Vector2();
  let joyPointer = null, lookPointer = null;
  const targetRing = new THREE.Mesh(new THREE.RingGeometry(.12, .16, 32),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide,
      transparent: true, opacity: .85, depthWrite: false }));
  targetRing.visible = false; scene.add(targetRing);
  const walkState = window.walkState = { pos: camera.position.toArray(), yaw, pitch,
    speed: 0, moving: false, lastInputToMoveMs, clickTarget: null,
    steer: { mode: 'free', side: 0, angleDeg: 0 } };
  const steering = { side: 0, flips: 0, angle: 0, target: 0, input: null };
  function resetSteering() {
    steering.side = steering.flips = steering.angle = steering.target = 0; steering.input = null;
    Object.assign(walkState.steer, { mode: 'free', side: 0, angleDeg: 0 });
  }
  const heightControl = document.querySelector('#height');
  let eyeHeight = 1.6;
  let renderEyeZ;
  let fallSpeed = 0, fallAcceleration = 0;
  let automaticHeight = true;
  const ray = new THREE.Raycaster();
  function intersections(origin, direction, distance) {
    ray.set(origin, direction); ray.near = .001; ray.far = distance;
    return ray.intersectObject(collision, true);
  }
  function meshFloorAt(position, expected = position.z - eyeHeight, tolerance = .35, below = false) {
    let floor;
    for (const { a, b, c, normal } of walkGrid.get(`${Math.floor(position.x)},${Math.floor(position.y)}`) ?? []) {
      const z = a.z - (normal.x * (position.x - a.x) + normal.y * (position.y - a.y)) / normal.z;
      if (Math.abs(z - expected) > tolerance + 1e-6 || (below && z > expected + 1e-6)) continue;
      const cross = (p, q) => (q.x - p.x) * (position.y - p.y) - (q.y - p.y) * (position.x - p.x);
      if (cross(a, b) < -1e-7 || cross(b, c) < -1e-7 || cross(c, a) < -1e-7) continue;
      if (floor === undefined || Math.abs(z - expected) < Math.abs(floor - expected)) floor = z;
    }
    return floor;
  }
  function floorBelowAt(position) {
    const expected = position.z - eyeHeight;
    let floor = meshFloorAt(position, expected, Infinity, true);
    for (const p of floorGrid.get(`${Math.floor(position.x)},${Math.floor(position.y)}`) ?? []) {
      const z = p.origin.z - (p.normal.x * (position.x - p.origin.x) +
        p.normal.y * (position.y - p.origin.y)) / p.normal.z;
      if (z <= expected + 1e-6 && (floor === undefined || z > floor) && supportedFloorOn(p, position, z)) floor = z;
    }
    return floor;
  }
  function stepLimit(position, floor) {
    return floor !== undefined && meshFloorAt(position, floor, .025) !== undefined ? .35 : maxStep;
  }
  function directFloorAt(position) {
    const expected = position.z - eyeHeight;
    const mesh = meshFloorAt(position, expected);
    if (mesh !== undefined) return mesh;
    // Bridge small holes between real treads before considering broad slabs,
    // whose simplified footprint can extend underneath the staircase.
    for (const [dx, dy] of [[1,0], [0,1], [.7071,.7071], [.7071,-.7071]]) {
      const sides = [];
      for (const sign of [-1, 1]) for (let distance = .05; distance <= .30 + 1e-6; distance += .05) {
        const probe = position.clone(); probe.x += sign * dx * distance; probe.y += sign * dy * distance;
        const z = meshFloorAt(probe, expected);
        if (z !== undefined) { sides.push({ z, distance }); break; }
      }
      if (sides.length === 2 && Math.abs(sides[0].z - sides[1].z) <= .35) {
        return (sides[0].z * sides[1].distance + sides[1].z * sides[0].distance) /
          (sides[0].distance + sides[1].distance);
      }
    }
    let floor, nearest = Infinity;
    for (const p of floorGrid.get(`${Math.floor(position.x)},${Math.floor(position.y)}`) ?? []) {
      const z = p.origin.z - (p.normal.x * (position.x - p.origin.x) +
        p.normal.y * (position.y - p.origin.y)) / p.normal.z;
      const distance = Math.abs(z - expected) + (p.surface === 'ceiling' ? .4 : 0);
      if (z > position.z - .35 || distance >= nearest || !supportedFloorOn(p, position, z)) continue;
      nearest = distance; floor = z;
    }
    return floor;
  }
  function floorAt(position) {
    const direct = directFloorAt(position), expected = position.z - eyeHeight;
    if (direct !== undefined && Math.abs(direct - expected) <= stepLimit(position, direct)) return direct;
    // Close only short gaps bounded by physical floor on opposite sides.
    // Door thresholds and raster seams must not become invisible walls;
    // an exterior edge has no opposite support and remains blocked.
    for (const [dx, dy] of [[1,0], [0,1], [.7071,.7071], [.7071,-.7071]]) {
      const sides = [];
      for (const sign of [-1, 1]) {
        for (let distance = .05; distance <= .30 + 1e-6; distance += .05) {
          const probe = position.clone(); probe.x += sign * dx * distance; probe.y += sign * dy * distance;
          const z = directFloorAt(probe);
          if (z !== undefined && Math.abs(z - expected) <= .35) {
            sides.push({ z, distance, mesh: meshFloorAt(probe, z, .025) !== undefined }); break;
          }
        }
      }
      const limit = sides.some(s => s.mesh) ? .35 : maxStep;
      if (sides.length === 2 && sides.every(s => Math.abs(s.z - expected) <= limit) && Math.abs(sides[0].z - sides[1].z) <= limit) {
        return (sides[0].z * sides[1].distance + sides[1].z * sides[0].distance) /
          (sides[0].distance + sides[1].distance);
      }
    }
    return direct;
  }
  const captureFloors = captures.map(c => ({ z: floorAt(c.point), captureZ: c.point.z }))
    .filter(c => c.z !== undefined && c.captureZ - c.z >= .5 && c.captureZ - c.z <= 2.5);
  function captureEyeHeight(position) {
    const floor = floorAt(position);
    if (floor === undefined) return eyeHeight;
    const heights = captureFloors.filter(c => Math.abs(c.z - floor) < .35)
      .map(c => c.captureZ - c.z).sort((a, b) => a - b);
    const middle = Math.floor(heights.length / 2);
    return heights.length ? (heights[middle] + heights[Math.floor((heights.length - 1) / 2)]) / 2 : eyeHeight;
  }
  function showHeight() {
    heightControl.min = Math.min(1.1, eyeHeight); heightControl.max = Math.max(2, eyeHeight);
    heightControl.step = '.01'; heightControl.value = eyeHeight;
    document.querySelector('#heightValue').textContent = eyeHeight.toFixed(2);
  }
  function reset() {
    renderEyeZ = undefined;
    fallSpeed = fallAcceleration = 0;
    clearInput();
    camera.position.fromArray(config.position);
    if (automaticHeight) eyeHeight = captureEyeHeight(camera.position);
    showHeight();
    const floor = floorAt(camera.position);
    if (floor !== undefined && (!stableHF || Math.abs(camera.position.z - floor - eyeHeight) < .35)) camera.position.z = floor + eyeHeight;
    yaw = Math.atan2(config.forward[1], config.forward[0]); pitch = 0;
  }
  reset();
  const previewPose = (params?.get('pose') || '').split(',').map(Number);
  if (previewPose.length === 5 && previewPose.every(Number.isFinite)) {
    camera.position.fromArray(previewPose); yaw = previewPose[3]; pitch = previewPose[4];
  }
  if (params?.get('diagnostics') === '1') {
    const proof = document.createElement('pre'); proof.id = 'pose-init-proof'; proof.hidden = true;
    proof.textContent = JSON.stringify({search: location.search, params: params.toString(), previewPose,
      position: camera.position.toArray(), yaw, pitch}); document.body.append(proof);
  }
  document.querySelector('#reset').onclick = reset;
  document.querySelector('#mesh').onchange = e => { overlay.visible = e.target.checked; };
  heightControl.oninput = () => {
    if (cameraOwned()) return;
    const value = Number(heightControl.value);
    automaticHeight = false;
    camera.position.z += value - eyeHeight; eyeHeight = value;
    document.querySelector('#heightValue').textContent = value.toFixed(2);
  };
  const initialClassic = !['w21', 'w12e'].includes((params?.get('mode') || params?.get('profile') || '').toLowerCase());
  const photoLayer = createPhotoLayer(renderer, collision, captures, initialClassic, stableHF);
  const measure = typeof installMeasure === 'function' ? installMeasure({ THREE, camera, renderer, collision,
    getDepthMeshes: () => photoLayer.depthMeshes() }) : null;
  window.demoReview ??= {};
  window.demoReview.merge = { modes: ['Tour', 'Walk 3D', 'Measure'], walkUrlOk: true,
    get measureOk() { return !!measure?.state.enabled; } };
  const photoControl = document.querySelector('#photos');
  let previewLayer = ['both', 'photo', 'splat'].includes(params?.get('layer')) ? params.get('layer') : 'both';
  photoControl.checked = previewLayer !== 'splat';
  window.previewView = {
    getPose: () => [...camera.position.toArray(), yaw, pitch],
    getLayer: () => previewLayer,
    setLayer: value => { if (['both', 'photo', 'splat'].includes(value)) {
      previewLayer = value; photoControl.checked = value !== 'splat';
    } },
  };
  window.dispatchEvent(new Event('preview-ready'));
  let lastWalkPosition = camera.position.clone();
  let furnitureCameraOwned = false, furniture = null;
  const aerialState = { active: false, distance: 12, azimuth: 0, elevation: .85 };
  let aerialTransition = null, savedWalk = null, orbitPointer = null;
  const homeBounds = new THREE.Box3().setFromObject(collision);
  const orbitCentre = homeBounds.getCenter(new THREE.Vector3());
  const homeSize = homeBounds.getSize(new THREE.Vector3());
  const neighbourhoodState = { loading: false, visible: false, error: null, info: null };
  let neighbourhoodLayer = null;
  const neighbourhoodControl = document.querySelector('#neighbourhood');
  const neighbourhoodLabel = document.querySelector('#neighbourhoodLabel');
  function updateNeighbourhood() {
    neighbourhoodState.visible = !!neighbourhoodLayer && aerialState.active && neighbourhoodControl.checked;
    if (neighbourhoodLayer) neighbourhoodLayer.visible = neighbourhoodState.visible;
    neighbourhoodLabel.hidden = !neighbourhoodState.visible;
  }
  async function loadNeighbourhood() {
    if (neighbourhoodLayer || neighbourhoodState.loading || typeof addNeighbourhood !== 'function') return;
    neighbourhoodState.loading = true; neighbourhoodState.error = null;
    document.querySelector('#neighbourhoodStatus').textContent = 'Đang tải khu phố…';
    try {
      const { group, info } = await addNeighbourhood(scene, './', THREE);
      group.userData.skipRoofCut = true;
      neighbourhoodLayer = group; neighbourhoodState.info = info;
      neighbourhoodLabel.textContent = info.label;
      document.querySelector('#neighbourhoodStatus').textContent = '';
    } catch (error) {
      neighbourhoodState.error = error.message;
      document.querySelector('#neighbourhoodStatus').textContent = 'Không tải được khu phố. Bật lại để thử.';
    } finally { neighbourhoodState.loading = false; updateNeighbourhood(); }
  }
  neighbourhoodControl.onchange = () => {
    updateNeighbourhood();
    if (aerialState.active && neighbourhoodControl.checked) void loadNeighbourhood();
  };
  window.demoReview.neighbourhood = { state: neighbourhoodState };
  function cameraOwned() { return furnitureCameraOwned || aerialState.active || !!aerialTransition; }
  function orbitPosition() {
    const { distance: r, azimuth: a, elevation: e } = aerialState;
    return orbitCentre.clone().add(new THREE.Vector3(r * Math.cos(e) * Math.cos(a),
      r * Math.cos(e) * Math.sin(a), r * Math.sin(e)));
  }
  function cameraTween(position, target, returning) {
    const aim = camera.clone(); aim.position.copy(position); aim.lookAt(target);
    aerialTransition = { start: performance.now(), from: camera.position.clone(), to: position,
      fromQ: camera.quaternion.clone(), toQ: aim.quaternion.clone(), returning };
  }
  function enterAerial() {
    if (cameraOwned()) return false;
    clearInput(); measure?.setEnabled(false);
    furniture?.setMode(false);
    savedWalk = { position: camera.position.clone(), yaw, pitch, far: camera.far };
    lastWalkPosition.copy(savedWalk.position);
    aerialState.active = true; aerialState.azimuth = yaw + Math.PI;
    aerialState.distance = Math.max(8, Math.hypot(homeSize.x, homeSize.y) * .9);
    camera.far = 1000; camera.updateProjectionMatrix();
    cameraTween(orbitPosition(), orbitCentre, false);
    document.querySelector('#aerial').textContent = 'Về đi bộ';
    document.querySelector('#aerialZoom').hidden = false; joystick.hidden = true;
    updateNeighbourhood();
    if (neighbourhoodControl.checked) void loadNeighbourhood();
    return true;
  }
  function exitAerial() {
    if (!aerialState.active || !savedWalk) return false;
    clearInput(); orbitPointer = null; aerialState.active = false;
    const p = savedWalk.position.clone(), d = new THREE.Vector3(Math.cos(savedWalk.yaw) * Math.cos(savedWalk.pitch),
      Math.sin(savedWalk.yaw) * Math.cos(savedWalk.pitch), Math.sin(savedWalk.pitch));
    cameraTween(p, p.clone().add(d), true);
    document.querySelector('#aerial').textContent = 'Nhìn từ trên';
    document.querySelector('#aerialZoom').hidden = true;
    updateNeighbourhood();
    return true;
  }
  function zoomAerial(factor) {
    if (!aerialState.active || aerialTransition) return;
    aerialState.distance = THREE.MathUtils.clamp(aerialState.distance * factor, 2, 240);
  }
  function updateAerial(time) {
    if (aerialTransition) {
      const transition = aerialTransition;
      const t = THREE.MathUtils.clamp((time - transition.start) / 400, 0, 1), eased = t * t * (3 - 2 * t);
      camera.position.lerpVectors(transition.from, transition.to, eased);
      camera.quaternion.slerpQuaternions(transition.fromQ, transition.toQ, eased);
      if (t === 1) {
        aerialTransition = null;
        if (transition.returning) {
          yaw = savedWalk.yaw; pitch = savedWalk.pitch; renderEyeZ = camera.position.z;
          camera.far = savedWalk.far; camera.updateProjectionMatrix();
          clearInput(); joystick.hidden = !controls.checked; savedWalk = null;
        }
      }
    } else if (aerialState.active) {
      camera.position.copy(orbitPosition()); camera.lookAt(orbitCentre);
    }
  }
  document.querySelector('#aerial').onclick = () => aerialState.active ? exitAerial() : enterAerial();
  document.querySelector('#walkReturn').onclick = exitAerial;
  document.querySelector('#zoomIn').onclick = () => zoomAerial(1 / 1.3);
  document.querySelector('#zoomOut').onclick = () => zoomAerial(1.3);
  renderer.domElement.addEventListener('wheel', e => {
    if (!aerialState.active) return;
    e.preventDefault(); zoomAerial(Math.exp(THREE.MathUtils.clamp(e.deltaY, -1000, 1000) * .001));
  }, { passive: false });
  window.demoReview.aerial = { state: aerialState, enter: enterAerial, exit: exitAerial, zoom: zoomAerial };
  const roofState = { hidden: false, cutZ: 1e6 };
  const roofPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 1e6);
  const roofUniform = typeof dyno === 'undefined' ? null : dyno.dynoFloat(1e6);
  const roofModifier = roofUniform && dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    const { center, opacity } = dyno.splitGsplat(gsplat).outputs;
    return { gsplat: dyno.combineGsplat({ gsplat, opacity: dyno.select(
      dyno.greaterThan(dyno.split(center).outputs.z, roofUniform), dyno.dynoConst('float', 0), opacity) }) };
  });
  const roofSplats = new WeakSet(), roofMaterials = new WeakSet();
  function applyRoof(root = scene) {
    if (root.userData?.skipRoofCut) return;
    if (roofModifier && root instanceof SplatMesh && !roofSplats.has(root)) {
      root.worldModifiers = [...(root.worldModifiers ?? []), roofModifier];
      root.updateGenerator(); roofSplats.add(root);
    }
    for (const mat of Array.isArray(root.material) ? root.material : [root.material]) {
      if (mat?.isMeshBasicMaterial && !roofMaterials.has(mat)) {
        mat.clippingPlanes = [roofPlane]; mat.needsUpdate = true; roofMaterials.add(mat);
      }
    }
    for (const child of root.children ?? []) applyRoof(child);
  }
  function updateRoof(position = lastWalkPosition) {
    let cutZ = 1e6;
    if (roofState.hidden) {
      const floor = floorAt(position) ?? position.z - eyeHeight;
      const candidates = holes.planes.filter(p => p.surface === 'ceiling' && Math.abs(p.normal[2]) > .3)
        .map(p => {
          const z = p.origin[2] - ((position.x - p.origin[0]) * p.normal[0] +
            (position.y - p.origin[1]) * p.normal[1]) / p.normal[2];
          const delta = [position.x - p.origin[0], position.y - p.origin[1], z - p.origin[2]];
          const u = delta.reduce((s, d, i) => s + d * p.u[i], 0), v = delta.reduce((s, d, i) => s + d * p.v[i], 0);
          return { z, inside: u >= 0 && v >= 0 && u <= p.width * p.step && v <= p.height * p.step };
        }).filter(p => p.z > floor + 1.7 && p.z < floor + 5);
      const local = candidates.filter(p => p.inside);
      const ceilings = local.length ? local : candidates;
      cutZ = (ceilings.length ? Math.min(...ceilings.map(p => p.z)) : floor + 3.2) - .04;
    }
    if (Math.abs(cutZ - roofState.cutZ) > .01) {
      roofState.cutZ = cutZ; roofPlane.constant = cutZ;
      if (roofUniform) roofUniform.value = cutZ;
      photoLayer.setRoof(cutZ);
      scene.traverse(o => { if (roofSplats.has(o)) o.updateGenerator(); });
    }
    applyRoof();
  }
  const roofButton = document.querySelector('#roof');
  function setRoofHidden(hidden) {
    roofState.hidden = !!hidden; roofButton.textContent = hidden ? 'Hiện mái' : 'Ẩn mái';
    roofButton.setAttribute?.('aria-pressed', String(hidden)); updateRoof();
  }
  roofButton.onclick = () => setRoofHidden(!roofState.hidden);
  window.demoReview.roof = { state: roofState, setHidden: setRoofHidden };
  function suspendWalk(value) {
    furnitureCameraOwned = value; clearInput();
    joystick.hidden = value || !controls.checked;
  }
  function setCameraPose(position, target) {
    const direction = target.clone().sub(position).normalize();
    camera.position.copy(position); yaw = Math.atan2(direction.y, direction.x);
    pitch = Math.asin(THREE.MathUtils.clamp(direction.z, -1, 1));
    renderEyeZ = position.z; camera.lookAt(target);
  }
  // Tool selection saves a walk pose, including when opened from aerial mode.
  addEventListener('viewer-tool', e => {
    if (e.detail !== 'furniture' || !savedWalk) return;
    const pose = savedWalk;
    aerialState.active = false; aerialTransition = null; orbitPointer = null; savedWalk = null;
    camera.position.copy(pose.position); yaw = pose.yaw; pitch = pose.pitch; renderEyeZ = pose.position.z;
    camera.far = pose.far; camera.updateProjectionMatrix();
    camera.lookAt(pose.position.clone().add(new THREE.Vector3(Math.cos(yaw) * Math.cos(pitch),
      Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch))));
    document.querySelector('#aerial').textContent = 'Nhìn từ trên';
    document.querySelector('#aerialZoom').hidden = true;
    updateNeighbourhood(); clearInput();
  });
  if (typeof installFurniture === 'function') {
    furniture = installFurniture({ THREE, scene, camera, renderer, collision, walkMesh, splat, SplatMesh, PackedSplats,
      getPhotoEntries: () => photoLayer.entries(), contactGrid, obstacleGrid, suspendWalk, setCameraPose,
      lazy: initialClassic, autoFrame: false, parent: document.querySelector('#furnitureTools') });
    furniture.ready.catch(error => console.error('Furniture unavailable:', error));
  }
  const classicToggleLabel = document.createElement('p');
  classicToggleLabel.innerHTML = '<label><input id="classic0200" type="checkbox"> Bản 02:00 sáng (mượt &amp; liền mạch)</label>';
  photoControl.closest('p')?.after(classicToggleLabel);
  const classicToggle = classicToggleLabel.querySelector('#classic0200');
  if (classicToggle) {
    classicToggle.checked = initialClassic;
    classicToggle.onchange = () => {
      const use0200 = classicToggle.checked;
      photoLayer.setClassic0200(use0200);
      if (use0200) furniture?.deactivateHeavy?.();
      else void furniture?.ensureHeavyActivated?.();
    };
  }
  const helpToggle = document.querySelector('#helpToggle'), helpPanel = document.querySelector('#helpPanel');
  if (params?.get('embed') === '1' || (typeof window !== 'undefined' && window.self !== window.top)) {
    helpPanel.hidden = true; helpToggle.hidden = false;
    helpToggle.onclick = () => {
      helpPanel.hidden = !helpPanel.hidden;
      helpToggle.setAttribute('aria-expanded', String(!helpPanel.hidden));
    };
  }
  function cancelTarget() {
    if (clickTarget) ringFade = .3;
    clickTarget = null;
  }
  function clearInput() {
    resetSteering();
    keys.clear(); velocity.set(0, 0, 0); turnVelocity = 0;
    step = turnStep = moveGesture = turnGesture = moveCarry = null; turnCarry = 0;
    pendingInput = null; joy.set(0, 0); joyPointer = lookPointer = null;
    stick.style.transform = ''; cancelTarget();
  }
  function moveDirection() {
    const f = Number(keys.has('KeyW') || keys.has('ArrowUp')) - Number(keys.has('KeyS') || keys.has('ArrowDown')) - joy.y;
    const r = Number(keys.has('KeyD')) - Number(keys.has('KeyA')) + joy.x;
    const direction = new THREE.Vector3(Math.cos(yaw) * f + Math.sin(yaw) * r,
      Math.sin(yaw) * f - Math.cos(yaw) * r, 0);
    if (direction.lengthSq() > 1) direction.normalize();
    return direction;
  }
  function turnDirection() { return Number(keys.has('ArrowLeft')) - Number(keys.has('ArrowRight')); }
  function typing(target) {
    return target instanceof Element && !!target.closest('input,button,select,textarea,[contenteditable="true"]');
  }
  addEventListener('keydown', e => {
    if (cameraOwned()) return;
    cancelTarget();
    if (typing(e.target)) return;
    if (moveCodes.has(e.code) || turnCodes.has(e.code) || e.code === 'Space') e.preventDefault();
    if (keys.has(e.code)) return;
    keys.add(e.code);
    if (moveCodes.has(e.code)) {
      // Rapid taps add up: keep what the interrupted step had left.
      if (step) moveCarry = { direction: step.direction, distance: remainingOf(step, step.distance) };
      step = null;
      if (!moveGesture) {
        moveGesture = { start: performance.now(), distance: 0, direction: moveDirection().normalize(), blocked: false };
        pendingInput = performance.now();
      }
    }
    if (turnCodes.has(e.code)) {
      if (turnStep) turnCarry = remainingOf(turnStep, turnStep.angle);
      turnStep = null;
      if (!turnGesture) {
        turnGesture = { start: performance.now(), angle: 0, direction: turnDirection() };
        pendingInput = performance.now();
      }
    }
  });
  addEventListener('keyup', e => {
    if (cameraOwned()) { clearInput(); return; }
    keys.delete(e.code);
    if (moveCodes.has(e.code) && moveGesture && ![...moveCodes].some(code => keys.has(code))) {
      const age = (performance.now() - moveGesture.start) / 1000;
      if (age <= .150001 && !moveGesture.blocked) {
        const carry = moveCarry && moveCarry.direction.dot(moveGesture.direction) > .9 ? moveCarry.distance : 0;
        step = { direction: moveGesture.direction, distance: Math.max(0, .30 - moveGesture.distance) + carry,
          elapsed: 0, duration: Math.max(.03, .18 - age), initialSpeed: velocity.length() };
        velocity.set(0, 0, 0);
      }
      moveGesture = null; moveCarry = null;
    }
    if (turnCodes.has(e.code) && turnGesture && ![...turnCodes].some(code => keys.has(code))) {
      const age = (performance.now() - turnGesture.start) / 1000;
      if (age <= .150001) {
        const carry = Math.sign(turnCarry) === turnGesture.direction ? turnCarry : 0;
        turnStep = { angle: turnGesture.direction * Math.max(0, Math.PI / 12 - Math.abs(turnGesture.angle)) + carry,
          elapsed: 0, duration: Math.max(.03, .18 - age), initialSpeed: Math.abs(turnVelocity) };
        turnVelocity = 0;
      }
      turnGesture = null; turnCarry = 0;
    }
  });
  addEventListener('blur', clearInput);
  document.addEventListener('visibilitychange', () => { if (document.hidden) clearInput(); });
  renderer.domElement.style.touchAction = 'none';
  renderer.domElement.addEventListener('pointerdown', e => {
    if (aerialState.active && !aerialTransition && e.button === 0) {
      orbitPointer = { id: e.pointerId, x: e.clientX, y: e.clientY };
      renderer.domElement.setPointerCapture(e.pointerId); return;
    }
    if (cameraOwned()) return;
    if (lookPointer || e.button !== 0) return;
    lookPointer = { id: e.pointerId, x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, dragged: false };
    renderer.domElement.setPointerCapture(e.pointerId);
  });
  renderer.domElement.addEventListener('pointermove', e => {
    if (orbitPointer?.id === e.pointerId) {
      aerialState.azimuth -= (e.clientX - orbitPointer.x) * .005;
      aerialState.elevation = THREE.MathUtils.clamp(aerialState.elevation + (e.clientY - orbitPointer.y) * .005, .08, 1.53);
      orbitPointer.x = e.clientX; orbitPointer.y = e.clientY; return;
    }
    if (cameraOwned()) return;
    const p = lookPointer; if (!p || p.id !== e.pointerId) return;
    if (Math.hypot(e.clientX - p.startX, e.clientY - p.startY) > 5) p.dragged = true;
    if (p.dragged) {
      yaw -= (e.clientX - p.x) * .003;
      pitch = THREE.MathUtils.clamp(pitch - (e.clientY - p.y) * .003, -80 * Math.PI / 180, 80 * Math.PI / 180);
    }
    p.x = e.clientX; p.y = e.clientY;
  });
  renderer.domElement.addEventListener('pointerup', e => {
    if (orbitPointer?.id === e.pointerId) { orbitPointer = null; return; }
    if (cameraOwned()) return;
    const p = lookPointer; if (!p || p.id !== e.pointerId) return;
    lookPointer = null;
    if (p.dragged) return;
    const rect = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1,
      1 - (e.clientY - rect.top) / rect.height * 2), camera);
    ray.near = .001; ray.far = 150;
    const hit = ray.intersectObject(collision, true)[0];
    // The nearest hit must be a supported floor; never click through a wall.
    if (!hit || Math.abs(hit.face.normal.z) <= .65 || !supportedFloor(hit.point, hit.point.z)) return;
    keys.clear(); step = turnStep = moveGesture = turnGesture = null;
    velocity.set(0, 0, 0); turnVelocity = 0;
    clickTarget = hit.point.clone(); pendingInput = performance.now();
    targetRing.position.copy(hit.point); targetRing.position.z += .025;
    targetRing.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), hit.face.normal.clone().normalize());
    targetRing.visible = true; targetRing.material.opacity = .85; ringFade = 0;
  });
  renderer.domElement.addEventListener('lostpointercapture', () => { lookPointer = orbitPointer = null; });
  renderer.domElement.addEventListener('pointercancel', () => { lookPointer = orbitPointer = null; });
  function releaseJoystick() {
    joyPointer = null; joy.set(0, 0); stick.style.transform = '';
  }
  joystick.addEventListener('pointerdown', e => {
    if (cameraOwned()) return;
    if (joyPointer !== null) return;
    e.preventDefault(); cancelTarget(); step = null; moveGesture = null;
    joyPointer = e.pointerId; pendingInput = performance.now();
    joystick.setPointerCapture(e.pointerId); updateJoystick(e);
  });
  function updateJoystick(e) {
    if (joyPointer !== e.pointerId) return;
    const rect = joystick.getBoundingClientRect();
    joy.set((e.clientX - rect.left - rect.width / 2) / 42, (e.clientY - rect.top - rect.height / 2) / 42);
    if (joy.length() > 1) joy.normalize();
    stick.style.transform = `translate(${joy.x * 42}px, ${joy.y * 42}px)`;
  }
  joystick.addEventListener('pointermove', updateJoystick);
  for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    joystick.addEventListener(event, e => { if (e.pointerId === joyPointer) releaseJoystick(); });
  }
  const touchDevice = matchMedia('(any-pointer: coarse)').matches || navigator.maxTouchPoints > 0;
  const controls = document.querySelector('#controls'); controls.checked = touchDevice;
  joystick.hidden = !controls.checked;
  controls.onchange = () => { joystick.hidden = !controls.checked || cameraOwned(); releaseJoystick(); };
  function closestOnEdge(point, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const t = THREE.MathUtils.clamp(((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy || 1), 0, 1);
    return a.clone().lerp(b, t);
  }
  function clipHeight(points, z, above) {
    const result = [];
    for (let i = 0; i < points.length; i++) {
      const a = points[i], b = points[(i + 1) % points.length];
      const aInside = above ? a.z >= z : a.z <= z, bInside = above ? b.z >= z : b.z <= z;
      if (aInside) result.push(a);
      if (aInside !== bInside) result.push(a.clone().lerp(b, (z - a.z) / (b.z - a.z)));
    }
    return result;
  }
  const ghost = document.querySelector('#ghost');
  const ghostNote = document.querySelector('#ghostNote');
  let pendingSolid = false;
  function updatePassThrough() {
    if (pendingSolid && !fallSpeed && !canMove(new THREE.Vector3(), true).normal) pendingSolid = false;
    ghostNote.textContent = pendingSolid ? 'Inside geometry: walls stay passable until you step clear' : '';
  }
  ghost.onchange = () => {
    if (cameraOwned()) return;
    resetSteering();
    pendingSolid = !ghost.checked && !!canMove(new THREE.Vector3(), true).normal;
    updatePassThrough();
  };
  function canMove(delta, solid = false) {
    const next = camera.position.clone().add(delta), floor = floorAt(next);
    const currentFloor = camera.position.z - eyeHeight;
    const allowedStep = stepLimit(next, floor);
    if (!solid && (ghost?.checked || pendingSolid)) {
      if (floor !== undefined && Math.abs(floor - currentFloor) <= allowedStep + 1e-6 &&
          (!fallSpeed || floor >= currentFloor)) return { floor };
      const below = floorBelowAt(next);
      // Preserve W6 recovery for a solid-mode start already outside all support.
      if (below === undefined && pendingSolid && !ghost.checked && floorBelowAt(camera.position) === undefined) return { floor: currentFloor };
      return below === undefined ? { normal: delta.clone().negate().normalize() } : { floor: currentFloor, fall: below };
    }
    const validFloor = floor !== undefined && Math.abs(floor - currentFloor) <= allowedStep + 1e-6 && supportedFloor(next, floor);
    const bodyFloor = validFloor ? Math.max(currentFloor, floor) : currentFloor;
    let footClearance = maxStep;
    if (validFloor) for (const [dx, dy] of [[1,0], [-1,0], [0,1], [0,-1], [.7071,.7071], [-.7071,.7071], [.7071,-.7071], [-.7071,-.7071]]) {
      const probe = next.clone(); probe.x += dx * (bodyRadius + .05); probe.y += dy * (bodyRadius + .05);
      const tread = meshFloorAt(probe, bodyFloor);
      if (tread !== undefined && tread > bodyFloor + .025) { footClearance = .35; break; }
    }
    let contact = null, deepest = 0;
    for (let x = Math.floor((next.x - bodyRadius) / .1); x <= Math.floor((next.x + bodyRadius) / .1); x++) {
      for (let y = Math.floor((next.y - bodyRadius) / .1); y <= Math.floor((next.y + bodyRadius) / .1); y++) {
        for (const cell of obstacleGrid.get(`${x},${y}`) ?? []) {
          if (cell.high < bodyFloor + footClearance || cell.low > bodyFloor + eyeHeight) continue;
          const nearest = new THREE.Vector3(THREE.MathUtils.clamp(next.x, x * .1, (x + 1) * .1),
            THREE.MathUtils.clamp(next.y, y * .1, (y + 1) * .1), next.z);
          let normal = next.clone().sub(nearest), distance = normal.length();
          if (distance < 1e-10) {
            const faces = [[next.x - x * .1, -1, 0], [(x + 1) * .1 - next.x, 1, 0],
              [next.y - y * .1, 0, -1], [(y + 1) * .1 - next.y, 0, 1]].sort((a,b) => a[0] - b[0]);
            distance = -faces[0][0]; normal.set(faces[0][1], faces[0][2], 0);
          } else normal.divideScalar(distance);
          if (bodyRadius - distance > deepest + 1e-7) { deepest = bodyRadius - distance; contact = normal; }
        }
      }
    }
    for (const item of contactGrid.get(`${Math.floor(next.x)},${Math.floor(next.y)}`) ?? []) {
      if (next.x < item.minX || next.x > item.maxX || next.y < item.minY || next.y > item.maxY) continue;
      let normal, distance;
      if (item.boundary) {
        const closest = closestOnEdge(next, item.a, item.b);
        if (Math.abs(closest.z - (floor ?? currentFloor)) > .15) continue;
        distance = Math.hypot(next.x - closest.x, next.y - closest.y);
        if (distance >= bodyRadius) continue;
        // Captured neighbours, including a supported step, are not a mask edge.
        const across = closest.clone().addScaledVector(item.normal, -.001); across.z = camera.position.z;
        const acrossFloor = floorAt(across);
        if (acrossFloor !== undefined && Math.abs(acrossFloor - closest.z) <= stepLimit(across, acrossFloor) + 1e-6 && supportedFloor(across, acrossFloor)) continue;
        normal = item.normal;
        if (next.clone().sub(closest).dot(normal) < 0) distance = -distance;
      } else {
        const bottom = bodyFloor + footClearance + 1e-6, top = bodyFloor + eyeHeight;
        if (item.maxZ < bottom || item.minZ > top) continue;
        const polygon = clipHeight(clipHeight(item.points, bottom, true), top, false);
        if (polygon.length < 2) continue;
        let nearest = null, nearestDistance = Infinity, inside = false;
        for (let i = 0; i < polygon.length; i++) {
          const a = polygon[i], b = polygon[(i + 1) % polygon.length];
          const closest = closestOnEdge(next, a, b), d = Math.hypot(next.x - closest.x, next.y - closest.y);
          if (d < nearestDistance) { nearestDistance = d; nearest = closest; }
          if ((a.y > next.y) !== (b.y > next.y) && next.x < (b.x - a.x) * (next.y - a.y) / (b.y - a.y) + a.x) inside = !inside;
        }
        distance = inside ? -nearestDistance : nearestDistance;
        normal = next.clone().sub(nearest); normal.z = 0;
        if (inside) normal.negate();
        if (normal.lengthSq() < 1e-12) normal.copy(delta).negate();
        normal.normalize();
      }
      const penetration = bodyRadius - distance;
      if (penetration > deepest + 1e-7) { deepest = penetration; contact = normal.clone(); }
    }
    if (contact) return { normal: contact };
    if (!validFloor) return { normal: delta.clone().negate().normalize() };
    return { floor };
  }
  let sweepOrigin = '';
  const sweepResults = new Map();
  function movementCheck(delta, solid = false) {
    const check = canMove(delta, solid);
    if (check.normal || (!solid && (ghost.checked || pendingSolid))) return check;
    // On a descent, the old floor can clip a wall above the final body base.
    // Validate the body after settling onto its destination tread as well.
    const origin = camera.position.clone();
    try {
      camera.position.add(delta); camera.position.z = check.floor + eyeHeight;
      const settled = canMove(new THREE.Vector3(), true);
      return settled.normal ? settled : check;
    } finally { camera.position.copy(origin); }
  }
  function sweep(direction, length = .6) {
    const origin = camera.position.clone();
    const pose = `${origin.x},${origin.y},${origin.z},${eyeHeight}`;
    if (pose !== sweepOrigin) { sweepOrigin = pose; sweepResults.clear(); }
    const key = `${direction.x},${direction.y},${length}`;
    if (sweepResults.has(key)) return sweepResults.get(key);
    let distance = 0, normal = null;
    const count = Math.ceil(length / .04), part = direction.clone().multiplyScalar(length / count);
    try {
      for (let i = 0; i < count; i++) {
        const check = canMove(part, true);
        if (check.normal) { normal = check.normal; break; }
        camera.position.add(part); camera.position.z = check.floor + eyeHeight;
        distance += length / count;
      }
    } finally { camera.position.copy(origin); }
    const result = { distance, normal };
    if (sweepResults.size >= 24) sweepResults.clear();
    sweepResults.set(key, result);
    return result;
  }
  function move(delta, straight = false, dt = 1 / 60) {
    updatePassThrough();
    const before = camera.position.clone();
    let intended = delta.clone(), mode = 'free';
    const length = delta.length();
    if (ghost.checked || pendingSolid || length < 1e-10) resetSteering();
    else {
      const input = delta.clone().normalize();
      if (steering.input && steering.input.dot(input) < Math.cos(Math.PI / 6)) resetSteering();
      if (!steering.input) steering.input = input.clone();
      const forward = { ...sweep(input) };
      // A short frame can hit a seam that a longer sweep sample straddles.
      const frameContact = movementCheck(delta, true).normal;
      if (frameContact) { forward.normal = frameContact; forward.distance = 0; }
      const tangent = input.clone();
      if (forward.normal) tangent.addScaledVector(forward.normal, -Math.min(0, tangent.dot(forward.normal)));
      // A corner may leave a tangent at the first contact and none at the next.
      // Measure the same three-contact slide before deciding whether to steer.
      let slideFraction = 1;
      if (forward.normal) {
        const saved = camera.position.clone(), probeLength = Math.min(length, .04);
        const probe = input.clone().multiplyScalar(probeLength);
        try {
          for (let attempt = 0; attempt < 3 && probe.lengthSq() > 1e-14; attempt++) {
            const check = movementCheck(probe, true);
            if (!check.normal) { slideFraction = probe.dot(input) > 0 ? probe.length() / probeLength : 0; break; }
            probe.addScaledVector(check.normal, -Math.min(0, probe.dot(check.normal)));
            slideFraction = 0;
          }
        } finally { camera.position.copy(saved); }
      }
      let angle = 0;
      if (forward.normal && (tangent.length() < .35 || slideFraction < .35 || steering.side)) {
        const candidates = [];
        for (const degrees of [25, 50, 75]) for (const side of [-1, 1]) {
          const radians = side * degrees * Math.PI / 180;
          const heading = input.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), radians);
          const result = sweep(heading);
          if (result.distance >= .6 - 1e-8 &&
              !movementCheck(heading.clone().multiplyScalar(length * .8), true).normal) {
            candidates.push({ side, radians, degrees, heading });
          }
        }
        let allowed = steering.side ? candidates.filter(c => c.side === steering.side) : candidates;
        if (!allowed.length && steering.side) {
          // Keep a previously cleared heading until it actually blocks. A
          // shorter remaining lookahead alone must not spend the one side flip.
          const heading = input.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), steering.target);
          if (!movementCheck(heading.clone().multiplyScalar(length * .8), true).normal) {
            allowed = [{ side: steering.side, radians: steering.target,
              degrees: Math.abs(steering.target * 180 / Math.PI), heading }];
          }
        }
        if (!allowed.length && steering.side && steering.flips < 1) allowed = candidates;
        if (allowed.length) {
          const smallest = Math.min(...allowed.map(c => c.degrees));
          allowed = allowed.filter(c => c.degrees === smallest);
          if (allowed.length > 1) allowed.sort((a,b) => sweep(b.heading, 1.2).distance - sweep(a.heading, 1.2).distance);
          const chosen = allowed[0];
          if (steering.side && chosen.side !== steering.side) steering.flips++;
          steering.side = chosen.side; steering.target = angle = chosen.radians; mode = 'steer';
        } else mode = 'blocked';
      } else if (forward.normal) mode = 'slide';
      steering.angle += (angle - steering.angle) * (1 - Math.exp(-dt / .04));
      intended.applyAxisAngle(new THREE.Vector3(0, 0, 1), steering.angle);
      if (mode === 'steer' || Math.abs(steering.angle) > 1e-6) intended.multiplyScalar(.8);
      // A finite stopping distance prevents a hard hit at a dead end.
      if (mode === 'blocked') intended.multiplyScalar(Math.min(1, forward.distance / .12));
      Object.assign(walkState.steer, { mode, side: steering.side, angleDeg: steering.angle * 180 / Math.PI });
    }
    const steps = Math.max(1, Math.ceil(delta.length() / .04));
    const part = intended.divideScalar(steps);
    let blocked = false;
    const normals = [];
    for (let i = 0; i < steps; i++) {
      const slide = part.clone();
      for (let attempt = 0; attempt < 3 && slide.lengthSq() > 1e-14; attempt++) {
        if (slide.dot(part) <= 0 || slide.dot(delta) <= 0) break;
        const check = movementCheck(slide);
        if (!check.normal) {
          camera.position.add(slide); camera.position.z = check.floor + eyeHeight;
          updatePassThrough();
          break;
        }
        blocked = true; normals.push(check.normal);
        slide.addScaledVector(check.normal, -Math.min(0, slide.dot(check.normal)));
      }
    }
    updatePassThrough();
    return { distance: Math.hypot(camera.position.x - before.x, camera.position.y - before.y), blocked, normals };
  }
  // The initial reset happens before the toggle is wired. Also recover on Reset.
  ghost.onchange();
  document.querySelector('#reset').onclick = () => { if (!cameraOwned()) { reset(); ghost.onchange(); } };
  // Integrate a linear acceleration ramp, including any time spent at full speed.
  function accelerate(current, target, rate, dt) {
    const difference = target.clone().sub(current), distance = difference.length();
    if (!distance) return current.clone().multiplyScalar(dt);
    const rampTime = Math.min(dt, distance / rate), before = current.clone();
    current.addScaledVector(difference, Math.min(1, rate * dt / distance));
    return before.add(current).multiplyScalar(rampTime / 2).addScaledVector(current, dt - rampTime);
  }
  function easeOf(action, amount) {
    const slope = amount ? Math.min(3, action.initialSpeed * action.duration / Math.abs(amount)) : 0;
    return t => (slope - 2) * t ** 3 + (3 - 2 * slope) * t ** 2 + slope * t;
  }
  function easedPortion(action, amount, dt) {
    const ease = easeOf(action, amount);
    const before = action.elapsed / action.duration;
    action.elapsed = Math.min(action.duration, action.elapsed + dt);
    return amount * (ease(action.elapsed / action.duration) - ease(before));
  }
  function remainingOf(action, amount) {
    return amount * (1 - easeOf(action, amount)(action.elapsed / action.duration));
  }
  let previous = performance.now(), frames = 0, sampleStart = previous;
  const samples = []; let lastFps = 0;
  renderer.setAnimationLoop(time => {
    const dt = Math.max(0, Math.min(.05, (time - previous) / 1000)); previous = time;
    furniture?.update(time);
    let groundedZ;
    if (!cameraOwned()) {
    const oldYaw = yaw;
    if (turnStep) {
      yaw += easedPortion(turnStep, turnStep.angle, dt);
      if (turnStep.elapsed >= turnStep.duration) turnStep = null;
    } else {
      const target = turnDirection() * Math.PI / 2;
      const rate = Math.PI / 2 / (target ? .12 : .10);
      const change = THREE.MathUtils.clamp(target - turnVelocity, -rate * dt, rate * dt);
      const rampTime = Math.abs(change) / rate;
      yaw += (turnVelocity + change / 2) * rampTime + (turnVelocity + change) * (dt - rampTime);
      turnVelocity += change;
    }
    if (turnGesture) turnGesture.angle += yaw - oldYaw;
    let delta;
    const straight = !!clickTarget;
    if (step) {
      delta = step.direction.clone().multiplyScalar(easedPortion(step, step.distance, dt));
      if (step.elapsed >= step.duration) step = null;
    } else {
      let direction = moveDirection(), speed = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 2.5 : 1.2;
      if (clickTarget) {
        direction = clickTarget.clone().sub(camera.position); direction.z = 0;
        // Brake to the target without crossing it.
        speed = Math.min(1.2, Math.sqrt(2 * (1.2 / .10) * direction.length()));
        direction.normalize();
      }
      const target = direction.multiplyScalar(speed);
      if (target.lengthSq()) stopRate = Math.max(1.2, velocity.length(), speed) / .10;
      const rate = target.lengthSq() ? Math.max(1.2, speed) / .12 : stopRate;
      delta = accelerate(velocity, target, rate, dt);
      if (clickTarget) {
        const remaining = Math.hypot(clickTarget.x - camera.position.x, clickTarget.y - camera.position.y);
        if (delta.length() > remaining) delta.setLength(remaining);
      }
    }
    if (!step && !clickTarget && !moveDirection().lengthSq()) resetSteering();
    const movement = move(delta, straight, dt);
    const fall = ghost.checked || pendingSolid ? canMove(new THREE.Vector3()).fall : undefined;
    if (fall !== undefined) {
      const drop = camera.position.z - eyeHeight - fall;
      if (!fallSpeed) fallAcceleration = Math.max(9.8, 8 * drop);
      const descent = fallSpeed * dt + .5 * fallAcceleration * dt * dt;
      camera.position.z -= Math.min(drop, descent);
      fallSpeed = descent >= drop ? 0 : fallSpeed + fallAcceleration * dt;
    } else fallSpeed = 0;
    if (!stableHF && automaticHeight && movement.distance > 0) {
      const value = captureEyeHeight(camera.position);
      camera.position.z += value - eyeHeight; eyeHeight = value; showHeight();
    }
    actualSpeed = dt ? movement.distance / dt : 0;
    if (moveGesture) {
      moveGesture.distance += movement.distance;
      moveGesture.blocked ||= movement.blocked;
    }
    if (movement.blocked) {
      step = null;
    }
    if (clickTarget && Math.hypot(clickTarget.x - camera.position.x, clickTarget.y - camera.position.y) < .001) {
      cancelTarget(); velocity.set(0, 0, 0);
    }
    if (pendingInput !== null && (movement.distance > 0 || yaw !== oldYaw)) {
      lastInputToMoveMs = Math.max(0, performance.now() - pendingInput); pendingInput = null;
    }
    if (ringFade > 0) {
      ringFade = Math.max(0, ringFade - dt); targetRing.material.opacity = .85 * ringFade / .3;
      if (!ringFade) targetRing.visible = false;
    }
    Object.assign(walkState, { pos: camera.position.toArray(), yaw, pitch, speed: actualSpeed,
      moving: actualSpeed > .0001 || Math.abs(yaw - oldYaw) > .0001,
      lastInputToMoveMs, clickTarget: clickTarget ? clickTarget.toArray() : null });
    groundedZ = camera.position.z;
    lastWalkPosition.copy(camera.position);
    renderEyeZ = fall !== undefined || renderEyeZ === undefined ? groundedZ : renderEyeZ + (groundedZ - renderEyeZ) * (1 - Math.exp(-dt / .05));
    camera.position.z = renderEyeZ;
    camera.lookAt(camera.position.clone().add(new THREE.Vector3(
      Math.cos(yaw) * Math.cos(pitch), Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch))));
    } else {
      updateAerial(time);
      Object.assign(walkState, { speed: 0, moving: false, clickTarget: null });
    }
    if (roofState.hidden) updateRoof();
    if (previewLayer !== 'splat' && photoControl.checked) photoLayer.update(camera, true);
    spark.material.stencilWrite = stableHF && previewLayer !== 'splat' && photoControl.checked;
    if (stableHF && previewLayer !== 'splat' && photoControl.checked) {
      renderer.clear();
      photoLayer.render(camera);
      if (previewLayer !== 'photo') {
        const autoClear = renderer.autoClear;
        try { renderer.autoClear = false; renderer.render(scene, camera); }
        finally { renderer.autoClear = autoClear; }
      }
    } else {
      if (previewLayer === 'photo') renderer.clear(); else renderer.render(scene, camera);
      if (previewLayer !== 'splat' && photoControl.checked) photoLayer.render(camera);
    }
    furniture?.render(); measure?.render(); frames++;
    if (groundedZ !== undefined) camera.position.z = groundedZ;
    if (time - sampleStart >= 1000) {
      lastFps = frames * 1000 / (time - sampleStart);
      walkState.fps = lastFps;
      samples.push(lastFps); frames = 0; sampleStart = time;
      const cost = photoLayer.state.gpuMs;
      status.textContent = `${config.smoke ? 'CPU smoke asset · ' : ''}${lastFps.toFixed(1)} FPS · ${innerWidth} × ${innerHeight}` +
        (captures.length ? ` · photo GPU ${cost === null ? 'unmeasured' : cost.toFixed(2) + ' ms'}` : '');
    }
  });
  document.querySelector('#save').onclick = () => {
    const data = { utc: new Date().toISOString(), user_agent: navigator.userAgent,
      viewport: [innerWidth, innerHeight], pixel_ratio: renderer.getPixelRatio(), fps_samples: samples,
      fps_mean: samples.length ? samples.reduce((a, b) => a + b, 0) / samples.length : null,
      photo_projection: photoLayer.state,
      note: 'Actual desktop samples; visual alignment and collision still require human inspection.' };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = 'browser_fps.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  addEventListener('resize', () => { renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); });
} catch (error) {
  status.textContent = `Load failed: ${error.message}. Serve this folder over HTTP; see run.log.`;
  console.error(error);
}
