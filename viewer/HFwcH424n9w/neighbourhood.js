/** W18 GLB subset loader. Pass the viewer's THREE instance and model-frame scene.
 * baseUrl is the directory containing neighbourhood.json and neighbourhood.glb.
 * Display info.label verbatim beside the layer toggle (W17 owns that UI).
 */
export async function addNeighbourhood(scene, baseUrl, THREE) {
  if (!scene?.add || !THREE?.Group) throw new TypeError('A scene and its THREE instance are required');
  const base = String(baseUrl).replace(/\/?$/, '/');
  const get = async (name) => {
    const response = await fetch(base + name);
    if (!response.ok) throw new Error(`Neighbourhood ${name}: HTTP ${response.status}`);
    return response;
  };
  const [info, raw] = await Promise.all([
    get('neighbourhood.json').then(r => r.json()),
    get('neighbourhood.glb').then(r => r.arrayBuffer()),
  ]);
  if (info.schema !== 'w18-neighbourhood-v1' || info.frame?.up !== 'Z' ||
      info.frame?.units !== 'metres' || !info.label?.startsWith('Neighbourhood: Vexcel aerial DSM + orthophoto (') ||
      !info.label.endsWith('; not captured by Matterport')) throw new Error('Unsupported neighbourhood provenance/frame');
  if (raw.byteLength !== info.bytes || raw.byteLength > 15_000_000) throw new Error('Neighbourhood size mismatch');
  if (globalThis.crypto?.subtle) {
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', raw));
    const hash = Array.from(digest, b => b.toString(16).padStart(2, '0')).join('');
    if (hash !== info.sha256) throw new Error('Neighbourhood checksum mismatch');
  }
  const header = new DataView(raw);
  if (header.getUint32(0, true) !== 0x46546c67 || header.getUint32(4, true) !== 2 ||
      header.getUint32(8, true) !== raw.byteLength) throw new Error('Invalid neighbourhood GLB');
  const jsonSize = header.getUint32(12, true);
  if (header.getUint32(16, true) !== 0x4e4f534a || 28 + jsonSize > raw.byteLength ||
      header.getUint32(24 + jsonSize, true) !== 0x004e4942) throw new Error('Invalid GLB chunks');
  const doc = JSON.parse(new TextDecoder().decode(new Uint8Array(raw, 20, jsonSize)));
  const binOffset = 28 + jsonSize;
  const binSize = header.getUint32(20 + jsonSize, true);
  if (binOffset + binSize !== raw.byteLength || doc.asset?.version !== '2.0' ||
      doc.extensionsRequired?.length || doc.buffers?.length !== 1 || doc.buffers[0].uri ||
      doc.nodes?.some(n => n.matrix || n.rotation || n.translation || n.scale || n.children)) {
    throw new Error('Unsupported neighbourhood GLB structure');
  }
  const slice = (index) => {
    const view = doc.bufferViews[index];
    const offset = view?.byteOffset || 0;
    if (!view || view.buffer !== 0 || view.byteStride || offset < 0 ||
        view.byteLength < 0 || offset + view.byteLength > binSize) throw new Error('Invalid GLB buffer view');
    return {view, offset: binOffset + offset};
  };
  const attribute = (index, type, component) => {
    const a = doc.accessors[index];
    if (!a || a.type !== type || a.componentType !== component || a.sparse || a.normalized ||
        !Number.isSafeInteger(a.count) || a.count <= 0) throw new Error('Unsupported GLB accessor');
    const {view, offset} = slice(a.bufferView);
    const width = {SCALAR: 1, VEC2: 2, VEC3: 3}[type];
    const start = a.byteOffset || 0;
    if (start < 0 || start % 4 || start + a.count * width * 4 > view.byteLength) throw new Error('Invalid GLB accessor bounds');
    const ArrayType = component === 5125 ? Uint32Array : Float32Array;
    return new THREE.BufferAttribute(new ArrayType(raw, offset + start, a.count * width), width);
  };
  const resources = [];
  const group = new THREE.Group();
  group.name = `Neighbourhood ${info.sid}`;
  // glTF stores Y up. This single conversion restores captures.json's Z-up frame.
  group.rotation.x = Math.PI / 2;
  group.userData.neighbourhood = info;
  try {
    const textures = [];
    for (const image of doc.images) {
      if (image.uri || !['image/jpeg', 'image/png'].includes(image.mimeType)) throw new Error('Unsupported embedded image');
      const {view, offset} = slice(image.bufferView);
      const blob = new Blob([new Uint8Array(raw, offset, view.byteLength)], {type: image.mimeType});
      const bitmap = await createImageBitmap(blob, {imageOrientation: 'none', premultiplyAlpha: 'none'});
      resources.push({dispose: () => bitmap.close()});
      const texture = new THREE.Texture(bitmap);
      resources.push(texture);
      texture.flipY = false;
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.needsUpdate = true;
      textures.push(texture);
    }
    const materials = doc.materials.map(m => {
      const t = doc.textures[m.pbrMetallicRoughness.baseColorTexture.index];
      const texture = textures[t.source];
      const sampler = doc.samplers?.[t.sampler];
      if (!texture) throw new Error('Missing GLB texture');
      texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.magFilter = sampler?.magFilter === 9728 ? THREE.NearestFilter : THREE.LinearFilter;
      texture.minFilter = sampler?.minFilter === 9728 ? THREE.NearestFilter : THREE.LinearMipmapLinearFilter;
      // Aerial imagery already contains real illumination and shadows.
      const material = new THREE.MeshBasicMaterial({map: texture, side: THREE.DoubleSide});
      material.name = m.name || 'Vexcel aerial';
      resources.push(material);
      return material;
    });
    for (const mesh of doc.meshes) for (const primitive of mesh.primitives) {
      if ((primitive.mode ?? 4) !== 4) throw new Error('Unsupported GLB primitive');
      const geometry = new THREE.BufferGeometry();
      resources.push(geometry);
      geometry.setAttribute('position', attribute(primitive.attributes.POSITION, 'VEC3', 5126));
      geometry.setAttribute('uv', attribute(primitive.attributes.TEXCOORD_0, 'VEC2', 5126));
      geometry.setIndex(attribute(primitive.indices, 'SCALAR', 5125));
      if (geometry.attributes.position.count !== geometry.attributes.uv.count ||
          geometry.index.count % 3 || !materials[primitive.material]) throw new Error('Invalid GLB primitive attributes');
      geometry.computeBoundingSphere();
      const object = new THREE.Mesh(geometry, materials[primitive.material]);
      object.name = mesh.name || 'Neighbourhood surface';
      group.add(object);
    }
    // Idempotent cleanup is available to the integration's layer toggle.
    let disposed = false;
    group.userData.dispose = () => {
      if (disposed) return;
      disposed = true;
      group.removeFromParent();
      for (const resource of resources) resource.dispose();
    };
    scene.add(group);
    return {group, info};
  } catch (error) {
    for (const resource of resources) resource.dispose();
    throw error;
  }
}
