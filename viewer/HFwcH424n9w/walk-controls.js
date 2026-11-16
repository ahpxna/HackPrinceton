const query = new URLSearchParams(location.search);
const sid = location.pathname.match(/\/walk\/([^/]+)/)?.[1];
const panel = document.createElement('section');
panel.style.cssText = 'position:fixed;right:12px;top:12px;z-index:100;background:white;color:#111;padding:10px;width:300px;font:13px/1.5 system-ui;border:1px solid #aaa;border-radius:4px';
panel.innerHTML = `<strong>HF · Đi bộ 3D</strong><br>
<label>Lớp <select id="w25-layer"><option value="both">Ảnh + Splat</option><option value="photo">Chỉ ảnh</option><option value="splat">Chỉ Splat</option></select></label>
<br><button id="w25-compare">So với r5 trước sửa ở cùng góc</button><br>
<div id="w25-wall-pass"></div>
<small>Giữ phím W/S đi, A/D sang ngang, ←/→ xoay.</small>
<pre id="w25-state" style="white-space:pre-wrap;font-size:11px"></pre>
<button id="w25-proof">Lưu số liệu lượt đi</button><br>
<button id="w25-clean">Ẩn giao diện · H</button><button id="w25-fullscreen">Toàn màn hình · F</button>`;
document.body.append(panel);
const ghost = document.querySelector('#ghost');
const ghostLabel = ghost.closest('label');
ghostLabel.replaceChildren(ghost, document.createTextNode(' Đi xuyên tường và đồ đạc'));
panel.querySelector('#w25-wall-pass').append(
  ghostLabel, document.createElement('br'), document.querySelector('#ghostNote'));
const layer = panel.querySelector('#w25-layer');
layer.value = query.get('layer') || 'both';
const setLayer = () => { window.previewView?.setLayer(layer.value);
  query.set('layer', layer.value); history.replaceState(null, '', `?${query}`); };
layer.onchange = setLayer;
window.addEventListener('preview-ready', setLayer);
const refined = sid === 'HFwcH424n9w' && query.get('refine') === '1';
panel.querySelector('#w25-compare').textContent = 'So với Ảnh + Splat 8765 ở cùng góc';
panel.querySelector('#w25-compare').hidden = true;
const recordingStyle = document.createElement('style');
recordingStyle.textContent = 'body.w25-clean>section,body.w25-clean>aside,body.w25-clean>#helpToggle,body.w25-clean>#joystick,body.w25-clean>#aerialZoom,body.w25-clean>#neighbourhoodLabel{display:none!important}';
document.head.append(recordingStyle);
const toggleClean = () => document.body.classList.toggle('w25-clean');
const fullscreen = () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()).catch(() => {});
panel.querySelector('#w25-clean').onclick = toggleClean;
panel.querySelector('#w25-fullscreen').onclick = fullscreen;
addEventListener('keydown', event => {
  if (event.repeat || event.target.matches('select,input,textarea')) return;
  if (event.code === 'KeyH') toggleClean();
  if (event.code === 'KeyF') fullscreen();
});
if (query.get('clean') === '1') toggleClean();
panel.querySelector('#w25-compare').onclick = () => {
  const pose = window.previewView?.getPose();
  if (pose) query.set('pose', pose.join(','));
  query.set('layer', layer.value);
  query.delete('refine'); query.delete('model'); query.delete('raw');
  location.href = `http://localhost:8765/walk/${sid}/?${query}`;
};
const samples = [];
window.addEventListener('error', event => { window.previewError = event.message; });
window.addEventListener('unhandledrejection', event => {
  window.previewError = event.reason?.message || String(event.reason);
});
setInterval(() => {
  const p = window.photoState, w = window.walkState;
  const error = window.previewError || p?.error;
  if (!p || !w) {
    panel.querySelector('#w25-state').textContent = error ? `Lỗi tải: ${error}` : 'Đang tải model và dữ liệu đi bộ…';
    return;
  }
  const sample = { time: performance.now(), pose: window.previewView?.getPose(),
    moving: w.moving, speed: w.speed, handoff: structuredClone(p.handoff),
    hiRes: [...p.hiRes], hitches: p.hitches, cpuMs: p.cpuMs, gpuMs: p.gpuMs, error };
  samples.push(sample); if (samples.length > 2400) samples.shift();
  panel.querySelector('#w25-state').textContent =
    `Model: ${sid === 'HFwcH424n9w' ? 'HF r5 · 94.454 bước' : 'VN r5'}${refined ? ' · bộ lọc thử nghiệm' : ' · nguyên bản'}\n` +
    (layer.value === 'splat' ? 'Chỉ Splat · không tải/chiếu ảnh theo điểm\n' : `Điểm: ${p.handoff.primary || 'đang tải'}\nChuyển: ${p.handoff.switches} · chờ: ${p.handoff.waitingFor || '—'}\n`) +
    `Lưới 256: ${p.handoff.fullGrids?.length || 0}/2 · ảnh 2K: ${p.hiRes.length}/2\n` +
    `Vị trí: ${w.pos.map(v => v.toFixed(2)).join(', ')}\n` +
    `Tốc độ: ${w.speed.toFixed(2)} m/s · photo CPU: ${p.cpuMs.toFixed(1)} ms\n` +
    `FPS thực: ${w.fps?.toFixed(1) || 'đang đo'} · tỉ lệ pixel: ${w.pixelRatio || '—'}\n` +
    (layer.value === 'splat' ? '' : `Photo GPU: ${p.gpuMs?.toFixed(1) || '—'} ms · photo khung >33ms/10s: ${p.hitches}\n`) +
    `Lỗi: ${error || 'không'}`;
}, 100);
panel.querySelector('#w25-proof').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify({ sid, samples }, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = `w25-${sid}-walk.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
