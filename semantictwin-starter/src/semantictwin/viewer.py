"""Portable no-dependency 3D point viewer; it does NOT render Gaussian footprints.

Useful for inspecting class IDs/uncertainty before selecting a production splat viewer.
"""
from pathlib import Path
import json
import numpy as np


def export_html(output_path, xyz, rgb, labels, confidence, classes, max_points=16000):
    xyz = np.asarray(xyz)
    rgb = np.asarray(rgb)
    labels = np.asarray(labels)
    confidence = np.asarray(confidence)
    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz)-1, max_points).astype(int)
        xyz, rgb, labels, confidence = xyz[idx], rgb[idx], labels[idx], confidence[idx]
    payload = {"xyz":xyz.round(4).tolist(), "rgb":np.clip(rgb*255,0,255).astype('uint8').tolist(),
               "labels":labels.astype(int).tolist(), "conf":confidence.round(3).tolist(),
               "classes":{str(k):v for k,v in classes.items()}}
    js = json.dumps(payload).replace('</', '<\\/')
    html = r'''<!doctype html><html><head><meta charset="utf-8"><title>SemanticTwin | 3D inspection</title>
<style>body{margin:0;background:#091323;color:#dde8fa;font:14px system-ui,sans-serif}aside{position:absolute;top:16px;left:16px;background:#17273aee;border:1px solid #506076;border-radius:14px;padding:16px;max-width:250px;z-index:2}button,select{background:#253c5a;color:#fff;border:1px solid #577197;border-radius:5px;padding:5px}label{display:block;margin:9px 0}small{color:#a1b4cf}#info{margin-top:10px;color:#e6c578}canvas{width:100vw;height:100vh;display:block}h3{margin:0 0 10px}</style></head>
<body><aside><h3>SemanticTwin | 3D preview</h3><small>Point centers, not photorealistic splatting. Drag to rotate; scroll to zoom; click to inspect.</small><label>Color mode <select id="mode"><option value="semantic">Semantic</option><option value="rgb">RGB</option><option value="confidence">Confidence</option></select></label><div id="classes"></div><div id="info">Select a point</div><button id="reset">Reset camera</button></aside><canvas id="scene"></canvas>
<script>const D=__PAYLOAD__;const canvas=document.querySelector('#scene');const ctx=canvas.getContext('2d');const labels=[...new Set(D.labels)].sort((a,b)=>a-b);const palettes=['#f59e0b','#22c55e','#3b82f6','#ec4899','#a78bfa','#e879f9','#14b8a6'];const palette=Object.fromEntries(labels.map((c,i)=>[c,c<0?'#68738a':palettes[i%palettes.length]]));const controls=document.querySelector('#classes');const visible={};labels.forEach(c=>{visible[c]=true;const l=document.createElement('label');const ck=document.createElement('input');ck.type='checkbox';ck.checked=true;ck.onchange=()=>{visible[c]=ck.checked;render()};l.append(ck,document.createTextNode(' '+(c<0?'Unknown':D.classes[c]||('Class '+c))));controls.append(l)});const center=[0,1,2].map(j=>D.xyz.reduce((s,p)=>s+p[j],0)/D.xyz.length);let az=-0.6,el=0.45,zoom=1.0;let screen=[];const mode=document.querySelector('#mode');mode.onchange=()=>render();function resize(){const ratio=window.devicePixelRatio||1;canvas.width=Math.floor(innerWidth*ratio);canvas.height=Math.floor(innerHeight*ratio);render()}function render(){const W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);ctx.fillStyle='#0b1524';ctx.fillRect(0,0,W,H);const cr=Math.cos(az),sr=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),scale=Math.min(W,H)*0.18*zoom;screen=[];for(let i=0;i<D.xyz.length;i++){const l=D.labels[i];if(!visible[l])continue;const p=D.xyz[i],x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];const xr=cr*x-sr*y,yr=sr*x+cr*y;const yy=se*yr+ce*z,depth=ce*yr-se*z;screen.push({i,x:W/2+xr*scale,y:H/2-yy*scale,depth})}screen.sort((a,b)=>b.depth-a.depth);for(const p of screen){const i=p.i;let c;if(mode.value==='rgb'){const t=D.rgb[i];c=`rgb(${t[0]},${t[1]},${t[2]})`}else if(mode.value==='confidence'){const k=D.conf[i];c=`hsl(${Math.round(115*k)} 76% 52%)`}else{c=palette[D.labels[i]]}ctx.fillStyle=c;ctx.fillRect(p.x-1.6,p.y-1.6,3.2,3.2)}}let dragging=false,lastX=0,lastY=0;canvas.onpointerdown=e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!dragging)return;az+=(e.clientX-lastX)*0.009;el=Math.max(-1.5,Math.min(1.5,el+(e.clientY-lastY)*0.007));lastX=e.clientX;lastY=e.clientY;render()};canvas.onpointerup=()=>dragging=false;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(0.15,Math.min(8,zoom*Math.exp(-e.deltaY*0.001)));render()};canvas.onclick=e=>{if(Math.abs(e.movementX)>4)return;const ratio=window.devicePixelRatio||1;const x=e.clientX*ratio,y=e.clientY*ratio;let winner=null,dist=50*ratio;for(const p of screen){const d=Math.hypot(p.x-x,p.y-y);if(d<dist){dist=d;winner=p}}if(winner){const i=winner.i;document.querySelector('#info').textContent=`ID ${i} · ${D.classes[D.labels[i]]||'unknown'} · vote confidence ${D.conf[i]} · xyz ${D.xyz[i]}`}};document.querySelector('#reset').onclick=()=>{az=-.6;el=.45;zoom=1;render()};addEventListener('resize',resize);resize();</script></body></html>'''.replace('__PAYLOAD__', js)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(html, encoding='utf8')
    return path
