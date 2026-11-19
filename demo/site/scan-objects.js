// Objects located in the 3D scan (W34): confirmed OWLv2 clusters placed on the mesh, drawn on the plan.
import {$,element} from './common.js?v=3';
const PALETTE=['#d1495b','#00798c','#edae49','#66a182','#2e4057','#8d6a9f','#c46d2c','#3c91e6','#9b5de5','#f15bb5','#4f772d'];
const code=label=>label.split(' ').map(w=>w[0]).join('').slice(0,2).toUpperCase();

export async function mountScanObjects(sid,{redraw}){
  let doc;
  try{const r=await fetch(`/api/scan-objects/${sid}`);if(!r.ok)return null;doc=await r.json();}catch{return null;}
  if(!doc.available||!doc.features.length)return null;
  const labels=[...new Set(doc.features.map(f=>f.label))].sort();
  const color=label=>PALETTE[labels.indexOf(label)%PALETTE.length];
  const objects=doc.features.flatMap(f=>f.objects.map(o=>({...o,label:f.label,room:f.room,floor_id:f.floor_id,mapping:f.room_mapping})));
  const floorName=id=>[...$('floor').options].find(o=>o.value===id)?.textContent||'?';
  let selected=null;

  const panel=element('section',null,'panel');panel.id='scan-objects-panel';
  panel.append(element('h2','Objects located in the 3D scan'),
    element('p',`${objects.length} confirmed detector clusters in ${doc.features.length} label/room groups (${doc.source_version}). Zero-shot OWLv2 boxes cast onto the scan mesh and merged across camera positions. Not human verified; a physical object can appear as more than one cluster, so counts are upper bounds. Room = the object's position in Matterport's room outlines; "room unknown" where an outline is open or its room type conflicts with the object. Click a row to see its best view and its marker on the plan.`,'caption'));
  const legend=element('div',null,'legend');legend.style.padding='0 0 10px';
  for(const label of labels){const item=element('span',`${code(label)} ${label}`),dot=element('i',null,'dot');dot.style.background=color(label);item.prepend(dot);legend.append(item);}
  const preview=element('figure');preview.style.margin='0 0 12px';preview.hidden=true;
  const view=document.createElement('canvas');view.width=view.height=360;view.style.cssText='width:100%;max-width:360px;border-radius:4px;background:#eef0f1';
  const caption=element('figcaption',null,'caption');preview.append(view,caption);
  const wrap=element('div',null,'table-scroll'),table=element('table'),head=element('thead'),body=element('tbody');
  const hr=element('tr');for(const h of ['Object','Room','Floor','Height above floor','Size','Views','Best score'])hr.append(element('th',h));head.append(hr);table.append(head,body);wrap.append(table);
  const rows=new Map();
  for(const o of objects.sort((a,b)=>a.label.localeCompare(b.label)||b.score_max-a.score_max)){
    const tr=element('tr');tr.style.cursor='pointer';
    const name=element('td');const dot=element('i',null,'dot');dot.style.background=color(o.label);const strong=element('strong',o.label);strong.prepend(dot);name.append(strong,element('small',o.id));
    tr.append(name,element('td',o.room),element('td',floorName(o.floor_id)),element('td',`${o.height_above_floor_m.toFixed(2)} m`),
      element('td',`${o.size_m.toFixed(2)} m`),element('td',`${o.support_captures} / ${o.n_captures}`),element('td',o.score_max.toFixed(2)));
    tr.addEventListener('click',()=>select(o));body.append(tr);rows.set(o.id,tr);
  }
  const notes=element('details');notes.append(element('summary','Caveats and review exclusions'));
  for(const c of doc.caveats)notes.append(element('p',c));
  for(const x of doc.excluded)notes.append(element('p',`Excluded ${x.id} (${x.label}): ${x.reason}`));
  panel.append(legend,preview,wrap,notes);
  $('rooms-panel').after(panel);

  async function select(o){
    selected=o.id;for(const [id,tr] of rows)tr.classList.toggle('active',id===o.id);
    if(o.floor_id&&[...$('floor').options].some(x=>x.value===o.floor_id))$('floor').value=o.floor_id;
    redraw();preview.hidden=false;
    caption.textContent=`${o.label} · ${o.room} · best view ${o.best_view.capture} face ${o.best_view.face} · score ${o.best_view.score.toFixed(2)} (not a probability) · ${o.mapping}`;
    const img=new Image();img.src=o.image_url;
    try{await img.decode();}catch{caption.textContent='Best-view image unavailable · '+caption.textContent;return;}
    if(selected!==o.id)return;
    const g=view.getContext('2d'),s=view.width,[x0,y0,x1,y1]=o.best_view.box;g.drawImage(img,0,0,s,s);
    g.strokeStyle=color(o.label);g.lineWidth=3;g.strokeRect(x0*s,y0*s,(x1-x0)*s,(y1-y0)*s);
  }

  return {draw(ctx,xy,floor){
    ctx.save();ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='bold 9px system-ui';
    for(const o of objects){
      if(o.floor_id!==floor)continue;
      const [px,py]=xy(o.position_m),big=o.id===selected;
      ctx.beginPath();ctx.arc(px,py,big?11:7.5,0,Math.PI*2);ctx.fillStyle=color(o.label);ctx.fill();
      ctx.lineWidth=big?3:1.5;ctx.strokeStyle=big?'#111':'#fff';ctx.stroke();ctx.fillStyle='#fff';ctx.fillText(code(o.label),px,py+.5);
    }
    ctx.restore();
  }};
}
