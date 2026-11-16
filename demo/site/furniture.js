import {element, dateLabel} from './common.js?v=3';

const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const edges=p=>p.map((a,i)=>[a,p[(i+1)%p.length]]);
function inside(p,segments){let hit=false;for(const [a,b] of segments)if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<a[0]+(p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1]))hit=!hit;return hit;}
function intersects(a,b,c,d){return cross(a,b,c)*cross(a,b,d)<-1e-12&&cross(c,d,a)*cross(c,d,b)<-1e-12;}
function overlap(a,b){return [a,b].every(p=>edges(p).every(([u,v])=>{const n=[v[1]-u[1],u[0]-v[0]],pa=a.map(q=>q[0]*n[0]+q[1]*n[1]),pb=b.map(q=>q[0]*n[0]+q[1]*n[1]);return Math.min(...pa)<Math.max(...pb)-1e-8&&Math.min(...pb)<Math.max(...pa)-1e-8;}));}
function box(p,w,d,angle){const c=Math.cos(angle),s=Math.sin(angle);return [[-w/2,-d/2],[w/2,-d/2],[w/2,d/2],[-w/2,d/2]].map(([x,y])=>[p[0]+x*c-y*s,p[1]+x*s+y*c]);}
function within(poly,geometry){const segments=[...geometry.segments,...(geometry.holes||[]).flat()];return poly.every(p=>inside(p,segments))&&!edges(poly).some(([a,b])=>segments.some(([c,d])=>intersects(a,b,c,d)))&&!segments.some(([a])=>inside(a,edges(poly)));}
export function parseDimensions(text,unit='cm'){
 const matches=[...text.trim().matchAll(/(\d+(?:\.\d+)?|\.\d+)\s*(cm|inches|inch|in|["″])?\s*([wdh])?/gi)];
 if(matches.length!==3||text.replace(/(\d+(?:\.\d+)?|\.\d+)\s*(cm|inches|inch|in|["″])?\s*([wdh])?/gi,'').replace(/[\s×x,;]+/gi,'')!=='')throw new Error('Enter W × D × H, for example 213 x 97 x 84 cm');
 const globalUnit=/cm\s*$/i.test(text)?'cm':/(inches|inch|in|["″])\s*$/i.test(text)?'in':unit;
 const labelled=matches.some(m=>m[3]);if(labelled&&new Set(matches.map(m=>m[3]?.toLowerCase())).size!==3)throw new Error('Use one W, D and H label');
 const values=matches.map(m=>Number(m[1])*((m[2]||globalUnit).toLowerCase()==='cm'?.01:.0254));
 const result=labelled?['w','d','h'].map(k=>values[matches.findIndex(m=>m[3]?.toLowerCase()===k)]):values;
 if(!result.every(v=>v>0&&v<=20))throw new Error('Each dimension must be above 0 and at most 20 m');return result;
}

export function mountFurniture(data,section,iframe){
 const panel=element('section',null,'furniture-panel');panel.id='furniture-panel';panel.append(element('h3','Try furniture'));
 const status=element('p','Upload a photo for category suggestions, or enter exact dimensions.','caption');status.id='furniture-status';status.setAttribute('aria-live','polite');
 function input(label,id,type='text'){const wrap=element('label',label+' '),field=element('input');field.id=id;field.type=type;wrap.append(field);return [wrap,field];}
 const [photoLabel,photo]=input('Photo (up to 12 MB)','furniture-photo','file');photo.accept='image/*';photo.setAttribute('capture','environment');
 const categories=element('div');categories.id='furniture-categories';
 const row=element('div',null,'row');const unit=element('select');unit.id='furniture-unit';unit.setAttribute('aria-label','Dimension units');for(const v of ['cm','in']){const o=element('option',v);o.value=v;unit.append(o);}
 const sizes=['Width','Depth','Height'].map((name,i)=>{const [label,field]=input(name,'furniture-'+name.toLowerCase(),'number');field.min='.01';field.step='any';field.value=[213,97,84][i];row.append(label);return field;});row.append(unit);
 const [pasteLabel,paste]=input('Paste W × D × H','furniture-dimensions');paste.placeholder='84"W x 38"D x 33"H';
 const apply=element('button','Use dimensions');apply.id='furniture-parse';
 const controls=element('div',null,'row'),place=element('button','Place at current sweep'),click=element('button','Click a floor spot');place.id='furniture-place';click.id='furniture-click';place.disabled=click.disabled=true;controls.append(place,click);
 const fitText=element('p','Place the box to check fit.');fitText.id='furniture-fit';fitText.setAttribute('aria-live','polite');
 const sun=element('div');sun.id='furniture-sun';
 const caveat=element('p',"Checked against the scan's room outline; existing furniture is not considered.",'caption');
 const scanDetails=element('details'),scanSummary=element('summary','Scan with video (beta)'),[videoLabel,video]=input('Video (up to 100 MB)','furniture-video','file'),[knownLabel,known]=input('Optional known width (metres)','furniture-known-width','number'),scan=element('button','Scan with video (beta)');scan.id='furniture-scan';video.accept='video/*';known.min='.01';known.step='any';scanDetails.append(scanSummary,videoLabel,knownLabel,scan);
 panel.append(status,photoLabel,categories,element('p','typical size for this category, enter your exact size','caption'),row,pasteLabel,apply,controls,fitText,caveat,sun,scanDetails);section.append(panel);
 const state=window.demoFurniture={ready:false,position:null,dimensions:[2.13,.97,.84],angle:0,color:'#8a8178',fit:null,sun:[],category:'3-seat sofa'};
 let sdk,fit,mesh,three,node,currentSweep,pointer,armed=false,stopped=false,doorData={available:false,zones:[]};const subscriptions=[];
 const request=async(path,body)=>{const r=await fetch(path,body?{method:'POST',body}:undefined),value=await r.json();if(!r.ok)throw new Error(value.error||'Furniture request failed');return value;};
 request('/api/furniture/openings/'+data.sid).then(value=>{doorData=value;update();}).catch(()=>{});
 const inverse=p=>fit.rotation[0].map((_,i)=>fit.rotation.reduce((sum,row,j)=>sum+row[i]*(p[j]-fit.translation[j]),0));
 const forward=p=>fit.rotation.map((row,i)=>row.reduce((sum,v,j)=>sum+v*p[j],fit.translation[i]));
 const setSize=values=>{if(!values.every(v=>Number.isFinite(v)&&v>0&&v<=20))throw new Error('Each dimension must be above 0 and at most 20 m');state.dimensions=values;sizes.forEach((field,i)=>field.value=(values[i]/(unit.value==='cm'?.01:.0254)).toFixed(2));update();};
 let previousUnit='cm';unit.onchange=()=>{try{const values=sizes.map(f=>Number(f.value)*(previousUnit==='cm'?.01:.0254));previousUnit=unit.value;setSize(values);}catch{}};
 function readSize(){const values=sizes.map(f=>Number(f.value)*(unit.value==='cm'?.01:.0254));if(!values.every(v=>v>0&&v<=20))throw new Error('Each dimension must be above 0 and at most 20 m');state.dimensions=values;}
 apply.onclick=()=>{try{setSize(parseDimensions(paste.value,unit.value));status.textContent='Exact dimensions entered. Confirm they match your furniture.';}catch(e){status.textContent=e.message;}};
 sizes.forEach(f=>f.onchange=()=>{try{readSize();update();}catch(e){status.textContent=e.message;}});
 photo.onchange=async()=>{const file=photo.files[0];if(!file)return;if(file.size>12_000_000){status.textContent='Photo must be under 12 MB';photo.value='';return;}photo.disabled=true;status.textContent='Looking up the photo locally…';categories.replaceChildren();try{const body=new FormData();body.append('file',file);const result=await request('/api/furniture/lookup',body);state.color=result.color;const table=element('table'),head=element('tr');for(const title of ['Category','Cosine score','Typical W × D × H (cm)','Range W / D / H (cm)'])head.append(element('th',title));table.append(head);for(const item of result.categories){const tr=element('tr'),td=element('td'),choose=element('button',item.category);choose.onclick=()=>{state.category=item.category;setSize(item.size_cm.map(v=>v/100));status.textContent=result.label;};td.append(choose);tr.append(td,element('td',item.cosine_score.toFixed(3)),element('td',item.size_cm.join(' × ')),element('td',item.range_cm.min.map((v,i)=>v+'–'+item.range_cm.max[i]).join(' / ')));table.append(tr);}categories.append(table);const first=result.categories[0];state.category=first.category;setSize(first.size_cm.map(v=>v/100));status.textContent=result.label+' · scores are similarities, not probabilities.';}catch(e){status.textContent=e.message;}finally{photo.value='';photo.disabled=false;}};
 scan.onclick=async()=>{const file=video.files[0];if(!file){status.textContent='Choose a video first.';return;}if(file.size>100_000_000){status.textContent='Video must be under 100 MB';return;}scan.disabled=true;status.textContent='Scanning video locally…';try{const body=new FormData();body.append('file',file);if(known.value)body.append('known_dim','width='+known.value);const result=await request('/api/furniture/scan',body);setSize([result.width_m,result.depth_m,result.height_m]);status.textContent=`Video scan · anchor: ${JSON.stringify(result.anchor)} · uncertainty: ${result.uncertainty_pct}% · ${result.frames_used} frames · ${result.seconds} s`;}catch(e){status.textContent=e.message;}finally{video.value='';scan.disabled=false;}};
 function roomAt(p){return data.rooms.find(r=>Math.abs(r.floor_z_m-p[2])<.35&&inside(p,[...data.geometry[r.id].segments,...(data.geometry[r.id].holes||[]).flat()]));}
 function put(p,source){const room=roomAt(p);if(!room){status.textContent='No scanned room floor at this spot. Choose another floor spot.';return;}state.position=[p[0],p[1],room.floor_z_m];state.source=source;status.textContent='Box placed · '+source;update();}
 place.onclick=()=>{try{readSize();const sweep=currentSweep?.id?currentSweep:null;if(!sweep)throw new Error('Wait for the current tour sweep');const p=sweep.puckPosition||sweep.position;const model=inverse([p.x,p.y,p.z]);if(!sweep.puckPosition){const room=data.rooms.filter(r=>r.floor_z_m<model[2]&&model[2]-r.floor_z_m<2.5&&inside(model,data.geometry[r.id].segments)).sort((a,b)=>b.floor_z_m-a.floor_z_m)[0];if(!room)throw new Error('Current sweep floor unavailable');model[2]=room.floor_z_m;}put(model,'current sweep floor');}catch(e){status.textContent=e.message;}};
 click.onclick=()=>{armed=true;status.textContent='Click a floor surface inside the tour. Use current sweep if the surface cannot be selected.';};
 function onFloorClick(){if(!armed||!pointer||stopped)return;const p=pointer.position,n=pointer.normal;if(!p||!n||pointer.object!==sdk.Pointer.Colliders.MODEL)return;const normal=fit.rotation[0].map((_,i)=>fit.rotation.reduce((sum,row,j)=>sum+row[i]*[n.x,n.y,n.z][j],0));if(normal[2]<.85){status.textContent='Choose a horizontal floor surface.';return;}try{readSize();put(inverse([p.x,p.y,p.z]),'SDK pointer floor intersection');armed=false;}catch(e){status.textContent=e.message;}}
 function button(label,id,action){const b=element('button',label);b.id=id;b.onclick=()=>{if(state.position){action();update();}};controls.append(b);}
 button('Rotate 15°','furniture-rotate15',()=>state.angle+=Math.PI/12);button('Rotate 90°','furniture-rotate90',()=>state.angle+=Math.PI/2);
 for(const [label,dx,dy,id] of [['← 10 cm',-.1,0,'left'],['→ 10 cm',.1,0,'right'],['↑ 10 cm',0,.1,'up'],['↓ 10 cm',0,-.1,'down']])button(label,'furniture-'+id,()=>{state.position[0]+=dx;state.position[1]+=dy;});
 function update(){
  if(!state.position||stopped)return;
  const p=state.position,[w,d,h]=state.dimensions,poly=box(p,w,d,state.angle),room=roomAt(p),geometry=room&&data.geometry[room.id];
  const zones=doorData.zones.filter(z=>z.floor_id===room?.floor_id).map(z=>box(z.center,z.width,1.8,Math.atan2(z.tangent[1],z.tangent[0])));
  const blocked=zones.some(z=>overlap(poly,z));let color='red',message='Crosses a wall or leaves the scanned room.';
  if(geometry&&within(poly,geometry)){
   const angle=state.angle+(w<d?Math.PI/2:0),long=Math.max(w,d),short=Math.min(w,d),normal=[-Math.sin(angle),Math.cos(angle)];
   const clearance=[-1,1].some(sign=>{const strip=box([p[0]+normal[0]*sign*(short/2+.45),p[1]+normal[1]*sign*(short/2+.45)],long,.9,angle);return within(strip,geometry)&&!zones.some(z=>overlap(strip,z));});
   color=blocked?'red':clearance&&doorData.available?'green':'amber';message=blocked?'Blocks a door or opening zone (0.9 m each side).':!doorData.available?'Opening data unavailable; full clearance check withheld.':clearance?'Inside room · at least 0.9 m clearance along one long side.':'Inside room · clearance below 0.9 m on both long sides.';
  }
  state.fit={color,message,room_id:room?.id};fitText.textContent=`${color.toUpperCase()}: ${message}`;fitText.dataset.fit=color;
  state.sun=[];sun.replaceChildren();
  if(room&&data.placement_status==='well'){
   const indices=data.geometry[room.id].grid.flatMap((q,i)=>overlap(poly,box(q,.1,.1,0))?[i]:[]);
   for(const [date,day] of Object.entries(data.days)){
    let minutes=0,mean=0;for(const [frame,runs] of (day.room_lit_runs[room.id]||[]).entries()){const count=indices.filter(i=>runs.some(([a,b])=>i>=a&&i<b)).length,weight=day.weights_minutes[frame];if(count)minutes+=weight;if(indices.length)mean+=weight*count/indices.length;}
    state.sun.push({date,minutes,mean_minutes:mean,cells:indices.length});sun.append(element('p',`${dateLabel(date)}: ${indices.length?Math.round(minutes)+' direct-sun min on any part · '+Math.round(mean)+' min cell average':'No sampled floor cells under footprint; sun unknown'}`));
   }
   const exposed=state.sun.some(day=>day.minutes>120);sun.append(element('p',exposed&&/sofa|rug|armchair|recliner/.test(state.category)?'Fade risk: more than 2 h of direct sun reaches this footprint.':/television|TV/.test(state.category)?'Glare can occur near sunlit windows; screen-facing direction is not measured.':'Fade/glare depends on materials and orientation; direct-sun minutes are a screening estimate.','caption'));
   sun.append(element('p','Existing 10 cm engine floor cells, weighted time steps; clear glass and clear sky. Floor exposure does not model sun on the furniture surface.','caption'));
  }else sun.append(element('p','Sun estimate withheld: room or aerial placement is uncertain.'));
  if(mesh){const vertices=[...poly.map(q=>[...q,p[2]+.015]),...poly.map(q=>[...q,p[2]+h])].flatMap(forward);const g=new three.BufferGeometry();g.setAttribute('position',new three.Float32BufferAttribute(vertices,3));g.setIndex([0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7]);g.computeVertexNormals();mesh.geometry.dispose();mesh.geometry=g;mesh.material.color.set(state.color);mesh.visible=true;}
 }
 async function ready(api,transform){sdk=api;fit=transform;try{
  let initialized;const init=new Promise(resolve=>initialized=resolve);
  await sdk.Scene.register('furniture.box',()=>({onInit(){three=this.context.three;mesh=new three.Mesh(new three.BufferGeometry(),new three.MeshBasicMaterial({color:state.color,transparent:true,opacity:.6,side:three.DoubleSide,depthWrite:false}));mesh.visible=false;this.outputs.objectRoot=mesh;initialized();},onDestroy(){mesh?.geometry.dispose();mesh?.material.dispose();}}));
  const [object]=await sdk.Scene.createObjects(1);if(stopped)return;node=object.addNode();node.addComponent('furniture.box');node.start();await Promise.race([init,new Promise((_,reject)=>setTimeout(()=>reject(new Error()),10000))]);if(stopped)return;
  subscriptions.push(sdk.Sweep.current.subscribe(value=>{currentSweep=value;}),sdk.Pointer.intersection.subscribe(value=>pointer=value));iframe.contentDocument.addEventListener('pointerup',onFloorClick,true);place.disabled=click.disabled=false;state.ready=true;
 }catch{status.textContent='Furniture scene unavailable. Exact size entry remains available.';}}
 function unavailable(){state.ready=false;place.disabled=click.disabled=true;}
 function destroy(){stopped=true;unavailable();subscriptions.forEach(s=>s?.cancel());iframe.contentDocument?.removeEventListener('pointerup',onFloorClick,true);node?.stop();}
 return {ready,unavailable,destroy};
}
