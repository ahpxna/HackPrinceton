import {element, dateLabel, minute, timeLabel} from './common.js?v=3';
import {mountFlight} from './sun-flight.js?v=6';
import {mountFurniture} from './furniture.js?v=1';
import {mountTourKeys} from './tour-keys.js?v=3';
import {mountRoofFix} from './roof-fix.js?v=1';

export const bundleVersion='26.10.4_webgl-1728-g1cd7e0997a';
const vector=p=>({x:p[0],y:p[1],z:p[2]});
const coords=p=>[p.x,p.y,p.z];
const finite=p=>p.every(Number.isFinite);
function deadline(promise,ms){let timer;return Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Tour timeout')),ms);})]).finally(()=>clearTimeout(timer));}
async function request(path,options){const r=await fetch(path,options);if(!r.ok)throw new Error('Tour data unavailable');return r.json();}
export function applyFit(fit,p,direction=false){return fit.rotation.map((row,i)=>row.reduce((sum,v,j)=>sum+v*p[j],direction?0:fit.translation[i]));}

export function mountTour(data,merge){
 const furnitureStyle=element('link');furnitureStyle.rel='stylesheet';furnitureStyle.href='furniture.css';document.head.append(furnitureStyle);
 const section=element('section',null,'panel');section.id='tour-panel';
 const status=element('p','Loading Matterport tour…');status.id='tour-status';status.setAttribute('aria-live','polite');
 const fitText=element('p','Tags wait for a fit between matching Model API and SDK sweep positions.','caption');fitText.id='tour-fit';
 const fallback=element('p','The plan view below remains available while the tour loads.','notice');fallback.id='tour-fallback';
 const iframe=element('iframe');iframe.id='tour-frame';iframe.title='Matterport tour with measured window findings';iframe.allow='fullscreen; xr-spatial-tracking';iframe.referrerPolicy='no-referrer';iframe.hidden=true;
 const controls=element('div',null,'tour-controls'),planOnly=element('button','Use plan only'),patchLabel=element('label'),patchCheck=element('input');patchCheck.type='checkbox';patchCheck.disabled=true;patchCheck.id='tour-patch';patchLabel.append(patchCheck,' Draw direct-sun floor cells in the tour');controls.append(planOnly,patchLabel);
 const patchText=element('p','Choose date, time and floor in the plan controls below. Yellow cells use the saved engine output, clear glass and the same assumptions as the plan.','caption');patchText.id='tour-patch-status';
 const details=element('details'),summary=element('summary','Window findings and agent notes'),findings=element('ol');findings.id='tour-findings';const notesList=element('ul');notesList.id='tour-notes';details.append(summary,findings,notesList);
 section.append(element('h2','Findings inside the Matterport tour'),status,fitText,controls,fallback,iframe,patchText,details);
 document.querySelector('.home-grid').before(section);
 const windows=data.windows.filter(w=>w.type==='window');
 const state=window.demoTour={sid:data.sid,status:'loading',attempted:false,loaded:false,windowTagCount:0,expectedWindows:windows.length,noteTagCount:0,fit:null,sdkVersion:bundleVersion,moves:0,reason:null,patchStatus:'off'};
 let sdk,fit,sweeps=[],sceneNode,patchMesh,three,stopped=false,latestFrame,pendingTarget,subscription,currentSubscription,modeSubscription,currentSweep,sweepIdMap={},stage='opening the model';
 function publishSweep(sweep=currentSweep){
  currentSweep=sweep;
  const ids=new Set([sweep?.id,sweep?.uuid,sweep?.sid,sweep?.map_id].filter(id=>typeof id==='string'&&id));
  for(const s of sweeps)if([s.id,s.uuid,s.sid,s.map_id].some(id=>ids.has(id)))for(const id of [s.id,s.uuid,s.sid,s.map_id])if(typeof id==='string')ids.add(id);
  for(const [a,b] of Object.entries(sweepIdMap))if(ids.has(a)||ids.has(b)){ids.add(a);ids.add(b);}
  for(const pair of fit?.pairs||[])if(ids.has(pair.sdk_id))ids.add(pair.model_id);
  state.currentSweepIds=[...ids];if(ids.size)state.lastSweepIds=[...ids];
 }
 const pointForWindow=w=>[...w.center,w.lower_z_m+w.height/2];
 function roomPoint(id){const room=data.rooms.find(r=>r.id===id)||data.audit_rooms?.find?.(r=>r.id===id),grid=data.geometry[id]?.grid||data.claim_room_outlines?.[id]?.segments?.flat();if(!grid?.length)return null;const floorZ=room?.floor_z_m??data.rooms.find(r=>r.floor_id===data.claim_room_outlines?.[id]?.floor_id)?.floor_z_m;if(!Number.isFinite(floorZ))return null;return [grid.reduce((s,p)=>s+p[0],0)/grid.length,grid.reduce((s,p)=>s+p[1],0)/grid.length,floorZ+1.4];}
 async function goTo(rooms=[],openingIds=[]){
  const w=data.windows.find(w=>openingIds.includes(w.id)),p=w?pointForWindow(w):rooms.map(roomPoint).find(Boolean);
  if(!p)return false;
  if(!fit||!sdk){pendingTarget=[rooms,openingIds];return false;}
  const target=applyFit(fit,p),nearest=sweeps.filter(s=>s.enabled!==false).sort((a,b)=>Math.hypot(...coords(a.position).map((v,i)=>v-target[i]))-Math.hypot(...coords(b.position).map((v,i)=>v-target[i])))[0];
  if(!nearest||stopped)return false;
  if(merge&&!await merge.showTour())return false;
  try{await deadline(sdk.Sweep.moveTo(nearest.id,{transition:sdk.Sweep.Transition.FLY,transitionTime:750}),10000);if(stopped)return false;publishSweep(nearest);state.moves++;state.lastSweep=nearest.id;status.textContent='Tour ready · moved to the nearest available sweep.';return true;}catch{if(!stopped)status.textContent='This sweep could not be reached. The selected finding remains visible in the plan.';return false;}
 }
 const flight=mountFlight(data,section,goTo,merge);
 const furniture=mountFurniture(data,section,iframe);
 const game=mountTourKeys(section,iframe,controls,state,()=>sweeps,()=>stopped);
 function sweepForCapture(id){
  if(!sdk||stopped||!id)return null;
  const ids=new Set([id]);
  for(const [a,b] of Object.entries(sweepIdMap))if(a===id||b===id){ids.add(a);ids.add(b);}
  for(const pair of fit?.pairs||[])if(pair.model_id===id)ids.add(pair.sdk_id);
  return sweeps.find(s=>s.enabled!==false&&[s.id,s.uuid,s.sid,s.map_id].some(alias=>ids.has(alias)));
 }
 async function returnToCapture(id){
  const sweep=sweepForCapture(id);
  if(!sweep)return false;
  try{await deadline(sdk.Sweep.moveTo(sweep.id,{transition:sdk.Sweep.Transition.INSTANT}),10000);publishSweep(sweep);state.lastSweep=sweep.id;state.moves++;return true;}catch{return false;}
 }
 async function prepareWalk(ids){
  const current=ids.find(id=>state.currentSweepIds?.includes(id));if(current)return current;
  const capture=ids.find(id=>state.lastSweepIds?.includes(id)&&sweepForCapture(id))||ids.find(id=>sweepForCapture(id));
  return capture&&await returnToCapture(capture)?capture:null;
 }
 async function setMode(name){
  const target=sdk?.Mode?.Mode?.[name];if(!target||stopped)return false;
  if(state.sdkMode===target)return true;
  try{await deadline(sdk.Mode.moveTo(target),10000);state.sdkMode=target;return true;}catch{return false;}
 }
 merge?.attachTour({iframe,viewport:iframe.parentElement,returnToCapture,prepareWalk,setMode,canReturnToCapture:id=>Boolean(sweepForCapture(id)),setActive(active){if(!active)flight?.pause();game.setActive(active);}});
 const roof=mountRoofFix(data,applyFit);
 function fail(reason){if(stopped)return;stopped=true;roof.destroy();furniture?.unavailable();flight?.unavailable(reason);subscription?.cancel();currentSubscription?.cancel();modeSubscription?.cancel();state.currentSweepIds=[];sceneNode?.stop();iframe.hidden=true;iframe.removeAttribute('src');patchCheck.disabled=true;patchCheck.checked=false;state.status='fallback';state.reason=reason;state.patchStatus='unavailable';status.textContent='Tour unavailable · plan view available';fallback.hidden=false;fallback.textContent=reason+' The existing plan view remains available below.';}
 planOnly.onclick=()=>fail('Tour disabled for this visit.');
 function windowDescription(w){
  const lines=[`Window ${w.id} · ${data.rooms.find(r=>r.id===w.room_id)?.label||'Room'}`];
  for(const date of Object.keys(data.days))lines.push(`${dateLabel(date)} ${date.slice(0,4)}: ${data.placement_status==='well'?minute(w.dates[date]?.unobstructed_any_aperture_minutes)+' clear-sky sun minutes':'sun estimate withheld'}`);
  const claims=(data.claims?.claims||[]).filter(c=>(c.evidence||[]).some(e=>e.window_id===w.id||e.window_ids?.includes(w.id)));
  lines.push(claims.length?'Claim verdicts explicitly citing this window:':'No supplied claim verdict explicitly cites this window.');
  for(const claim of claims)lines.push(`${claim.verdict}: “${claim.text}” — ${claim.explanation||''}`);
  lines.push('At least 1 of 9 aperture rays clear. Geometric direct sun; does not measure brightness. Clear glazing, roof/neighbour/tree assumptions; no weather, diffuse light, blinds or furniture.');
  return lines.join('\n');
 }
 for(const w of windows){const li=element('li'),button=element('button',`Go to window ${w.id}`);button.onclick=()=>goTo([], [w.id]);li.append(button,element('p',windowDescription(w)));findings.append(li);}
 function renderNotes(notes){
  if(!notes){notesList.append(element('li','No agent annotation file is present for this home.'));return [];}
  const mode=notes.approval_mode==='scripted_dry_run'?'Scripted dry-run annotations; not a live model or human approval.':`Annotation approval mode: ${notes.approval_mode||'unspecified'}.`;
  notesList.append(element('li',mode));
  if(notes.published!==true){notesList.append(element('li','Unpublished annotations are withheld.'));return [];}
  const annotations=notes.annotations||[];
  if(data.placement_status!=='well'&&annotations.length){notesList.append(element('li','Agent notes are withheld because placement is uncertain; no sun findings are placed from these notes.'));return [];}
  if(!annotations.length)notesList.append(element('li','The published annotation file contains no placed notes.'));
  for(const note of annotations){const li=element('li',`${note.title||'Agent note'} · ${note.verdict||'note'}\n${note.text||''}\nEvidence: ${(note.evidence_ids||[]).join(', ')||'none supplied'}`);notesList.append(li);}
  for(const caveat of notes.caveats||[])notesList.append(element('li',caveat));
  return annotations.filter(n=>finite([n.x,n.y,n.z]));
 }
 function setFrame(date,index,floor){latestFrame={date,index,floor};drawPatch();flight?.setFrame(latestFrame);}
 function drawPatch(){
  if(!patchMesh||!latestFrame||stopped)return;
  patchMesh.visible=patchCheck.checked;
  if(!patchCheck.checked){state.patchStatus='off';return;}
  const {date,index,floor}=latestFrame,vertices=[];
  for(const room of data.rooms.filter(r=>r.floor_id===floor)){
   const grid=data.geometry[room.id].grid;
   for(const [start,end] of data.days[date].room_lit_runs[room.id][index]||[])for(let i=start;i<end;i++){
    const [x,y]=grid[i],z=room.floor_z_m+.025,corners=[[-.05,-.05],[.05,-.05],[.05,.05],[-.05,.05]].map(([dx,dy])=>applyFit(fit,[x+dx,y+dy,z]));
    for(const j of [0,1,2,0,2,3])vertices.push(...corners[j]);
   }
  }
  const geometry=new three.BufferGeometry();geometry.setAttribute('position',new three.Float32BufferAttribute(vertices,3));patchMesh.geometry.dispose();patchMesh.geometry=geometry;
  state.patchStatus='drawn';state.patchCells=vertices.length/18;
  patchText.textContent=`${state.patchCells} engine floor cells · ${dateLabel(date)} ${timeLabel(data.days[date].local_times[index])} · selected floor. Clear-sky direct sun; furniture is not modelled. Cells may be hidden by real geometry.`;
 }
 patchCheck.onchange=drawPatch;
 async function addPatch(){
  if(data.placement_status!=='well'){state.patchStatus='withheld';patchText.textContent='Sun cells are withheld because aerial placement is uncertain.';return;}
  state.patchStatus='initializing';
  try{
   await deadline(sdk.Scene.register('sun.floorCells',()=>({onInit(){three=this.context.three;patchMesh=new three.Mesh(new three.BufferGeometry(),new three.MeshBasicMaterial({color:0xffdf36,transparent:true,opacity:.72,side:three.DoubleSide,depthWrite:false}));patchMesh.visible=false;this.outputs.objectRoot=patchMesh;},onDestroy(){patchMesh?.geometry.dispose();patchMesh?.material.dispose();}})),10000);
   if(stopped)return;
   const [object]=await deadline(sdk.Scene.createObjects(1),10000);if(stopped)return;sceneNode=object.addNode();sceneNode.addComponent('sun.floorCells');sceneNode.start();patchCheck.disabled=false;state.patchStatus='ready';drawPatch();
  }catch{state.patchStatus='unavailable';patchText.textContent='Tour floor overlay unavailable. The same engine cells remain visible in the plan.';}
 }
 async function start(){
  const configElement=document.getElementById('tour-config');let config={};
  try{config=JSON.parse(configElement?.textContent||'{}');}catch{}configElement?.remove();
  const planOnlyMode=new URLSearchParams(location.search).get('tour')==='off';
  const anchorsPromise=request(`/tour/data/${data.sid}${!config.configured||planOnlyMode?'?offline=1':''}`).catch(()=>null);
  if(planOnlyMode){config.key='';const anchors=await anchorsPromise;renderNotes(anchors?.notes);fail('Tour disabled for this visit.');return;}
  if(!config.configured||!config.key){const anchors=await anchorsPromise;renderNotes(anchors?.notes);fail('Matterport SDK is not configured for this server.');return;}
  const url=new URL('vendor/showcase-26.10.4/showcase.html',location.href);url.searchParams.set('m',data.sid);url.searchParams.set('applicationKey',config.key);url.searchParams.set('play','1');url.searchParams.set('qs','1');url.searchParams.set('log','0');config.key='';
  const load=new Promise((resolve,reject)=>{iframe.onload=resolve;iframe.onerror=reject;});state.attempted=true;iframe.src=url.href;url.searchParams.delete('applicationKey');iframe.hidden=false;
  try{
   await deadline(load,25000);if(stopped)return;
   sdk=await deadline(iframe.contentWindow.MP_SDK.connect(iframe.contentWindow),30000);if(stopped)return;
   await deadline(sdk.App.state.waitUntil(s=>s.phase===sdk.App.Phase.PLAYING),30000);if(stopped)return;
   state.loaded=true;game.ready(sdk);status.textContent='Tour loaded · fitting matching sweep positions…';
   currentSubscription=sdk.Sweep.current.subscribe(publishSweep);
   modeSubscription=sdk.Mode.current.subscribe(mode=>{state.sdkMode=mode;});
   stage='reading Model API anchors';const anchors=await anchorsPromise;const notes=renderNotes(anchors?.notes);
   if(!anchors?.model_sweeps?.length){fail('Model API sweep positions are unavailable, so no coordinate transform or findings can be placed safely.');return;}
   stage='matching SDK sweeps';
   await deadline(new Promise(resolve=>{const update=collection=>{sweeps=Object.entries(collection).map(([map_id,s])=>({...s,map_id}));if(sweeps.length>=4)resolve();};subscription=sdk.Sweep.data.subscribe({onAdded(_id,_s,collection){update(collection);},onCollectionUpdated:update});}),15000);
   if(stopped)return;
   let idMap={};try{idMap=await deadline(sdk.Sweep.Conversion.createIdMap(),5000);}catch{}sweepIdMap=idMap;publishSweep();
   const aliases=id=>Object.entries(idMap).flatMap(([a,b])=>a===id?[b]:b===id?[a]:[]);
   stage='fitting matching sweep IDs';fit=await request(`/tour/fit/${data.sid}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(sweeps.map(s=>({id:s.id,uuid:s.uuid,sid:s.sid,map_id:s.map_id,aliases:[...new Set([s.id,s.uuid,s.sid,s.map_id].flatMap(aliases))],enabled:s.enabled,position:s.position})))});state.fit=fit;
   if(stopped)return;
   fitText.textContent=`Model API → SDK: ${fit.count} matching sweep IDs · RMS ${fit.rms_m.toFixed(6)} m · max ${fit.max_m.toFixed(6)} m. Rigid fit in metres; no scale or guessed axis swaps. These are fit residuals, not independent validation.`;
   if(!fit.accepted){fail('Sweep transform exceeds the placement tolerance (RMS 0.05 m or maximum 0.15 m); findings have not been placed.');return;}
   publishSweep();
   void roof.ready(sdk,fit);
   stage='placing SDK tags';const stem=vector(applyFit(fit,[0,0,.25],true));
   const tags=windows.map(w=>({label:`Window ${w.id}`,description:windowDescription(w),anchorPosition:vector(applyFit(fit,pointForWindow(w))),stemVector:stem,color:{r:.95,g:.7,b:.1}}));
   if(tags.length){const ids=await deadline(sdk.Tag.add(...tags),15000);if(stopped)return;state.windowTagCount=ids.length;}
   if(notes.length){const ids=await deadline(sdk.Tag.add(...notes.map(n=>({label:n.title||'Agent note',description:`${n.verdict||'note'}: ${n.text||''}\nEvidence: ${(n.evidence_ids||[]).join(', ')}\n${anchors.notes.approval_mode==='scripted_dry_run'?'Scripted dry run; no live model or human approval.':''}`,anchorPosition:vector(applyFit(fit,[n.x,n.y,n.z])),stemVector:stem,color:{r:.2,g:.6,b:.95}}))),15000);if(stopped)return;state.noteTagCount=ids.length;}
   await flight?.ready(sdk,fit);
   if(stopped)return;state.status='ready';fallback.hidden=true;status.textContent=`Tour ready · ${state.windowTagCount}/${windows.length} exterior windows tagged · ${state.noteTagCount} agent notes. Click a room, claim or window finding to move to its nearest sweep.`;
   await addPatch();await furniture.ready(sdk,fit);if(pendingTarget){const target=pendingTarget;pendingTarget=null;goTo(...target);}
  }catch{if(!notesList.children.length){const anchors=await anchorsPromise;renderNotes(anchors?.notes);}fail(`Matterport tour unavailable while ${stage}. The model may be private, the SDK key may not allow this domain, or the network/SDK may be unavailable. No unregistered findings are shown.`);}
 }
 start();window.addEventListener('pagehide',()=>{stopped=true;roof.destroy();game.destroy();furniture.destroy();flight?.destroy();subscription?.cancel();currentSubscription?.cancel();modeSubscription?.cancel();sceneNode?.stop();},{once:true});
 return {goTo,setFrame};
}
