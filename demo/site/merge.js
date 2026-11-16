import {$,element} from './common.js?v=3';

const TOOLBAR='[class*="bottom-controls-"],.bottom-controls';
const RULER='button[data-testid="measure-mode-button"]';
const PERSON='button[data-testid="panorama"],[data-mpclid="click_inside_mode_button"],button[data-testid="mesh"],[data-mpclid="click_mesh_mode_button"]';
const DOLLHOUSE='button[data-testid="dollhouse"],button[data-mpclid="click_dollhouse_mode_button"]';
const FLOORPLAN='button[data-testid="floorplan"],button[data-mpclid="click_floorplan_mode_button"]';
function label(button,text){button.title=text;button.setAttribute('aria-label',text);}
function resetClone(button){for(const node of [button,...button.querySelectorAll('*')])for(const attr of [...node.attributes])if(attr.name==='id'||attr.name.startsWith('data-')||attr.name.startsWith('aria-'))node.removeAttribute(attr.name);button.type='button';}

export function mountMerge(sid){
 window.demoReview??={};
 const review=window.demoReview.merge={modes:['Walk 3D','Dollhouse','Floor plan','Measure'],walkUrlOk:false,measureOk:false,toolbar:false,mode:'tour',flight:{lastAction:null,ok:false,error:null}};
 const frame=$('walk-frame'),comparison=$('measure-panel'),availability=$('walk-availability');
 let available=false,captures=[],tour,nativeButton,observer,mirror,rulerClone,poll,walking=false,opening=false,transition,defaultOpened=false,watchedDoc,startCapture;
 function api(){try{return frame.contentWindow?.demoReview?.measure;}catch{return null;}}
 function syncMeasure(){
  const measure=api(),ready=typeof measure?.setEnabled==='function',enabled=ready&&Boolean(measure.state?.enabled);
  if(rulerClone){rulerClone.disabled=!ready;rulerClone.setAttribute('aria-disabled',String(!ready));rulerClone.setAttribute('aria-pressed',String(enabled));rulerClone.classList.toggle('mp-nova-active',enabled);label(rulerClone,ready?'Measure reconstructed geometry':'measure in Walk 3D: coming');}
  if(ready){const panel=frame.contentDocument?.querySelector('section[aria-label="Measure reconstructed geometry"]');if(panel){panel.hidden=!enabled;panel.style.bottom='64px';const toggle=panel.querySelector('button');if(toggle)toggle.hidden=true;}}
  comparison.hidden=!walking||!enabled;review.measureOk=ready;review.mode=walking?(enabled?'measure':'walk'):'tour';
 }
 function walkUrl(capture){
  const params=new URLSearchParams();params.set('embed','1');if(capture)params.set('capture',capture);params.set('v','w26-1');
  return '/walk/'+encodeURIComponent(sid)+'/index.html?'+params;
 }
 function nearestCapture(){
  let position;try{position=frame.contentWindow?.walkState?.pos;}catch{}
  if(!Array.isArray(position)||position.length!==3||!position.every(Number.isFinite))return startCapture;
  return captures.filter(c=>Array.isArray(c.position)&&c.position.length===3&&c.position.every(Number.isFinite)&&Math.abs(c.position[2]-position[2])<1.2&&(tour.canReturnToCapture?.(c.id)??true))
   .sort((a,b)=>Math.hypot(a.position[0]-position[0],a.position[1]-position[1])-Math.hypot(b.position[0]-position[0],b.position[1]-position[1]))[0]?.id;
 }
 async function showTour(mode){
  if(transition)await transition;
  if(!tour||tour.iframe.hidden)return false;
  transition=(async()=>{
   if(walking){
    const capture=nearestCapture();api()?.setEnabled(false);
    walking=false;tour.viewport.classList.remove('merge-walking');tour.iframe.style.visibility='';tour.iframe.inert=false;frame.hidden=true;frame.removeAttribute('src');if(mirror)mirror.hidden=true;tour.setActive(true);syncMeasure();
    await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    tour.iframe.contentWindow?.dispatchEvent(new Event('resize'));
    const moved=capture?await tour.returnToCapture(capture):false;
    review.returnCapture=capture||null;review.returnMapped=moved;
    availability.textContent=capture&&!moved?'Tour restored; the nearest walk capture could not be mapped to an available Matterport sweep.':'The person icon opens the reconstructed Walk 3D.';
   }
   if(mode&&!await tour.setMode(mode))return false;
   const bounds=tour.iframe.getBoundingClientRect();return bounds.width>0&&bounds.height>0;
  })();
  try{return await transition;}finally{transition=null;}
 }
 async function showWalk(){
  if(!tour||walking||opening||tour.iframe.hidden)return;
  opening=true;
  try{
   await capturesReady;if(transition)await transition;if(!available)return;
   startCapture=await tour.prepareWalk(captures.map(c=>c.id));
   if(!startCapture){availability.textContent='Walk 3D could not match an available Matterport sweep to a capture.';return;}
   const group=nativeButton?.closest(TOOLBAR);mirrorToolbar(tour.iframe.contentDocument,group);
   defaultOpened=true;walking=true;tour.setActive(false);tour.viewport.classList.add('merge-walking');tour.iframe.style.visibility='hidden';tour.iframe.inert=true;
   frame.src=walkUrl(startCapture);frame.hidden=false;if(mirror)mirror.hidden=false;review.startCapture=startCapture;review.walkUrlOk=frame.getAttribute('src').startsWith('/walk/'+encodeURIComponent(sid)+'/index.html');syncMeasure();
  }catch{availability.textContent='Walk 3D could not restore its starting tour sweep.';}finally{opening=false;}
 }
 function mirrorToolbar(doc,group){
  mirror?.remove();mirror=element('div');mirror.className='merge-toolbar';mirror.setAttribute('aria-label','Walk 3D toolbar');mirror.hidden=!walking;
  const shadow=mirror.attachShadow({mode:'open'});
  for(const sheet of doc.querySelectorAll('link[rel="stylesheet"]')){const link=element('link');link.rel='stylesheet';link.href=sheet.href;shadow.append(link);}
  for(const font of doc.fonts||[])document.fonts?.add(font);
  const style=element('style');style.textContent=':host{display:block}.bar{display:flex;align-items:center;gap:4px;pointer-events:auto}.bar button{flex:none;pointer-events:auto;border-radius:50%;width:34px;height:34px;margin:0;padding:0!important}.bar button>*{scale:.9}.bar button:focus-visible{outline:2px solid white;outline-offset:2px}.bar button[aria-pressed=true]{background:#fff!important;color:#222!important;box-shadow:none!important;outline:0}.bar button:disabled{opacity:.45}.bar svg{display:block;margin:auto}';shadow.append(style);
  const bar=element('div');bar.className='bar';bar.setAttribute('role','toolbar');bar.setAttribute('aria-label','Walk 3D controls');shadow.append(bar);
  rulerClone=null;
  function fallback(testid,text){const button=element('button',text);button.setAttribute('data-testid',testid);return button;}
  nativeButton??=fallback('panorama','Walk');
  const originals=[nativeButton,doc.querySelector(DOLLHOUSE)||fallback('dollhouse','⌂'),doc.querySelector(FLOORPLAN)||fallback('floorplan','▦'),doc.querySelector(RULER)||fallback('measure-mode-button','↔')];
  for(const original of new Set(originals)){
   const button=original.tagName.toLowerCase()==='button'?original.cloneNode(true):element('button',original.textContent||'Walk');resetClone(button);button.disabled=false;button.className=button.className.split(/\s+/).filter(name=>!/^active(?:-|$)/.test(name)&&!['mp-nova-active','mp-nova-disabled'].includes(name)).join(' ');button.setAttribute('aria-pressed','false');if(button.getAttribute('role')==='radio')button.setAttribute('aria-checked','false');
   const computed=doc.defaultView.getComputedStyle(original);
   for(const key of ['backgroundColor','color','border','boxShadow','font','display','alignItems','justifyContent','padding','boxSizing'])button.style[key]=computed[key];
   button.style.backgroundColor='rgba(0,0,0,.5)';button.style.color='#fff';button.style.boxShadow='none';button.style.outline='0';
   if(original===nativeButton){label(button,'Walk 3D (reconstructed)');button.setAttribute('aria-pressed','true');if(button.getAttribute('role')==='radio')button.setAttribute('aria-checked','true');button.addEventListener('click',showWalk);}
   else if(original.matches(RULER)){rulerClone=button;button.addEventListener('click',()=>{const measure=api();if(typeof measure?.setEnabled==='function'){measure.setEnabled(!measure.state?.enabled);syncMeasure();}});}
   else{const mode=original.matches(DOLLHOUSE)?'DOLLHOUSE':'FLOORPLAN';label(button,mode==='DOLLHOUSE'?'Dollhouse (Matterport)':'Floor plan (Matterport)');button.addEventListener('click',()=>showTour(mode));}
   bar.append(button);
  }
  tour.viewport.append(mirror);review.toolbar=true;syncMeasure();
 }
 function installToolbar(){
  let doc;try{doc=tour.iframe.contentDocument;}catch{return;}
  const person=doc?.querySelector(PERSON),group=person?.closest(TOOLBAR)||doc?.querySelector(TOOLBAR);
  if(!group||!person)return;
  if(nativeButton!==person){nativeButton=person;mirrorToolbar(doc,group);}
  nativeButton.disabled=!available;nativeButton.setAttribute('aria-disabled',String(!available));label(nativeButton,available?'Walk 3D (reconstructed)':'No splat data: 3D walk not built for this home yet.');review.toolbar=true;if(!doc.getElementById('merge-pill-size')){const pill=doc.createElement('style');pill.id='merge-pill-size';pill.textContent='[class*="pill-value-"]{clip-path:inset(4px round 999px)}';doc.head.append(pill);}
 }
 function interceptPerson(event){if(!available)return;const person=event.target.closest?.(PERSON);if(person){event.preventDefault();event.stopImmediatePropagation();showWalk();}}
 function watchToolbar(){observer?.disconnect();watchedDoc?.removeEventListener('click',interceptPerson,true);let doc;try{doc=tour.iframe.contentDocument;}catch{return;}if(!doc?.documentElement)return;watchedDoc=doc;doc.addEventListener('click',interceptPerson,true);observer=new MutationObserver(installToolbar);observer.observe(doc.documentElement,{childList:true,subtree:true});installToolbar();}
 frame.addEventListener('load',syncMeasure);
 poll=setInterval(()=>{if(!defaultOpened&&!review.flight.lastAction&&available&&window.demoTour?.status==='ready')showWalk();if(walking)syncMeasure();},400);
 window.addEventListener('pagehide',()=>{observer?.disconnect();watchedDoc?.removeEventListener('click',interceptPerson,true);clearInterval(poll);},{once:true});
 const capturesReady=fetch('/walk/'+encodeURIComponent(sid)+'/index.html',{method:'HEAD',cache:'no-store'}).then(async response=>{
  available=response.ok;review.walkUrlOk=available;
  if(available)try{const r=await fetch('/walk/'+encodeURIComponent(sid)+'/captures.json',{cache:'no-store'});if(r.ok)captures=(await r.json()).captures?.filter(c=>typeof c.id==='string')||[];}catch{}
  availability.textContent=available?'The person icon opens the reconstructed Walk 3D.':'No splat data: 3D walk not built for this home yet.';if(tour)installToolbar();
 }).catch(()=>{availability.textContent='Could not check 3D walk availability.';});
 fetch('/measure/data/'+encodeURIComponent(sid),{cache:'no-store'}).then(async response=>{
  if(!response.ok)throw Error();const data=await response.json(),rooms=data.rooms||[],box=$('matterport-dimensions');box.replaceChildren();
  if(!rooms.length){box.append(element('p',"Matterport's room dimensions are unavailable for this home.",'caption'));return;}
  const table=element('table'),head=element('tr');for(const text of ['Room',"Matterport's dimension estimates"])head.append(element('th',text));const thead=element('thead');thead.append(head);table.append(thead);const body=element('tbody');
  const labels={length:'Length',width:'Width',height:'Height',minHeight:'Minimum ceiling',maxHeight:'Maximum ceiling'};
  for(const room of rooms){const row=element('tr'),values=Object.entries(room.dimensions_m||{}).filter(([key,value])=>labels[key]&&Number.isFinite(value)&&value>0);const text=values.map(([key,value])=>{const inches=Math.round(value/.0254);return labels[key]+' '+value.toFixed(2)+' m ('+Math.floor(inches/12)+' ft '+inches%12+' in)';}).join(' · ');row.append(element('td',room.label),element('td',text));body.append(row);}
  table.append(body);box.append(table);
 }).catch(()=>{$('matterport-dimensions').textContent="Matterport's room dimensions are unavailable for this home.";});
 return {showTour,attachTour(value){tour=value;tour.viewport.append(frame);const panel=tour.viewport.closest('#tour-panel');panel?.after(comparison);tour.iframe.addEventListener('load',watchToolbar);watchToolbar();}};
}
