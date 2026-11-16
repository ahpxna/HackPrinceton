import {element} from './common.js?v=3';

export function mountTourKeys(panel,iframe,controls,state,getSweeps,isStopped){
 Object.assign(state,{hops:0,lastHopMs:null,lastInputToMoveMs:null,vfov:null,cleanMoves:true});
 const viewport=element('div',null,'tour-viewport');iframe.before(viewport);viewport.append(iframe);
 viewport.tabIndex=0;viewport.setAttribute('aria-label','Tour navigation: W/S forward/back, A/D strafe, arrows move and turn');
 const cleanLabel=element('label'),clean=element('input');clean.type='checkbox';clean.id='tour-clean-moves';clean.checked=true;
 cleanLabel.append(clean,' Clean moves');cleanLabel.title='Instant photo-to-photo hops avoid mesh morphing. Uncheck for 350 ms fly transitions.';
 const wideLabel=element('label'),wide=element('input');wide.type='checkbox';wide.id='tour-wide';wideLabel.append(wide,' Wide view (unchecked: Normal)');
 const padToggle=element('button','Show movement pad');padToggle.id='tour-pad-toggle';padToggle.setAttribute('aria-expanded','false');
 controls.append(cleanLabel,wideLabel,padToggle);
 const pad=element('div',null,'tour-pad');pad.id='tour-pad';pad.setAttribute('aria-label','Tour movement pad');
 const touch=navigator.maxTouchPoints>0||matchMedia('(any-pointer: coarse)').matches;pad.hidden=!touch;padToggle.setAttribute('aria-expanded',String(touch));
 padToggle.textContent=touch?'Hide movement pad':'Show movement pad';
 viewport.append(pad);
 const held=new Map(),listeners=[],subscriptions=[];
 let sdk,pose,current,zoomLevel=1,fovBusy=false,fovPending=false,hopBusy=false,queued=null,hopTiming=null,nextHopAt=0,raf=0,turnLast=0,turnBusy=false,destroyed=false,nudgeTimer,active=true;
 const keys={ArrowUp:'forward',w:'forward',ArrowDown:'back',s:'back',a:'left',d:'right',ArrowLeft:'turnLeft',ArrowRight:'turnRight'};
 const flightOwnsCamera=()=>['loading','flying','complete'].includes(window.demoFlight?.status);
 const available=()=>active&&sdk&&state.status==='ready'&&!isStopped()&&!destroyed&&!flightOwnsCamera();
 const editable=target=>Boolean(target?.closest?.('input,textarea,select,[contenteditable]:not([contenteditable="false"]),[role="textbox"]'));
 const on=(target,type,handler,options)=>{target.addEventListener(type,handler,options);listeners.push(()=>target.removeEventListener(type,handler,options));};
 const fovOf=p=>p?.projection?.[5]>0?2*Math.atan(1/p.projection[5])*180/Math.PI:null;
 function observePose(p){
  pose=p;state.vfov=fovOf(p);
  if(hopTiming&&Math.hypot(...['x','y','z'].map(axis=>p.position[axis]-hopTiming.position[axis]))>.001){
   state.lastInputToMoveMs=performance.now()-hopTiming.inputAt;hopTiming=null;
  }
 }
 function nudge(action){viewport.dataset.blocked=action;clearTimeout(nudgeTimer);nudgeTimer=setTimeout(()=>delete viewport.dataset.blocked,180);}
 function clear(){held.clear();queued=null;}
 async function applyFov(){
  if(!sdk||destroyed||isStopped()||flightOwnsCamera())return;
  if(fovBusy){fovPending=true;return;}
  fovBusy=true;
  try{
   const target=wide.checked?75:60;
   // Verify the projection, rather than assuming a zoom factor yields the requested FOV.
   for(let i=0;i<5;i++){
    const measured=fovOf(await sdk.Camera.getPose());state.vfov=measured;
    if(!measured||Math.abs(measured-target)<.35)break;
    if(flightOwnsCamera()||isStopped()||destroyed)break;
    zoomLevel=await sdk.Camera.zoomTo(zoomLevel*measured/target);
   }
   const after=await sdk.Camera.getPose();state.vfov=fovOf(after);
  }catch{state.fovUnavailable=true;}
  finally{fovBusy=false;if(fovPending){fovPending=false;applyFov();}}
 }
 const matches=(s,id)=>id&&[s.id,s.sid,s.uuid,s.map_id].includes(id);
 const deltaAngle=(a,b)=>(((a-b)%360+540)%360)-180;
 // Where the user is actually looking, measured by projecting probe points through the live camera
 // instead of trusting a yaw sign convention. Bearing is clockwise from -Z seen from above.
 function viewBearing(){
  const w=iframe.clientWidth,h=iframe.clientHeight,project=sdk?.Conversion?.worldToScreen;
  if(project&&w&&h){
   const probe=b=>{const r=b*Math.PI/180;
    try{return project.call(sdk.Conversion,{x:pose.position.x+2*Math.sin(r),y:pose.position.y,z:pose.position.z-2*Math.cos(r)},pose,{w,h});}catch{return null;}};
   let best=null;
   for(let b=0;b<360;b+=5){
    const c=probe(b),side=probe(b+20);
    if(!c||!side||!Number.isFinite(c.x)||!Number.isFinite(side.x)||side.x<=c.x)continue;
    const error=Math.abs(c.x-w/2);
    if(!best||error<best.error)best={b,error};
   }
   if(best&&best.error<w/4){state.heading=best.b;state.headingSource='projection';return best.b;}
  }
  state.heading=-pose.rotation.y;state.headingSource='yaw';
  return -pose.rotation.y;
 }
 function candidate(action){
  if(!pose||!current)return null;
  const sweeps=getSweeps(),origin=sweeps.find(s=>matches(s,current.id)||matches(s,current.sid)||matches(s,pose.sweep));
  if(!origin)return null;
  const heading=viewBearing()+({forward:0,back:180,left:-90,right:90}[action]||0);
  // A walking step: Matterport neighbours include far line-of-sight sweeps (yard through a door), so keep
  // the hop short, on the same floor and level, with the same alignment, and indoors only if starting indoors.
  const sameFloor=s=>!origin.floorInfo?.id||!s.floorInfo?.id||s.floorInfo.id===origin.floorInfo.id;
  const sameSide=s=>!origin.roomInfo?.id||!!s.roomInfo?.id;
  const level=s=>Math.abs((s.puckPosition?.y??s.position.y)-(origin.puckPosition?.y??origin.position.y))<.5;
  return sweeps.filter(s=>s.enabled!==false&&s!==origin&&origin.neighbors?.some(id=>matches(s,id))
    &&sameFloor(s)&&sameSide(s)&&level(s)&&s.alignmentType===origin.alignmentType)
   .map(s=>{const dx=s.position.x-origin.position.x,dz=s.position.z-origin.position.z;
    // SDK yaw is clockwise from -Z in the Y-up sweep coordinate system.
    const angle=Math.atan2(dx,-dz)*180/Math.PI,distance=Math.hypot(dx,dz);
    return {s,angle:Math.abs(deltaAngle(angle,heading)),distance};})
   .filter(c=>c.distance>.05&&c.distance<=4&&c.angle<=35)
   .sort((a,b)=>(a.angle/35+a.distance/4)-(b.angle/35+b.distance/4))[0]?.s;
 }
 function hop(action,inputAt){
  if(!available())return;
  if(hopBusy){queued={action,inputAt};return;}
  const target=candidate(action);
  if(!target){nudge(action);nextHopAt=performance.now()+350;return;}
  hopBusy=true;queued=null;const start=performance.now();state.lastInputDispatchMs=start-inputAt;
  hopTiming={inputAt,position:{...pose.position}};state.lastInputToMoveMs=null;
  let move;
  try{move=sdk.Sweep.moveTo(target.id,{rotation:{...pose.rotation},transition:state.cleanMoves?sdk.Sweep.Transition.INSTANT:sdk.Sweep.Transition.FLY,transitionTime:350});}
  catch{nudge(action);hopBusy=false;hopTiming=null;return;}
  Promise.resolve(move).then(async()=>{
   state.hops++;state.lastHopMs=performance.now()-start;state.lastHop=target.id;
   // Refresh both before chaining: instant transitions can resolve before observables publish.
   observePose(await sdk.Camera.getPose());current=target;await applyFov();
  }).catch(()=>nudge(action)).finally(()=>{
   hopBusy=false;hopTiming=null;nextHopAt=Math.max(nextHopAt,start+350);
   const pending=queued;queued=null;
   if(pending&&available())hop(pending.action,pending.inputAt);
  });
 }
 function turn(degrees){
  if(turnBusy||!available())return;
  turnBusy=true;
  try{Promise.resolve(sdk.Camera.rotate(degrees,0,{speed:90})).catch(()=>{}).finally(()=>{turnBusy=false;});}
  catch{turnBusy=false;}
 }
 function press(token,action){
  if(!available()||held.has(token))return;
  const now=performance.now();held.set(token,{action,at:now});
  if(action.startsWith('turn')){turnLast=now+167;turn(action==='turnLeft'?-15:15);}
  else hop(action,now);
 }
 function release(token){held.delete(token);if(!held.size)queued=null;}
 function keydown(event){
  const action=keys[event.key]||keys[event.key?.toLowerCase()];
  if(!action||editable(event.target)||event.altKey||event.ctrlKey||event.metaKey||!available())return;
  const doc=event.target.ownerDocument||document;
  if(doc===document&&!panel.matches(':hover')&&!panel.contains(document.activeElement))return;
  event.preventDefault();event.stopImmediatePropagation();
  if(!event.repeat)press('key:'+event.code,action);
 }
 function keyup(event){const token='key:'+event.code;if(held.has(token)){event.preventDefault();event.stopImmediatePropagation();release(token);}}
 function bindDocument(doc){
  on(doc,'keydown',keydown,true);on(doc,'keyup',keyup,true);on(doc.defaultView,'blur',clear);
  on(doc,'focusin',event=>{if(editable(event.target)||(doc===document&&!panel.contains(event.target)&&!panel.matches(':hover')))clear();});
 }
 bindDocument(document);
 for(const [action,label,title] of [['turnLeft','↶','Turn left'],['forward','↑','Step forward'],['turnRight','↷','Turn right'],['left','A','Strafe left'],['back','↓','Step back'],['right','D','Strafe right']]){
  const button=element('button',label);button.type='button';button.dataset.action=action;button.setAttribute('aria-label',title);pad.append(button);
  on(button,'pointerdown',event=>{if(!available())return;event.preventDefault();button.setPointerCapture(event.pointerId);viewport.focus({preventScroll:true});press('pointer:'+event.pointerId,action);});
  for(const type of ['pointerup','pointercancel','lostpointercapture'])on(button,type,event=>release('pointer:'+event.pointerId));
  on(button,'keydown',event=>{if([' ','Enter'].includes(event.key)){event.preventDefault();press('button:'+action,action);}});
  on(button,'keyup',event=>{if([' ','Enter'].includes(event.key)){event.preventDefault();release('button:'+action);}});
 }
 on(panel,'mouseleave',()=>{if(!panel.contains(document.activeElement))clear();});
 on(document,'visibilitychange',()=>{if(document.hidden)clear();});
 clean.onchange=()=>{state.cleanMoves=clean.checked;};wide.onchange=applyFov;
 padToggle.onclick=()=>{pad.hidden=!pad.hidden;padToggle.textContent=pad.hidden?'Show movement pad':'Hide movement pad';padToggle.setAttribute('aria-expanded',String(!pad.hidden));};
 function frame(now){
  if(destroyed)return;
  if(!available())clear();
  else{
   const actions=[...held.values()],movement=actions.find(v=>!v.action.startsWith('turn'));
   if(movement&&!hopBusy&&now>=nextHopAt)hop(movement.action,now);
   // The first 15° executes immediately; after its 167 ms, continue at 90°/s.
   const left=actions.some(v=>v.action==='turnLeft'&&now-v.at>=167),right=actions.some(v=>v.action==='turnRight'&&now-v.at>=167);
   if(left!==right&&!turnBusy){const dt=Math.min(Math.max(0,(now-turnLast)/1000),.05);turnLast=now;if(dt)turn((right?1:-1)*90*dt);}
   else if(!left&&!right)turnLast=now;
  }
  raf=requestAnimationFrame(frame);
 }
 return {
  setActive(value){active=Boolean(value);if(!active)clear();},
  ready(api){
   sdk=api;
   try{bindDocument(iframe.contentDocument);}catch{state.keyCaptureUnavailable=true;}
   subscriptions.push(sdk.Camera.pose.subscribe(observePose));
   subscriptions.push(sdk.Camera.zoom.subscribe(z=>{zoomLevel=z.level;}));
   subscriptions.push(sdk.Sweep.current.subscribe(s=>{const changed=s.id&&s.id!==current?.id;current=s;if(changed)applyFov();}));
   applyFov();raf=requestAnimationFrame(frame);
  },
  destroy(){destroyed=true;clear();cancelAnimationFrame(raf);clearTimeout(nudgeTimer);listeners.forEach(remove=>remove());subscriptions.forEach(s=>s.cancel());}
 };
}
