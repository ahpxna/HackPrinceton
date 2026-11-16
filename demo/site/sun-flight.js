import {element, timeLabel} from './common.js?v=3';

const homes=new Set(['khbuN7q853q','VN4b1MGnnCR']);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
function deadline(promise,ms=10000){let timer;return Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Flight timeout')),ms);})]).finally(()=>clearTimeout(timer));}
function safeError(error){return {name:['Error','TypeError','RangeError','SyntaxError'].includes(error?.name)?error.name:'SDK error',message:String(error?.message||'No message').replace(/https?:\/\/[^\s"'<>]+/gi,'[URL]').replace(/(?:applicationKey|token|secret|authorization|signature|api[_-]?key)\s*[:=]\s*[^\s,;]+/gi,'[credential removed]').replace(/[A-Za-z0-9_-]{12,}/g,'[redacted]').slice(0,240)};}
const transform=(fit,p)=>fit.rotation.map((row,i)=>dot(row,p)+fit.translation[i]);
function inverseXY(data,e,n){const [[a,b],[c,d]]=data.transform.registered_matrix,det=a*d-b*c;return [(d*e-b*n)/det,(-c*e+a*n)/det];}
function horizonAt(values,az){const x=((az-.5)%360+360)%360,i=Math.floor(x);return values[i]*(1-x+i)+values[(i+1)%360]*(x-i);}

// Use the engine's saved aperture verdicts and floor grid, with its ray/plane
// intersection equations. No new solar-position calculation is performed.
export function chooseFlight(data,frame,context={horizons:{},daily:[]}){
 if(!frame||data.placement_status!=='well')return null;
 const day=data.days[frame.date],index=frame.index,hour=day.local_times[index].slice(11,13),hz=context.horizons||{};
 const clear=(w,i)=>!hz[w.id]||day.sun_apparent_elevation_deg[i]>horizonAt(hz[w.id],day.sun_azimuth_deg[i]);
 const ranked=data.windows.filter(w=>w.type==='window'&&w.floor_id===frame.floor).map(w=>({w,minutes:day.local_times.reduce((s,t,i)=>s+(t.slice(11,13)===hour&&clear(w,i)?day.weights_minutes[i]*(day.window_sample_codes[w.id][i].filter(c=>c===0).length/9):0),0)})).sort((a,b)=>b.minutes-a.minutes||a.w.id.localeCompare(b.w.id));
 const best=ranked[0];if(!best||best.minutes<=0)return null;
 const {w,minutes}=best,room=data.rooms.find(r=>r.id===w.room_id),codes=day.window_sample_codes[w.id][index];
 if(!room||!clear(w,index))return null;
 const az=day.sun_azimuth_deg[index]*Math.PI/180,el=day.sun_apparent_elevation_deg[index]*Math.PI/180;
 const xy=inverseXY(data,Math.sin(az)*Math.cos(el),Math.cos(az)*Math.cos(el)),direction=[...xy,Math.sin(el)];
 const tangent=w.tangent,normal=[-tangent[1],tangent[0]],den=dot(direction.slice(0,2),normal);
 if(Math.abs(den)<1e-8||direction[2]<=0)return null;
 const grid=data.geometry[room.id].grid,lit=new Set();
 for(const [a,b] of day.room_lit_runs[room.id][index]||[])for(let i=a;i<b;i++)lit.add(i);
 const candidates=[];
 for(let i=0;i<grid.length;i++){
  if(!lit.has(i))continue;
  const p=grid[i],delta=p.map((v,j)=>v-w.center[j]),travel=-dot(delta,normal)/den;
  const x=dot(delta,tangent)+travel*dot(direction.slice(0,2),tangent),z=room.floor_z_m+travel*direction[2];
  const u=(x/w.width+.5)*3,v=(z-w.lower_z_m)/w.height*3;
  if(travel>0&&u>=0&&u<3&&v>=0&&v<3&&codes[Math.floor(v)*3+Math.floor(u)]===0){
   candidates.push({patch:[...p,room.floor_z_m+.025],aperture:[p[0]+travel*direction[0],p[1]+travel*direction[1],z],distance:Math.hypot(...delta)});
  }
 }
 if(!candidates.length)return null;
 candidates.sort((a,b)=>a.distance-b.distance);
 return {...candidates[Math.floor(candidates.length/2)],window:w.id,room:room.id,minutes,date:frame.date,index,floor:frame.floor,time:day.local_times[index],dsm:Boolean(hz[w.id]),dsmDaily:context.daily?.find(r=>r.window_id===w.id&&r.date===frame.date)?.minutes??null};
}

export function mountFlight(data,section,goTo,merge){
 if(!homes.has(data.sid))return null;
 const box=element('div',null,'notice');box.id='sun-flight';
 const label=element('strong','modelled: geometry, clear sky; not measured');label.id='flight-label';
 const description=element('p','Aerial-to-window sun flight · selected date, hour and floor. The hour ranks windows by clear-aperture area-weighted minutes. The endpoint must be a saved lit floor cell reached through that window at the selected time.','caption');
 const button=element('button','Fly to the sun'),find=element('button','Find a sunlit time'),stop=element('button','Return to tour'),fallbackButton=element('button','Move by nearest sweep');button.disabled=true;stop.disabled=true;
 button.id='flight-start';stop.id='flight-stop';fallbackButton.id='flight-sweep';
 const status=element('p','Flight waits for the tour and a measured coordinate fit.');status.id='flight-status';status.setAttribute('aria-live','polite');
 const provenance=element('p','Checking optional DSM horizon…','caption');provenance.id='flight-dsm';
 find.id='flight-find';box.append(label,description,button,' ',find,' ',stop,' ',fallbackButton,status,provenance);section.append(box);
 const state=window.demoFlight={status:'waiting',registered:false,componentInitialized:false,pathPointCount:0,cameraMoved:false,cameraDistance:0,ray:false,aerialLoaded:false,fallback:false,selection:null,reason:null};
 let sdk,fit,frame,context={horizons:{},daily:[]},node,stopped=false,selection,component,object,texture,contextReady=false,initialize,busy=false,flightStage='waiting for the tour';
 const contextPromise=fetch('/flight/data/'+data.sid).then(r=>{if(!r.ok)throw Error();return r.json();}).then(c=>{
  context=c;provenance.textContent=c.status==='available'?'DSM horizon applied to aperture samples before hourly ranking and floor-ray selection. Centre-of-window, 1° conservative opaque DSM horizon; canopy transmission and leaf state unknown.':'DSM horizon unavailable for this home (daily DSM values are empty). Using saved geometry only.';
  if(c.daily.length&&!Object.keys(c.horizons).length){state.contextError=true;provenance.textContent='DSM daily column exists, but hourly horizon arrays are unavailable. Flight withheld rather than use a daily total as an hourly correction.';}
 }).catch(()=>{provenance.textContent='DSM context unavailable; flight withheld until its availability can be checked.';state.contextError=true;}).finally(()=>{contextReady=true;refresh();});
 function updateButtons(){button.disabled=busy||!sdk||!contextReady||state.contextError;find.disabled=busy;fallbackButton.disabled=busy;}
 function fallback(reason){state.status='fallback';state.fallback=true;state.reason=reason;updateButtons();status.textContent=reason+' Nearest-sweep navigation and plan remain available.';}
 function release(){if(node){node.stop();node=null;}component=null;stop.disabled=true;updateButtons();}
 function refresh(){
  if(stopped)return;
  if(node){release();state.status='ready';status.textContent='Selection changed; flight stopped. Choose Fly to the sun to restart.';}
  selection=chooseFlight(data,frame,context);state.selection=selection;
  updateButtons();
  if(!sdk)return;
  if(!state.registered)return;
  if(!selection){state.status='no-sun';status.textContent='No floor patch through the highest-sun window at this selected time and floor. Change date/time; no flight or sun ray is invented.';return;}
  state.status='ready';state.fallback=false;
  status.textContent=`Window ${selection.window} · ${selection.minutes.toFixed(1)} area-weighted clear minutes in the selected hour · floor target at ${timeLabel(selection.time)}.${selection.dsm?' DSM horizon applied.':''}${selection.dsmDaily!==null?' DSM daily total '+selection.dsmDaily.toFixed(1)+' minutes.':''} ${data.sid==='khbuN7q853q'?'Hand-checked aerial placement.':'Registered aerial placement; not hand checked.'} Camera path is illustrative; furniture and collision clearance are not modelled.`;
 }
 async function action(name,run){
  if(busy)return;
  const proof=window.demoReview?.merge?.flight;if(proof)Object.assign(proof,{lastAction:name,ok:false,error:null});
  busy=true;updateButtons();status.textContent='Opening Tour…';
  try{
   if(merge&&!await merge.showTour('INSIDE'))throw Error('Tour could not be made visible.');
   await contextPromise;
   if(!sdk||stopped)throw Error('Tour is not ready. Wait for the tour to load and try again.');
   if(state.contextError)throw Error('Sun context is unavailable; no sun target can be verified.');
   await run();if(proof)proof.ok=true;
  }catch(error){const detail=safeError(error);if(proof)proof.error=detail.message;state.status='action-error';status.textContent=detail.message;}
  finally{busy=false;updateButtons();}
 }
 function nextSun(){
  if(!frame)throw Error('Date, time and floor are not ready yet.');
  const day=data.days[frame.date];
  for(let offset=1;offset<=day.local_times.length;offset++){
   const index=(frame.index+offset)%day.local_times.length,chosen=chooseFlight(data,{...frame,index},context);
   if(chosen){frame={...frame,index};const slider=document.getElementById('time');slider.value=index;slider.dispatchEvent(new Event('input'));selection=chosen;state.selection=chosen;return chosen;}
  }
  throw Error('No qualifying window-to-floor sun target on this date and floor. Choose another date or floor.');
 }
 async function moveSweep(windowId,text){
  release();if(!windowId||!await goTo([], [windowId]))throw Error(text+' The tour could not reach the nearest sweep.');
  state.fallback=true;state.status='sweep';status.textContent=text+' Moved to the nearest available Matterport sweep.';
 }
 fallbackButton.onclick=()=>action('sweep',()=>moveSweep(selection?.window||data.windows.find(w=>w.type==='window'&&w.floor_id===frame?.floor)?.id,'Sun ray and custom camera are off.'));
 find.onclick=()=>action('find',async()=>{const chosen=nextSun();await moveSweep(chosen.window,`Found ${timeLabel(chosen.time)} · window ${chosen.window} · saved sunlit floor target.`);});
 stop.onclick=()=>action('stop',async()=>{release();refresh();});
 async function ready(connection,measuredFit){
  sdk=connection;fit=measuredFit;
  if(!fit?.accepted||fit.rms_m>.05||fit.max_m>.15){fallback('Flight withheld: coordinate fit is unverified.');return;}
  try{
   await deadline(sdk.Scene.register('sun.flight',()=>({outputs:{camera:null},onInit(){
    try{
     flightStage='initializing the Three.js camera';
     const T=this.context.three,root=new T.Group(),camera=new T.PerspectiveCamera(65,1,.03,2000);camera.up.set(...fit.rotation.map(row=>dot(row,[0,0,1])));
     this.outputs.objectRoot=root;
     component={T,root,camera,outputs:this.outputs,start:performance.now(),resources:[],aerial:null,active:false};state.componentInitialized=true;initialize?.resolve(component);
    }catch{initialize?.reject(Error('Scene initialization failed'));}
   },onTick(){
    const c=component;if(!c?.curve||!c.active)return;const t=Math.min(1,(performance.now()-c.start)/14000),p=c.curve.getPointAt(t);
    c.camera.aspect=Math.max(1,document.getElementById('tour-frame').clientWidth)/Math.max(1,document.getElementById('tour-frame').clientHeight);c.camera.updateProjectionMatrix();c.camera.position.copy(p);
    const target=t<.98?c.curve.getPointAt(Math.min(1,t+.035)):c.target;c.camera.lookAt(target);
    state.cameraDistance=c.startPoint.distanceTo(p);state.cameraMoved=state.cameraDistance>.1;
    if(c.aerial){c.aerial.material.opacity=Math.max(0,1-t/.32);c.aerial.visible=t<.32;}
    if(t===1&&state.status!=='complete'){state.status='complete';status.textContent='Arrived above the modelled floor patch. Thin yellow tube: window → floor ray. Return to tour restores normal navigation.';}
   },onDestroy(){this.outputs.camera=null;if(component){for(const r of component.resources)r.dispose();component=null;}}})));
   if(stopped)return;state.registered=true;await contextPromise;refresh();
  }catch{fallback('Bundle camera component unavailable.');}
 }
 button.onclick=()=>action('fly',async()=>{
  const chosen=selection||nextSun();
  if(!state.registered){await moveSweep(chosen.window,`Custom flight unavailable. Sun target at ${timeLabel(chosen.time)} · window ${chosen.window}.`);return;}
  release();button.disabled=true;state.status='loading';state.reason=null;state.cameraMoved=false;state.ray=false;state.pathPointCount=0;state.aerialLoaded=false;state.componentInitialized=false;
  try{
   flightStage='Scene.createObjects';state.stage=flightStage;const objects=await deadline(sdk.Scene.createObjects(1));if(stopped)return;
   const initialized=new Promise((resolve,reject)=>{initialize={resolve,reject};});
   object=objects[0];flightStage='object.addNode';state.stage=flightStage;node=object.addNode();flightStage='node.addComponent';state.stage=flightStage;node.addComponent('sun.flight');flightStage='node.start / onInit';state.stage=flightStage;node.start();
   const c=await deadline(initialized);initialize=null;if(stopped||component!==c)return;flightStage='building the fitted camera path';const T=c.T,toV=p=>new T.Vector3(...transform(fit,p));
   const room=data.rooms.find(r=>r.id===chosen.room),grid=data.geometry[room.id].grid,center=[grid.reduce((s,p)=>s+p[0],0)/grid.length,grid.reduce((s,p)=>s+p[1],0)/grid.length];
   const ground=data.assumptions.model_ground_z_m,roof=Math.max(...data.rooms.map(r=>r.ceiling_z_m),ground+4)+.5;
   const [a,p]=[chosen.aperture,chosen.patch],out=a.map((v,i)=>v-p[i]),length=Math.hypot(out[0],out[1]);
   if(length<1e-6)throw Error();
   const points=[[...center,roof+38],[...center,roof+13],[a[0]+out[0]/length*4,a[1]+out[1]/length*4,a[2]+2],[a[0]+out[0]/length*.5,a[1]+out[1]/length*.5,a[2]+.3],[a[0]-out[0]/length*.4,a[1]-out[1]/length*.4,a[2]],[p[0],p[1],p[2]+.65]].map(toV);
   c.curve=new T.CatmullRomCurve3(points,false,'centripetal');c.target=toV(p);c.startPoint=points[0].clone();state.pathPointCount=points.length;
   const mesh=(geometry,material)=>{const m=new T.Mesh(geometry,material);c.root.add(m);c.resources.push(geometry,material);return m;};
   mesh(new T.TubeGeometry(c.curve,100,.012,5,false),new T.MeshBasicMaterial({color:0x76c8e8,transparent:true,opacity:.3}));
   mesh(new T.TubeGeometry(new T.LineCurve3(toV(a),toV(p)),1,.018,6,false),new T.MeshBasicMaterial({color:0xffd52b}));state.ray=true;
   const marker=mesh(new T.SphereGeometry(.10,12,8),new T.MeshBasicMaterial({color:0xffd52b}));marker.position.copy(toV(p));
   // Vexcel is a registered 2D photo plane, never a reconstructed 3D roof.
   const [west,south,east,north]=data.aerial.bounds_enu_m,shift=data.transform.registered_shift;
   const corners=[[west,south],[east,south],[east,north],[west,north]].map(([e,n])=>[...inverseXY(data,e-shift[0],n-shift[1]),roof]);
   const positions=[0,1,2,0,2,3].flatMap(i=>transform(fit,corners[i])),geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(positions,3));geometry.setAttribute('uv',new T.Float32BufferAttribute([0,0,1,0,1,1,0,0,1,1,0,1],2));
   const aerialURL=new URL(data.aerial.image,location.href);
   flightStage='loading the registered aerial texture';texture=await deadline(new Promise((resolve,reject)=>new T.TextureLoader().load(aerialURL.href,resolve,undefined,reject)));
   if(stopped||component!==c){texture.dispose();return;}
   c.resources.push(texture);c.aerial=mesh(geometry,new T.MeshBasicMaterial({map:texture,side:T.DoubleSide,transparent:true,opacity:1,depthWrite:false}));state.aerialLoaded=true;
   c.start=performance.now();c.active=true;c.outputs.camera=c.camera;stop.disabled=false;state.status='flying';
   const patchToggle=document.getElementById('tour-patch');if(patchToggle&&!patchToggle.disabled){patchToggle.checked=true;patchToggle.dispatchEvent(new Event('change'));}
   status.textContent=`Vexcel registered aerial → window ${chosen.window} → modelled floor patch. Photo fades before entering; its height is a presentation plane. Use Return to tour to stop.`;
  }catch(error){state.error=safeError(error);state.stage=flightStage;initialize=null;await moveSweep(chosen.window,`Custom flight unavailable while ${flightStage}. Sun target at ${timeLabel(chosen.time)} · window ${chosen.window}.`);}
 });
 return {ready,setFrame(value){frame=value;refresh();},pause(){release();refresh();},unavailable(reason){release();sdk=null;fallback(reason);},destroy(){stopped=true;release();}};
}
