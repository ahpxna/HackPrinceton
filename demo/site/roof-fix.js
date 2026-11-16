const CUT_MODES=new Set(['mode.dollhouse','mode.floorplan']);
const MARGIN_M=.15;
const median=values=>{const v=values.slice().sort((a,b)=>a-b),i=Math.floor(v.length/2);return v.length%2?v[i]:(v[i-1]+v[i])/2;};
function timeout(promise,ms){let timer;return Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Roof control timeout')),ms);})]).finally(()=>clearTimeout(timer));}

function floorCuts(data,fit,transform){
 const up=transform(fit,[0,0,1],true);
 if(up[1]<.999||Math.hypot(up[0],up[2])>.02)throw new Error('Roof cut requires SDK Y-up alignment');
 const samples=new Map();
 for(const room of data.rooms){
  if(!room.floor_id||!Number.isFinite(room.floor_z_m)||!Number.isFinite(room.ceiling_z_m)||room.ceiling_z_m-room.floor_z_m<.5)continue;
  const grid=data.geometry?.[room.id]?.grid;
  if(!grid?.length)continue;
  const center=grid.reduce((sum,p)=>[sum[0]+p[0]/grid.length,sum[1]+p[1]/grid.length],[0,0]);
  const height=transform(fit,[...center,room.ceiling_z_m])[1];
  if(!Number.isFinite(height))continue;
  if(!samples.has(room.floor_id))samples.set(room.floor_id,[]);
  samples.get(room.floor_id).push(height);
 }
 return new Map([...samples].map(([id,values])=>[id,Math.round((median(values)-MARGIN_M)*1000)/1000]));
}

export function mountRoofFix(data,transform){
 const review=window.demoReview??(window.demoReview={});
 const state=review.roofFix={mechanism:'World-height roofTrimHeight on streamed model chunks; camera-independent cut',mode:null,hiddenChunks:0,toggles:null,status:'waiting-for-fit',active:false,cutHeight:null,marginM:MARGIN_M,modelMeshes:0,trackedChunks:0,fullyHiddenChunks:0,unsupportedChunks:0,uniformWrites:0,uniformCorrections:0,totalToggles:0,observedFrames:0,observedSeconds:0,measurement:'Above-cut shader visibility before render; live visual confirmation required'};
 const records=new Map(),lastVisibility=new Map(),flips=[];
 let node,scene,three,previousRender,beforeRender,modeSub,floorSub,stopped=false,rendering=false,cuts,idMap={},floor,cut=null,lastCutMode=null,policy=null,started=null;

 function reset(){flips.length=0;lastVisibility.clear();started=null;state.toggles=null;state.observedSeconds=0;state.observedFrames=0;state.totalToggles=0;}
 state.reset=reset;
 function chooseFloor(){
  if(floor?.sequence===-1){cut=Math.max(...cuts.values());state.floor='all';return;}
  if(floor?.id==='')return;
  const id=floor?.id,alias=idMap[id],reverse=Object.keys(idMap).find(key=>idMap[key]===id);
  const match=[id,alias,reverse].find(key=>cuts.has(key));
  cut=match?cuts.get(match):null;state.floor=match??null;
 }
 function restore(chunk,record){
  if(record.target!==null&&chunk.uniformCache?.roofTrimHeight?.value===record.target){
   try{chunk.setMaterialsUniform({roofTrimHeight:record.original});}catch{state.restoreErrors=(state.restoreErrors||0)+1;}
  }
 }
 function release(){
  rendering=false;
  if(scene&&scene.onBeforeRender===beforeRender)scene.onBeforeRender=previousRender;
  for(const [chunk,record] of records)restore(chunk,record);
  records.clear();lastVisibility.clear();state.active=false;state.hiddenChunks=0;state.fullyHiddenChunks=0;state.trackedChunks=0;state.status='released';
 }
 function scan(){
  const active=(CUT_MODES.has(state.mode)||(state.mode==='mode.transitioning'&&lastCutMode!==null))&&Number.isFinite(cut);
  const nextPolicy=active?`${lastCutMode}:${cut}`:'off';
  if(nextPolicy!==policy){reset();policy=nextPolicy;}
  state.active=active;state.cutHeight=active?cut:null;
  if(!active){
   for(const [chunk,record] of records)restore(chunk,record);
   records.clear();state.hiddenChunks=0;state.fullyHiddenChunks=0;state.trackedChunks=0;state.modelMeshes=0;state.unsupportedChunks=0;
   state.status=CUT_MODES.has(state.mode)?'unmapped-floor':'inactive';return;
  }
  const seen=new Set(),now=performance.now(),box=new three.Box3();
  let hidden=0,fullyHidden=0,meshes=0,unsupported=0,errors=0,exposed=0;
  scene.traverse(mesh=>{
   if(!mesh.isModelMesh||!Array.isArray(mesh.chunks))return;
   meshes++;
   for(const chunk of mesh.chunks){
    if(seen.has(chunk))continue;
    seen.add(chunk);
    const uniform=chunk.uniformCache?.roofTrimHeight;
    if(!uniform||typeof chunk.setMaterialsUniform!=='function'){unsupported++;continue;}
    let record=records.get(chunk);
    if(!record){record={original:uniform.value,target:null};records.set(chunk,record);}
    if(uniform.value!==cut){
     if(record.target===null||uniform.value!==record.target){record.original=uniform.value;if(record.target!==null)state.uniformCorrections++;}
     try{chunk.setMaterialsUniform({roofTrimHeight:cut});record.target=cut;state.uniformWrites++;}catch{errors++;}
    }
    const geometry=chunk.geometry;
    if(!geometry)continue;
    if(!geometry.boundingBox)geometry.computeBoundingBox();
    if(!geometry.boundingBox)continue;
    box.copy(geometry.boundingBox).applyMatrix4(mesh.matrixWorld);
    if(box.max.y<=cut)continue;
    const clipped=uniform.value>-10000&&uniform.value<=cut;
    if(clipped){hidden++;if(box.min.y>cut)fullyHidden++;}else exposed++;
    let visible=true;for(let p=mesh;p;p=p.parent)if(!p.visible){visible=false;break;}
    const roofVisible=visible&&(chunk.getOpacity?.()??1)>0&&!clipped;
    if((lastVisibility.has(chunk)&&lastVisibility.get(chunk)!==roofVisible)||(!lastVisibility.has(chunk)&&roofVisible)){
     flips.push(now);state.totalToggles++;
    }
    lastVisibility.set(chunk,roofVisible);
   }
  });
  for(const [chunk,record] of records)if(!seen.has(chunk)){restore(chunk,record);records.delete(chunk);lastVisibility.delete(chunk);}
  if(started===null)started=now;
  while(flips.length&&flips[0]<=now-1000)flips.shift();
  const seconds=(now-started)/1000;
  state.toggles=seconds>=1?flips.length:null;state.observedSeconds=Math.round(seconds*10)/10;state.observedFrames++;
  state.hiddenChunks=hidden;state.fullyHiddenChunks=fullyHidden;state.modelMeshes=meshes;state.trackedChunks=records.size;state.unsupportedChunks=unsupported;
  state.status=errors||unsupported?'partial':records.size?'active':'awaiting-model-chunks';
  state.writeErrors=errors;
  state.exposedRoofChunks=exposed;
 }
 function destroy(){
  if(stopped)return;
  stopped=true;modeSub?.cancel();floorSub?.cancel();node?.stop();node=null;release();state.status='stopped';
 }
 async function ready(sdk,fit){
  try{
   cuts=floorCuts(data,fit,transform);
   if(!cuts.size)throw new Error('No measured floor ceilings');
   state.floorCuts=Object.fromEntries(cuts);
   try{idMap=await timeout(sdk.Floor.Conversion.createIdMap(),5000);}catch{}
   if(stopped)return;
   modeSub=sdk.Mode.current.subscribe(mode=>{state.mode=mode;if(CUT_MODES.has(mode))lastCutMode=mode;else if(mode!=='mode.transitioning')lastCutMode=null;});
   floorSub=sdk.Floor.current.subscribe(value=>{floor=value;chooseFloor();});
   await timeout(sdk.Scene.register('sun.roofFix',()=>({
    onInit(){
     ({scene,three}=this.context);previousRender=scene.onBeforeRender;rendering=true;
     beforeRender=function(...args){
      previousRender?.apply(this,args);
      if(stopped||!rendering)return;
      try{scan();}catch{release();state.status='unavailable';state.reason='Model chunk interface changed';}
     };
     scene.onBeforeRender=beforeRender;this.outputs.objectRoot=new three.Group();state.status='awaiting-render';
    },
    onDestroy:release
   })),10000);
   if(stopped)return;
   const [object]=await timeout(sdk.Scene.createObjects(1),10000);
   if(stopped)return;
   node=object.addNode();node.addComponent('sun.roofFix');node.start();
  }catch{
   modeSub?.cancel();floorSub?.cancel();node?.stop();release();state.status='unavailable';state.reason='Floor alignment, SDK floor mapping or model chunk interface unavailable';
  }
 }
 return {ready,destroy};
}
