import {json,containedLabel} from './common.js?v=3';
import {chooseFlight} from './sun-flight.js?v=5';
import {parseDimensions} from './furniture.js?v=1';
const results=[],filterTimings=[],tours=[],frame=document.getElementById('target'),output=document.getElementById('results');
const pause=ms=>new Promise(r=>setTimeout(r,ms));
function check(name,condition){results.push({name,passed:Boolean(condition)});output.textContent=JSON.stringify({finished:false,checks:results},null,2);}
async function until(fn,attempts=400){for(let i=0;i<attempts;i++){if(fn())return;await pause(50);}throw new Error('Timed out waiting for page state');}
async function page(url,ready){frame.src=url;await new Promise(r=>frame.onload=r);try{await until(()=>ready(frame.contentDocument));}catch(error){throw new Error(url+': '+error.message+' · '+frame.contentDocument.getElementById('address')?.textContent+' · '+frame.contentDocument.getElementById('window-detail')?.textContent);}return frame.contentDocument;}
function input(doc,id,value){const start=performance.now(),e=doc.getElementById(id);e.value=value;e.dispatchEvent(new frame.contentWindow.Event('input',{bubbles:true}));filterTimings.push(performance.now()-start);}
try{
 const index=await(await fetch('data/index.json')).json();
 let d=await page('index.html',d=>d.querySelectorAll('.card').length>0);
 check('All supplied homes shown in result count',d.getElementById('count').textContent.includes('1,874'));
 check('Results paged to at most24 cards',d.querySelectorAll('.card').length===24);
 check('Search fits the browser width',d.documentElement.scrollWidth<=frame.contentWindow.innerWidth);
 check('Search loads exactly one compact index',frame.contentWindow.performance.getEntriesByType('resource').filter(r=>r.name.endsWith('/data/index.json')).length===1);
 const first=d.querySelector('.card').dataset.sid;d.getElementById('next').click();check('Next page changes results',d.querySelector('.card').dataset.sid!==first);
 input(d,'topic',String(index.topics.indexOf('ceilings')));input(d,'verdict','contradicted');check('Ceiling contradictions filter returns cases',d.querySelectorAll('.card').length>0&&[...d.querySelectorAll('.card')].every(e=>e.textContent.includes('contradicted')));
 d.getElementById('reset').click();input(d,'quality','poor');check('Poor homes remain searchable',d.querySelectorAll('.card').length>0);check('Poor cards show no sun numbers',d.querySelectorAll('.sun-number').length===0);
 d.getElementById('reset').click();input(d,'room','kitchen');input(d,'period','1');input(d,'minimum','120');check('Morning kitchen sun filter returns homes',d.querySelectorAll('.card').length>0);
 const count=()=>Number(d.getElementById('count').textContent.split(' of ')[0].replaceAll(',',''));
 for(const [id,value,predicate] of [['city','Houston',h=>h.city==='Houston'],['price-max','250000',h=>h.price!=null&&h.price<=250000],['ceiling','3',h=>h.ceiling!=null&&h.ceiling>=3],['pool','yes',h=>h.pool===true]]){d.getElementById('reset').click();input(d,id,value);check(id+' filter matches the complete supplied index',count()===index.homes.filter(predicate).length);}
 for(const period of [1,2]){d.getElementById('reset').click();input(d,'room','main');input(d,'date','2026-03-20');input(d,'period',String(period));input(d,'minimum','120');check((period===1?'Before':'After')+' solar noon filter matches supplied minutes',count()===index.homes.filter(h=>h.sun.main?.[3+period]>=120).length);}
 check('Every scripted filter responds within half a second',Math.max(...filterTimings)<500);
 await pause(1500);const liveStatus=await(await fetch('/agent/status')).json();check('Search live input follows configured status',d.getElementById('agent-form').hidden===!liveStatus.configured);
 const light=index.homes.find(h=>!index.heavy.includes(h.sid)&&h.computed&&h.placement==='well');
 for(const sid of [...index.featured,light.sid]){
  d=await page('home.html?sid='+sid,d=>d.querySelectorAll('#rooms-body tr').length>0&&d.querySelectorAll('#listing-fields .claim').length>0);
  check(sid+' room table and claims loaded',d.querySelectorAll('#rooms-body tr').length>0);
  check(sid+' buyer-facing terminology',!/(registration|cohort|convention)/i.test(d.body.innerText));
  check(sid+' home fits the browser width',d.documentElement.scrollWidth<=frame.contentWindow.innerWidth);
  const styles=[...d.querySelectorAll('#listing-fields > p .verdict')].map(e=>frame.contentWindow.getComputedStyle(e).borderBottomStyle);check(sid+' four verdict underline styles',styles.length===4&&new Set(styles).size===4);
  const range=d.getElementById('time'),previous=d.getElementById('clock').textContent;range.value=String(Math.max(0,Number(range.value)-5));range.dispatchEvent(new frame.contentWindow.Event('input'));check(sid+' time slider updates clock',d.querySelector('.sun-controls').hidden||d.getElementById('clock').textContent!==previous);
  const roomClaim=[...d.querySelectorAll('#listing-fields .claim')].find(e=>{const id=Number(e.dataset.claim);return id>=0;});roomClaim?.click();check(sid+' claim explanation opens',!d.getElementById('claim-evidence').hidden&&d.getElementById('claim-evidence').textContent.length>30);
  const home=await json('data/'+sid+'.json'),ci=home.claims.claims.findIndex(c=>c.evidence?.some(e=>home.plan_room_ids.includes(e.room_id)));
  if(ci>=0){d.querySelector(`#listing-fields [data-claim="${ci}"]`).click();const ids=d.getElementById('claim-evidence').dataset.rooms;check(sid+' evidence selects real plan room',Boolean(ids)&&d.getElementById('plan').dataset.claimRooms===ids);}
  if(sid===light.sid)check('Lighter home has no aerial image request',!d.querySelector('#window-images img'));
 }
 const poor=index.homes.find(h=>h.placement==='poor'&&h.computed);
 d=await page('home.html?sid='+poor.sid,d=>d.querySelectorAll('#rooms-body tr').length>0);check('Poor home hides sun controls',d.querySelector('.sun-controls').hidden);check('Poor room estimates withheld',!d.getElementById('rooms-body').innerText.includes(' min'));
 const failed=index.homes.find(h=>!h.computed);d=await page('home.html?sid='+failed.sid,d=>d.querySelectorAll('#listing-fields .claim').length>0);check('Failed calculation retains the listing',d.getElementById('address').textContent===failed.address&&d.querySelector('.map-panel').hidden);
 const partials=index.homes.filter(h=>h.contained_partial);check('Contained group retains sun withholding',partials.length===372&&partials.every(h=>h.placement!=='well'&&Object.keys(h.sun).length===0));
 const partial=partials.find(h=>h.computed);d=await page('home.html?sid='+partial.sid,d=>d.querySelectorAll('#rooms-body tr').length>0);
 check('Contained unit uses requested wording',d.getElementById('home-tags').textContent.includes(containedLabel)&&d.getElementById('map-label').textContent===containedLabel);
 check('Contained unit has no poor placement label',!d.getElementById('home-tags').textContent.includes('Poorly placed'));
 check('Contained unit hides placement selector and sun controls',d.getElementById('alignment').closest('label').hidden&&d.querySelector('.sun-controls').hidden);
 check('Contained unit has no aerial background request',!frame.contentWindow.performance.getEntriesByType('resource').some(r=>r.name.includes('/img/'+partial.sid+'/aerial')));
 check('Contained room sun stays withheld',!d.getElementById('rooms-body').textContent.includes(' min'));
 d=await page('claims.html',d=>d.querySelectorAll('#topic-counts tr').length>0);check('Topic counts and ceiling comparison loaded',d.getElementById('ceiling-data').querySelectorAll('table tr').length===3);check('Light comparison states little difference',d.getElementById('light-comparison').textContent.includes('nearly the same'));
 d=await page('method.html',d=>d.getElementById('full-counts'));check('Method shows all selected and failures',d.getElementById('full-counts').textContent.includes('1,874')&&d.getElementById('full-counts').textContent.includes('invalid numerical'));
 check('Method buyer-facing terminology',!/(registration|cohort|convention)/i.test(d.body.innerText));
 const recording=await(await fetch('data/agent-replay.json')).json();
 d=await page('agent.html',d=>d.getElementById('position')?.textContent.includes('of '+recording.events.length)&&!d.getElementById('reconnect').disabled);
 check('Replay clearly identifies framework and no live model',d.getElementById('mode-label').textContent==='scripted replay, real Deep Agents framework, no live model');
 check('Replay starts with live input hidden',d.getElementById('live-form').hidden);
 check('Replay fits the browser width',d.documentElement.scrollWidth<=frame.contentWindow.innerWidth);
 let parallel=false,planChanged=false,previousPlan='';
 function advanceToApproval(){for(let i=0;i<recording.events.length;i++){if(!d.getElementById('approval-card').hidden)break;d.getElementById('step').click();parallel ||= Number(d.getElementById('agent-lanes').dataset.running)>=3;const plan=d.getElementById('agent-plan').textContent;if(previousPlan&&plan!==previousPlan)planChanged=true;if(plan)previousPlan=plan;}}
 advanceToApproval();check('Three auditor lanes visibly overlap',parallel);check('Replay plan is populated',Boolean(previousPlan));
 check('Approval pauses playback with editable draft',!d.getElementById('approval-card').hidden&&d.getElementById('step').disabled&&d.getElementById('approval-text').value.length>100);
 check('Tool calls contain short results',Array.from(d.querySelectorAll('#agent-tools li')).some(e=>e.textContent.includes('→')));
 const fileButton=d.querySelector('[data-path="needs.md"]');fileButton?.click();await until(()=>d.getElementById('file-content').textContent!=='Loading…');check('Workspace tree opens a recorded file',Boolean(fileButton)&&d.getElementById('file-content').textContent===recording.files['needs.md'].text);
 d.getElementById('approve').click();await until(()=>{if(!d.getElementById('approval-card').hidden&&!d.getElementById('approve').disabled)d.getElementById('approve').click();return d.getElementById('final-answer').textContent.length>0&&d.querySelectorAll('#comparison a').length>=3;});
 check('Approved replay completes with home-page comparison links',Array.from(d.querySelectorAll('#comparison a')).every(a=>a.getAttribute('href').startsWith('home.html?sid=')));
 check('Replay final answer explicitly remains recorded',d.getElementById('answer-title').textContent==='Recorded final answer'&&d.getElementById('run-status').textContent.includes('No live model'));
 check('Replay todo plan updates',planChanged||d.getElementById('agent-plan').textContent!==previousPlan);
 for(const decision of ['edit','reject']){d.getElementById('restart').click();advanceToApproval();if(decision==='edit')d.getElementById('approval-text').value='A browser-only edit';d.getElementById(decision).click();await until(()=>d.getElementById('final-answer').textContent.includes(decision==='edit'?'Your edit':'Rejected locally'));check('Replay '+decision+' does not invent a model continuation',d.getElementById('play').disabled&&d.getElementById('step').disabled);}
 // Exercise the live UI transport with the recorded events, never a paid model.
 // This proves browser streaming/approval wiring only, not the real agent server.
 const win=frame.contentWindow,nativeFetch=win.fetch.bind(win),requests=[];
 function stream(events){const encoder=new win.TextEncoder();return new win.Response(new win.ReadableStream({async start(controller){for(const event of events){const bytes=encoder.encode(JSON.stringify(event)+'\n');const cut=Math.floor(bytes.length/2);controller.enqueue(bytes.slice(0,cut));await pause(2);controller.enqueue(bytes.slice(cut));}controller.close();}}),{headers:{'Content-Type':'application/x-ndjson'}});}
 const approvalIndex=recording.events.findIndex(e=>e.type==='approval_request');let liveCursor=0;
 win.fetch=async(url,options)=>{const path=String(url);if(path==='/agent/status')return new win.Response(JSON.stringify({configured:true}),{headers:{'Content-Type':'application/json'}});if(path==='/agent/ask'||path==='/agent/approve'){requests.push({path,body:JSON.parse(options.body)});let end=recording.events.findIndex((e,i)=>i>=liveCursor&&e.type==='approval_request');end=end<0?recording.events.length:end+1;const events=recording.events.slice(liveCursor,end);liveCursor=end;return stream(events);}if(path.startsWith('/agent/files/')){const relative=decodeURIComponent(path.split('/').slice(4).join('/'));const entry=recording.files[relative];return entry?.text!=null?new win.Response(entry.text):new win.Response('',{status:404});}return nativeFetch(url,options);};
 d.getElementById('reconnect').click();await until(()=>!d.getElementById('live-mode').hidden);d.getElementById('live-mode').click();check('Configured status reveals live question form',!d.getElementById('live-form').hidden);check('Availability checks do not start agent work',requests.length===0);
 d.getElementById('question').value='Browser transport review';d.getElementById('live-form').dispatchEvent(new win.Event('submit',{cancelable:true,bubbles:true}));
 await until(()=>!d.getElementById('approval-card').hidden&&!d.getElementById('edit').disabled);check('Live transport renders streamed events before approval',d.querySelectorAll('#agent-tools li').length>5&&d.querySelectorAll('.lane').length>=3);
 d.getElementById('approval-text').value='Reviewed edit';d.getElementById('edit').click();await until(()=>{if(!d.getElementById('approval-card').hidden&&!d.getElementById('approve').disabled)d.getElementById('approve').click();return d.querySelectorAll('#comparison a').length>=3&&d.getElementById('final-answer').textContent.length>0;});
 check('Live transport POSTs question and matching approval identity',requests[0]?.path==='/agent/ask'&&requests[0]?.body.question==='Browser transport review'&&requests[1]?.path==='/agent/approve'&&requests[1]?.body.thread===recording.thread&&requests[1]?.body.approval_id===recording.events[approvalIndex].approval.id&&requests[1]?.body.decision==='edit'&&requests[1]?.body.text==='Reviewed edit');
 win.fetch=nativeFetch;
 // Known rigid transform of real cached anchors: transport/math check only.
 // This is not evidence that an actual Showcase model loads or matches.
 const anchors=await(await fetch('/tour/data/khbuN7q853q')).json(),angle=.37,c=Math.cos(angle),s=Math.sin(angle),rotation=[[0,-c,s],[1,0,0],[0,s,c]],shift=[4,-7,2];
 const synthetic=anchors.model_sweeps.slice(0,16).map(p=>{const a=[p.position.x,p.position.y,p.position.z],b=rotation.map((row,i)=>row.reduce((sum,v,j)=>sum+v*a[j],shift[i]));return {id:p.id,enabled:true,position:{x:b[0],y:b[1],z:b[2]}};});
 const fitted=await(await fetch('/tour/fit/khbuN7q853q',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(synthetic)})).json();
 check('Sweep fit recovers a known rigid transform (synthetic, not live proof)',fitted.accepted&&fitted.count===synthetic.length&&fitted.rms_m<1e-8&&fitted.rotation.every((row,i)=>row.every((v,j)=>Math.abs(v-rotation[i][j])<1e-8))&&fitted.translation.every((v,i)=>Math.abs(v-shift[i])<1e-8));
 const unmatched=await fetch('/tour/fit/khbuN7q853q',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(synthetic.map((p,i)=>({...p,id:'unmatched-'+i})))});
 check('Sweep fit refuses unmatched IDs instead of guessing axes',unmatched.status===400);
 const distorted=await(await fetch('/tour/fit/khbuN7q853q',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(synthetic.map((p,i)=>i? p:{...p,position:{...p.position,x:p.position.x+2}}))})).json();
 check('Sweep fit rejects excessive measured residuals',distorted.accepted===false&&distorted.max_m>.15);
 d=await page('home.html?sid=khbuN7q853q&tour=off',d=>d.getElementById('tour-status')&&frame.contentWindow.demoTour?.status==='fallback');
 check('Explicit plan-only fallback explains the unavailable tour',!d.getElementById('tour-fallback').hidden&&d.getElementById('tour-fallback').textContent.includes('plan view')&&d.getElementById('tour-frame').hidden&&d.getElementById('plan').clientWidth>0);
 check('Plan-only mode neither opens Showcase nor invents a fit',frame.contentWindow.demoTour.attempted===false&&frame.contentWindow.demoTour.fit===null&&frame.contentWindow.demoTour.windowTagCount===0);
 check('Flight fallback preserves the modelling label and disables the custom camera',frame.contentWindow.demoFlight.fallback&&d.getElementById('flight-start').disabled&&d.getElementById('flight-label').textContent==='modelled: geometry, clear sky; not measured'&&d.getElementById('flight-status').textContent.includes('Round 4'));
 for(const sid of index.featured){
  d=await page('home.html?sid='+sid,d=>Boolean(frame.contentWindow.demoTour));
  await until(()=>['ready','fallback'].includes(frame.contentWindow.demoTour.status),3600);
  check(sid+': furniture panel and privacy-safe upload controls exist',Boolean(d.getElementById('furniture-panel'))&&d.getElementById('furniture-photo').accept==='image/*'&&d.getElementById('furniture-photo').getAttribute('capture')==='environment');
  const furniture=frame.contentWindow.demoFurniture,home=await json('data/'+sid+'.json');
  if(sid==='khbuN7q853q'){
   const blob=await(await fetch(home.aerial.image)).blob(),file=new frame.contentWindow.File([blob],'local-photo.jpg',{type:'image/jpeg'}),transfer=new frame.contentWindow.DataTransfer();transfer.items.add(file);
   const upload=d.getElementById('furniture-photo');upload.files=transfer.files;upload.dispatchEvent(new frame.contentWindow.Event('change'));
   await until(()=>!upload.disabled,2400);
   check('Furniture photo travels through the real local lookup API (existing aerial fixture; category accuracy not tested)',d.querySelectorAll('#furniture-categories tr').length===4&&upload.files.length===0&&d.getElementById('furniture-status').textContent.includes('typical size'));
   d.getElementById('furniture-dimensions').value='84"W x 38"D x 33"H';d.getElementById('furniture-parse').click();
   check('Furniture pasted exact size replaces category preset',Math.abs(furniture.dimensions[0]-2.1336)<1e-8);
  }
  check(sid+': furniture units parse inches and centimetres',Math.abs(parseDimensions('84"W x 38"D x 33"H')[0]-2.1336)<1e-8&&parseDimensions('213 x 97 x 84 cm')[1]===.97);
  if(frame.contentWindow.demoTour.status==='ready'){
   await until(()=>furniture.ready,600);
   d.getElementById('furniture-place').click();
   check(sid+': furniture places a real SDK box at current sweep floor',Boolean(furniture.position)&&Boolean(furniture.fit));
   if(furniture.position){
    const x=furniture.position[0],angle=furniture.angle;d.getElementById('furniture-rotate15').click();d.getElementById('furniture-right').click();
    check(sid+': furniture rotate and nudge controls work',Math.abs(furniture.angle-angle-Math.PI/12)<1e-8&&Math.abs(furniture.position[0]-x-.1)<1e-8);
    check(sid+': furniture reports fit and existing-engine sun days',['green','amber','red'].includes(furniture.fit.color)&&furniture.sun.length===Object.keys(home.days).length&&furniture.sun.every(day=>day.minutes>=0&&day.cells>0));
   }
  }
  const state=frame.contentWindow.demoTour,expected=home.windows.filter(w=>w.type==='window').length;
  const outcome={sid,address:home.listing.address,attempted:state.attempted,loaded:state.loaded,status:state.status,windows:expected,tags:state.windowTagCount,note_tags:state.noteTagCount,reason:state.reason,fit:state.fit?{count:state.fit.count,rms_m:state.fit.rms_m,max_m:state.fit.max_m,rotation:state.fit.rotation,translation:state.fit.translation,accepted:state.fit.accepted}:null};tours.push(outcome);
  check(sid+': one window finding per exterior window (doors excluded)',d.querySelectorAll('#tour-findings>li').length===expected&&state.expectedWindows===expected);
  check(sid+': SDK config is removed from the page after handoff',!d.getElementById('tour-config'));
  check(sid+': tour or explicit fallback preserves the plan',d.getElementById('plan').clientWidth>0&&(state.status==='ready'?!d.getElementById('tour-frame').hidden:!d.getElementById('tour-fallback').hidden&&d.getElementById('tour-frame').hidden&&Boolean(state.reason)));
  if(state.status==='ready'){
   const viewport=d.querySelector('.tour-viewport'),wide=d.getElementById('tour-wide');
   await until(()=>Math.abs(state.vfov-60)<1,400);
   check(sid+': Normal view measures 60 degree vertical FOV from SDK projection',Math.abs(state.vfov-60)<1);
   check(sid+': Clean moves defaults to instant photo hops',state.cleanMoves&&d.getElementById('tour-clean-moves').checked);
   const hopsBefore=state.hops;
   const tourKey=(type,key,code)=>viewport.dispatchEvent(new frame.contentWindow.KeyboardEvent(type,{key,code,bubbles:true,cancelable:true}));
   const walkSdk=await d.getElementById('tour-frame').contentWindow.MP_SDK.connect(d.getElementById('tour-frame').contentWindow);
   const hopDistances=[];let forwardAngle=null,projected=true;
   async function step(key,code){
    const before=await walkSdk.Camera.getPose(),hops=state.hops;
    tourKey('keydown',key,code);tourKey('keyup',key,code);
    // candidate() records the projected pre-hop heading synchronously on input.
    const heading=state.heading;projected&&=state.headingSource==='projection';
    await pause(800);
    if(state.hops>hops){
     const after=await walkSdk.Camera.getPose(),dx=after.position.x-before.position.x,dy=after.position.y-before.position.y,dz=after.position.z-before.position.z;
     hopDistances.push(Math.hypot(dx,dy,dz));
     if(key==='w'&&Math.hypot(dx,dz)>.05){
      const bearing=Math.atan2(dx,-dz)*180/Math.PI;
      forwardAngle=Math.abs(((bearing-heading)%360+540)%360-180);
     }
    }
   }
   viewport.focus();
   for(let heading=0;heading<2&&state.hops===hopsBefore;heading++){
    if(heading)for(let tap=0;tap<2;tap++){tourKey('keydown','ArrowRight','ArrowRight');tourKey('keyup','ArrowRight','ArrowRight');await pause(250);}
    for(const [key,code] of [['w','KeyW'],['a','KeyA'],['d','KeyD'],['s','KeyS']]){
     await step(key,code);if(state.hops>hopsBefore)break;
    }
   }
   check(sid+': captured direction key completes a real adjacent sweep hop',state.hops>hopsBefore&&Boolean(state.lastHop)&&Number.isFinite(state.lastHopMs)&&Number.isFinite(state.lastInputToMoveMs)&&state.lastInputDispatchMs<50);
   // Find a forward neighbour without accepting a strafe/back hop as forward proof.
   for(let turn=0;turn<24&&forwardAngle===null;turn++){
    tourKey('keydown','ArrowRight','ArrowRight');tourKey('keyup','ArrowRight','ArrowRight');await pause(250);
    await step('w','KeyW');
   }
   check(sid+': ready tour movement heading comes from projection',projected&&state.headingSource==='projection');
   check(sid+': forward hop camera delta is within 45 degrees of pre-hop heading',Number.isFinite(forwardAngle)&&forwardAngle<=45);
   check(sid+': no measured keyboard hop exceeds 4 metres',hopDistances.length>0&&hopDistances.every(distance=>distance<=4+1e-6));
   await until(()=>Math.abs(state.vfov-60)<1,400);
   check(sid+': hop preserves Normal FOV',Math.abs(state.vfov-60)<1);
   wide.checked=true;wide.dispatchEvent(new frame.contentWindow.Event('change'));await until(()=>Math.abs(state.vfov-75)<1,400);
   check(sid+': Wide view measures 75 degree vertical FOV',Math.abs(state.vfov-75)<1);
   wide.checked=false;wide.dispatchEvent(new frame.contentWindow.Event('change'));await until(()=>Math.abs(state.vfov-60)<1,400);
   check(sid+': actual SDK window tag count equals exterior windows',state.loaded&&state.windowTagCount===expected);
   check(sid+': measured sweep fit within tolerance',state.fit.accepted&&state.fit.count>=4&&state.fit.rms_m<=.05&&state.fit.max_m<=.15&&d.getElementById('tour-fit').textContent.includes('RMS'));
   const before=state.moves;d.querySelector('#rooms-body tr')?.click();await until(()=>state.moves>before);check(sid+': room click reaches nearest sweep with Sweep.moveTo',Boolean(state.lastSweep)&&state.moves>before);
   await until(()=>state.patchStatus!=='initializing',800);
   const notes=await(await fetch('/tour/data/'+sid+'?offline=1')).json(),expectedNotes=home.placement_status==='well'&&notes.notes?.published===true?(notes.notes.annotations||[]).filter(n=>[n.x,n.y,n.z].every(Number.isFinite)).length:0;
   check(sid+': placed note count matches published anchored annotations',state.noteTagCount===expectedNotes);
   const ci=home.claims.claims.findIndex(c=>c.evidence?.some(e=>home.rooms.some(r=>r.id===e.room_id)||home.windows.some(w=>w.id===e.window_id)));
   if(ci>=0){const prior=state.moves;d.querySelector(`#listing-fields [data-claim="${ci}"]`)?.click();await until(()=>state.moves>prior);check(sid+': anchored claim reaches a sweep with Sweep.moveTo',state.moves>prior);}
   if(!d.getElementById('tour-patch').disabled){
    const toggle=d.getElementById('tour-patch');toggle.checked=true;toggle.dispatchEvent(new frame.contentWindow.Event('change'));await until(()=>state.patchStatus==='drawn');
    const date=d.getElementById('sun-date').value,index=Number(d.getElementById('time').value),floor=d.getElementById('floor').value,count=home.rooms.filter(r=>r.floor_id===floor).reduce((n,r)=>n+home.days[date].room_lit_runs[r.id][index].reduce((a,[start,end])=>a+end-start,0),0);
    check(sid+': tour overlay uses exactly the engine cell count',state.patchCells===count);
   }
   if(['khbuN7q853q','VN4b1MGnnCR'].includes(sid)){
    const flight=frame.contentWindow.demoFlight,context=await(await fetch('/flight/data/'+sid)).json();
    await until(()=>flight.registered||flight.fallback,600);
    check(sid+': sun flight camera component registered',flight.registered===true);
    check(sid+': modelling label and DSM provenance remain visible',d.getElementById('flight-label').textContent==='modelled: geometry, clear sky; not measured'&&d.getElementById('flight-dsm').textContent.includes('DSM'));
    let chosen;
    outer:for(const date of Object.keys(home.days))for(const floor of [...new Set(home.rooms.map(r=>r.floor_id))])for(let i=0;i<home.days[date].local_times.length;i++){const candidate={date,index:i,floor};if(chooseFlight(home,candidate,context)){chosen=candidate;break outer;}}
    check(sid+': highest-hour-sun window has a saved floor-cell target',Boolean(chosen));
    if(chosen&&flight.registered){
     const event=new frame.contentWindow.Event('change');d.getElementById('sun-date').value=chosen.date;d.getElementById('sun-date').dispatchEvent(event);d.getElementById('floor').value=chosen.floor;d.getElementById('floor').dispatchEvent(new frame.contentWindow.Event('change'));d.getElementById('time').value=chosen.index;d.getElementById('time').dispatchEvent(new frame.contentWindow.Event('input'));
     await until(()=>!d.getElementById('flight-start').disabled);d.getElementById('flight-start').click();
     await until(()=>flight.status==='complete'||flight.status==='fallback',600);
     check(sid+': flight has six control points, registered aerial and a sun ray',flight.pathPointCount===6&&flight.aerialLoaded&&flight.ray);
     check(sid+': actual custom camera moved and completed its flight',flight.cameraMoved&&flight.cameraDistance>.1&&flight.status==='complete');
     outcome.flight={status:flight.status,registered:flight.registered,initialized:flight.componentInitialized,path_points:flight.pathPointCount,camera_moved:flight.cameraMoved,camera_distance:flight.cameraDistance,window:flight.selection?.window,dsm:flight.selection?.dsm,aerial:flight.aerialLoaded,reason:flight.reason,stage:flight.stage,error:flight.error};
     d.getElementById('flight-stop').click();const prior=state.moves;d.getElementById('flight-sweep').click();await until(()=>state.moves>prior);
     check(sid+': nearest-sweep fallback releases custom camera',flight.status==='sweep'&&flight.fallback&&state.moves>prior&&d.getElementById('flight-stop').disabled);
     d.getElementById('time').value=0;d.getElementById('time').dispatchEvent(new frame.contentWindow.Event('input'));
     check(sid+': no-sun time withholds flight instead of inventing a patch',d.getElementById('flight-start').disabled&&flight.status==='no-sun');
    }
   }
  }else check(sid+': fallback reports observed inability without fabricated tags',Boolean(state.reason)&&state.status==='fallback');
 }
 window.demoReview={finished:true,passed:results.every(x=>x.passed),checks:results,max_filter_ms:Math.max(...filterTimings),tours};
}catch(error){window.demoReview={finished:true,passed:false,checks:results,tours,error:error.message};}
output.textContent=JSON.stringify(window.demoReview,null,2);
