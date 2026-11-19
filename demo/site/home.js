import {$,dates,dateLabel,minute,timeLabel,compass,colors,element,json,options,money,showError,containedLabel} from './common.js?v=3';
import {renderClaims,buyerText} from './claims.js?v=2';
import {connectAgent} from './agent.js?v=4';
import {mountTour} from './tour.js?v=8';
import {mountMerge} from './merge.js?v=w31-1';
import {mountScanObjects} from './scan-objects.js?v=w34-1';
let tour,scan;
const canvas=$('plan'),ctx=canvas.getContext('2d');
let data,image,day,frame=0,selectedWindow=null,hoverRoom=null,timer=null,zoom=1,roomPaths=[],windowLines=[],claimRooms=new Set(),claimWindows=new Set(),view,imageMessage='Plan only · aerial photos are included for the 36 selected examples';
const currentDate=()=>$('sun-date').value;
const currentFloor=()=>$('floor').value;
const registered=()=>$('alignment').value==='registered';
const roomNames=new Map();
function roomLabel(room){return roomNames.get(room.id)||room.label.replace('Unlabelled room','Room (no label)');}
function transform(point,reg=true){const tr=data.transform,m=reg?tr.registered_matrix:tr.delivered_matrix,s=reg?tr.registered_shift:[0,0];return [m[0][0]*point[0]+m[0][1]*point[1]+s[0],m[1][0]*point[0]+m[1][1]*point[1]+s[1]];}
function pixel(en){return [(en[0]-view.left)*view.scale,(view.top-en[1])*view.scale];}
function xy(point,reg=true){return pixel(transform(point,reg));}
function pathFor(segments,holes=[],reg=true){const p=new Path2D();for(const group of [segments,...holes])for(const [a,b] of group){const aa=xy(a,reg),bb=xy(b,reg);p.moveTo(...aa);p.lineTo(...bb);}return p;}
// Segment boundaries may not be ordered. Canvas hit testing uses the engine's
// even-odd rule directly, rather than pretending independent segments close.
function contains(point,segments){let odd=false;for(const [a,b] of segments)if((a[1]>point[1])!==(b[1]>point[1])&&point[0]<(b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0])odd=!odd;return odd;}
function polygon(points,style,dash=[],width=1.3){if(!points.length)return;ctx.beginPath();points.map(pixel).forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.strokeStyle=style;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
function draw(){
 if(!data||!day)return;
 const width=canvas.clientWidth,height=canvas.clientHeight,dpr=window.devicePixelRatio||1;
 if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
 const bounds=Object.entries(data.claim_room_outlines||{}).filter(([,r])=>r.floor_id===currentFloor());
 const extra=bounds.filter(([id])=>claimRooms.has(id));
 const points=[...data.rooms.filter(r=>r.floor_id===currentFloor()).flatMap(r=>data.geometry[r.id].grid),...bounds.flatMap(([,r])=>r.segments.flat())].map(p=>transform(p));
 const [minX,minY,maxX,maxY]=points.reduce((b,p)=>[Math.min(b[0],p[0]),Math.min(b[1],p[1]),Math.max(b[2],p[0]),Math.max(b[3],p[1])],[Infinity,Infinity,-Infinity,-Infinity]);
 const center=[(minX+maxX)/2,(minY+maxY)/2];
 const scale=Math.min(width/(maxX-minX+3),height/(maxY-minY+3))*zoom;
 view={left:center[0]-width/scale/2,top:center[1]+height/scale/2,scale};
 const [west,south,east,north]=data.aerial?.bounds_enu_m||[minX-2,minY-2,maxX+2,maxY+2];
 const topLeft=pixel([west,north]);
 if(image?.complete&&image.naturalWidth)ctx.drawImage(image,...topLeft,(east-west)*scale,(north-south)*scale);
 else {ctx.fillStyle='#34445a';ctx.font='14px system-ui';ctx.textAlign='left';ctx.fillText(imageMessage,17,26);}
 ctx.fillStyle='#0a142414';ctx.fillRect(0,0,width,height);
 for(const o of data.obstructions)polygon(o.ring_model_xy_m.map(p=>transform(p)),o.kind==='tree'?'#7d9278':image?'#ffffffd9':'#8b949e',[4,4],1.1);
 for(const roof of data.own_roof_layers)polygon(roof.ring.map(p=>transform(p)),image?'#fff':'#657382',[],1.5);
 roomPaths=[];windowLines=[];
 for(const room of data.rooms.filter(r=>r.floor_id===currentFloor())){
  const geom=data.geometry[room.id],reg=registered(),path=pathFor(geom.segments,geom.holes,reg);
  const grid=geom.grid;
  // Draw the engine's actual 0.10 m cells; no client solar computation.
  if(reg){ctx.fillStyle=claimRooms.has(room.id)?'#205fbc66':room.id===hoverRoom?'#ffffff88':'#ffffff32';for(const p of grid){const [x,y]=xy(p);ctx.fillRect(x-.05*scale,y-.05*scale,.1*scale+.15,.1*scale+.15);}
   ctx.fillStyle='#ffe24af0';for(const [start,end] of (data.placement_status==='well'?day.room_lit_runs[room.id][frame]:[]))for(let i=start;i<end;i++){const [x,y]=xy(grid[i]);ctx.fillRect(x-.05*scale,y-.05*scale,.1*scale+.2,.1*scale+.2);}
  }
  ctx.strokeStyle=claimRooms.has(room.id)?'#132f61':reg?'#148253':'#bd3b36';ctx.lineWidth=claimRooms.has(room.id)?5:room.id===hoverRoom?3.5:1.7;ctx.stroke(path);
  roomPaths.push({room,geom,reg});
  if(grid.length){const mid=grid.reduce((s,p)=>[s[0]+p[0]/grid.length,s[1]+p[1]/grid.length],[0,0]);const [x,y]=xy(mid,reg);const label=room.label.replace('Living Room','Living').replace('Bedroom','Bed');ctx.font='11px system-ui';const tw=ctx.measureText(label).width;ctx.fillStyle='#ffffffe8';ctx.fillRect(x-tw/2-3,y-8,tw+6,15);ctx.fillStyle='#1c2630';ctx.textAlign='center';ctx.fillText(label,x,y+3);}
 }
 for(const [,room] of extra){ctx.strokeStyle='#132f61';ctx.lineWidth=5;ctx.setLineDash([7,3]);ctx.stroke(pathFor(room.segments,room.holes,registered()));ctx.setLineDash([]);}
 canvas.dataset.claimRooms=[...claimRooms].join(' ');canvas.dataset.claimWindows=[...claimWindows].join(' ');
 for(const w of data.windows.filter(w=>w.floor_id===currentFloor())){const c=w.center,t=w.tangent.map(n=>n*w.width/2),a=xy([c[0]-t[0],c[1]-t[1]],registered()),b=xy([c[0]+t[0],c[1]+t[1]],registered());ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle='#fff';ctx.lineWidth=(w.id===selectedWindow||claimWindows.has(w.id))?9:6;ctx.stroke();ctx.strokeStyle=colors[compass(w.outward_azimuth_deg)];ctx.lineWidth=(w.id===selectedWindow||claimWindows.has(w.id))?6:4;ctx.stroke();windowLines.push({w,a,b});}
 if(data.placement_status==='well'){
 const az=day.sun_azimuth_deg[frame]*Math.PI/180;
 const ray=Math.min(width,height)*.4,anchor=[width/2+ray*Math.sin(az),height/2-ray*Math.cos(az)],tip=[width/2+ray*.45*Math.sin(az),height/2-ray*.45*Math.cos(az)];
 ctx.beginPath();ctx.moveTo(...anchor);ctx.lineTo(...tip);ctx.strokeStyle='#332b14';ctx.lineWidth=4;ctx.stroke();ctx.strokeStyle='#ffdf3c';ctx.lineWidth=2;ctx.stroke();
 const angle=Math.atan2(tip[1]-anchor[1],tip[0]-anchor[0]);ctx.beginPath();ctx.moveTo(...tip);ctx.lineTo(tip[0]-12*Math.cos(angle-.45),tip[1]-12*Math.sin(angle-.45));ctx.lineTo(tip[0]-12*Math.cos(angle+.45),tip[1]-12*Math.sin(angle+.45));ctx.closePath();ctx.fillStyle='#ffdf3c';ctx.fill();
 ctx.beginPath();ctx.arc(...anchor,6,0,Math.PI*2);ctx.fill();ctx.textAlign='center';ctx.font='11px system-ui';ctx.fillStyle='#fff';ctx.fillText('Sun',anchor[0],anchor[1]-12);
 }
 const bar=5*scale;ctx.fillStyle='#ffffffee';ctx.fillRect(17,height-31,bar,3);ctx.textAlign='left';ctx.fillText('5 m',17,height-38);
 scan?.draw(ctx,p=>xy(p,registered()),currentFloor());
 $('map-label').textContent=data.contained_partial?containedLabel:data.placement_status!=='well'?'Placement on the aerial photo uncertain · sun estimate withheld':registered()?`Placed on the aerial photo · ${dateLabel(currentDate())} ${timeLabel(day.local_times[frame])} · yellow = direct floor sun`:'As delivered · red walls · sun cells hidden';
}
function updateFrame(){
 if(!data||!day)return;
 if(data.placement_status!=='well'){$('sun-summary').textContent='Sun estimate withheld because placement on the aerial photo is uncertain.';draw();return;}
 frame=Math.max(0,Math.min(Number($('time').value),day.local_times.length-1));
 $('clock').textContent=timeLabel(day.local_times[frame]);
 const visible=data.rooms.filter(r=>r.floor_id===currentFloor());const area=visible.reduce((s,r)=>s+day.room_floor_area_m2[r.id][frame],0);
 if(!visible.length){$('sun-summary').textContent='Sun estimate unavailable for these scan boundaries.';draw();return;}
 $('sun-summary').textContent=`${area.toFixed(2)} m² directly lit · floor ${$('floor').selectedOptions[0]?.textContent} · elevation ${day.sun_apparent_elevation_deg[frame].toFixed(1)}°`;
 tour?.setFrame(currentDate(),frame,currentFloor());updateBlockers();draw();
}
function setDate(){if(!data)return;day=data.days[currentDate()];$('time').max=day.local_times.length-1;const middle=Math.floor(day.local_times.length/2);$('time').value=middle;frame=middle;$('daylight').textContent=`Sunrise ${timeLabel(day.sunrise)} · sunset ${timeLabel(day.sunset)}`;$('noon').textContent=`Solar noon ${timeLabel(day.solar_noon)} · ${data.assumptions.timezone}`;updateFrame();}
function updateBlockers(){if(data.placement_status!=='well'){$('window-blockers').textContent='Sun estimate withheld.';return;}if(!selectedWindow||!day)return;const codes=day.window_sample_codes[selectedWindow][frame].map(c=>day.code_legend[c]),clear=codes.includes('clear'),names={sun_inside_wall:'the sun is behind this wall',own_roof:'blocked by this home’s roof',neighbour:'blocked by a neighbouring building',tree:'blocked by a tree',below_horizon:'the sun is below the horizon'};const reasons=[...new Set(codes)].filter(c=>c!=='clear').map(c=>names[c]||c);$('window-blockers').textContent=`At ${timeLabel(day.local_times[frame])}: `+(clear?'direct sun reaches this window'+(reasons.length?' (partly obstructed).':'.'):reasons.join('; ')+'.');}
function setWindow(id){
 const w=data.windows.find(w=>w.id===id);if(!w)return;selectedWindow=id;$('window-select').value=id;
 if(currentFloor()!==w.floor_id)$('floor').value=w.floor_id;
 const room=data.rooms.find(r=>r.id===w.room_id);$('window-title').textContent=`${compass(w.outward_azimuth_deg)} · ${roomLabel(room)}`;
 const detail=$('window-detail');detail.replaceChildren(element('span',w.type==='window'?'Window':'Exterior doorway · glazing unknown',w.type==='window'?'tag':'tag warn'),element('p',w.id,'source'));
 const facts=element('div',null,'facts');for(const [label,value] of [['Width × height',`${w.width.toFixed(2)} × ${w.height.toFixed(2)} m`],['Sill above room floor',`${w.sill_above_room_floor_m.toFixed(2)} m`],['Outward direction',`${w.outward_azimuth_deg.toFixed(1)}° ${compass(w.outward_azimuth_deg)}`]]){const fact=element('span',label);fact.append(element('strong',value));facts.append(fact);}detail.append(facts);
 const table=element('table'),head=element('tr');head.append(element('th','Date'),element('th','Direct sun at this window'));table.append(head);for(const date of Object.keys(data.days)){const tr=element('tr');tr.append(element('td',dateLabel(date)),element('td',data.placement_status==='well'?`${minute(w.dates[date].unobstructed_any_aperture_minutes)} min`:'Withheld'));table.append(tr);}$('window-sun').replaceChildren(table);
 let technical=$('opening-technical');if(!technical){technical=element('p');technical.id='opening-technical';$('assumptions').append(technical);}technical.textContent=`Selected opening ${w.id}: model aperture base Z ${w.lower_z_m.toFixed(2)} m. Window minutes count at least 1 of 9 clear aperture rays. Room-floor totals require at least 0.1 m² lit. Blocker labels follow the engine’s precedence; they describe geometric rays, not visible brightness.`;
 const imgs=$('window-images');imgs.replaceChildren();const asset=data.window_assets?.[id];
 if(data.heavy_assets){for(const [key,label] of [['inside','Inside · pano'],['outside','Outside · oblique']]){if(asset?.[key]){const fig=element('figure'),img=element('img');img.src=asset[key];img.alt=`${label}: graph-projected ${id}`;fig.append(img,element('figcaption',`${label}. ${asset[key+'_status']}`));imgs.append(fig);}else imgs.append(element('p',`${label}: ${asset?.[key+'_status']||'Crop unavailable.'}`,'caption'));}}
 else imgs.append(element('p','Photo crops are included for the six featured homes and 30 selected listing examples.','caption'));
 updateFrame();
}
function roomTable(){
 const head=element('tr');head.append(element('th','Room / windows'));for(const date of Object.keys(data.days))head.append(element('th',dateLabel(date)));$('rooms-head').replaceChildren(head);
 for(const room of data.rooms){const row=element('tr');row.dataset.room=room.id;const label=element('td');label.append(element('strong',roomLabel(room)));const ws=data.windows.filter(w=>w.room_id===room.id);label.append(element('small',ws.map(w=>`${compass(w.outward_azimuth_deg)}${w.type==='doorway'?' (door)':''}`).join(', ')||'No assigned exterior opening'));row.append(label);
  for(const date of Object.keys(data.days)){const v=room.dates[date],cell=element('td');if(data.placement_status!=='well'){cell.textContent='Withheld';row.append(cell);continue;}cell.append(element('strong',`${minute(v.lit_minutes_ge_0_1m2)} min`),element('small',`${minute(v.before_solar_noon_minutes)} / ${minute(v.after_solar_noon_minutes)}`));cell.title=`Windows only: ${minute(v.windows_only_lit_minutes_ge_0_1m2)} min. Floor ${room.floor_z_m.toFixed(2)} m model Z. Area ${room.area_m2.toFixed(1)} m².`;row.append(cell);}
  row.tabIndex=0;row.setAttribute('role','button');row.setAttribute('aria-label',`View ${roomLabel(room)} in tour`);row.addEventListener('click',()=>tour?.goTo([room.id],[]));row.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();tour?.goTo([room.id],[]);}});
  row.addEventListener('mouseenter',()=>{hoverRoom=room.id;$('floor').value=room.floor_id;highlightRows();updateFrame();});row.addEventListener('mouseleave',()=>{hoverRoom=null;highlightRows();draw();});$('rooms-body').append(row);
 }
}
function highlightRows(){for(const row of $('rooms-body').rows)row.classList.toggle('active',row.dataset.room===hoverRoom);}
function pointOnSegment(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)));return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);}
function eventPoint(e){const r=canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];}
canvas.addEventListener('click',e=>{const p=eventPoint(e);const line=windowLines.map(w=>({...w,distance:pointOnSegment(p,w.a,w.b)})).sort((a,b)=>a.distance-b.distance)[0];if(line?.distance<14){setWindow(line.w.id);tour?.goTo([], [line.w.id]);}else if(hoverRoom)tour?.goTo([hoverRoom],[]);});
canvas.addEventListener('mousemove',e=>{if(!view||!data)return;const p=eventPoint(e),en=[view.left+p[0]/view.scale,view.top-p[1]/view.scale];const tr=data.transform,m=registered()?tr.registered_matrix:tr.delivered_matrix,s=registered()?tr.registered_shift:[0,0],det=m[0][0]*m[1][1]-m[0][1]*m[1][0],a=en[0]-s[0],b=en[1]-s[1],model=[(a*m[1][1]-b*m[0][1])/det,(b*m[0][0]-a*m[1][0])/det];const hit=roomPaths.find(({geom})=>contains(model,geom.segments)&&!geom.holes.some(h=>contains(model,h)));if(hoverRoom!==hit?.room.id){hoverRoom=hit?.room.id||null;highlightRows();draw();}});
canvas.addEventListener('mouseleave',()=>{hoverRoom=null;highlightRows();draw();});
$('time').addEventListener('input',updateFrame);$('sun-date').addEventListener('change',setDate);$('alignment').addEventListener('change',draw);$('floor').addEventListener('change',updateFrame);$('window-select').addEventListener('change',e=>setWindow(e.target.value));
$('zoom-in').addEventListener('click',()=>{zoom=Math.min(3,zoom*1.2);draw();});$('zoom-out').addEventListener('click',()=>{zoom=Math.max(.55,zoom/1.2);draw();});$('reset-view').addEventListener('click',()=>{zoom=1;draw();});
$('play').addEventListener('click',()=>{if(!day)return;if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';}else{$('play').textContent='Pause';timer=setInterval(()=>{$('time').value=(frame+1)%day.local_times.length;updateFrame();},220);}});
window.addEventListener('resize',draw);window.addEventListener('pagehide',()=>clearInterval(timer));
try{
 const sid=new URLSearchParams(location.search).get('sid')||'khbuN7q853q';if(!/^[A-Za-z0-9]+$/.test(sid))throw new Error('Invalid home identifier');
 const merge=mountMerge(sid);
 data=await json(`data/${sid}.json`);document.title=`${data.listing.address} · Sun on the floor`;$('address').textContent=data.listing.address;
 if(data.contained_partial){$('home-tags').append(element('p',containedLabel,'notice'));data.aerial=null;imageMessage='Partial building scan · plan only · sun estimate withheld';$('alignment').closest('label').hidden=true;}
 $('home-meta').textContent=data.contained_partial?'Partial building scan · sun estimate withheld':data.featured?'Featured home':data.heavy_assets?'Selected listing example':'A scan, placed on the aerial photo';
 if(!data.computed){$('home-meta').textContent='Sun estimate unavailable';$('price').textContent=money(data.listing.price);renderClaims(data);$('home-tags').append(element('p','This home remains listed. No usable direct-sun calculation is available.','notice'));for(const selector of ['.map-panel','#rooms-panel','#window-panel','#plan-guide'])document.querySelector(selector)?.setAttribute('hidden','');throw Object.assign(new Error('Plan and sun calculation unavailable for this home.'),{expected:true});}
 if(data.placement_status!=='well'&&!data.contained_partial){$('home-tags').append(element('span','Poorly placed on the aerial photo','tag warn'));const warning=element('div','Placement on the aerial photo is uncertain for this home; sun estimates are withheld.','notice');$('home-tags').append(warning);}
 $('home-tags').append(element('span',`${data.rooms.length} rooms`,'tag'),element('span',`${data.windows.filter(w=>w.type==='window').length} windows`,'tag'),element('span',`${data.windows.filter(w=>w.type==='doorway').length} exterior doors (glazing unknown)`,'tag'));
 const r=data.registration;for(const [value,label] of [[`${Math.round(r.iou_before*100)}% → ${Math.round(r.iou_after*100)}%`,'Footprint overlap · fitted'],[`${r.fix_shift_m.toFixed(1)} m`,'Horizontal shift'],[`${r.fix_rot_deg.toFixed(1)}°`,'Rotation correction'],[`${Math.round(r.contain_after*100)}%`,'Outline inside aerial footprint']]){const cell=element('div',null,'stat');cell.append(element('strong',value),element('span',label));$('stats').append(cell);}
 options($('sun-date'),Object.keys(data.days).map(d=>[d,dateLabel(d)]));const floors=[...new Set([...data.rooms.map(r=>r.floor_id),...Object.values(data.claim_room_outlines||{}).map(r=>r.floor_id)])];options($('floor'),floors.map((f,i)=>[f,`${i+1}`]));
 options($('window-select'),data.windows.map(w=>[w.id,`${w.id} · ${compass(w.outward_azimuth_deg)} · ${data.rooms.find(r=>r.id===w.room_id)?.label}`]));
 for(const [direction,color] of Object.entries(colors)){const item=element('span',direction),dot=element('i',null,'dot');dot.style.background=color;item.prepend(dot);$('legend').append(item);}
 const centers=data.rooms.map(r=>{const grid=data.geometry[r.id].grid;return grid.reduce((s,p)=>[s[0]+p[0]/grid.length,s[1]+p[1]/grid.length],[0,0]);});
 const middle=centers.reduce((s,p)=>[s[0]+p[0]/centers.length,s[1]+p[1]/centers.length],[0,0]);
 data.rooms.forEach((r,i)=>{const c=transform(centers[i]),m=transform(middle),az=(Math.atan2(c[0]-m[0],c[1]-m[1])*180/Math.PI+360)%360,side={N:'north',NE:'northeast',E:'east',SE:'southeast',S:'south',SW:'southwest',W:'west',NW:'northwest'}[compass(az)];roomNames.set(r.id,`${r.label.replace('Unlabelled room','Room (no label)')} · ${side} · ${Math.round(r.area_m2)} m²`);});
 function selectClaim(rooms,windows){claimRooms=new Set(rooms);claimWindows=new Set(windows);const room=data.rooms.find(r=>claimRooms.has(r.id))||Object.entries(data.claim_room_outlines||{}).find(([id])=>claimRooms.has(id))?.[1];if(room)$('floor').value=room.floor_id;const w=data.windows.find(w=>claimWindows.has(w.id));if(w)setWindow(w.id);tour?.goTo(rooms,windows);draw();}
 tour=mountTour(data,merge);roomTable();mountScanObjects(sid,{redraw:()=>updateFrame()}).then(s=>{scan=s;draw();});$('price').textContent=money(data.listing.price);const claimUI=renderClaims(data,selectClaim);
 $('assumptions').append(element('p','Scan room labels and ceiling dimensions come from the supplied model and may be incomplete. Windows are treated as clear glazing. Exterior doors are conditionally transparent, with actual glazing unknown.'),element('p','Own roof and neighbouring buildings block direct sun. Eave height and floor-to-ground height are assumed. Trees, weather, diffuse light, blinds and furniture are mostly not modelled.'),element('p',`Floor cells come from the same Python engine as the room totals. Maximum cell-area difference checked against saved output: ${data.engine_provenance.max_cell_area_difference_m2} m². Five-minute steps; 0.10 m floor grid.`));
 setDate();if(data.placement_status!=='well')document.querySelector('.sun-controls').hidden=true;
 const initial=data.windows.find(w=>w.id==='F0E4O0')||data.windows.find(w=>data.window_assets?.[w.id]?.inside&&data.window_assets?.[w.id]?.outside)||data.windows[0];if(initial)setWindow(initial.id);
 const requestedClaim=new URLSearchParams(location.search).get('claim');const chosenClaim=(data.claims?.claims||[]).find(c=>c.text===requestedClaim);if(chosenClaim)claimUI.show(chosenClaim);
 if(data.aerial){image=new Image();image.src=data.aerial.image;try{await image.decode();}catch{imageMessage='Aerial image unavailable · floor plan shown below';}}draw();
 connectAgent().then(connected=>{if(!connected)return;const button=element('button','Ask the assistant to check this listing');button.onclick=async()=>{button.disabled=true;try{const result=await window.demoCheck(sid);data.claims=result;renderClaims(data,selectClaim);button.textContent='Listing check updated';}catch(e){button.textContent=e.message;}finally{button.disabled=false;}};$('listing-fields').prepend(button);});
}catch(e){if(!e.expected)$('address').textContent='Home unavailable';showError($('window-detail'),e);}
