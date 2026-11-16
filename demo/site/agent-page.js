import {$,element} from './common.js?v=3';
import {agentStatus,streamAgent,endpoint} from './agent.js?v=4';
let bundle,mode='replay',cursor=0,timer,thread,approval,busy=false,available=new Set(),lanes=new Map(),calls=[],selectedFile='',generation=0,todoState=[];
const short=value=>{const s=typeof value==='string'?value:JSON.stringify(value);return (s||'').slice(0,340);};
const clean=path=>String(path||'').replace(/^\//,'');
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play';}
function buttons(){for(const id of ['approve','edit','reject'])$(id).disabled=busy||!approval||id==='edit'&&approval?.actions?.[0]?.name==='publish_annotations';$('ask').disabled=busy||Boolean(approval);$('replay-mode').disabled=busy||Boolean(approval)&&mode==='live';$('live-mode').disabled=busy||Boolean(approval)&&mode==='live';$('step').disabled=Boolean(approval)||cursor>=bundle.events.length;$('play').disabled=Boolean(approval)||cursor>=bundle.events.length;}
function lane(agent,sid,state,seq){
 const key=agent+':'+(sid||'shared');let row=lanes.get(key);
 if(!row){const node=element('article',null,'lane');node.append(element('h3',agent),element('p',sid||'Shared workspace'),element('p','Waiting','lane-state'));$('agent-lanes').append(node);row={node};lanes.set(key,row);}
 row.node.dataset.state=state;row.node.querySelector('.lane-state').textContent=state+(seq==null?'':' · event '+seq);
 $('agent-lanes').dataset.running=String([...lanes.values()].filter(x=>x.node.dataset.state==='running').length);
}
function reset(){stop();generation++;cursor=0;approval=null;thread=mode==='replay'?bundle.thread:null;available=new Set();lanes=new Map();calls=[];selectedFile='';todoState=[];
 for(const id of ['agent-plan','agent-tools','agent-lanes','agent-files','comparison'])$(id).replaceChildren();
 $('file-title').textContent='Select a file';$('file-content').textContent='Files appear as the run writes them.';$('file-content').hidden=false;$('file-image').hidden=true;$('final-answer').textContent='';$('final-panel').hidden=true;$('approval-card').hidden=true;$('approval-status').textContent='';
 $('mode-label').textContent=mode==='replay'?'scripted replay, real Deep Agents framework, no live model':'Live agent · model calls enabled by the local server';
 $('replay-source').hidden=mode!=='replay';$('replay-controls').hidden=mode!=='replay';$('live-form').hidden=mode!=='live';$('replay-mode').setAttribute('aria-pressed',mode==='replay');$('live-mode').setAttribute('aria-pressed',mode==='live');
 $('trace-provenance').textContent=mode==='replay'?`${bundle.source} · SHA-256 ${bundle.source_sha256} · thread ${thread}`:'Live events from the local agent server. Workspace files remain on that server.';
 if(mode==='replay')for(const e of bundle.events)if(e.type==='subagent_start')lane(e.agent,e.sid,'waiting');
 $('run-status').textContent=mode==='replay'?'Ready to replay the recorded run.':'Ready for a question.';position();buttons();
}
function position(){$('position').textContent=`Event ${cursor} of ${bundle.events.length}`;}
async function file(path){
 path=clean(path);if(!path||path.split('/').some(x=>x==='..'||!x))throw new Error('Invalid workspace path');
 if(mode==='replay'){const entry=bundle.files[path];if(!entry)throw new Error('File was not included in this recording.');return entry;}
 const url=endpoint+'/files/'+encodeURIComponent(thread)+'/'+path.split('/').map(encodeURIComponent).join('/');const r=await fetch(url);if(!r.ok)throw new Error('Workspace file is not available ('+r.status+').');
 if((r.headers.get('Content-Type')||'').startsWith('image/'))return {image:URL.createObjectURL(await r.blob()),temporary:true};
 return {text:await r.text()};
}
async function openFile(path){const token=generation;selectedFile=path;$('file-title').textContent=path;$('file-content').hidden=false;$('file-content').textContent='Loading…';const img=$('file-image');if(img.dataset.temporary)URL.revokeObjectURL(img.src);img.hidden=true;delete img.dataset.temporary;
 try{const entry=await file(path);if(token!==generation||selectedFile!==path){if(entry.temporary)URL.revokeObjectURL(entry.image);return;}if(entry.image){img.src=entry.image;img.hidden=false;img.dataset.temporary=entry.temporary?'yes':'';$('file-content').hidden=true;}else $('file-content').textContent=entry.text;}catch(e){$('file-content').textContent=e.message;}
}
function addFile(path){path=clean(path);if(!path||path.split('/').some(x=>x==='..'||!x)||available.has(path))return;available.add(path);const root=Object.create(null);for(const name of [...available].sort()){const parts=name.split('/');let node=root;for(const part of parts.slice(0,-1))node=node[part]??=Object.create(null);node[parts.at(-1)]=name;}
 function tree(node,parent){for(const [name,value] of Object.entries(node)){if(typeof value==='string'){const button=element('button',name);button.dataset.path=value;button.onclick=()=>openFile(value);parent.append(button,element('br'));}else{const dir=element('details');dir.open=true;dir.append(element('summary',name+'/'));tree(value,dir);parent.append(dir);}}}
 $('agent-files').replaceChildren();tree(root,$('agent-files'));
}
function plan(todos){todoState=(todos||[]).map(t=>({...t}));$('agent-plan').replaceChildren();for(const todo of todoState){const item=element('li',`${todo.content||todo.task||todo.title||'Task'} · ${todo.status||'pending'}`);item.dataset.state=todo.status||'pending';$('agent-plan').append(item);}}
function apply(e){if(e.thread)thread=e.thread;
 if(e.type==='plan_update')plan(e.todos);
 if(e.type==='tool_call'){
  const args=e.arguments||{},item=element('li');item.append(element('strong',e.tool),element('small',short(e.tool==='write_file'?{file_path:args.file_path}:args)));$('agent-tools').append(item);calls.push({tool:e.tool,item,done:false});item.scrollIntoView({block:'nearest'});
  if(e.tool==='write_todos')plan(args.todos);if(e.tool==='write_file'&&mode==='replay')addFile(args.file_path);
  if(mode==='replay'&&e.tool==='prepare_comparison')plan(todoState.map((t,i)=>({...t,status:i===0?'completed':'in_progress'})));
 }
 if(e.type==='tool_result'){const call=calls.find(x=>x.tool===e.tool&&!x.done);if(call){call.done=true;call.item.append(element('small','→ '+short(e.summary??e.result??'Completed')));}if(e.result_file)addFile(e.result_file);}
 if(e.type==='subagent_start'||e.type==='subagent_finish')lane(e.agent,e.sid,e.type==='subagent_start'?'running':'complete',e.seq);
 if(e.type==='file_written'){if(e.path)addFile(e.path);else if(typeof e.result==='string'){const match=e.result.match(/\/[\w./-]+\.(?:md|json|png|csv|txt)/);if(match)addFile(match[0]);}}
 if(e.type==='approval_request'){
  stop();approval=e.approval;const action=approval.actions?.[0];$('approval-card').hidden=false;$('approval-description').textContent=(mode==='replay'?'Replay pause. ':'Live execution paused. ')+(action?.description||action?.name||'Review proposed action');
  const publication=action?.name==='publish_annotations';const draft=publication?'Publish measured annotations to the local agent output. Review the annotations/ files in the workspace tree before approving. This approval is separate from the draft. Evidence must be revised before requesting different annotations.':action?.name==='save_preferences'?action.args:e.draft??approval.draft??action?.args?.draft??action?.args?.text??action?.args??'';
  $('approval-text').readOnly=publication;$('approval-text').value=typeof draft==='string'?draft:JSON.stringify(draft,null,2);$('approval-status').textContent=(mode==='replay'?'Approve continues the recorded branch. Edit or reject stops here; no model will recompute it.':'Review the proposal, then choose a decision.')+(publication?' Measured annotations accept approve or reject only.':'');
  $('run-status').textContent='Waiting for your approval.';
 }
 if(e.type==='final_answer'){if(mode==='replay')plan(todoState.map(t=>({...t,status:'completed'})));$('final-answer').textContent=e.answer||'';$('answer-title').textContent=mode==='replay'?'Recorded final answer':'Final answer';$('run-status').textContent=mode==='replay'?'Recorded run complete. No live model was used.':'Agent run complete.';comparison();}
 if(e.type==='error'){$('run-status').textContent='Agent error: '+short(e.message||e.error||e);stop();}
 buttons();
}
async function comparison(){const token=generation;$('final-panel').hidden=false;$('comparison').textContent='Loading comparison…';
 try{const entry=await file('comparison.md');let homes=[];try{homes=JSON.parse((await file('candidates.json')).text);homes=homes.homes||homes.results||homes;}catch{}if(token!==generation)return;
  const lines=entry.text.split('\n').filter(s=>s.trim().startsWith('|'));if(lines.length<2){$('comparison').textContent=entry.text;return;}
  const table=element('table');for(const [i,line] of lines.entries()){const cells=line.trim().replace(/^\||\|$/g,'').split('|').map(s=>s.trim());if(cells.every(s=>/^[-: ]+$/.test(s)))continue;const tr=element('tr');cells.forEach((value,j)=>{const cell=element(i===0?'th':'td');const home=i>0&&j===0&&Array.isArray(homes)?homes.find(h=>value.includes(h.address)||value.includes(h.sid)):null;if(home?.sid){const a=element('a',value);a.href='home.html?sid='+encodeURIComponent(home.sid);cell.append(a);}else cell.textContent=value;tr.append(cell);});table.append(tr);}$('comparison').replaceChildren(table);
 }catch(e){if(token===generation)$('comparison').textContent='Comparison unavailable: '+e.message;}
}
function step(){if(mode!=='replay'||approval||cursor>=bundle.events.length)return;const e=bundle.events[cursor++];apply(e);position();if(cursor===bundle.events.length)stop();}
$('step').onclick=step;$('play').onclick=()=>{if(timer)stop();else{$('play').textContent='Pause';timer=setInterval(step,200);step();}};$('restart').onclick=reset;
$('replay-mode').onclick=()=>{mode='replay';reset();};$('live-mode').onclick=()=>{mode='live';reset();};
async function run(path,body){busy=true;buttons();$('run-status').textContent='Agent is running…';try{await streamAgent(path,body,apply);}catch(e){$('run-status').textContent=e.message;}finally{busy=false;buttons();}}
$('live-form').onsubmit=async e=>{e.preventDefault();if(busy||approval)return;const question=$('question').value.trim();if(!question)return;reset();await run('/ask',{question});};
async function decide(decision){if(!approval||busy)return;const saved=approval;let body={thread,approval_id:saved.id,decision};
 if(decision==='edit'){if(saved.actions?.[0]?.name==='save_preferences'){try{body.preferences=JSON.parse($('approval-text').value);}catch{$('approval-status').textContent='Preferences must be valid JSON.';return;}}else body.text=$('approval-text').value;}
 approval=null;$('approval-status').textContent=(mode==='replay'?'Browser-only replay decision: ':'Sent decision: ')+decision;buttons();
 if(mode==='replay'){if(decision==='approve'){$('approval-card').hidden=true;$('run-status').textContent='Continuing recorded approval branch.';$('play').click();}else{$('run-status').textContent=`Replay stopped after ${decision}. The recording only contains an approved continuation. Restart to replay it.`;await comparison();$('answer-title').textContent='Replay decision';$('final-answer').textContent=decision==='edit'?'Your edit stays in this browser. A live model is required to produce a revised answer.':'Rejected locally. The recorded approved answer has not been replayed.';$('play').disabled=true;$('step').disabled=true;}}else{$('approval-card').hidden=true;await run('/approve',body);}
}
for(const id of ['approve','edit','reject'])$(id).onclick=()=>decide(id);
async function status(){$('reconnect').disabled=true;const s=await agentStatus();$('connection').textContent=s.configured?'Live agent configured':'Live agent unavailable · replay ready';$('live-mode').hidden=!s.configured;if(!s.configured&&mode==='live'&&!busy){mode='replay';reset();}$('reconnect').disabled=false;return s;}
$('reconnect').onclick=status;
window.addEventListener('pagehide',stop);
try{const r=await fetch('data/agent-replay.json');if(!r.ok)throw new Error('Recorded run unavailable');bundle=await r.json();reset();const s=await status();const query=new URLSearchParams(location.search);if(s.configured&&query.get('mode')==='live'){mode='live';reset();$('question').value=query.get('question')||'';}}catch(e){$('run-status').textContent=e.message;}
