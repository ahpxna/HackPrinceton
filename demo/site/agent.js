import {element} from './common.js?v=3';
export const endpoint='/agent';
export async function agentStatus(){
 try{const r=await fetch(endpoint+'/status',{signal:AbortSignal.timeout(2500)});if(!r.ok)return {configured:false};const s=await r.json();return {...s,configured:s.configured===true&&s.real_model_run!==false&&s.dry_run!==true};}catch{return {configured:false};}
}
export async function streamAgent(path,body,onEvent){
 const r=await fetch(endpoint+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(300000)});
 if(!r.ok)throw new Error('Agent request failed ('+r.status+'). Recheck the connection before retrying.');
 if(!(r.headers.get('Content-Type')||'').includes('ndjson'))throw new Error('Expected a streaming agent response.');
 const reader=r.body.getReader(),decoder=new TextDecoder();let buffer='',failure;
 function line(s){if(!s.trim())return;const e=JSON.parse(s);onEvent(e);if(e.type==='error')failure=e;}
 try{while(true){const {done,value}=await reader.read();buffer+=decoder.decode(value,{stream:!done});const lines=buffer.split('\n');buffer=lines.pop();for(const s of lines)line(s);if(done){line(buffer);break;}}}finally{reader.releaseLock();}
 if(failure)throw new Error(failure.message||failure.error||'The agent could not finish.');
}
export async function connectAgent(){
 const status=await agentStatus();delete window.demoAgent;delete window.demoCheck;if(!status.configured)return false;
 window.demoAgent=async question=>{location.href='agent.html?mode=live&question='+encodeURIComponent(question);return {answer:'Opening the live agent workspace…'};};
 window.demoCheck=async sid=>{const r=await fetch(endpoint+'/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid})});if(!r.ok)throw new Error('The agent could not check this home.');return r.json();};
 window.dispatchEvent(new Event('demo-agent-ready'));return true;
}
export function renderAgentAnswer(target,result){target.replaceChildren(element('p',typeof result.answer==='string'?result.answer:JSON.stringify(result)));}
