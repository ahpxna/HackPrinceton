import {$,element} from './common.js?v=2';
export const verdictLabels={supported:'✓ Supported',contradicted:'× Contradicted',unclear:'? Unclear',not_checkable:'— Not checkable'};
export const buyerText=value=>String(value??'').replace(/registration/gi,'placement on the aerial photo').replace(/cohort/gi,'supplied homes').replace(/convention/gi,'comparison rule');
const fieldLabels={bedsMin:'Bedrooms',bedsMax:'Bedrooms (maximum)',bathsMin:'Bathrooms',bathsMax:'Bathrooms (maximum)',bathsFull:'Full bathrooms',bathsHalf:'Half bathrooms',sqftMin:'Living area (sq ft)',sqftMax:'Living area maximum (sq ft)'};
export function renderClaims(data,onSelect=()=>{}){
 const claims=data.claims?.claims||[],chars=Array.from(data.listing.remarks||'');
 const target=$('remarks');target.replaceChildren();
 function show(claim){
  const box=$('claim-evidence');box.hidden=false;box.replaceChildren(element('strong',verdictLabels[claim.verdict]||'Unclear'),element('h3',buyerText(claim.text)),element('p',buyerText(claim.explanation)));
  const ids=new Set(),windows=new Set();
  const evidence=element('ul');
  for(const e of claim.evidence||[]){
   if(e.room_id)ids.add(e.room_id);if(e.window_id)windows.add(e.window_id);for(const r of e.rooms||[])ids.add(typeof r==='string'?r:r.room_id);
   if(data.placement_status!=='well'&&/sun|lit_minutes|noon|floor_area_m2_at/.test(e.fact||'')){evidence.append(element('li','Sun estimate withheld because placement on the aerial photo is uncertain.'));continue;}
   const value=typeof e.value==='number'?(Number.isInteger(e.value)?e.value:e.value.toFixed(3)):typeof e.value==='object'?JSON.stringify(e.value):e.value??'Unavailable';
   const label=buyerText((e.fact||'Evidence').replaceAll('_',' '));
   evidence.append(element('li',buyerText(`${e.room||e.room_id||'Home'} · ${label}: ${value} ${e.unit||''}${e.date?' · '+e.date:''}`)));
  }
  box.append(evidence);
  if(claim.caveats?.length){box.append(element('strong','Limits of this check'));const caveats=element('ul');for(const c of claim.caveats)caveats.append(element('li',buyerText(c)));box.append(caveats);}
  const planIds=new Set(data.plan_room_ids||Object.keys(data.geometry||{})),located=[...ids].filter(id=>planIds.has(id));
  const missing=[...ids].filter(id=>!planIds.has(id));
  const link=element('p',located.length?`${located.length} referenced room(s) outlined on the plan.`:windows.size?'Referenced opening outlined on the plan.':ids.size?'The referenced room is in the scan records but has no usable plan outline.':'This check applies to the home as a whole; no individual room or window is identified.','caption');box.append(link);
  if(missing.length)box.append(element('p',`${missing.length} referenced room(s) have no usable plan outline.`,'caption'));
  if(located.some(id=>data.claim_room_outlines?.[id]))box.append(element('p','Dashed room outlines come from the scan boundary. These rooms have no usable sun estimate.','caption'));
  box.dataset.rooms=located.join(' ');box.dataset.windows=[...windows].join(' ');onSelect(located,[...windows]);
  document.querySelectorAll('.claim.selected').forEach(e=>e.classList.remove('selected'));
  document.querySelectorAll(`[data-claim="${claims.indexOf(claim)}"]`).forEach(e=>e.classList.add('selected'));
 }
 function mark(text,c){const button=element('button',buyerText(text),`claim ${c.verdict}`);button.type='button';button.dataset.claim=claims.indexOf(c);button.setAttribute('aria-label',`${verdictLabels[c.verdict]}: ${buyerText(text)}`);button.title=`${verdictLabels[c.verdict]} · ${buyerText(c.explanation)}`;button.onclick=()=>show(c);return button;}
 // Partition overlapping spans at all boundaries, preserving every checked character.
 const spans=claims.filter(c=>c.source==='remarks'&&Number.isInteger(c.start)&&Number.isInteger(c.end)&&c.start>=0&&c.end<=chars.length&&chars.slice(c.start,c.end).join('')===c.text);
 const boundaries=[...new Set([0,chars.length,...spans.flatMap(c=>[c.start,c.end])])].sort((a,b)=>a-b);
 for(let i=0;i<boundaries.length-1;i++){const a=boundaries[i],b=boundaries[i+1],cover=spans.filter(c=>c.start<=a&&c.end>=b),text=chars.slice(a,b).join('');if(!cover.length)target.append(document.createTextNode(buyerText(text)));else{const button=mark(text,cover[0]);if(cover.length>1)button.title+=' · More checks for this phrase are listed below.';target.append(button);}}
 if(!chars.length)target.textContent='No listing remarks supplied.';
 $('listing-status').textContent=claims.length?'Click an underlined phrase or a checked field to see its evidence and locate it on the plan.':'No checked phrases supplied for this listing.';
 const list=$('listing-fields');list.replaceChildren();
 const legend=element('p');for(const [v,label] of Object.entries(verdictLabels))legend.append(element('span',label,`tag verdict ${v}`));list.append(legend);
 const checked=element('ul');checked.className='checked-list';
 for(const c of claims){const li=element('li');let text=c.text;if(c.source==='listing_field'){const key=c.text.replace('property.','');text=`${fieldLabels[key]||key}: ${data.listing.fields?.[key]??'Unknown'}`;}li.append(mark(text,c),element('small',` · ${verdictLabels[c.verdict]} · ${buyerText(c.source==='remarks'?'Remarks':c.source==='feature'?'Listing highlight / feature':'Listing field')}`));checked.append(li);}list.append(checked);
 const details=element('details'),summary=element('summary','All listing highlights and fields');details.append(summary);const all=element('ul');
 const entries=[...(data.listing.highlights||[]).map(text=>({text,claim:claims.find(c=>c.source==='feature'&&c.text===text)})),...(data.listing.features||[]).map(f=>({text:`${f.name}${f.doubleValue!=null?': '+f.doubleValue:f.textValue?': '+f.textValue:f.boolValue!=null?': '+f.boolValue:''}`,claim:claims.find(c=>c.source==='feature'&&c.text===f.name)})),...Object.entries(data.listing.fields||{}).filter(([,v])=>v!=null).map(([k,v])=>({text:`${fieldLabels[k]||k}: ${v}`,claim:claims.find(c=>c.source==='listing_field'&&c.text==='property.'+k)}))];
 const seen=new Set();for(const {text,claim} of entries){if(seen.has(text))continue;seen.add(text);const li=element('li');li.append(claim?mark(text,claim):document.createTextNode(buyerText(text)));if(claim)li.append(element('small',` · ${verdictLabels[claim.verdict]}`));all.append(li);}details.append(all);list.append(details);
 return {show};
}
