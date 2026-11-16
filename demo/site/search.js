import {$,dates,dateLabel,minute,element,json,options,money,showError,containedLabel} from './common.js?v=3';
import {connectAgent,renderAgentAnswer} from './agent.js?v=4';
let index,page=0,filtered=[];const pageSize=24;
const labels={ceilings:'High / tall ceilings',light:'Light / bright rooms',open_plan:'Open plan',living_area:'Living area',bedrooms:'Bedrooms',bathrooms:'Bathrooms',pool:'Pool',garage:'Garage',solar:'Solar panels',appliances:'Appliances',renovations_finishes:'Renovations / finishes',roof_age:'Roof age',schools_location:'Schools / location'};
options($('date'),dates.map(d=>[d,dateLabel(d)]));
function selectedMinutes(h){const v=h.sun[$('room').value];return v?.[index.dates.indexOf($('date').value)*3+Number($('period').value)]??null;}
function claimMatches(h){const topic=$('topic').value,verdict=$('verdict').value,mask=verdict?1<<index.verdicts.indexOf(verdict):15;return topic!==''?Boolean((h.claims[topic]||0)&mask):!verdict||Object.values(h.claims).some(v=>v&mask);}
function render(reset=true){
 if(reset)page=0;
 const min=Number($('minimum').value)||0,height=Number($('ceiling').value)||0;
 filtered=index.homes.map(h=>({...h,selected:selectedMinutes(h)})).filter(h=>{
 const priceMin=$('price-min').value,priceMax=$('price-max').value,room=$('room').value;
 return (!$('city').value||h.city===$('city').value)&&(!priceMin||(h.price!=null&&h.price>=Number(priceMin)))&&(!priceMax||(h.price!=null&&h.price<=Number(priceMax)))&&
 (!$('quality').value||h.placement===$('quality').value)&&(!height||(h.ceiling!=null&&h.ceiling>=height))&&
 (!$('pool').value||h.pool===($('pool').value==='yes'))&&claimMatches(h)&&
 (min>0?h.selected!=null&&h.selected>=min:(room==='any'||h.sun[room]!=null||h.placement!=='well'||!h.computed));
 });
 const sort=$('sort').value;
 const WALK=['VN4b1MGnnCR','HFwcH424n9w'];
 if(sort==='featured')filtered.sort((a,b)=>Number(WALK.includes(b.sid))-Number(WALK.includes(a.sid))||Number(index.featured.includes(b.sid))-Number(index.featured.includes(a.sid))||a.address.localeCompare(b.address));
 else{const [key,dir]=sort.split('-'),field=key==='sun'?'selected':key;filtered.sort((a,b)=>a[field]==null?(b[field]==null?0:1):b[field]==null?-1:(a[field]-b[field])*(dir==='asc'?1:-1)||a.sid.localeCompare(b.sid));}
 page=Math.min(page,Math.max(0,Math.ceil(filtered.length/pageSize)-1));
 $('count').textContent=`${filtered.length.toLocaleString()} of ${index.homes.length.toLocaleString()} homes`;
 $('page').textContent=`Page ${page+1} of ${Math.max(1,Math.ceil(filtered.length/pageSize))}`;$('previous').disabled=page===0;$('next').disabled=(page+1)*pageSize>=filtered.length;
 $('cards').replaceChildren(...filtered.slice(page*pageSize,(page+1)*pageSize).map(h=>{
 const card=element('a',null,'card');card.href=`home.html?sid=${h.sid}`;card.dataset.sid=h.sid;
 if(h.image){const img=element('img');img.src=h.image;img.alt='Floor plan placed on the aerial photo';img.loading='lazy';card.append(img);}
 const body=element('div',null,'card-body');
 if(['VN4b1MGnnCR','HFwcH424n9w'].includes(h.sid)){const t=element('span','Walk 3D · trained','tag featured');t.style.cssText='background:#1a7f37;color:#fff;margin-right:4px';body.append(t);}
 if(index.featured.includes(h.sid))body.append(element('span','Featured','tag featured'));
 const placement=h.contained_partial?containedLabel:h.placement==='poor'?'Poorly placed on the aerial photo':h.placement==='unknown'?'Placement on the aerial photo unavailable':!h.computed?'Sun estimate unavailable':'Placed on the aerial photo';
 body.append(element('span',placement,h.placement==='well'?'tag':'tag warn'),element('h2',h.address),element('p',h.city,'caption'),element('strong',money(h.price)));
 if(h.placement==='well'&&h.computed){const p=element('p',h.selected==null?'No matching room measurement':`${minute(h.selected)} min · ${dateLabel($('date').value)} · ${$('period').selectedOptions[0].textContent}`,'sun-number');body.append(p);}
 else body.append(element('p','Sun estimate withheld.','caption'));
 body.append(element('p',`${h.ceiling==null?'Ceiling height unavailable':h.ceiling.toFixed(2)+' m highest usable room ceiling'} · ${h.pool===true?'Pool detected':h.pool===false?'No pool detected':'Pool detection unavailable'}`,'caption'));
 for(const [t,mask] of Object.entries(h.claims)){if($('topic').value!==''&&t!==$('topic').value)continue;for(let i=0;i<4;i++)if(mask&(1<<i)){if($('verdict').value&&index.verdicts[i]!==$('verdict').value)continue;body.append(element('span',`${['✓','×','?','—'][i]} ${labels[index.topics[t]]||index.topics[t].replaceAll('_',' ')}: ${index.verdicts[i].replaceAll('_',' ')}`,`tag verdict ${index.verdicts[i]}`));}}
 card.append(body);return card;
 }));
 if(!filtered.length)$('cards').append(element('p','No homes match these filters.','empty'));
}
for(const id of ['city','price-min','price-max','room','date','period','minimum','ceiling','pool','topic','verdict','quality','sort'])$(id).addEventListener('input',()=>index&&render());
$('previous').onclick=()=>{page--;render(false);};$('next').onclick=()=>{page++;render(false);};
$('reset').onclick=()=>{for(const id of ['city','price-min','price-max','ceiling','pool','topic','verdict','quality'])$(id).value='';$('minimum').value=0;$('room').value='any';$('period').value='0';$('sort').value='featured';render();};
$('agent-form').onsubmit=async e=>{e.preventDefault();if(!window.demoAgent)return;try{$('agent-status').textContent='Looking through the homes…';const result=await window.demoAgent($('agent-text').value);renderAgentAnswer($('agent-results'),result);$('agent-status').textContent='Answer from the connected home assistant.';}catch(e){$('agent-status').textContent=e.message;}};
try{index=await json('data/index.json');options($('city'),[['','All cities'],...[...new Set(index.homes.map(h=>h.city).filter(Boolean))].sort().map(c=>[c,c])]);options($('topic'),[['','Any checked topic'],...index.topics.map((t,i)=>[String(i),labels[t]||t.replaceAll('_',' ')])]);$('feature-rule').textContent=index.selection_rule;render();}catch(e){showError($('cards'),e);}
connectAgent().then(connected=>{if(connected){$('agent-form').hidden=false;$('agent-form').classList.add('connected');$('agent-status').hidden=false;$('agent-status').classList.add('connected');}});
