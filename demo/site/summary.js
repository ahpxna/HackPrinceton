import {$,json,element,showError} from './common.js?v=2';
import {buyerText,verdictLabels} from './claims.js?v=2';
const labels={ceilings:'Ceilings',light:'Light',open_plan:'Open plan',living_area:'Living area',renovations_finishes:'Renovations and finishes',schools_location:'Schools and location',roof_age:'Roof age'};
const n=x=>x==null?'Unavailable':Number(x).toLocaleString();
const topic=t=>labels[t]||t.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
function comparison(target,groups,key,unit,scale=1){
 const table=element('table'),head=element('tr');for(const h of ['Listing wording','Homes with measurement','Median','Mean'])head.append(element('th',h));table.append(head);
 for(const [group,label] of [['claim','Makes the claim'],['no_claim','Does not make the claim']]){const values=groups[group][key],tr=element('tr');for(const v of [label,n(values.n),(values.median*scale).toFixed(2)+' '+unit,(values.mean*scale).toFixed(2)+' '+unit])tr.append(element('td',v));table.append(tr);}target.append(table);
}
try{
 const s=await json('data/claims-summary.json');$('summary-count').textContent=`${n(s.homes_with_claims_file)} home check files · ${n(s.homes_with_listing)} homes with listing text · ${n(s.homes_with_sun)} homes with a sun calculation`;
 for(const [t,v] of Object.entries(s.topics)){const tr=element('tr');tr.append(element('th',topic(t)),element('td',n(v.listings_making_claim)));for(const verdict of ['supported','contradicted','unclear','not_checkable'])tr.append(element('td',n(v.verdict_counts[verdict]||0)));$('topic-counts').append(tr);}
 const ceiling=s.base_rates.ceilings;comparison($('ceiling-data'),ceiling.groups,'max_main_ceiling_m','m');
 $('ceiling-data').append(element('p',`Listings making ceiling claims have a ${(ceiling.median_differences_claim_minus_no_claim.max_main_ceiling_m).toFixed(2)} m higher median maximum ceiling in a usable main indoor room. This is each home's highest eligible room ceiling, not a whole-home minimum; garages, stairs, closets, outdoor spaces and implausible heights are excluded.`));
 const light=s.base_rates.light;const box=$('light-data');box.append(element('h3','Window area relative to floor area'));comparison(box,light.groups,'window_floor_ratio','%',100);
 box.append(element('p','The median window-to-floor ratios are nearly the same. Light wording shows little separation in this measure; the means are also very similar.'));
 box.append(element('h3','20 March direct sun in the best main living room'));comparison(box,light.groups,'march_windows_only_best_main_room_minutes','min');
 box.append(element('p','The direct-sun medians differ modestly, with a smaller difference in means. This uses windows only and homes well placed on the aerial photo. It does not establish that light wording predicts a brighter room.'));
 for(const [t,v] of Object.entries(s.topics)){if(!v.clearest)continue;const details=element('details'),head=element('summary',topic(t));details.append(head);for(const verdict of ['contradicted','supported']){const cases=v.clearest[verdict]||[];if(!cases.length)continue;details.append(element('h3',verdictLabels[verdict]));const ul=element('ul');for(const c of cases){const li=element('li'),a=element('a',buyerText(c.address||c.sid));a.href=`home.html?sid=${c.sid}&claim=${encodeURIComponent(c.text)}`;li.append(a,document.createTextNode(' · “'+buyerText(c.text)+'”'));ul.append(li);}details.append(ul);}$('clear-cases').append(details);}
}catch(e){showError($('topic-counts'),e);}
