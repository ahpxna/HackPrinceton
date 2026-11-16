export const $ = (id) => document.getElementById(id);
export const containedLabel = 'one unit in a larger building; not placed on the aerial photo';
export const dates = ['2026-12-21','2026-03-20','2026-06-21','2026-11-14'];
export const dateLabel = (date) => { const [,m,d]=date.split('-');return `${Number(d)} ${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][Number(m)-1]}`; };
export const minute = (n) => n==null ? '—' : Math.round(n).toLocaleString();
export const timeLabel = (iso) => iso ? iso.slice(11,16) : '—';
export const compass = (az) => ['N','NE','E','SE','S','SW','W','NW'][Math.floor((az+22.5)/45)%8];
export const colors = {N:'#1764aa',NE:'#188b93',E:'#338b31',SE:'#9c931d',S:'#d88414',SW:'#c94527',W:'#a52d87',NW:'#5548a3'};
export const element = (tag,text,className) => {const e=document.createElement(tag);if(text!=null)e.textContent=text;if(className)e.className=className;return e;};
export async function json(path){const packed=/^data\/[A-Za-z0-9]{11}\.json$/.test(path);const url=packed?path+'.gz':path,r=await fetch(url);if(!r.ok)throw new Error(`Could not load ${path} (${r.status})`);if(!packed)return r.json();return JSON.parse(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text());}
export function options(select,values){select.replaceChildren(...values.map(([value,label])=>{const o=element('option',label);o.value=value;return o;}));}
export function money(value){if(value==null||typeof value==='object')return value?.amount!=null ? money(value.amount) : 'Price unavailable';return new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(Number(value));}
export function showError(target,error){target.replaceChildren(element('p',error.message,'error'));}
