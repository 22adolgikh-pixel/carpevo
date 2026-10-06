// Сбор классификации и подписей azerbaijanrugs.com/guide — запускать в консоли браузера на https://www.azerbaijanrugs.com/guide/
// Результат: window.AR3 = {regions:[{name,url,lm,title,caps:[[подпись, страница, миниатюра]],cats:[{name,url,lm,title,caps}]}]}
// Затем JSON.stringify(AR3) → сохранить как azrugs_raw.json и прогнать build.py. Фото не скачиваются.
window.AR3={regions:[],done:false,err:[]};
(async()=>{
 const D=new DOMParser(), B='https://www.azerbaijanrugs.com/guide/';
 const get=async u=>{const r=await fetch(u); const b=await r.arrayBuffer(); return [r.headers.get('last-modified'), new TextDecoder('windows-1252').decode(b)]};
 const rel=u=>u.replace(B,'');
 const caps=(doc,url)=>{ const by=new Map();
   for(const a of doc.querySelectorAll('a[href]')){ const hr=a.getAttribute('href'); if(!hr.includes('/')||/^\.\.|^http/.test(hr)) continue; const abs=new URL(hr,url).href; if(!/\.htm/.test(abs)) continue;
     const e=by.get(abs)||{t:new Set(),img:null}; const tx=a.textContent.replace(/click to enlarge/ig,'').replace(/\s+/g,' ').trim(); if(tx) e.t.add(tx);
     const im=a.querySelector('img'); if(im&&!e.img) { e.img=new URL(im.getAttribute('src'),url).href; const td=a.closest('td'); if(td&&!td.querySelector('td')){ const t2=td.textContent.replace(/click to enlarge/ig,'').replace(/\s+/g,' ').trim(); if(t2) e.t.add(t2);} }
     by.set(abs,e); }
   return [...by].filter(([k,e])=>e.img).map(([k,e])=>[[...e.t].sort((x,y)=>y.length-x.length)[0]||'', rel(k), rel(e.img)]); };
 const idx=[...document.querySelectorAll('a')].map(a=>a.href).filter(h=>/\/guide\/[^/]+\.htm/.test(h));
 const names={}; [...document.querySelectorAll('a')].forEach(a=>{const t=a.textContent.replace(/\s+/g,' ').trim(); if(t&&!/^\(/.test(t)) names[a.href]=names[a.href]||t;});
 const seen=new Set();
 for(const url of idx){ if(seen.has(url)) continue; seen.add(url); const name=names[url]||rel(url);
  try{ const [lm,h]=await get(url); const doc=D.parseFromString(h,'text/html');
   const R={name,url:rel(url),lm,title:doc.title,caps:caps(doc,url),cats:[]};
   const links=[...doc.querySelectorAll('a[href]')].map(a=>[a.textContent.replace(/\s+/g,' ').trim(),a.getAttribute('href')]).filter(x=>x[0]&&!/click to enlarge|back to|main page/i.test(x[0])&&/^[^/]+\.htm$/.test(x[1])&&!/^index|guide_index/.test(x[1]));
   const s2=new Set();
   for(const [cn,ch] of links){ const cu=new URL(ch,url).href; if(s2.has(cu)||cu===url) continue; s2.add(cu);
     try{ const [lm2,h2]=await get(cu); const d2=D.parseFromString(h2,'text/html'); R.cats.push({name:cn,url:rel(cu),lm:lm2,title:d2.title,caps:caps(d2,cu)}); }catch(e){AR3.err.push(cu+' '+e)} }
   AR3.regions.push(R);
  }catch(e){AR3.err.push(url+' '+e)}
 }
 AR3.done=true;})();
