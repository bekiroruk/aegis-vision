const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const code=fs.readFileSync('web/redis-images.js','utf8');
const flush=async()=>{for(let i=0;i<5;i++) await new Promise(r=>setImmediate(r));};
function harness(handler,stored=null) {
  const nodes=new Map(),store=new Map(stored?[['aegis-redis-image-v1',JSON.stringify(stored)]]:[]),calls=[],timers=[];
  class Element {
    constructor(tag='div') {this.tag=tag;this.children=[];this.events={};this.hidden=false;this.disabled=false;this.textContent='';this._value='';}
    set innerHTML(html) {for(const m of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"([^>]*)>/g)) {const e=new Element(m[1]);e.hidden=/\bhidden\b/.test(m[3]);e.disabled=/\bdisabled\b/.test(m[3]);nodes.set(m[2],e);}}
    get value() {return this._value || (this.tag==='select'?this.children[0]?.value || '':'');}
    set value(v) {this._value=v;}
    append(e) {this.children.push(e);}
    replaceChildren(...c) {this.children=c;this._value='';}
    addEventListener(type,fn) {this.events[type]=fn;}
    removeAttribute(name) {delete this[name];}
  }
  nodes.set('redis-image-panel',new Element());
  const context={document:{getElementById:id=>nodes.get(id),createElement:t=>new Element(t)},
    localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},
    crypto:{randomUUID:()=> '12345678-1234-1234-1234-123456789abc'},AbortSignal,
    setInterval:fn=>timers.push(fn),
    fetch:async(path,options)=>{const body=options.body?JSON.parse(options.body):undefined;calls.push({path,body});
      const r=await handler(path,body);return {ok:(r.status||200)<400,status:r.status||200,json:async()=>r.value};}};
  vm.runInNewContext(code,context);
  return {nodes,store,calls,timers,el:id=>nodes.get('ri-'+id),click:async(id,type='click')=>{nodes.get('ri-'+id).events[type]({preventDefault(){}});await flush();}};
}
const status={enabled:true,queue:'images',context:'ctx',images:['bus.jpg'],worker_status:'unknown'};
(async()=>{
  let h=harness(async()=>({value:{enabled:false}}));await flush();assert.equal(h.el('submit').disabled,true);
  let job=null,failSubmit=true;
  h=harness(async(path,body)=>{
    if(path==='/api/redis-images' && body===undefined) return {value:status};
    if(path==='/api/redis-images') {
      if(failSubmit) {failSubmit=false;throw new Error('lost acknowledgement');}
      job={id:body.id,state:'queued',attempts:0};return {value:job};
    }
    if(path.endsWith('/cancel')) {job.state='cancelled';return {value:job};}
    return job?{value:job}:{status:404,value:{error:'missing'}};
  });await flush();assert.equal(h.el('submit').disabled,false);
  await h.click('form','submit');
  const saved=JSON.parse(h.store.get('aegis-redis-image-v1'));assert.match(saved.id,/^img-/);
  assert.equal(h.el('submit').disabled,true);assert.equal(h.el('retry').hidden,false);
  await h.click('refresh');assert.equal(h.el('retry').hidden,false);
  await h.click('retry');assert.equal(job.id,saved.id);
  assert.equal(h.calls.filter(c=>c.body?.image).length,2);
  job={...job,state:'succeeded',attempts:1,result:{context:'ctx',image:'bus.jpg',detections:[{label:'<img onerror=alert(1)>',confidence:.9}]}};
  h.timers[0]();await flush();
  assert.equal(h.el('results').children[0].textContent,'<img onerror=alert(1)> · skor 0.900');
  assert.equal(h.el('preview').src,`/api/redis-images/${saved.id}/preview.jpg`);
  h.el('preview').onload();assert.equal(h.el('preview').hidden,false);
  await h.click('new');assert.equal(h.el('preview').hidden,true);assert.equal(h.store.size,0);
  await h.click('form','submit');await h.click('cancel');assert.equal(h.el('new').hidden,false);
  // Refresh restores the same accepted ID; never auto-submits it under a fresh ID.
  h=harness(async(path)=>path==='/api/redis-images'?{value:status}:{value:{id:saved.id,state:'running',attempts:1}},saved);
  await flush();assert.match(h.el('status').textContent,/analiz ediliyor/);assert.equal(h.calls.filter(c=>c.body).length,0);
  // Do not restore an old model/media namespace.
  h=harness(async()=>({value:status}),{...saved,context:'old'});await flush();assert.equal(h.el('submit').disabled,false);
  h=harness(async()=>({value:status}),{...saved,queue:'different'});await flush();assert.equal(h.el('submit').disabled,false);
  console.log('Redis UI: disabled state, stable retry ID, refresh recovery, cancellation, safe labels and preview passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
