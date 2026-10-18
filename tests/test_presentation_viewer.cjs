// Offline presentation controls, no browser or model dependencies.
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../docs/sunum/index.html'),'utf8');
const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes=new Map(['slide','counter','prev','next','full'].map(id=>[id,{}]));
const handlers={};
const doc={getElementById:id=>nodes.get(id),addEventListener:(event,fn)=>handlers[event]=fn,
  documentElement:{requestFullscreen:async()=>{throw new Error('unsupported');}},fullscreenElement:null};
vm.runInNewContext(source,{document:doc});
const next=nodes.get('next'),prev=nodes.get('prev');
next.onclick();assert.equal(nodes.get('slide').src,'assets/slide-2.png');
for(let i=0;i<10;i++)next.onclick();
assert.equal(nodes.get('counter').textContent,'6 / 6');assert.equal(next.disabled,true);
for(let i=0;i<10;i++)prev.onclick();
assert.equal(nodes.get('counter').textContent,'1 / 6');assert.equal(prev.disabled,true);
handlers.keydown({key:'ArrowRight',target:{tagName:'VIDEO'}});
assert.equal(nodes.get('counter').textContent,'1 / 6','Video keys must not navigate slides');
handlers.keydown({key:'ArrowRight',target:{tagName:'BODY'}});
assert.equal(nodes.get('counter').textContent,'2 / 6');
assert.equal(/https?:\/\//.test(html),false,'Viewer must use local assets only');
(async()=>{await nodes.get('full').onclick();assert.match(nodes.get('full').textContent,/F11/);console.log('Presentation viewer passed: bounds, keyboard, video controls, offline assets, fullscreen fallback.');})().catch(e=>{console.error(e);process.exitCode=1;});
