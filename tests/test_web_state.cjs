// Development-only regression test. The C++ service does not require Node.js.
// Run from any directory: node tests/test_web_state.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const base = 'http://127.0.0.1:8090/';
const ordinary = 'pedestrians.mp4';
const archived = 'live-archive/live-200-1/segment-0001/clip.mp4';
const later = 'live-archive/live-300-1/segment-0001/clip.mp4';
const url = value => new URL('/media/' + value.split('/').map(encodeURIComponent).join('/'),base).href;
function deferred() {
  let resolve;
  const promise = new Promise(value => { resolve = value; });
  return {promise,resolve};
}
function segment(media, session = 'live-200-1') {
  return {session_id:session,segment_index:1,media_path:media,index_state:'succeeded',
    frames:62,source_fps:10,playback_duration_seconds:6.2,source_session:1};
}
class Node {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.events = new Map(); this.dataset = {};
    this.selectedIndex = -1; this._value = ''; this._src = ''; this.disabled = false;
    this.checked = false; this.hidden = false; this.textContent = ''; this.duration = 6.2;
  }
  get value() { return this.tag === 'select' ? this.children[this.selectedIndex]?.value || '' : this._value; }
  set value(value) {
    if (this.tag === 'select') this.selectedIndex = this.children.findIndex(option => option.value === value);
    else this._value = value;
  }
  get src() { return this._src; }
  set src(value) { this._src = new URL(value,base).href; }
  get options() { return this.children; }
  append(...nodes) {
    this.children.push(...nodes);
    // Native single-select automatically chooses its first option when added.
    if (this.tag === 'select' && this.selectedIndex < 0 && this.children.length) this.selectedIndex = 0;
  }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  addEventListener(name,callback,options) {
    if (!this.events.has(name)) this.events.set(name,[]);
    this.events.get(name).push({callback,once:!!options?.once});
  }
  async emit(name,event = {}) {
    const entries = [...(this.events.get(name) || [])];
    this.events.set(name,entries.filter(entry => !entry.once));
    await Promise.all(entries.map(entry => entry.callback(event)));
  }
  play() { return Promise.resolve(); }
  scrollIntoView() {}
  removeAttribute(name) { if (name === 'src') this._src = ''; }
  set innerHTML(value) { throw new Error('Metadata must be text, never HTML'); }
}
function harness({delayMedia = false, videos = [ordinary], segmentation=false, ocr=false, delayMask=false} = {}) {
  const nodes = new Map(), calls = [], media = deferred();
  const maskReply=deferred(), jobs=[];
  const get = id => {
    if (!nodes.has(id)) nodes.set(id,new Node(['media','live-source','search-scope'].includes(id) ? 'select' : 'div'));
    return nodes.get(id);
  };
  const catalog = {enabled:true,segments:[segment(archived)],config:{segment_seconds:10,
    max_segments_per_session:4,max_total_segments:8,max_bytes:268435456}};
  let mediaRequested = false;
  const fetch = async (pathname,options = {}) => {
    calls.push({pathname,body:options.body ? JSON.parse(options.body) : undefined});
    let data;
    if (pathname === '/api/health') data = {segmentation_enabled:segmentation,ocr_enabled:ocr};
    else if (pathname === '/api/live/sources') data = {sources:[],archive_available:true,archive_config:catalog.config};
    else if (pathname === '/api/live') data = {session:null};
    else if (pathname === '/api/live/archive') data = catalog;
    else if (pathname === '/api/jobs' && options.body) {
      const request=JSON.parse(options.body);
      data={id:`1-${jobs.length+1}`,request,state:'queued',progress:{}};jobs.unshift(data);
      if (delayMask) await maskReply.promise;
    }
    else if (pathname === '/api/jobs') data = {jobs};
    else if (pathname === '/api/media') {
      mediaRequested = true;
      if (delayMedia) await media.promise;
      data = {videos:videos.map(value => ({path:value}))};
    } else throw new Error('Unexpected test request: ' + pathname);
    return {ok:true,status:200,json:async () => data};
  };
  const context = {document:{hidden:false,getElementById:get,createElement:tag => new Node(tag),
    addEventListener(){},querySelectorAll(){return [];}},window:{addEventListener(){}},
    fetch,console,URL,location:{href:base},performance,AbortController,setTimeout,clearTimeout,setInterval(){}};
  vm.createContext(context);
  const source = fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  assert.match(source,/start\(\);\s*$/,'App bootstrap must remain visible to the regression harness');
  vm.runInContext(source.replace(/start\(\);\s*$/,'globalThis.started = start();') +
    ';globalThis.ui = {showArchive,chooseMedia,playAt,refresh};',context);
  return {get,calls,catalog,ui:context.ui,started:context.started,releaseMedia:media.resolve,
    mediaRequested:() => mediaRequested,jobs,releaseMask:maskReply.resolve};
}
async function waitingForMedia(test) {
  for (let i = 0; i < 20 && !test.mediaRequested(); ++i) await new Promise(resolve => setImmediate(resolve));
  assert.equal(test.mediaRequested(),true,'Startup should be waiting for its media response');
}
function consistent(test,value,managed = false) {
  assert.equal(test.get('media').value,value,'Selector must describe the actual player source');
  assert.equal(test.get('player').src,url(value),'Player must show the selected media');
  for (const id of ['index-button','stride','max-frames'])
    assert.equal(test.get(id).disabled,managed,'Archive media must not enable ordinary indexing');
}
async function mediaFirst() {
  const test = harness(); await test.started;
  consistent(test,ordinary);
  assert(test.get('media').options.some(option => option.value === archived));
  test.ui.showArchive({...test.catalog,segments:[segment(later,'live-300-1'),segment(archived)]});
  consistent(test,ordinary);
  test.ui.playAt(archived,3); await test.get('player').emit('loadedmetadata');
  consistent(test,archived,true); assert.equal(test.get('player').currentTime,3);
  test.ui.showArchive({...test.catalog,segments:[segment(later,'live-300-1')]});
  consistent(test,archived,true); // A newer poll must not steal the chosen clip.
  const before = test.calls.length;
  await test.get('index-form').emit('submit',{preventDefault(){}});
  assert.equal(test.calls.length,before,'Managed MP4 cannot submit a generic index job');
  test.get('media').value = ordinary; await test.get('media').emit('change');
  consistent(test,ordinary);
  test.ui.showArchive(test.catalog); consistent(test,ordinary);
}
async function archiveFirst() {
  const test = harness({delayMedia:true}); await waitingForMedia(test);
  test.ui.showArchive(test.catalog); // Simulate a visibility/poll response before media initialization.
  assert.equal(test.get('media').value,''); assert.equal(test.get('player').src,'');
  assert.equal(test.get('index-button').disabled,true);
  test.releaseMedia(); await test.started;
  consistent(test,ordinary); // Ordinary files win the default selection, never background archives.
}
async function lateInitializationAfterUserChoice() {
  const test = harness({delayMedia:true}); await waitingForMedia(test);
  test.ui.showArchive(test.catalog);
  test.ui.playAt(archived,3); await test.get('player').emit('loadedmetadata');
  consistent(test,archived,true);
  test.releaseMedia(); await test.started;
  consistent(test,archived,true); assert.equal(test.get('player').currentTime,3);
}
async function archiveOnly() {
  const test = harness({videos:[]}); await test.started;
  consistent(test,archived,true);
  test.ui.showArchive({...test.catalog,segments:[segment(later,'live-300-1'),segment(archived)]});
  consistent(test,archived,true);
}
async function segmentationPanel() {
  const disabled=harness();await disabled.started;
  assert.equal(disabled.get('segmentation-button').disabled,true);
  const test=harness({segmentation:true});await test.started;
  assert.equal(test.get('segmentation-button').disabled,false);
  test.get('segmentation-frame').value='2';
  await test.get('segmentation-form').emit('submit',{preventDefault(){}});
  assert.equal(test.jobs.length,1);assert.equal(test.jobs[0].request.frame_index,2);
  assert.equal(test.get('segmentation-button').disabled,true);
  const job=test.jobs[0];job.state='succeeded';
  job.result={path:ordinary,frame_index:2,width:640,height:480,preview_data_url:'data:image/jpeg;base64,/9j/',
    instances:[{label:'<script>not markup</script>',score:.9,mask_pixels:123}]};
  await test.ui.refresh();
  assert.equal(test.get('segmentation-image').hidden,false);
  assert.match(test.get('segmentation-status').textContent,/Kare 2/);
  assert.equal(test.get('segmentation-instances').children.length,1);
  assert.equal(test.get('segmentation-button').disabled,false);
  test.ui.chooseMedia(archived,true);
  assert.equal(test.get('segmentation-image').hidden,true);
  assert.equal(test.get('segmentation-image').src,'');
  assert.equal(test.get('segmentation-button').disabled,true);
  const reopen=test.get('jobs').children[0].children.find(node=>node.textContent==='Maskeyi göster');
  await reopen.emit('click');
  assert.equal(test.get('media').value,ordinary);
  assert.equal(test.get('segmentation-image').hidden,false);
  test.ui.chooseMedia(ordinary,true);
  await test.get('segmentation-form').emit('submit',{preventDefault(){}});
  test.jobs[0].state='failed';test.jobs[0].error='model failure';await test.ui.refresh();
  assert.equal(test.get('segmentation-image').hidden,true);
  assert.equal(test.get('segmentation-button').disabled,false);
  assert.equal(test.get('segmentation-status').textContent,'model failure');
  await test.get('segmentation-form').emit('submit',{preventDefault(){}});
  test.jobs[0].state='succeeded';test.jobs[0].result={...job.result,preview_data_url:'https://untrusted/image.jpg'};
  await test.ui.refresh();assert.equal(test.get('segmentation-image').hidden,true);
}
async function staleSegmentationSubmission() {
  const test=harness({segmentation:true,delayMask:true});await test.started;
  test.get('segmentation-frame').value='0';
  const pending=test.get('segmentation-form').emit('submit',{preventDefault(){}});
  test.ui.chooseMedia(archived,true);test.releaseMask();await pending;
  test.jobs[0].state='succeeded';test.jobs[0].result={path:ordinary,frame_index:0,preview_data_url:'data:image/jpeg;base64,/9j/',instances:[]};
  await test.ui.refresh();assert.equal(test.get('segmentation-image').hidden,true);
  assert.equal(test.get('segmentation-status').textContent,'');
}
async function ocrPanel() {
  const disabled=harness();await disabled.started;assert.equal(disabled.get('ocr-button').disabled,true);
  const test=harness({ocr:true});await test.started;
  assert.equal(test.get('ocr-button').disabled,false);test.get('ocr-frame').value='2';
  await test.get('ocr-form').emit('submit',{preventDefault(){}});
  assert.equal(test.jobs[0].request.type,'ocr_frame');assert.equal(test.jobs[0].request.frame_index,2);
  assert.equal(test.get('ocr-button').disabled,true);
  const job=test.jobs[0];job.state='succeeded';job.result={path:ordinary,frame_index:2,width:640,height:480,
    preview_data_url:'data:image/jpeg;base64,/9j/',regions:[{text:'<script>plain text</script>',detection_confidence:.9,recognition_confidence:.8}]};
  await test.ui.refresh();assert.equal(test.get('ocr-image').hidden,false);
  assert.equal(test.get('ocr-regions').children[0].children[0].textContent,'<script>plain text</script>');
  assert.equal(test.get('ocr-button').disabled,false);
  test.ui.chooseMedia(archived,true);assert.equal(test.get('ocr-image').hidden,true);assert.equal(test.get('ocr-button').disabled,true);
  const reopen=test.get('jobs').children[0].children.find(node=>node.textContent==='Yazıları göster');await reopen.emit('click');
  assert.equal(test.get('media').value,ordinary);assert.equal(test.get('ocr-image').hidden,false);
  for (const state of ['failed','cancelled']) {
    await test.get('ocr-form').emit('submit',{preventDefault(){}});test.jobs[0].state=state;
    await test.ui.refresh();assert.equal(test.get('ocr-image').hidden,true);assert.equal(test.get('ocr-button').disabled,false);
  }
  await test.get('ocr-form').emit('submit',{preventDefault(){}});test.jobs[0].state='succeeded';
  test.jobs[0].result={...job.result,regions:[]};await test.ui.refresh();
  assert.match(test.get('ocr-regions').children[0].textContent,/metin bulunamadı/);
  await test.get('ocr-form').emit('submit',{preventDefault(){}});test.jobs[0].state='succeeded';
  test.jobs[0].result={...job.result,preview_data_url:'https://untrusted/image.jpg'};
  await test.ui.refresh();assert.equal(test.get('ocr-image').hidden,true);
  const delayed=harness({ocr:true,delayMask:true});await delayed.started;delayed.get('ocr-frame').value='0';
  const pending=delayed.get('ocr-form').emit('submit',{preventDefault(){}});
  delayed.ui.chooseMedia(archived,true);delayed.releaseMask();await pending;
  delayed.jobs[0].state='succeeded';delayed.jobs[0].result=job.result;
  await delayed.ui.refresh();assert.equal(delayed.get('ocr-image').hidden,true);assert.equal(delayed.get('ocr-status').textContent,'');
}
(async () => {
  await mediaFirst(); await archiveFirst(); await lateInitializationAfterUserChoice(); await archiveOnly();
  await segmentationPanel();await staleSegmentationSubmission();
  await ocrPanel();console.log('OCR UI passed: capability, request, safe text/URL, source change, history, empty, failure, cancellation, stale reply.');
  console.log('Segmentation UI passed: capability, submit, result, source change, failure, unsafe URL, delayed submission.');
  console.log('Web media state passed: media-first, archive-first, preserved user selection, seek and managed-index guards.');
})().catch(error => { console.error(error); process.exitCode = 1; });
