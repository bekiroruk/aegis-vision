(() => {
  'use strict';
  const panel = document.getElementById('redis-image-panel');
  if (!panel) return;
  // Static markup only; filenames, labels and server errors always use textContent.
  panel.innerHTML = `<div class="panel-title"><h2>Fotoğrafta nesne bul</h2><span class="badge subtle">Redis · C++ · CPU</span></div>
    <p>Fotoğraf kuyruğa alınır; ayrı C++ işçisi nesneleri bulur. Video ve diğer analizlerin kuyruğu değişmez.</p>
    <p id="ri-connection" role="status">Bağlantı kontrol ediliyor…</p>
    <form id="ri-form" class="index-controls"><div><label for="ri-image">Yerel fotoğraf</label><select id="ri-image" disabled></select></div>
    <button id="ri-submit" disabled>Fotoğrafı analiz et</button></form>
    <div class="demo-actions"><button id="ri-refresh" type="button">Bağlantıyı / sonucu yenile</button>
    <button id="ri-retry" type="button" hidden>Aynı işi tekrar gönder</button><button id="ri-cancel" type="button" hidden>İşi iptal et</button>
    <button id="ri-new" type="button" hidden>Yeni analiz</button></div>
    <p class="hint">Redis bağlantısı, işçinin çalıştığı anlamına gelmez. İş sırada kalırsa C++ worker'ı kontrol edin. Yalnızca medya klasöründeki JPG/PNG dosyaları listelenir.</p>
    <p id="ri-status" role="status" aria-live="polite"></p><img id="ri-preview" alt="Fotoğraftaki tespit edilmiş nesneler" hidden>
    <ul id="ri-results"></ul>`;
  const el = id => document.getElementById('ri-' + id);
  const idOK = id => typeof id === 'string' && /^[A-Za-z0-9_-]{1,64}$/.test(id);
  const terminal = state => ['succeeded', 'failed', 'cancelled'].includes(state);
  const storageKey = 'aegis-redis-image-v1';
  let context = '', queueName = '', enabled = false, active = null, pending = false, revision = 0, shown = '';
  async function api(path, body) {
    const r = await fetch(path, {signal: AbortSignal.timeout(10000), ...(body === undefined ? {} :
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
    const value = await r.json();
    if (!r.ok) { const e = new Error(value.error || `HTTP ${r.status}`); e.status=r.status; throw e; }
    return value;
  }
  function save() {
    try { if (active) localStorage.setItem(storageKey, JSON.stringify(active)); else localStorage.removeItem(storageKey); }
    catch { /* The open page still retains its stable ID if storage is disabled. */ }
  }
  function sync() {
    el('submit').disabled=!enabled || pending || !!active || !el('image').value;
    el('image').disabled=!enabled || pending || !!active;
    el('cancel').hidden=!active || terminal(active.state);
    el('cancel').disabled=pending || !enabled;
    el('retry').disabled=pending || !enabled;
    el('new').hidden=!active || !terminal(active.state);
    el('refresh').disabled=pending;
  }
  function clearResult() { shown=''; el('preview').hidden=true; el('preview').removeAttribute('src'); el('results').replaceChildren(); }
  function show(job) {
    if (!active || job.id!==active.id) return;
    if (!['queued','running','succeeded','failed','cancelled'].includes(job.state)) throw new Error('Geçersiz iş durumu');
    active.state=job.state; save(); el('retry').hidden=true; sync();
    const labels={queued:'Sırada · C++ worker bekleniyor',running:'Fotoğraf analiz ediliyor',succeeded:'Tamamlandı',failed:'Analiz başarısız; worker kaydını kontrol edin',cancelled:'İptal edildi'};
    el('status').textContent=`${active.id} · ${labels[job.state]} · Deneme ${job.attempts}`;
    if (job.state!=='succeeded' || shown===job.id) return;
    const r=job.result;
    if (!r || r.context!==context || r.image!==active.image || !Array.isArray(r.detections) || r.detections.length>100)
      throw new Error('Sonuç bu görüntü veya model ile eşleşmiyor');
    clearResult();
    for (const d of r.detections) {
      if(typeof d.label!=='string' || d.label.length>64 || !Number.isFinite(d.confidence) || d.confidence<0 || d.confidence>1)
        throw new Error('Geçersiz tespit sonucu');
      const item=document.createElement('li');item.textContent=`${d.label} · skor ${d.confidence.toFixed(3)}`;el('results').append(item);
    }
    if (!r.detections.length) { const item=document.createElement('li');item.textContent='Bu eşikte nesne bulunamadı.';el('results').append(item); }
    const image=el('preview'), stamp=revision;
    image.onload=()=>{if(stamp===revision && active?.id===job.id) image.hidden=false;};
    image.onerror=()=>{if(stamp===revision && active?.id===job.id) {image.hidden=true;shown='';el('status').textContent='Sonuç var ama görüntü doğrulanamadı veya değişti. Yenile / yeni analiz kullanın.';}};
    shown=job.id; image.src=`/api/redis-images/${encodeURIComponent(job.id)}/preview.jpg`;
  }
  async function poll() {
    if (pending || !enabled || !active || terminal(active.state)) return;
    pending=true;sync();const stamp=revision;
    try { const job=await api('/api/redis-images/'+encodeURIComponent(active.id));if(stamp===revision) show(job); }
    catch(e) { if(stamp===revision) {
      el('status').textContent=e.status===404 ? 'İş bulunamadı. İsterseniz aynı kimlikle tekrar gönderin.' : `Durum alınamadı: ${e.message}. İş kimliği korundu.`;
      el('retry').hidden=e.status!==404;
    }} finally {pending=false;sync();}
  }
  async function connect() {
    if(pending) return;
    pending=true;sync();
    try {
      const info=await api('/api/redis-images');
      enabled=info.enabled===true;
      panel.hidden=!enabled && panel.dataset?.optional==='true';
      if(!enabled) {el('connection').textContent='Redis görüntü kuyruğu bu serviste etkin değil.';return;}
      if(typeof info.context!=='string' || typeof info.queue!=='string' || !Array.isArray(info.images)) throw new Error('Geçersiz kuyruk bilgisi');
      if(context && (context!==info.context || queueName!==info.queue)) {++revision;active=null;clearResult();save();}
      context=info.context;queueName=info.queue;
      el('connection').textContent=`Redis erişilebilir · ${info.queue} · CPU işçisi ayrı çalışır`;
      const previous=el('image').value;el('image').replaceChildren();
      for(const name of info.images.slice(0,128)) {if(typeof name!=='string') continue;const option=document.createElement('option');option.value=name;option.textContent=name;el('image').append(option);}
      if(info.images.includes(previous)) el('image').value=previous;
      if(!active) {try { const saved=JSON.parse(localStorage.getItem(storageKey));
        if(saved?.context===context && saved.queue===queueName && idOK(saved.id) && typeof saved.image==='string') active={...saved,state:'queued'};
      } catch {}}
      if(active) {active.state='queued';shown='';}
      if(!info.images.length) el('connection').textContent+=' · Medya klasöründe JPG/PNG bulunamadı.';
    } catch(e) {enabled=false;panel.hidden=false;clearResult();el('connection').textContent=`Redis bağlantısı yok: ${e.message}`;}
    finally {pending=false;sync();}
    await poll();
  }
  async function submit(retry=false) {
    if(pending || !enabled) return;
    if(!retry) {if(active || !el('image').value) return;++revision;clearResult();active={id:'img-'+crypto.randomUUID(),image:el('image').value,context,queue:queueName,state:'queued'};save();}
    if(!active) return;
    pending=true;sync();el('retry').hidden=true;
    try {show(await api('/api/redis-images',{id:active.id,image:active.image}));}
    catch(e) {el('status').textContent=`Gönderim doğrulanamadı: ${e.message}. Kimlik ${active.id} korundu; önce durumu yenileyin.`;
      // Never create a fresh ID after a potentially committed submit.
      el('retry').hidden=false;
      if([400,409,429].includes(e.status)) {active.state='failed';save();}
    } finally {pending=false;sync();}
  }
  el('form').addEventListener('submit',e=>{e.preventDefault();void submit();});
  el('retry').addEventListener('click',()=>void submit(true));
  el('refresh').addEventListener('click',()=>void connect());
  el('cancel').addEventListener('click',async()=>{
    if(pending || !active) return;pending=true;sync();
    try {show(await api(`/api/redis-images/${encodeURIComponent(active.id)}/cancel`,{}));}
    catch(e) {el('status').textContent=`İptal doğrulanamadı: ${e.message}. Durumu yenileyin.`;}
    finally {pending=false;sync();}
  });
  el('new').addEventListener('click',()=>{if(pending) return;++revision;active=null;save();clearResult();el('status').textContent='';el('retry').hidden=true;sync();});
  el('image').addEventListener('change',sync);
  setInterval(()=>void poll(),1000);
  void connect();
})();
