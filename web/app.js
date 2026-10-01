const $ = id => document.getElementById(id);
const terminal = state => ['succeeded', 'failed', 'cancelled'].includes(state);
const stateText = {queued:'Sırada',running:'Çalışıyor',succeeded:'Tamamlandı',failed:'Hata',cancelled:'İptal edildi'};
let displayedSearch = '', polling = false;
let liveSession = null, livePolling = false, liveSequence = '', liveObjectUrl = '', liveEnabled = false, liveAction = false, liveRevision = 0, liveExpiresAt = 0;
const mediaUrl = path => '/media/' + path.split('/').map(encodeURIComponent).join('/');
function notice(text) { $('notice').textContent = text; }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function element(tag, text, className) {
  const node = document.createElement(tag); if(text !== undefined) node.textContent = text;
  if(className) node.className = className; return node;
}
const liveStateText = {starting:'Model hazırlanıyor',running:'Analiz çalışıyor',stopping:'Durduruluyor',stopped:'Durduruldu',completed:'Süre tamamlandı',failed:'Hata'};
function hideLive(text) {
  $('live-image').hidden = true; $('live-placeholder').hidden = false; $('live-placeholder').textContent = text;
  if (liveObjectUrl) { URL.revokeObjectURL(liveObjectUrl); liveObjectUrl = ''; }
  $('live-image').removeAttribute('src'); liveSequence = '';
}
function showLive(session) {
  liveSession = session;
  const active = !!session?.active;
  $('live-start').disabled = !liveEnabled || active || liveAction;
  $('live-stop').disabled = !active || !!session?.cancel_requested || liveAction;
  $('live-source').disabled = active || liveAction;
  $('live-metrics').replaceChildren();
  if (!session) { $('live-state').textContent = liveEnabled ? 'Başlatılmaya hazır' : 'Kaynak tanımlanmamış'; hideLive(liveEnabled ? 'Yapılandırılmış kaynağı başlatın.' : 'Canlı kaynak bu sunucuda etkin değil.'); return; }
  if (session.source_id) $('live-source').value = session.source_id;
  const reconnecting = active && session.connection_state === 'reconnecting';
  $('live-state').textContent = reconnecting ? 'Yeniden bağlanıyor…' : liveStateText[session.state] || session.state;
  const entries = [
    ['Analiz edilen',session.processed_frames],['Okunan',session.decoded_frames],['Atlanan',session.dropped_frames],
    ['Bağlantı oturumu',session.sessions],['Kuyruk tepe / sınır',`${session.queue_high_watermark} / ${session.queue_capacity}`],
    ['Ortalama analiz',`${Number(session.mean_analysis_ms).toFixed(0)} ms`],
    ['Görüntü yaşı',session.has_preview ? `${Number(session.decode_age_ms).toFixed(0)} ms` : 'Güncel kare yok']
  ];
  entries.forEach(([label,value]) => { const cell = element('div'); cell.append(element('small',label),element('strong',String(value))); $('live-metrics').append(cell); });
  if (!session.has_preview) hideLive(session.error || (reconnecting ? 'Yayın kesildi. Yeniden bağlantı bekleniyor.' : active ? 'Güncel analiz karesi bekleniyor…' : 'Oturum kapandı. Yeniden başlatabilirsiniz.'));
}
async function refreshLive() {
  if (livePolling || document.hidden) return;
  livePolling = true;
  const revision = liveRevision;
  try {
    const data = await api('/api/live'); if (revision !== liveRevision) return; showLive(data.session);
    const session = data.session;
    if (session?.has_preview && `${session.id}:${session.preview_sequence}` !== liveSequence) {
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(),5000), fetchStarted = performance.now();
      let response;
      try { response = await fetch(`/api/live/${session.id}/preview.jpg`,{signal:controller.signal,cache:'no-store'}); }
      finally { clearTimeout(timer); }
      if (response.status === 204) hideLive('Güncel kare bekleniyor…');
      else if (response.ok && response.headers.get('Content-Type')?.startsWith('image/jpeg')) {
        const blob = await response.blob(), nextUrl = URL.createObjectURL(blob);
        const expiresAt = fetchStarted + 2000 - Number(response.headers.get('X-Decode-Age-Ms') || 2000);
        if (revision !== liveRevision || liveSession?.id !== session.id || liveSession.cancel_requested || performance.now() >= expiresAt) { URL.revokeObjectURL(nextUrl); return; }
        const oldUrl = liveObjectUrl; liveObjectUrl = nextUrl;
        liveExpiresAt = expiresAt;
        liveSequence = `${session.id}:${response.headers.get('X-Live-Sequence')}`;
        $('live-image').src = nextUrl; $('live-image').hidden = false; $('live-placeholder').hidden = true;
        if (oldUrl) URL.revokeObjectURL(oldUrl);
      } else hideLive('Önizleme alınamadı. Bağlantı yeniden kontrol ediliyor.');
    }
  } catch(error) { hideLive('Canlı servise ulaşılamıyor.'); $('live-state').textContent = 'Bağlantı yok'; }
  finally { livePolling = false; }
}
async function liveCommand(action) {
  if (liveAction) return; liveAction = true; ++liveRevision; showLive(liveSession); notice('');
  try {
    if (action === 'start') showLive(await api('/api/live/start',{source_id:$('live-source').value}));
    else if (liveSession) showLive(await api(`/api/live/${liveSession.id}/stop`,{}));
  } catch(error) { notice(error.message); }
  finally { ++liveRevision; liveAction = false; showLive(liveSession); await refreshLive(); }
}
$('live-start').addEventListener('click',() => liveCommand('start'));
$('live-stop').addEventListener('click',() => liveCommand('stop'));
document.addEventListener('visibilitychange',() => { if (!document.hidden) refreshLive(); });
window.addEventListener('pagehide',() => { if (liveObjectUrl) URL.revokeObjectURL(liveObjectUrl); });
function playAt(path, seconds) {
  const player = $('player');
  const target = new URL(mediaUrl(path), location.href).href;
  const seek = () => { player.currentTime = seconds; player.play().catch(() => {}); };
  if (player.src !== target) { player.src = target; player.addEventListener('loadedmetadata', seek, {once:true}); }
  else seek();
  $('media').value = path;
  player.scrollIntoView({behavior:'smooth',block:'center'});
}
function showResults(job) {
  displayedSearch = job.id;
  $('results').replaceChildren();
  const matches = job.result.results;
  $('result-title').textContent = `“${job.result.query}” · ${matches.length} sonuç`;
  if (!matches.length) { $('results').append(element('p','Sonuç bulunamadı. Önce bir video indeksleyin.','empty')); return; }
  matches.forEach((match, index) => {
    const card = element('button',undefined,'result'); card.type = 'button';
    const metadata = match.metadata, seconds = Number(metadata.timestamp_ms || 0) / 1000;
    if (match.media_path) {
      const img = element('img'); img.alt = `${metadata.label || 'Nesne'} · ${seconds.toFixed(1)} saniye`;
      img.loading = 'lazy'; img.src = `/api/preview/${job.id}/${index}.jpg`; card.append(img);
      card.addEventListener('click',() => playAt(match.media_path,seconds));
    } else card.disabled = true;
    const body = element('span',undefined,'result-body');
    body.append(element('strong',`${metadata.label || 'Nesne'} · ${seconds.toFixed(1)} sn`),
      element('small',match.media_path || 'Kaynak bu medya klasöründe değil'),
      element('small',`Kare ${metadata.frame_index ?? '—'} · Benzerlik ${Number(match.score).toFixed(3)}`));
    card.append(body); $('results').append(card);
  });
}
function showJobs(jobs) {
  const container = $('jobs'); container.replaceChildren();
  if (!jobs.length) container.append(element('p','Henüz iş yok. Bir video indeksleyerek başlayın.','empty'));
  jobs.slice(0,12).forEach(job => {
    const row = element('div',undefined,'job');
    const title = element('div',undefined,'job-title');
    title.append(element('strong',job.request.type === 'search' ? 'Metin araması' : 'Video indeksleme'),
      element('span',job.cancel_requested && !terminal(job.state) ? 'İptal bekleniyor' : stateText[job.state],'state'));
    row.append(title,element('p',job.request.path || job.request.query));
    if (job.recoveries > 0) row.append(element('p',`Yeniden başlatma sonrası kurtarıldı · Deneme ${job.attempts}`));
    if (job.request.type === 'index_video') row.append(element('p',`${job.progress.decoded_frames || 0} kare okundu · ${job.progress.indexed_items || 0} nesne kaydı`));
    if (job.error) row.append(element('p',job.error));
    if (!terminal(job.state)) {
      const cancel = element('button','İptal','cancel'); cancel.type = 'button'; cancel.disabled = job.cancel_requested;
      cancel.addEventListener('click',async() => { try { await api(`/api/jobs/${job.id}/cancel`,{}); await refresh(); } catch(error) { notice(error.message); } });
      row.append(cancel);
    }
    container.append(row);
  });
  const latest = jobs.find(job => job.request.type === 'search' && job.state === 'succeeded');
  if (latest && latest.id !== displayedSearch) showResults(latest);
}
async function refresh() {
  if (polling) return;
  polling = true;
  try {
    const data = await api('/api/jobs'); showJobs(data.jobs);
    await api('/api/health');
    $('health').textContent = '● Servis hazır';
  } catch(error) { $('health').textContent = 'Servise ulaşılamıyor'; notice(error.message); }
  finally { polling = false; }
}
$('index-form').addEventListener('submit',async event => {
  event.preventDefault(); notice('');
  try { await api('/api/jobs',{type:'index_video',path:$('media').value,stride:Number($('stride').value),max_frames:Number($('max-frames').value)}); await refresh(); }
  catch(error) { notice(error.message); }
});
$('search-form').addEventListener('submit',async event => {
  event.preventDefault(); notice('');
  try { await api('/api/jobs',{type:'search',query:$('query').value,limit:8}); await refresh(); }
  catch(error) { notice(error.message); }
});
$('media').addEventListener('change',() => { $('player').src = mediaUrl($('media').value); });
$('player').addEventListener('loadedmetadata',() => { $('duration').textContent = `${$('player').duration.toFixed(1)} saniye`; });
async function start() {
  try {
    await api('/api/health');
    const live = await api('/api/live/sources');
    live.sources.forEach(source => { const option = element('option',source.label); option.value = source.id; $('live-source').append(option); });
    liveEnabled = live.sources.length > 0;
    if (!liveEnabled) { hideLive('Canlı kaynak bu sunucuda etkin değil.'); $('live-help').textContent = 'Sunucuyu -LiveUrl seçeneğiyle başlatın; tarayıcıdan keyfî URL kabul edilmez.'; }
    else $('live-help').textContent = `${live.duration_seconds} saniyelik oturum · yaklaşık 2 önizleme/sn · kalıcı kayıt/indeksleme yapmaz.`;
    await refreshLive(); setInterval(refreshLive,500);
    setInterval(() => { if (liveObjectUrl && performance.now() >= liveExpiresAt) hideLive('Güncel analiz karesi bekleniyor…'); },250);
    const data = await api('/api/media');
    data.videos.forEach(video => { const option = element('option',video.path); option.value = video.path; $('media').append(option); });
    if (data.videos.length) $('player').src = mediaUrl(data.videos[0].path);
    else { $('index-button').disabled = true; notice('Medya klasöründe video yok. Bir MP4 ekleyip sayfayı yenileyin.'); }
    await refresh(); setInterval(refresh,1000);
  } catch(error) { notice(error.message); $('health').textContent = 'Servise ulaşılamıyor'; }
}
start();
