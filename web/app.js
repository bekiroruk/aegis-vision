const $ = id => document.getElementById(id);
const terminal = state => ['succeeded', 'failed', 'cancelled'].includes(state);
const stateText = {queued:'Sırada',running:'Çalışıyor',succeeded:'Tamamlandı',failed:'Hata',cancelled:'İptal edildi'};
let displayedSearch = '', polling = false;
let liveSession = null, livePolling = false, liveSequence = '', liveObjectUrl = '', liveEnabled = false, liveAction = false, liveRevision = 0, liveExpiresAt = 0;
let archiveAvailable = false, archivePolling = false, archiveRevision = 0;
let selectedMedia = '', mediaReady = false, mediaChoiceMade = false;
let segmentationEnabled=false, segmentationSubmitting=false, segmentationJob='', segmentationRevision=0;
function syncSegmentation() {
  const available=segmentationEnabled && !!selectedMedia && !selectedMedia.startsWith('live-archive/');
  $('segmentation-button').disabled=!available || segmentationSubmitting || !!segmentationJob;
  $('segmentation-frame').disabled=!available || segmentationSubmitting || !!segmentationJob;
  $('segmentation-state').textContent=segmentationEnabled ? 'Tek kare · C++' : 'Model etkin değil';
  $('segmentation-source').textContent=selectedMedia ? `Kaynak: ${selectedMedia}` : 'Video seçin.';
}
function clearSegmentation() {
  ++segmentationRevision;segmentationJob='';
  $('segmentation-image').hidden=true;$('segmentation-image').removeAttribute('src');
  $('segmentation-instances').replaceChildren();$('segmentation-status').textContent='';
}
function showSegmentation(job) {
  if (!segmentationJob || job.id!==segmentationJob) return;
  $('segmentation-status').textContent=`Kare ${job.request.frame_index} · ${stateText[job.state] || job.state}`;
  if (!terminal(job.state)) return;
  segmentationJob='';syncSegmentation();
  if (job.state!=='succeeded') { $('segmentation-status').textContent=job.error || stateText[job.state];return; }
  const result=job.result;
  if (result.path!==selectedMedia) return;
  const url=result.preview_data_url;
  if (typeof url!=='string' || url.length>750000 || !/^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/.test(url)) {
    $('segmentation-status').textContent='Bu sonuçta geçerli maske önizlemesi yok. Analizi yeniden başlatın.';return;
  }
  $('segmentation-image').src=url;$('segmentation-image').hidden=false;
  $('segmentation-status').textContent=`${result.path} · Kare ${result.frame_index} · ${result.instances.length} maske · ${result.width}×${result.height}`;
  for (const item of result.instances) {
    const cell=element('div');cell.append(element('strong',item.label),element('small',`Skor ${Number(item.score).toFixed(2)} · ${item.mask_pixels} piksel`));
    $('segmentation-instances').append(cell);
  }
}
const archiveActions = new Set();
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
const archiveStateText = {recording:'Kaydediliyor',active:'Kaydediliyor',disabled:'Kapalı',idle:'Hazır',stopped:'Kayıt durdu',completed:'Kayıt tamamlandı',finished:'Kayıt tamamlandı',limit_reached:'Kayıt sınırına ulaşıldı',quota_reached:'Kota doldu',quota:'Kota doldu',full:'Kota doldu',failed:'Arşiv hatası',error:'Arşiv hatası'};
function archiveLimits(config) {
  if (!config) return 'Arşiv bu sunucuda etkin değil.';
  return `${config.segment_seconds} sn/parça · oturumda en fazla ${config.max_segments_per_session} parça · toplam ${config.max_total_segments} parça / ${Math.round(config.max_bytes / 1048576)} MiB. Analiz kareleri kaydedilir; otomatik silme yok.`;
}
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
  $('live-archive').disabled = !archiveAvailable || active || liveAction;
  $('live-metrics').replaceChildren();
  if (!session) { $('live-state').textContent = liveEnabled ? 'Başlatılmaya hazır' : 'Kaynak tanımlanmamış'; hideLive(liveEnabled ? 'Yapılandırılmış kaynağı başlatın.' : 'Canlı kaynak bu sunucuda etkin değil.'); return; }
  if (session.source_id) $('live-source').value = session.source_id;
  if (session.archive && active) $('live-archive').checked = !!session.archive.enabled;
  const reconnecting = active && session.connection_state === 'reconnecting';
  $('live-state').textContent = reconnecting ? 'Yeniden bağlanıyor…' : liveStateText[session.state] || session.state;
  const entries = [
    ['Analiz',session.analysis_mode === 'segmentation' ? 'Piksel maskesi + takip' : 'Nesne tespiti + takip'],
    ['Analiz edilen',session.processed_frames],['Okunan',session.decoded_frames],['Atlanan',session.dropped_frames],
    ['Bağlantı oturumu',session.sessions],['Kuyruk tepe / sınır',`${session.queue_high_watermark} / ${session.queue_capacity}`],
    ['Ortalama analiz',`${Number(session.mean_analysis_ms).toFixed(0)} ms`],
    ['Görüntü yaşı',session.has_preview ? `${Number(session.decode_age_ms).toFixed(0)} ms` : 'Güncel kare yok']
  ];
  entries.forEach(([label,value]) => { const cell = element('div'); cell.append(element('small',label),element('strong',String(value))); $('live-metrics').append(cell); });
  if (session.archive?.enabled) {
    const cell = element('div');
    cell.append(element('small','Arşiv'),element('strong',archiveStateText[session.archive.state] || session.archive.state || 'Etkin'),element('small',`${session.archive.closed_segments || 0} kapatılan parça`));
    if (session.archive.index_queue_failures) cell.append(element('small',`${session.archive.index_queue_failures} parça için indekslemeyi tekrar deneyin.`));
    if (session.archive.error) cell.append(element('small',session.archive.error));
    $('live-metrics').append(cell);
  }
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
      if (revision !== liveRevision || liveSession?.id !== session.id) return;
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
  } catch(error) { if (revision === liveRevision) { hideLive('Canlı servise ulaşılamıyor.'); $('live-state').textContent = 'Bağlantı yok'; } }
  finally { livePolling = false; }
}
async function liveCommand(action) {
  if (liveAction) return; liveAction = true; ++liveRevision; showLive(liveSession); notice('');
  try {
    if (action === 'start') showLive(await api('/api/live/start',{source_id:$('live-source').value,archive:archiveAvailable && $('live-archive').checked}));
    else if (liveSession) showLive(await api(`/api/live/${liveSession.id}/stop`,{}));
  } catch(error) { notice(error.message); }
  finally { ++liveRevision; liveAction = false; showLive(liveSession); await refreshLive(); await refreshArchive(); }
}
$('live-start').addEventListener('click',() => liveCommand('start'));
$('live-stop').addEventListener('click',() => liveCommand('stop'));
document.addEventListener('visibilitychange',() => { if (!document.hidden) { refreshLive(); refreshArchive(); } });
window.addEventListener('pagehide',() => { if (liveObjectUrl) URL.revokeObjectURL(liveObjectUrl); });
function playAt(path, seconds) {
  const player = $('player');
  const target = new URL(mediaUrl(path), location.href).href;
  const seek = () => { if (selectedMedia !== path || player.src !== target) return; player.currentTime = seconds; player.play().catch(() => {}); };
  const changing = player.src !== target;
  if (changing) player.addEventListener('loadedmetadata',seek,{once:true});
  chooseMedia(path,true);
  if (!changing) seek();
  player.scrollIntoView({behavior:'smooth',block:'center'});
}
function syncMediaControls() {
  const managed = selectedMedia.startsWith('live-archive/');
  $('index-button').disabled = !selectedMedia || managed;
  $('stride').disabled = !selectedMedia || managed;
  $('max-frames').disabled = !selectedMedia || managed;
  syncSegmentation();
}
function addMedia(path) {
  if (!Array.from($('media').options).some(option => option.value === path)) {
    const option = element('option',path); option.value = path; $('media').append(option);
  }
  // Adding the first option implicitly selects it in browsers. Only chooseMedia
  // may change the selected source; background archive refresh must not do so.
  $('media').value = selectedMedia;
  syncMediaControls();
}
function chooseMedia(path, userChoice = false) {
  if (!path) return;
  if (path!==selectedMedia) clearSegmentation();
  addMedia(path); selectedMedia = path; $('media').value = path;
  if (userChoice) mediaChoiceMade = true;
  const target = new URL(mediaUrl(path),location.href).href;
  if ($('player').src !== target) { $('duration').textContent = 'Yükleniyor…'; $('player').src = target; }
  syncMediaControls();
}
async function retryArchive(segment) {
  const key = `${segment.session_id}:${segment.segment_index}`;
  if (archiveActions.has(key)) return;
  archiveActions.add(key); ++archiveRevision; notice('');
  document.querySelectorAll('[data-archive-retry]').forEach(button => { if (button.dataset.archiveRetry === key) button.disabled = true; });
  try { await api('/api/live/archive/index',{session_id:segment.session_id,segment_index:segment.segment_index}); await refresh(); }
  catch(error) { notice(error.message); }
  finally { archiveActions.delete(key); ++archiveRevision; await refreshArchive(); }
}
function showArchive(data) {
  const container = $('archive-segments'); container.replaceChildren();
  const allSegments = data.segments || [];
  // Arrival is session-relative, so it cannot order clips from different sessions.
  const segments = [...allSegments].sort((a,b) => String(b.session_id).localeCompare(String(a.session_id),undefined,{numeric:true}) || Number(b.segment_index) - Number(a.segment_index)).slice(0,8);
  $('archive-state').textContent = data.enabled ? `${allSegments.length} kayıt parçası` : 'Etkin değil';
  if (data.config && archiveAvailable) $('archive-help').textContent = archiveLimits(data.config);
  if (!segments.length) { container.append(element('p',data.enabled ? 'Kaydet ve arşivde ara seçeneğiyle başlayın. İlk parça kapandıktan sonra burada görünür.' : 'Sunucuda canlı arşiv etkin değil.','empty')); return; }
  segments.forEach(segment => {
    const card = element('article',undefined,'archive-card');
    const indexState = segment.index_state || 'pending';
    const seconds = Number(segment.playback_duration_seconds), fps = Number(segment.source_fps);
    const duration = segment.playback_duration_seconds != null && Number.isFinite(seconds) && seconds > 0 ? `${seconds.toFixed(1)} sn klip` : 'Klip hazır değil';
    const rate = segment.source_fps != null && Number.isFinite(fps) && fps > 0 ? `${fps} FPS oynatma` : 'Oynatma hızı bilinmiyor';
    const count = Number.isInteger(segment.frames) && segment.frames >= 0 ? `${segment.frames} analiz karesi` : 'Kare sayısı bilinmiyor';
    const part = Number.isInteger(segment.segment_index) ? `Parça ${segment.segment_index}` : 'Arşiv parçası';
    card.append(element('strong',`${part} · ${duration}`),element('small',`İndeks: ${stateText[indexState] || (indexState === 'pending' ? 'Bekliyor' : indexState)}`),element('small',`${count} · ${rate}`),element('small',`Oturum: ${segment.session_id || 'Bilinmiyor'}`,'archive-session'));
    if (segment.source_session !== undefined) card.append(element('small',`Bağlantı oturumu: ${segment.source_session}`));
    if (segment.error) card.append(element('small',segment.error,'archive-error'));
    const actions = element('div',undefined,'archive-actions');
    const open = element('button','Kaydı aç'); open.type = 'button'; open.disabled = !segment.media_path;
    if (segment.media_path) { addMedia(segment.media_path); open.addEventListener('click',() => playAt(segment.media_path,0)); }
    actions.append(open);
    if (['pending','failed','cancelled'].includes(indexState)) {
      const retry = element('button','İndekslemeyi dene','secondary'); retry.type = 'button';
      const key = `${segment.session_id}:${segment.segment_index}`; retry.dataset.archiveRetry = key;
      retry.disabled = archiveActions.has(key) || typeof segment.session_id !== 'string' || !Number.isInteger(segment.segment_index); retry.addEventListener('click',() => retryArchive(segment)); actions.append(retry);
    }
    card.append(actions); container.append(card);
  });
  if (mediaReady && !selectedMedia) {
    const first = segments.find(segment => segment.media_path);
    if (first) chooseMedia(first.media_path);
  }
}
async function refreshArchive() {
  if (archivePolling || document.hidden) return;
  archivePolling = true;
  const revision = archiveRevision;
  try { const data = await api('/api/live/archive'); if (revision === archiveRevision) showArchive(data); }
  catch(error) { if (revision === archiveRevision) $('archive-state').textContent = 'Arşive ulaşılamıyor'; }
  finally { archivePolling = false; }
}
function showResults(job) {
  displayedSearch = job.id;
  $('results').replaceChildren();
  const matches = job.result.results;
  $('result-title').textContent = `${job.request?.scope === 'live' ? 'Canlı arşiv · ' : ''}“${job.result.query}” · ${matches.length} sonuç`;
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
    if (metadata.origin === 'live_archive') body.append(element('span','Canlı arşiv','archive-badge'));
    body.append(element('strong',`${metadata.label || 'Nesne'} · ${seconds.toFixed(1)} sn`),
      element('small',match.media_path || 'Kaynak bu medya klasöründe değil'),
      element('small',`Kare ${metadata.frame_index ?? '—'} · Benzerlik ${Number(match.score).toFixed(3)}`));
    if (metadata.origin === 'live_archive') body.append(element('small',`Arşiv oturumu ${metadata.live_session_id || '—'} · Klip konumu, kamera zamanı değil`));
    card.append(body); $('results').append(card);
  });
}
function showJobs(jobs) {
  const container = $('jobs'); container.replaceChildren();
  if (!jobs.length) container.append(element('p','Henüz iş yok. Bir video indeksleyerek başlayın.','empty'));
  jobs.slice(0,12).forEach(job => {
    const row = element('div',undefined,'job');
    const title = element('div',undefined,'job-title');
    title.append(element('strong',job.request.type === 'segment_frame' ? 'Piksel maskesi analizi' : job.request.type === 'search' ? 'Metin araması' : job.request.type === 'index_live_archive' ? 'Canlı arşiv indeksleme' : 'Video indeksleme'),
      element('span',job.cancel_requested && !terminal(job.state) ? 'İptal bekleniyor' : stateText[job.state],'state'));
    row.append(title,element('p',job.request.path || job.request.query || (job.request.type === 'index_live_archive' ? `${job.request.session_id} · Parça ${job.request.segment_index}` : '')));
    if (job.recoveries > 0) row.append(element('p',`Yeniden başlatma sonrası kurtarıldı · Deneme ${job.attempts}`));
    if (['index_video','index_live_archive'].includes(job.request.type)) row.append(element('p',`${job.progress.decoded_frames || 0} kare okundu · ${job.progress.indexed_items || 0} nesne kaydı`));
    if (job.error) row.append(element('p',job.error));
    if (job.request.type==='segment_frame' && job.state==='succeeded') {
      const open=element('button','Maskeyi göster','cancel');open.type='button';
      open.addEventListener('click',()=> {
        chooseMedia(job.request.path,true);clearSegmentation();segmentationJob=job.id;showSegmentation(job);
        $('segmentation-image').scrollIntoView({behavior:'smooth',block:'center'});
      });row.append(open);
    }
    if (!terminal(job.state)) {
      const cancel = element('button','İptal','cancel'); cancel.type = 'button'; cancel.disabled = job.cancel_requested;
      cancel.addEventListener('click',async() => { try { await api(`/api/jobs/${job.id}/cancel`,{}); await refresh(); } catch(error) { notice(error.message); } });
      row.append(cancel);
    }
    container.append(row);
  });
  const maskJob=jobs.find(job=>job.id===segmentationJob);
  if (maskJob) showSegmentation(maskJob);
  const latest = jobs.find(job => job.request.type === 'search' && job.state === 'succeeded');
  if (latest && latest.id !== displayedSearch) showResults(latest);
}
async function refresh() {
  if (polling) return;
  polling = true;
  try {
    const data = await api('/api/jobs'); showJobs(data.jobs);
    const health=await api('/api/health');segmentationEnabled=health.segmentation_enabled===true;syncSegmentation();
    $('health').textContent = '● Servis hazır';
  } catch(error) { segmentationEnabled=false;syncSegmentation();$('health').textContent = 'Servise ulaşılamıyor'; notice(error.message); }
  finally { polling = false; }
}
$('index-form').addEventListener('submit',async event => {
  event.preventDefault(); notice('');
  if ($('media').value.startsWith('live-archive/')) { notice('Canlı kayıtları yeniden indekslemek için arşiv kartındaki “İndekslemeyi dene” düğmesini kullanın.'); return; }
  try { await api('/api/jobs',{type:'index_video',path:$('media').value,stride:Number($('stride').value),max_frames:Number($('max-frames').value)}); await refresh(); }
  catch(error) { notice(error.message); }
});
$('search-form').addEventListener('submit',async event => {
  event.preventDefault(); notice('');
  try { await api('/api/jobs',{type:'search',query:$('query').value,limit:8,scope:$('search-scope').value}); await refresh(); }
  catch(error) { notice(error.message); }
});
$('segmentation-form').addEventListener('submit',async event => {
  event.preventDefault();
  const frame=Number($('segmentation-frame').value), path=selectedMedia;
  if ($('segmentation-button').disabled || !Number.isInteger(frame) || frame<0 || frame>10000) return;
  clearSegmentation();const revision=segmentationRevision;
  segmentationSubmitting=true;syncSegmentation();$('segmentation-status').textContent='İş gönderiliyor…';
  try {
    const job=await api('/api/jobs',{type:'segment_frame',path,frame_index:frame});
    if (revision===segmentationRevision && selectedMedia===path) { segmentationJob=job.id;showSegmentation(job); }
    await refresh();
  } catch(error) { if (revision===segmentationRevision) $('segmentation-status').textContent=error.message; }
  finally { segmentationSubmitting=false;syncSegmentation(); }
});
$('media').addEventListener('change',() => chooseMedia($('media').value,true));
$('player').addEventListener('loadedmetadata',() => {
  if (selectedMedia && $('player').src === new URL(mediaUrl(selectedMedia),location.href).href)
    $('duration').textContent = `${$('player').duration.toFixed(1)} saniye`;
});
async function start() {
  try {
    await api('/api/health');
    const live = await api('/api/live/sources');
    live.sources.forEach(source => { const option = element('option',source.label); option.value = source.id; $('live-source').append(option); });
    liveEnabled = live.sources.length > 0;
    $('live-heading').textContent=live.analysis_mode === 'segmentation' ? 'Canlı RTSP maskeleri' : 'Canlı RTSP analizi';
    archiveAvailable = !!live.archive_available;
    $('live-archive').checked = archiveAvailable;
    $('archive-help').textContent = archiveAvailable ? archiveLimits(live.archive_config) : 'Canlı arşiv bu sunucuda etkin değil. Önizleme tek başına kullanılabilir.';
    if (!liveEnabled) { hideLive('Canlı kaynak bu sunucuda etkin değil.'); $('live-help').textContent = 'Sunucuyu -LiveUrl seçeneğiyle başlatın; tarayıcıdan keyfî URL kabul edilmez.'; }
    else $('live-help').textContent = `${live.duration_seconds} saniyelik oturum · yaklaşık 2 önizleme/sn · kayıt ve indeksleme yalnızca seçeneği açarsanız yapılır.`;
    await refreshLive(); setInterval(refreshLive,500);
    setInterval(() => { if (liveObjectUrl && performance.now() >= liveExpiresAt) hideLive('Güncel analiz karesi bekleniyor…'); },250);
    const data = await api('/api/media');
    data.videos.forEach(video => addMedia(video.path));
    mediaReady = true;
    if (!mediaChoiceMade) {
      const initial = data.videos[0]?.path || $('media').options[0]?.value;
      if (initial) chooseMedia(initial);
    }
    syncMediaControls();
    await refreshArchive(); setInterval(refreshArchive,2000);
    if (selectedMedia && !data.videos.length) notice('Yerel dosya yok; canlı arşiv kayıtlarını oynatabilirsiniz.');
    else if (!$('media').options.length) { $('index-button').disabled = true; notice('Medya klasöründe video yok. Bir MP4 ekleyin veya canlı arşiv oluşturun.'); }
    await refresh(); setInterval(refresh,1000);
  } catch(error) { notice(error.message); $('health').textContent = 'Servise ulaşılamıyor'; }
}
syncMediaControls();
start();
