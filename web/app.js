const $ = id => document.getElementById(id);
const terminal = state => ['succeeded', 'failed', 'cancelled'].includes(state);
const stateText = {queued:'Sırada',running:'Çalışıyor',succeeded:'Tamamlandı',failed:'Hata',cancelled:'İptal edildi'};
let displayedSearch = '', polling = false;
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
    const data = await api('/api/media');
    data.videos.forEach(video => { const option = element('option',video.path); option.value = video.path; $('media').append(option); });
    if (data.videos.length) $('player').src = mediaUrl(data.videos[0].path);
    else { $('index-button').disabled = true; notice('Medya klasöründe video yok. Bir MP4 ekleyip sayfayı yenileyin.'); }
    await refresh(); setInterval(refresh,1000);
  } catch(error) { notice(error.message); $('health').textContent = 'Servise ulaşılamıyor'; }
}
start();
