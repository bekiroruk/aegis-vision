# Yerel C++ video arama servisi

YOLO → nesne kırpması → CLIP → Qdrant zinciri artık uzun ömürlü bir C++ HTTP
servisi üzerinden çalışır. Tarayıcıda video seçilir, indeksleme başlatılır,
ilerleme/iptal izlenir ve metinle bulunan sonuca tıklanarak kaydın ilgili anı açılır.
Python veya Node.js çalışma zamanında gerekmez. Bu **yerel geliştirme servisi**dir;
internete açık, çok kullanıcılı production deployment değildir.

## Başlatma

Önce [arama kurulumu](search.md) ve [YOLO modeli](yolo.md) hazırlanmış olmalı.
Arama derlemesi `aegisvision_server` hedefini de üretir. Proje kökünde iki terminal:

```powershell
# Terminal 1 — zaten çalışıyorsa yeniden başlatmayın
./artifacts/deps/qdrant/qdrant.exe --config-path configs/qdrant-local.yaml
```

```powershell
# Terminal 2 — yalnızca artifacts/media içindeki videoları erişime açar
./scripts/start_service.ps1
```

Tarayıcı: <http://127.0.0.1:8090>. Başlatıcı eksik model/derleme veya çalışmayan
Qdrant için açıklayıcı hata verir; paket kurmaz, veri silmez, arka plan süreci açmaz.
Terminal açık kalmalıdır; kapatmak için Ctrl+C. Özel klasör/port:

```powershell
./scripts/start_service.ps1 -MediaDirectory C:/videos -Port 8091
```

İşler varsayılan olarak `artifacts/service/jobs.sqlite` içinde tutulur. Model,
Qdrant koleksiyonu veya medya kökü değiştirildiğinde farklı veritabanı seçin:
`./scripts/start_service.ps1 -JobDatabase artifacts/service/other-jobs.sqlite`.
Aynı iş veritabanını iki servis aynı anda açamaz; ikinci süreç hata verir.
Arşiv etkinse medya kökündeki `live-archive` ayrıca native tek-süreç sahiplik
kilidiyle korunur. Farklı DB seçilse bile aynı arşive ikinci yazıcı reddedilir;
OS kilidi süreç kapanınca serbest bırakır. Kilit dosyasını açık serviste silmeyin.
SQLite 3.53.4 kaynakları CMake ile sabit SHA3-256 kontrolünden geçirilerek derlenir;
ayrı SQLite kurulumu gerekmez.

Doğrudan Windows komutu (Linux'ta exe uzantısını ve Release dizinini kaldırın):

```powershell
./build/search/Release/aegisvision_server.exe configs/service-search.toml configs/image.toml artifacts/media web 8090
```

CLIP ve YOLO başlangıçta bir kez yüklenir. `service-search.toml` ayrı
`aegis_service` koleksiyonunu kullanır; yoksa oluşturur, mevcut verileri sıfırlamaz.
İngilizce sorgularla başlayın: `a person walking on the street`.

İsteğe bağlı `-SegmentationModel MODEL.onnx` seçeneği aynı kuyrukta tek-kare
`segment_frame` işini açar. Yeni bir `-JobDatabase` seçin; kullanım, piksel maskesi
şeması ve sınırlar [segmentation HTTP kılavuzunda](segmentation.md#kuyruk-tabanlı-http-tek-kare-maskesi).
Bu seçenek dashboard'a maske çizimi veya canlı segmentation eklemez.

## Süre sınırlı canlı önizleme ve aranabilir arşiv

`./scripts/start_service.ps1 -LiveUrl rtsp://127.0.0.1:8554/pedestrians` canlı paneli
etkinleştirir. Varsayılan profil `configs/live-preview.toml` ile 180 saniyedir;
`-LiveConfig` ile değiştirilebilir. Önce RTSP yayınını hazırlayın: üç terminalle
relay/publisher/servis kurulumu ve kesinti testi [canlı video kılavuzunda](live-video.md).
Canlı URL tarayıcıdan alınmaz, yalnızca sunucuda tanımlı kaynak ID'si seçilir.

Dosya/arama worker'ından ayrı tek canlı worker ve ayrı YOLO instance'ı kullanılır;
canlı capture SQLite iş kuyruğuna girmez. **Kaydet ve arşivde ara** seçeneği
yalnızca analiz karelerini sınırlı parçalara kaydeder; kapanan parçalar aynı dosya
worker'ında sonlu encoder/CLIP/Qdrant işlerine dönüşür. İlave CLIP modeli yüklenmez.
HTTP istemcileri aynı son JPEG'i paylaşır; uzun ömürlü MJPEG bağlantısı veya istemci
başına decoder yoktur. Ekran yaklaşık 2 FPS önizlemedir, tam video/ses değildir.
Disconnect/stop ve 2 saniyeden eski decode-arrival görüntüsünde önizleme temizlenir.
Bu yaş kamera-ağ gecikmesini ölçmez. En fazla bir aktif oturum, süre sınırı ve
tek son terminal özeti tutulur; yeni oturum eski ID'yi geçersiz kılar. Servis
yeniden açılınca canlı oturum/geçmiş kurtarılmaz ve otomatik kamera bağlantısı yapılmaz.

Arşiv parçası en fazla 10 saniyelik arrival aralığı/100 analiz karesidir; oturumda
4, toplamda 8 parça ve 256 MiB sınırı vardır. Kayıt/encoder sınırları ve gerçek
zaman ile klip zamanı ayrımı [canlı arşiv kılavuzundadır](live-video.md#canlı-yayını-kaydet-ve-arşivde-ara).
Kota dolduğunda önizleme devam eder; mevcut kayıtlar otomatik silinmez.
FFmpeg PATH'te olmalıdır; başlatıcıda `-Ffmpeg` ile özel executable verilebilir.
Sunucu başlangıcında sabit encoder seçilir; HTTP'den komut/URL kabul edilmez.

Arşiv kartları en yeni sekiz parçayı gösterir; doğrulanmış MP4'ü **Kaydı aç** ile
oynatın. Kuyruk doluysa pending parça için **İndekslemeyi dene** kullanın. Aramada
**Yalnızca canlı arşiv** seçeneği dosya indekslerini dışarıda bırakır. Kamera
kaydının tam FPS arşivi değildir; yalnızca analiz kareleri CFR klibe dönüşür.
Servis yeniden açıldığında kapalı parçalar manifestlerden listelenir ve kabul
edilmiş SQLite işleri kurtarılır. İş geçmişinden budanan başarılı parçaların
tamamlanma kaydı korunur. Kabul edilmemiş pending parçalar kendiliğinden kuyruğa
verilmez; kullanıcı retry isteği gerekir. Canlı capture hiçbir durumda auto-resume etmez.

## Gerçek kamera kaydı

Bu bilgisayarda OpenCV ile gelen
[`vtest.avi`](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi)
yerel olarak tarayıcı uyumlu H.264 MP4'e dönüştürüldü. Kayıt gerçek yaya hareketi
içerir; fotoğraftan oluşturulmuş animasyon değildir. Kaynak: OpenCV örnek verisi;
yeniden dağıtım için veri kaynağının lisans koşullarını ayrıca inceleyin. Videoyu
ve model dosyalarını repoya eklemiyoruz. Yerel dosya: `artifacts/media/pedestrians.mp4`.

Aynı dosya bilgisayarınızda varsa yeniden üretmek için (FFmpeg gerekir):

```powershell
New-Item -ItemType Directory -Force artifacts/media
ffmpeg -n -i C:/opencv/opencv/sources/samples/data/vtest.avi -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p -movflags +faststart -an artifacts/media/pedestrians.mp4
```

Kayıt 768×576, 10 FPS, 795 kare ve 79,5 saniyedir. Varsayılan stride 15 ile
53 karede tespit yapılır. Nesne sayısı **benzersiz kişi sayısı değildir**;
aynı kişi farklı karelerde ayrı kırpma kaydı oluşturabilir. Bu servis Re-ID yapmaz.
Arama skoru cosine benzerliğidir; başarı olasılığı veya etiketli doğruluk metriği değildir.

Yerel entegrasyon doğrulamasında bu kayıt 53 örnek kareden **410 nesne kaydı**
üretti. Metin sorgusu 8 sonuç döndürdü; tarayıcıda 25,5 saniyelik sonuca tıklama
videoyu o andan oynattı. Yeniden indekslemeyi iptal etme denemesinde 13 tamamlanan
yazım raporlandı ve koleksiyon 410 kayıtta kaldı. Bunlar bu dosya/model/ayarların
çalışma kontrolüdür, detection veya arama doğruluğu değerlendirmesi değildir.

## API

Tüm POST istekleri `Content-Type: application/json` gerektirir.

| İstek | Sonuç |
| --- | --- |
| `GET /api/health` | Süreç hazır, kuyruk/geçmiş sınırları; anlık Qdrant sağlık testi değildir |
| `GET /api/media` | Medya kökündeki video dosyaları |
| `POST /api/jobs` | `202` ve iş ID'si; dolu kuyrukta `429` |
| `GET /api/jobs` | En yeni önce olmak üzere tutulan işler |
| `GET /api/jobs/{id}` | Durum, ilerleme, sonuç veya hata |
| `POST /api/jobs/{id}/cancel` | Bekleyen işi kaldırır veya çalışan işten iptal ister |
| `GET /api/preview/{search_job_id}/{result_index}.jpg` | Kutusu çizilmiş gerçek sonuç karesi |
| `GET /media/{relative_path}` | Byte-range destekli video; tarayıcı codec desteği gerekir |
| `GET /api/live/sources` | Sunucudaki kaynak ID/etiketleri ve arşiv kullanılabilirliği/kotalar; URL döndürmez |
| `GET /api/live` | `session: null` veya anlık/son oturum özeti |
| `POST /api/live/start` | `{"source_id":"local-pedestrians","archive":true}`; `202`, başka aktif oturumda `409`; archive varsayılan false |
| `POST /api/live/{id}/stop` | `{}`; durdurma isteği `202`, bilinmeyen eski ID `404` |
| `GET /api/live/{id}/preview.jpg` | Taze JPEG `200`, görüntü yok/eski `204`, bilinmeyen ID `404` |
| `GET /api/live/archive` | Kapalı parçalar, oynatılabilir medya yolu, indeksleme durumu/iş ID'si ve kotalar |
| `POST /api/live/archive/index` | `{"session_id":"live-…","segment_index":1}`; `202`, zaten sırada/çalışıyorsa `409`, kuyruk doluysa `429` |

İndeksleme ve arama gövdeleri:

```json
{"type":"index_video","path":"pedestrians.mp4","stride":15,"max_frames":0}
```

```json
{"type":"search","query":"a person walking on the street","limit":8,"scope":"live"}
```

Durumlar: `queued`, `running`, `succeeded`, `failed`, `cancelled`.
`stride`: 1–10000; `max_frames`: 0–1000000 (0 = tüm kayıt); `limit`: 1–20.
`scope`: `all` (varsayılan) veya `live`; canlı kapsamı Qdrant `origin: live_archive`
filtresini kullanır. Canlı arşiv sonucunda `live_session_id`, `live_source_id`,
`source_session`, `tracking_epoch` metadata'sı ve MP4 medya yolu bulunur.
Yollar medya köküne göre verilmelidir. Mutlak yollar, kökten kaçış ve dışarıya
işaret eden symlink'ler reddedilir. UI yalnızca kökün doğrudan altındaki videoları listeler.
Canlı MP4 yolları arşiv kartları/sonuçlardan oynatıcıya eklenir. `live-archive`
alt ağacına sıradan `index_video` işi kabul edilmez; ham/staging dosyaları HTTP'de sunulmaz.

Canlı oturum durumları ayrı sözleşmedir: `starting`, `running`, `stopping`,
`stopped`, `completed`, `failed`; bağlantı `connecting`, `live`, `reconnecting`,
`stopped` olabilir. `decoded_frames`, `processed_frames`, `dropped_frames`,
`sessions`, `tracking_epochs`, `queue_high_watermark`, `mean_analysis_ms` ve
`has_preview` izlenir. JPEG başlıkları sequence/session/epoch ve decode yaşı verir.
Stop işbirlikçidir; in-flight model yüklemesi/inference ve backend deadline beklenir.
Detaylı model/decoder hata metni ve kaynak URL'si API yanıtlarına eklenmez.

Canlı özette `archive: {enabled,state,error,closed_segments,index_queue_failures}`
kayıt durumunu bildirir. `/api/live/archive` parçaları `session_id`, `source_id`,
1 tabanlı `segment_index`, `frames`, `source_fps`, `arrival_start_ms`,
`arrival_end_ms`, `media_path` (hazır değilse null), `playback_duration_seconds`,
`index_state`, `job_id` ve `error` taşır. İndeks durumları normal iş durumlarına
ek olarak `pending` olabilir. Arrival monotonic uygulama zamanıdır; MP4 içindeki
`timestamp_ms` kodlanmış CFR kare/FPS konumudur, kamera PTS değildir.

## Kuyruk, tutarlılık ve sınırlar

- Bir inference worker; en fazla 8 bekleyen iş ve 128 işlik kalıcı geçmiş.
  Eski tamamlanmış işler kapasitede diskte/bellekte birlikte atılır. Arama da
  indekslemenin arkasında bekler. Kurtarma sırasında 8 bekleyen işe eski aktif iş
  eklenebilir; bu geçici 9 işlik backlog boşalana kadar yeni iş kabul edilmez.
- HTTP 202 yalnızca SQLite commit sonrası döner. Durum, sonuç ve iptal isteği
  kalıcıdır; ilerleme yaklaşık saniyede bir diske kaydedilir. Ani kapanmada son
  checkpoint bir saniye kadar geriden gelebilir; tamamlanan vektör yazımları korunur.
  Bu kalıcılık sözü sonlu `/api/jobs` ve arşiv indeksleme işleri içindir; canlı
  start/stop `202` yalnızca geçici oturum isteğinin kabul edildiğini bildirir.
- Yeniden başlatmada bekleyen işler aynı ID ile çalışır; `running` işler baştan
  tekrar indekslenir. Kare konumundan devam edilmez. Sabit Qdrant ID'leri aynı
  kaydı günceller. Dağıtık exactly-once garantisi yoktur; teslimat at-least-once'dur.
- `attempts` çalıştırma sayısını, `recoveries` ani kesintiden kurtarma sayısını
  bildirir. Bir iş en fazla üç ani kesinti sonrası tekrar denenir; sonraki
  kesintide `failed` olur. Kullanıcı iptali yeniden başlatmada da korunur.
  Normal model/Qdrant hataları otomatik tekrar edilmez; yeni iş gönderilir.
- DB bağlamı medya kökü, detector imzası, CLIP alanı ve Qdrant koleksiyonunu içerir.
  Başka model/koleksiyonla aynı DB açılması reddedilir. Kabul edilen videonun
  boyutu veya değişiklik zamanı değişirse iş hata verir; yeni iş göndermek gerekir.
- SQLite WAL/FULL ve tek süreç sahipliği kullanılır. Varsayılan ana DB limiti
  65536 sayfadır (yeni DB'de 256 MiB). Disk/kota yazım hatasında kuyruk durur,
  sağlık endpoint'i ve yeni iş/iptal istekleri 503 verir. Yazılamayan sonuç başarılı
  gösterilmez; alan sorunu giderilip yeniden başlatıldığında kabul edilmiş iş kurtarılır.
- DB yerel diskte tutulmalıdır. Çalışırken SQLite dosyalarını silmeyin veya
  kopyalamayın; servis durduktan sonra yedekleyin. Redis, çoklu worker ve dağıtık
  lease/sahiplik desteği henüz yoktur.
- İptal işbirlikçidir: sürmekte olan model veya Qdrant çağrısı bitince kontrol edilir.
  Tamamlanan yazımlar korunur; rollback yoktur. Aynı dosya/model/ayarlarla tekrar
  çalıştırma sabit ID'leri günceller, kayıtları çoğaltmaz. Ayar/dosya değişikliği yeni
  ID oluşturabilir; eski kayıtlar otomatik temizlenmez.
- HTTP istekleri 8 KiB ile sınırlıdır; dört HTTP worker ve 16 bekleyen bağlantı.
  Video akışı 64 KiB parçalarla okunur; tüm dosya belleğe alınmaz.
- Yalnızca `127.0.0.1` dinlenir. Host/Origin kontrolleri, dosya yolu sınırlaması ve
  içerik güvenlik politikası vardır; kimlik doğrulama/TLS yoktur. Dış ağa açmayın.
- Kaynak dosyaları işlem boyunca değiştirmeyin. Video yükleme endpoint'i yoktur;
  kendi MP4 dosyanızı medya klasörüne koyup sayfayı yenileyin.
- Canlı arşiv disk/kare/parça limitleriyle sınırlandırılmıştır; sonsuz RTSP işi
  SQLite worker'ını işgal etmez. Sınırlı encoder alt süreci sonlu işler içinde
  çalışır; CPU yarışması ve kuyrukta aramanın beklemesi mümkündür. Arşiv hatası
  canlı önizlemeyi başarılı arşiv gibi göstermez ve capture'ı durdurmaz.
  Slot başına 32 MiB raw/encoded/manifest rezervasyonu uygulama kabul kontrolüdür;
  OS filesystem kotası değildir. Ham ve yayımlanan MP4 için 12 MiB kontrol edilir.
  FFmpeg `-fs` advisory olduğundan geçici dosya kısa süre aşabilir; son boyut/decode
  kontrolünü geçmeyen klip HTTP'de yayımlanmaz. Dış süreç yazımları kapsam dışıdır.
- Kare zamanı `frame_index / FPS` üzerinden hesaplanır; VFR/RTSP gerçek PTS desteği
  henüz yoktur. Önizleme arşivden yeniden çözülür, indeksleme anındaki dosya korunmalıdır.

## Test

Arayüzün medya seçimi regresyonu `node tests/test_web_state.cjs` ile kontrol
edilir: dosya/arşiv yanıt sırası, geç yanıtın kullanıcı seçimini bozmaması,
klipteki ana atlama ve yönetilen arşivde generic indekslemenin kapatılması.
Node yalnızca bu geliştirme testi içindir; C++ servisini çalıştırmak için gerekmez.

`ctest --test-dir build/search -C Release --output-on-failure` kuyruğun taşmasını,
bekleyen/çalışan iş iptalini, kısmi ilerlemeyi, hata sonrası devam etmeyi, geçmiş
budamayı, HTTP doğrulamasını, yol sınırlamasını, tekrar indekslemeyi, JPEG önizlemeyi
ve video byte-range isteklerini test eder. Kalıcılık testleri ayrıca tek sahipliği,
ID/geçmiş korunmasını, üç tekrar sınırını, yanlış bağlamı, kota hatasında durmayı
ve aynı işin kurtarılmasını doğrular. Ayrı süreç testi `std::_Exit` ile destructor
çalıştırmadan kapanır; sonraki test WAL üzerinden aktif/bekleyen işleri tamamlar.
HTTP testi servis yeniden açıldıktan sonra eski arama/önizlemeyi kontrol eder.
Servis testi sahte modellerle deterministiktir;
gerçek YOLO/CLIP/Qdrant kamera denemesi ayrı entegrasyon doğrulamasıdır.
`aegisvision_live_service` testi ayrı canlı modeli, tek aktif oturumu, stale/kopmuş
JPEG'i, yeni source session/epoch'u, stop ve cleanup'ı doğrular. Gerçek RTSP/web
kesinti testi için `scripts/test_live_dashboard.ps1` kullanılır.

Arşiv doğrulaması ayrıca parça/session sınırlarını, kota altında önizlemenin
devam etmesini, dolu kuyrukta pending parçayı, retry tekilleştirmesini, encoder
hata/iptal/deadline davranışını, raw dosya erişim reddini ve canlı arama filtresini
kapsar. Gerçek RTSP → MP4 → CLIP/Qdrant → tarayıcı denemesi bir entegrasyon
kontrolüdür; kamera PTS, detection doğruluğu veya performans benchmark'ı değildir.

Yerel relay/Qdrant/canlı servis açık, başka publisher/aktif oturum yok ve arşiv
kotasında en az iki boş slot varken gerçek uçtan uca kontrol:

```powershell
./scripts/test_live_archive.ps1 -Output outputs/live-archive-test
```

FFmpeg PATH'te, `artifacts/media/pedestrians.mp4` mevcut olmalı; yeni çıktı dizini
kullanın. Script kendi publisher/oturumunu yönetir; oluşturulan arşiv parçaları
medya kökünde kalır ve tekrarlar kotayı tüketir. Önkoşullar ve artefaktların
ayrıntıları [canlı arşiv testinde](live-video.md#gerçek-rtsp--aranabilir-arşiv-testi).

2026-10-02 gerçek YOLO/CLIP/Qdrant smoke doğrulaması **118 okunan, 63 analiz edilen
kareden 62 ve 1 karelik iki MP4** üretti (6,2 ve 0,1 sn). İki indeks işi tamamlandı;
canlı kapsamlı metin araması 8 sonuç döndürdü, MP4 range `206` doğrulandı. İlk
sonuç kodlanmış klibin 30. karesi / 3000 ms konumundaydı. Kanıt:
`outputs/live-archive-verified/report.json`, `search.json` ve JPEG önizlemeler.
Bu gerçek kayıt üzerinden yerel sistem kontrolüdür; fiziksel kamera, etiketli
doğruluk ölçümü veya throughput benchmark'ı değildir.

Teknik referanslar: [SQLite WAL](https://sqlite.org/wal.html),
[SQLite locking mode](https://sqlite.org/pragma.html#pragma_locking_mode),
[kaynak dağıtımı](https://sqlite.org/download.html).
