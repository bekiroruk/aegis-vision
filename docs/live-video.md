# Canlı RTSP analiz, önizleme ve kayıt

C++ `aegisvision_stream`, OpenCV FFmpeg backend'iyle RTSP okur. Ayrı capture thread'i
sürekli decode eder; YOLO ve takip başka thread'de analiz edilir. Yetişmeyen analiz
eski kareleri biriktirmez: drop-oldest kuyruğu varsayılan **1 kare**, en fazla 16 kare
ve 64 MiB piksel verisidir. Tek kabul edilen kare en fazla 32 MiB, CV_8UC3'tür.
Bu sınırlar decoder/FFmpeg tamponlarının veya modelin toplam belleğini sınırlamaz;
kaynak paket tamponu/gecikmesi ayrıca ölçülmelidir. Gerçek zaman garantisi yoktur.

## Tarayıcıda canlı analiz

Web ekranı ve kayıt CLI'ı aynı `analyze_stream` motorunu kullanır. Ekran, kutulu
JPEG'i yaklaşık saniyede iki kez yeniler; ses veya tam FPS video oynatıcı değildir.
Canlı worker kendi YOLO modelini yükler; dosya/CLIP kuyruğunun modelini paylaşmaz.
Bu nedenle canlı analiz sırasında metin araması çalışabilir, ancak CPU/RAM paylaşılır
ve toplam RAM kullanımı ek model nedeniyle artar. Canlı capture oturumu geçicidir;
servis yeniden açılınca kendiliğinden başlamaz. İsteğe bağlı arşiv seçeneği yalnızca
analiz edilen kareleri kısa kliplere kaydeder ve sonlu indeksleme işlerini mevcut
SQLite kuyruğuna verir. Böylece CLIP modeli çoğaltılmadan canlı kayıtlar Qdrant'ta aranabilir.

Önce [servis kurulumu](service.md) ile YOLO, CLIP ve Qdrant hazırlanmış olmalı.
Qdrant zaten çalışırken proje kökünde aşağıdaki üç terminali açık tutun:

```powershell
# Bir kez: sabit sürüm ve SHA256 kontrolüyle yerel relay hazırlama
./scripts/prepare_rtsp_relay.ps1
# Terminal 1: yalnızca loopback TCP relay
./artifacts/deps/mediamtx/mediamtx.exe configs/rtsp-local.yml
```

```powershell
# Terminal 2: gerçek kamera kaydını yerel RTSP olarak tekrar yayınlama
ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i artifacts/media/pedestrians.mp4 -an -c:v libx264 -preset ultrafast -tune zerolatency -g 10 -keyint_min 10 -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/pedestrians
```

```powershell
# Terminal 3: web servisinde önceden tanımlı canlı kaynağı etkinleştirme
./scripts/start_service.ps1 -LiveUrl rtsp://127.0.0.1:8554/pedestrians
```

<http://127.0.0.1:8090> adresinde **Canlı analizi başlat** düğmesini kullanın.
Varsayılan `configs/live-preview.toml` profili 180 saniyede durur; `-LiveConfig`
ile başka stream profili seçilebilir. Kaynak URL'si yalnızca servis başlangıcında
verilir; tarayıcı kaynak ID'si gönderir. Keyfî URL, kimlik bilgisi ve token kabul
edilmez. `-LiveUrl` verilmezse canlı bölüm devre dışıdır; dosya araması çalışır.
Başlatıcı `-Ffmpeg` ile encoder yolunu kabul eder; varsayılan `ffmpeg` PATH'te olmalıdır.

Bu demo fiziksel kameraya bağlandığımız anlamına gelmez: gerçek OpenCV yaya kaydı
FFmpeg → MediaMTX → RTSP → C++ YOLO/takip → tarayıcı zincirinden geçer.
Kaynak, MP4 hazırlama ve lisans notları [servis kılavuzundadır](service.md).

Her servis en fazla bir canlı oturum tutar; ikinci başlatma `409` verir. Stop isteği
hemen `202` döner ve önizlemeyi temizler. Worker sürmekte olan model yüklemesi,
inference veya open/read çağrısını zorla kesmez; bitiş deadline'ı beklenebilir.
Kopmada eski görüntü gösterilmez. Decode-arrival yaşı 2 saniyeyi aşan JPEG sunulmaz;
tarayıcı da kendi süre kontrolüyle donmuş görüntüyü kaldırır. Bu yaş kamera PTS'si
veya kamera-ağ gecikmesi değildir. Önizleme en fazla 960×720 ve 2 MiB'dir; istemci
başına yeni decoder/model kurulmaz, tek değişmez son JPEG paylaşılır.

## Canlı piksel maskeleri

Sunucuya `--live-segmentation` veya PowerShell başlatıcıya `-LiveSegmentation`
verilirse canlı worker YOLOv8-seg kullanır. Yukarıdaki relay ve yayıncı açıkken:

```powershell
./scripts/start_service.ps1 -LiveUrl rtsp://127.0.0.1:8554/pedestrians -SegmentationModel artifacts/models/yolov8n-seg/yolov8n-seg.onnx -LiveSegmentation -JobDatabase artifacts/service/segmentation-jobs.sqlite
```

Yeni derlemeyi başka klasörde hazırladıysanız `-ServerExecutable` ekleyin; örneğin
`-ServerExecutable build/live-seg/Release/aegisvision_server.exe`. Aynı porttaki
eski web servisini önce kendi terminalinde Ctrl+C ile durdurun; Qdrant açık kalır.
Web paneli **Canlı RTSP maskeleri** başlığını ve **Piksel maskesi + takip** modunu
gösterir. Başlatma/durdurma, 180 saniyelik süre ve iki saniyelik tazelik sınırı
önceki canlı analizle aynıdır. Kaynak/oturum API'leri `analysis_mode` alanını verir.

Canlı maskeler için ayrı model örneği yüklenir; dosya işlerinin segmenter'ı
paylaşılmaz. Her analiz karesi tek segmentation inference'ından geçer. Sınıf
duyarlı IoU .30/max_missed 20 ile maskeler doğrudan giriş sırasındaki takip
kimliğine bağlanır. Bu modda TOML'deki detection takipçisi yerine bu maske
takipçisi kullanılır. Reconnect, uzun decode-arrival aralığı veya çözünürlük
değişiminde pipeline sıfırlanır; görüntüde **Epoch** ve kişi/nesne yanında **#ID**
yazar. Kimlik yalnız aynı epoch içinde anlamlıdır. En fazla 1920 piksel kenar,
100 maske/kare; GPU/gerçek zaman garantisi yoktur. Boş tespitte orijinal kare
gösterilir, kaçırılan nesne için maske tahmini yapılmaz.

Arşiv açılırsa analiz edilen **orijinal kareler** kaydedilir; piksel maskeleri
arşive yazılmaz. Arşiv indeksleme mevcut dosya detection/CLIP zincirini kullanır.
Canlı kayıt/arama sözleşmesi değişmez.

Gerçek RTSP kopma/yeniden bağlanma kontrolü için (relay açık, yayıncıyı betik yönetir):

```powershell
./scripts/test_live_dashboard.ps1 -ExpectSegmentation -Output outputs/live-mask-rtsp-v1
```

Bu betik yeni çıktı dizini ister; başlangıç/yeniden bağlantı JPEG'lerini ve oturum
raporunu kaydeder. Deterministik HTTP testleri iki analiz modunda model izolasyonunu,
maskenin iç piksellerinin boyandığını, reconnect/çözünürlük değişiminde epoch
yenilendiğini, eski görüntünün temizlenmesini, arama eşzamanlılığını ve hatadan
sonra yeniden başlatmayı doğrular. Gerçek RTSP+maske ölçümü ayrıca raporlanmalıdır.

2026-10-10 yerel doğrulama: ayrı `build/live-seg` Release derlemesi başarılı;
`ctest --test-dir build/live-seg -C Release --output-on-failure` **35/35** geçti
(44,76 sn; gerçek CLIP referans testi dahil). İki modu kapsayan canlı HTTP testi
6,92 sn'de geçti. `node tests/test_web_state.cjs` başarılı. Bu sonuçlar gerçek
YOLOv8-seg modelinin RTSP üzerinde uçtan uca denendiği anlamına gelmez; canlı HTTP
testi kontrollü capture/model kullanır. Açık eski sunucu bu derlemeyle otomatik
değiştirilmez; yeni executable ile yeniden başlatılmalıdır.

## Canlı yayını kaydet ve arşivde ara

Bu özellik 2026-10-02'de eklendi; aşağıdaki eski RTSP smoke sonuçları kendi
2026-10-01 tarihleriyle korunmuştur.

Canlı paneldeki **Kaydet ve arşivde ara** seçeneğini kontrol edip oturumu başlatın.
Sunucuda arşiv kullanılabiliyorsa seçenek başlangıçta açıktır; işaretini kaldırınca
yalnızca geçici önizleme çalışır. API'de `archive` gönderilmezse kayıt yapılmaz.
Kaydın başladığı ve kota/hata durumları sayaçların yanında gösterilir.

- Parçalar 10 saniyelik decode-arrival aralığında, 100 analiz karesi sınırında,
  bağlantı/format sınırında veya stop sırasında kapanır. Tek parça farklı bağlantı
  oturumlarını birleştirmez; takip epoch/kimlikleri sidecar metadata'da korunur.
- Oturum başına en fazla 4, medya kökünde toplam en fazla 8 parça ve 256 MiB arşiv
  sınırı vardır. Kota dolunca yalnızca kayıt durur; canlı analiz ve önizleme sürer.
  Eski kayıtlar otomatik silinmez. Kapalı serviste kendi arşivinizi yedekleyip
  bilinçli olarak yönetin; açık servis sırasında dosyaları değiştirmeyin.
  Raw/encoded/manifest bütçesi için slot başına 32 MiB rezervasyon hesaplanır.
  Bu uygulama kabul kotasıdır, OS disk kotası değildir; dış süreç yazımları ve
  kısa süreli encoder taşması karşısında mutlak disk limiti garantisi vermez.
- Analiz kareleri ham MJPEG olarak kaydedilir; sonlu iş FFmpeg ile tarayıcı uyumlu
  MP4 üretip decode kare sayısını doğrular. Her ham/final MP4 en fazla 12 MiB'dir;
  encoder `-fs` ile sınırlandırılsa da bu advisory'dir ve geçici dosya kısa süre
  sınırı aşabilir. Final boyut kontrolünü geçmeyen dosya yayımlanmaz.
  encoder alt süreci 30 saniye deadline'ına ve iş iptaline tabidir. Ham/staging
  dosyaları HTTP'de açılmaz; doğrulanmış MP4 hazır olmadan **Kaydı aç** etkinleşmez.
- Otomatik indeksleme, dosya/arama worker'ının sınırlı kuyruğunu ve mevcut CLIP
  modelini kullanır. Varsayılan örnekleme her 10 analiz karesinde birdir. Kuyruk
  doluysa parça `pending` kalır; sonsuz backlog veya otomatik retry döngüsü yoktur.
  **İndekslemeyi dene** ile pending/failed/cancelled parçalara sonlu yeni iş verilir.
- **Canlı arşiv** kartlarında klip oynatma uzunluğu ve indeksleme durumu görünür.
  Arama kapsamını **Yalnızca canlı arşiv** seçin; sonuçtaki **Canlı arşiv** rozeti
  kaynağı ayırır. Sonuca tıklama MP4'ün ilgili kodlanmış kare konumunu açar.

Arşiv, kamera yayınını tam FPS kaydetmez: yalnızca analiz edilen kareler 10 FPS
CFR klibe dönüşür. Örneğin 10 saniyelik canlı aralıkta 65 analiz karesi varsa
klip 6,5 saniyedir; `timestamp_ms` bu **klipteki konumdur**, gerçek kamera zamanı
veya stream PTS değildir. `manifest.json` kare/kaynak sequence, source session,
tracking epoch ve monotonic arrival eşlemesini korur. Bu metadata zaman eşleme
altyapısıdır; kamera PTS, ses ve kayıpsız tam yayın arşivi henüz yoktur.

Kayıtlar medya kökünde `live-archive/<live-session>/segment-0001/` altında tutulur.
`manifest.json`, ham kayıt ve doğrulanmış `clip.mp4` yereldir; modeller ve kayıtlar
Git'e eklenmez. Yeniden başlatma canlı capture'ı başlatmaz; SQLite'a kabul edilmiş
sonlu işler normal kuyruk kurtarma kurallarına tabidir. Kapalı manifestler yeniden
listelenir; kabul edilmemiş pending parçalar otomatik kuyruğa girmez ve manuel
retry ister. Başarılı indekslemenin tamamlanma kaydı, SQLite iş geçmişi budansa
da korunur. Qdrant metadata'sındaki
`origin: live_archive`, live session/source ve bağlantı/epoch alanları dosya
indeksleriyle karışmadan arama filtresini sağlar. Generic `index_video` ile bu
alt ağacı yeniden indekslemek yerine arşiv kartındaki özel retry endpoint'ini kullanın.

Arşiv kökünde SQLite'tan bağımsız native tek-süreç sahiplik kilidi bulunur.
Aynı medya arşivini iki servis farklı iş DB'leriyle de olsa birlikte yazamaz;
ikinci başlangıç reddedilir. Süreç kapanınca OS kilidi bırakır. Kilit dosyasını
çalışan serviste silmeyin; dosyanın var olması tek başına aktif sahiplik değildir.

[API sözleşmesi](service.md#api) ve kararın gerekçesi
[ADR-0011](adr/0011-live-searchable-archive.md) içinde açıklanmıştır.

## Kullanım

OpenCV `videoio` + FFmpeg desteği ve yerel YOLOv8 ONNX modeli gerekir.
OpenCV derlemesi yeterlidir; CLIP/Qdrant gerekmez. Windows'ta FFmpeg plugin DLL'i
komutun yanına kopyalanır. Yeni veya boş çıktı klasörü kullanın:

```powershell
./build/search/Release/aegisvision_config.exe validate configs/stream.toml
./build/search/Release/aegisvision_stream.exe configs/stream.toml rtsp://127.0.0.1:8554/pedestrians outputs/live-run
```

Linux'ta aynı komut `./build/aegisvision_stream ...` biçimindedir. URL yalnızca
`rtsp://` olabilir. URL kullanıcı adı/şifre, query token ve fragment bilinçli olarak
reddedilir: FFmpeg hata logları hassas URL'leri gösterebilir. Bu sürüm anonim yerel
yayın içindir; kimlik doğrulamalı kamera için secret-safe adaptör henüz yoktur.
Şifreyi repo/konfigürasyona eklemeyin. URL bir HTTP isteğinden alınmaz; API'de
keyfî kaynak URL'si çalıştırma özelliği yoktur.

## Ayarlar ve kopma davranışı

`pipeline.mode = "stream"`, `[detector]`, `[tracking]`, `[stream]`, `[runtime]`
kullanılır; `[video]` uyumsuzdur. Kaynak URL'si komuta verilir, TOML'ye yazılmaz.

| Ayar | Varsayılan / sınır | İşlev |
|---|---|---|
| duration_seconds | 30 / 1–86400 | Bağlantı bekleme dahil kayıt süresi; örnek dosyada 45 |
| open_timeout_ms | 8000 / 1–15000 | FFmpeg bağlantı ve stream probing deadline |
| read_timeout_ms | 2000 / 1–5000 | FFmpeg read deadline |
| reconnect_initial_ms / reconnect_max_ms | 250 / 2000, 1–5000 | Üstel backoff; initial ≤ max |
| max_outage_ms | 15000 / 1–600000 | İlk/son decode'dan sonra kesinti sınırı |
| max_frame_age_ms | 1000 / 1–10000 | Kuyruktan alınırken decode-arrival yaş sınırı |
| tracking_gap_ms | 1000 / 1–10000 | İşlenen kareler arası boşluk aşılırsa tracker sıfırlanır |
| queue_capacity | 1 / 1–16 | Eski kareyi atarak yeni kareye yer açma |
| output_fps | 10 / 1–120 | Yalnızca işlenen karelerin çıktı oynatma hızı |

Read başarısız olduğunda bekleyen kareler silinir. Başarısız bağlantılar sınırlı
backoff ile denenir; ilk **başarılı kare** yeni source session açar. Takipçi yeni
session'da veya uzun analiz boşluğunda yeniden başlar. Kimlik `(tracking_epoch,
track_id)` çiftidir; tek başına `track_id` farklı oturumlarda aynı kişiyi göstermez.
Re-ID veya çoklu kamera eşleştirmesi yapılmaz. İki aşamalı hareket modeli frame-step
tabanlıdır; kare atlama altında doğruluğu ayrıca ölçülmelidir.

Stop, capture thread'ini birleştirir; mevcut open/read işleminin deadline'ını
bekleyebilir. Süre/kesinti sınırı da in-flight open/read ve inference nedeniyle
bir miktar aşılabilir. Süre bittiğinde en az bir kare varsa çıkış kodu 0;
kesinti/format/backend hatası veya sıfır kare için 1'dir. Ctrl+C zorla sonlandırma
bu CLI'da graceful değildir; kesilen AVI/CSV kısmi kalabilir. HTTP kuyruğuna
sonsuz RTSP işi eklenmez ve bu kayıt işi SQLite ile kurtarılmaz.

## Çıktılar ve zamanın anlamı

- `tracked.avi`: gerçekten analiz edilen kareler, MJPEG ve epoch/ID kutuları.
- `preview.jpg`, `latest.jpg`: ilk ve son analiz görselleri.
- `tracks.csv`: source sequence/session, tracking epoch, arrival zamanı, kutu/etiket/ID.
- `frames.csv`: her yazılan kare için kaynak sequence, arrival, decode yaşı ve analiz süresi.
- `summary.json`: decoded/delivered/dropped kareler, bağlantı denemeleri, session ve kuyruk tepe değeri.

`arrival_ms`, decode dönüşünde ölçülen monotonic süredir; **kamera PTS değildir**.
`decode_age_ms` inference başlamadan önceki uygulama yaşıdır; kamera/network
gecikmesini içermez. AVI kare atlamalarını ve bağlantı kopukluğunu zaman olarak
korumaz: 61 kare / 10 FPS = 6,1 saniyelik oynatma, 45 saniyelik canlı oturum değildir.
Senkronizasyon için kamera/stream PTS sonraki adımdır. Format/boyut değişirse
kayıt hata verir; sessiz resize veya aynı dosyada yeni format yapılmaz.
Hata halinde kısmi dosyalar kalabilir; rapor yazımı başarısızsa komut hata verir.

Önceki Linux transport çalışmasında 3 saniyelik open deadline, başarılı RTSP TCP
bağlantısından sonra FFmpeg stream probing aşamasını kesiyordu. Varsayılan 8 saniyeye
yükseltildi: FFmpeg'in varsayılan analiz süresi 5 saniyedir. Bu uygulama deadline'ıdır,
8 saniyelik açılma garantisi veya kamera gecikmesi ölçümü değildir.

## Tekrarlanabilir gerçek görüntülü Windows testi

Yerel gerçek yaya videosu `artifacts/media/pedestrians.mp4` ve FFmpeg PATH'te
olmalı. Video hazırlama/kaynak notları [servis kılavuzunda](service.md).

```powershell
./scripts/prepare_rtsp_relay.ps1
./scripts/test_rtsp_recovery.ps1 -Output outputs/rtsp-recovery
```

MediaMTX v1.21.1 resmî SHA256 doğrulamasıyla `artifacts/deps` içine açılır;
sistem kurulumu yapılmaz. Test relay'i yalnızca `127.0.0.1:8554` TCP dinler;
diğer protokoller/API kapalıdır. Test yalnızca kendi başlattığı süreçleri kapatır.
Port doluysa başka süreci sonlandırmaz. İlk analizden sonra publisher öldürülür,
5 saniye beklenir ve yeniden açılır. İkinci session'da analiz, takip reseti,
kuyruk sınırı, CSV/kare sayısı ve videonun bütünüyle decode edilmesi doğrulanır.

2026-10-01 bu bilgisayarda gerçek YOLOv8n/CPU çalışması: **347 decode, 61 analiz,
3 session, 2 read failure, 7 bağlantı denemesi; kuyruk tepe değeri 1**.
283 overflow, 3 kopma/stop nedeniyle atılan kare; 0 stale drop.
Analiz başlangıcındaki maksimum decode yaşı 101 ms, ortalama inference 458 ms.
İlk bağlantı aşamasında da timeout görüldüğü için session sayısı iki yerine üçtür.
Bu tek smoke çalışmasıdır; throughput benchmark, tracking doğruluğu veya fiziksel
IP kamera testi değildir. Artefaktlar `outputs/rtsp-recovery-verified/result` altında
yereldedir; video ve modeller Git'e eklenmez.

CI'da ayrı Linux transport testi FFmpeg hareketli test pattern'ini yayınlar,
publisher'ı kapatıp açar ve gerçek FFmpeg/RTSP decode + reconnect'i doğrular.
Model indirmez ve detection kalitesini test etmez. Deterministik fake capture
testleri ayrıca overflow, owned buffer, stale frame, outage, format, concurrent
stop ve analiz exception'ında reader cleanup kontrolü yapar.

## Canlı web ekranı kesinti testi

Relay ve yukarıdaki 180 saniyelik canlı servis zaten çalışırken, başka aktif
oturum ve publisher yoksa:

```powershell
./scripts/test_live_dashboard.ps1 -Output outputs/live-dashboard-recovery
```

Script yalnızca kendi FFmpeg süreçlerini başlatır/kapatır. Gerçek ilk JPEG,
publisher kesilince `204`, yeniden yayında yeni source session/tracking epoch,
en fazla 1 karelik kuyruk ve canlı analiz sürerken CLIP metin araması doğrulanır.
Sonunda kendi canlı oturumunu durdurur. `before.jpg`, `recovered.jpg` ve
`report.json` kanıtları yereldedir; repoya eklenmez. 2026-10-01 yerel doğrulamada
yeni session ve epoch ile analiz geri döndü, eşzamanlı arama dört sonuç tamamladı.
Bu sistem entegrasyon kontrolüdür; model doğruluğu veya performans benchmark'ı değildir.
Deterministik `aegisvision_live_service` testi ayrıca stale JPEG, HTTP doğrulaması,
paralel başlatma, stop, süre sınırı, model hatası ve worker cleanup'ı kontrol eder.

## Gerçek RTSP → aranabilir arşiv testi

Yukarıdaki yerel relay, Qdrant ve 180 saniyelik canlı servis çalışıyor olmalı;
publisher terminalini başlatmayın. Başka publisher/aktif canlı oturum bulunmamalı,
arşiv kotasında en az iki boş parça slotu olmalı. FFmpeg PATH'te ve gerçek yaya
kaydı `artifacts/media/pedestrians.mp4` yerinde olmalıdır. Yeni çıktı dizini kullanın:

```powershell
./scripts/test_live_archive.ps1 -Output outputs/live-archive-test
```

Script kendi gizli FFmpeg publisher'ını başlatır; opt-in arşivli oturumda ilk
JPEG ve kapalı parça bekler, kendi oturumunu durdurur, encoder/indeks işlerinin
tamamlanmasını kontrol eder. MP4 byte-range `206`, gerçek CLIP ile `scope: live`
araması, yalnızca `origin: live_archive` sonuçları ve kutulu arama JPEG'i doğrulanır.
Sonunda yalnızca kendi publisher'ını kapatır. `live-preview.jpg`,
`search-preview.jpg`, `search.json` ve `report.json` çıktı kanıtlarıdır.
Medya kökündeki üretilen arşiv parçaları tutulur; test tekrarları kotayı tüketir,
mevcut kayıtlar otomatik silinmez ve var olan çıktı dizini üzerine yazılmaz.

2026-10-02 gerçek Windows/CPU YOLOv8n + CLIP + Qdrant entegrasyonu geçti:
**118 decode, 63 analiz; 62 ve 1 karelik iki parça**, sırasıyla 6,2 ve 0,1 saniyelik
doğrulanmış MP4. İki parça da indekslendi; canlı kapsamlı arama **8 sonuç** döndürdü
ve MP4 range isteği `206` oldu. İlk sonuç klipteki 30. kare / `timestamp_ms: 3000`
konumundaydı; manifestteki kaynak arrival eşlemesi farklı bir zaman eksenidir.
Kanıtlar `outputs/live-archive-verified/report.json` ve `search.json` içindedir.

Bu, gerçek kamera kaydının yerel RTSP üzerinden uçtan uca smoke kontrolüdür;
fiziksel kameraya bağlanma, etiketli detection/arama doğruluğu, camera PTS veya
performans benchmark'ı değildir. Deterministik `aegisvision_archive_service`
testi ayrıca canlı/dosya kapsam ayrımını, retry tekilleştirmesini, ham/sahipsiz
medya erişim reddini, sealed içerik bütünlüğünü ve yeniden açılış katalog/geçmişini
kontrol eder.

Referanslar: [OpenCV open/read timeout özellikleri](https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html),
[FFmpeg stream analiz süresi](https://ffmpeg.org/ffmpeg-formats.html),
[MediaMTX yapılandırması](https://mediamtx.org/docs/references/configuration-file),
[MediaMTX FFmpeg yayını](https://mediamtx.org/docs/publish/ffmpeg).
