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
ve toplam RAM kullanımı ek model nedeniyle artar. Canlı oturum dosyaya, Qdrant'a
veya SQLite'a yazılmaz; servis yeniden açılınca kendiliğinden başlamaz.

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
Doğrudan executable komutunun son üç argümanı `JOB_DB STREAM.toml RTSP_URL`'dir.

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

Referanslar: [OpenCV open/read timeout özellikleri](https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html),
[FFmpeg stream analiz süresi](https://ffmpeg.org/ffmpeg-formats.html),
[MediaMTX yapılandırması](https://mediamtx.org/docs/references/configuration-file),
[MediaMTX FFmpeg yayını](https://mediamtx.org/docs/publish/ffmpeg).
