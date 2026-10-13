# Servis ve panelde isteğe bağlı FP16 GPU detection

## Kapsam

`--trt-bundle` / PowerShell `-TensorRtBundle`, dosya ve arşiv indekslemesindeki
nesne tespitini native TensorRT FP16'ya geçirir. Canlı **detection** oturumları
da aynı engine'den kendi model/context/stream örneğini oluşturur. Queue worker
ve canlı worker GPU nesnesini paylaşmaz; oturum yeniden başlayınca yeni örnek açılır.

CPU varsayılanı korunur. GPU seçilip engine/SDK/GPU/model uyuşmazsa sessiz CPU
fallback yoktur. FP32/INT8 engine'ler servis seçiminde reddedilir; INT8 deneyi
kalite kapısını geçmedi. Yerel engine güvenilir kaynaktan üretilmiş olmalıdır.

Maskeleme, maskeli video, OCR, CLIP ve VLM bu seçimle GPU'ya geçirilmez.
`-LiveSegmentation` açıkken canlı analiz CPU segmenter kullanır, dosya detection
GPU'da kalabilir. `/api/health.detector_backend`, `/api/live/sources.backend`
ve canlı oturumun `backend` alanı ayrıdır; panel ayrı motor bilgilerini gösterir.
Oturum başlamadan görünen canlı motor bilgisi yapılandırmayı belirtir, çalışan
kamera veya başarı kanıtı değildir. Canlı hata halinde CPU'ya geçilmez.

GPU indeks kimliği engine hash'i ve CPU referansının model/ayar kimliğini içerir;
CPU sonuçları GPU sonuçlarıyla aynı cache olarak kullanılmaz. GPU'ya geçişte yeni
job database kullanın: mevcut SQLite bağlamı model kimliğini korur. CPU DB'sini
silerek uyum zorlanmamalıdır. `service-search-gpu.toml` ayrı Qdrant demo collection'ı
kullanır; önceki CPU sonuçları değişmez.

## Başlatma

[SDK ve FP16 engine kurulumu](tensorrt.md) ön koşuldur. Qdrant çalışmalı.

```powershell
cmake --build build/tensorrt --config Release --target aegisvision_server --parallel 4
.\scripts\start_service.ps1 -TensorRtBundle artifacts/models/yolov8n-trt-fp16-v1 -SearchConfig configs/service-search-gpu.toml -Port 8091 -JobDatabase artifacts/service/gpu-jobs.sqlite
```

Script GPU seçildiğinde `build/tensorrt/Release/aegisvision_server.exe` seçer;
explicit `-ServerExecutable` verilirse onu kullanır. CPU build GPU bayrağını
reddeder. Çalışan CPU servisi otomatik kapatılmaz. Mevcut segmentation/OCR
seçenekleri ayrıca kullanılabilir. Yalnız detection ile başlatılırsa maskeli video
ve OCR düğmelerinin kapalı olması beklenen durumdur.

Canlı yayın için mevcut `-LiveUrl rtsp://127.0.0.1:8554/pedestrians` eklenir.
Relay ve publisher önceden çalışıyor olmalıdır; kaynak URL'si tarayıcıdan
değiştirilmez. Canlı önizleme yaklaşık 2 kez/sn yenilenir; GPU kernel FPS'i
veya sesli/tam FPS video yayını iddiası değildir.

Gerçek RTSP + GPU birleşik testi (yayın açıkken, boş oturum ve ayrı demo collection'ıyla):

```powershell
.\scripts\test_gpu_service.ps1 -Port 8091 -Output outputs/gpu-service-test-v1
```

Bu test canlı kare, eşzamanlı GPU dosya indeksleme, CLIP/Qdrant arama, ayrı
detector kimliği, bounded queue ve stop sonrası eski JPEG kaldırılmasını doğrular.
Var olan aktif canlı oturumu kesmez. RTSP yayınını kendisi başlatmaz.

## 2026-10-13 yerel kanıt

- RTX 3060 / FP16 engine `27bcd10dd28bf8e96753346d455d5b63d29ed4f9942078b26c9795c597373df0`.
- Gerçek HTTP servisi, pedestrians videosunun ilk 30 karesinden stride 10 ile
  üç kare örnekledi ve **16 nesne kaydı** oluşturdu. İngilizce metin araması
  **8 sonuç** döndürdü; her sonuç GPU'ya özgü detector imzasına sahipti.
- Tarayıcıda sonuç kartları görüldü, karta tıklayınca yerel video oynadı
  (`paused=false`, `readyState=4`, medya hatası yok). Motor etiketi GPU/FP16 idi.
- Gerçek FP16 ile iki ayrı GPU context'te dosya ve canlı worker eşzamanlı
  çalıştı; JPEG üretimi, kuyruk sınırı, stop/restart ve iki oturumda iki farklı
  model oluşturulması geçti. **Capture fixture kullanan bu test RTSP değildir.**
- GPU build'inde servis/live/help ve gerçek TensorRT testleri; CPU build'inde
  servis/live testleri; web motor ayrımı ve mevcut UI regression testleri geçti.
  INT8 bundle ile servis başlatma açık hata ile reddedildi.
- **Tam gerçek RTSP + GPU testi tamamlanmadı.** Otomatik publisher başlatma araç
  tarafından engellendi; kullanıcı örnek yayını açıp sonra PowerShell'i kapattı.
  Relay kaydı yayının açılıp kapandığını gösterir, GPU canlı analizin geçtiğini
  göstermez. `test_gpu_service.ps1` henüz başarılı koşu kanıtına sahip değil.

Yerel HTTP kayıtları: `outputs/gpu-service-runtime-v1/jobs-evidence.json`
(SHA256 `1730b6675fb1d51ab0a1ff27da5e1c52497723b19b6a5b56a7f76da6b9b25880`),
ekran görüntüsü aynı klasörde `browser-proof.png`. Demo 8091'de ayrı media root,
SQLite ve collection kullanır; 8090'daki mevcut CPU servisi korunmuştur.
Bu iş GPU Docker/Linux deployment veya üretim yük testi değildir.

## Kamera yayını olmadan video demosu — 2026-10-14

`-ReplayFile` seçeneği publisher/MediaMTX olmadan **yerel videoyu döngüde analiz
eder**, gerçek RTSP kamera yayını değildir. Panel bunu açıkça etiketler;
API kaynak ve oturumlarında `source_kind=file_replay` döner. RTSP için aynı
alan `rtsp` olur. Tarayıcıdan dosya yolu/URL kabul edilmez.

```powershell
.\scripts\start_service.ps1 -TensorRtBundle artifacts/models/yolov8n-trt-fp16-v1 -SearchConfig configs/service-search-gpu.toml -Port 8091 -JobDatabase artifacts/service/gpu-jobs.sqlite -ReplayFile artifacts/media/pedestrians.mp4
```

Qdrant, yerel model dosyaları ve derlenmiş servis ön koşulları değişmez.
`LiveUrl` ve `ReplayFile` birlikte seçilemez. Native CLI'da stream config ve
kaynak konumuna video yolu verilip `--replay-file` eklenir. Varsayılan RTSP
doğrulaması korunur. Replay yalnız mevcut mutlak yerel dosya kabul eder; server
başlangıcında göreli yol mutlaklaştırılır. UNC ve URL kaynakları reddedilir.

Dosya `stream.output_fps` hızında (varsayılan 10 kare/sn) okunur. Bu hız kamera
FPS'i veya dosyanın özgün PTS akışı değildir. Dosya sonunda yeniden açılır,
yeni kaynak oturumu/takip epoch'u oluşur; normal döngü kamera okuma hatası
sayılmaz. Açılmayan/çözülemeyen dosyada mevcut outage sınırı geçerlidir.
Oturum 180 saniye sonra biter; **Video demosunu başlat** ile yeniden açılabilir.
Arşiv demo modunda varsayılan kapalıdır; isteğe bağlı sınırlı analiz-kare arşivi
aynı kurallarla kullanılabilir. Bekleme durdurulabilir; kuyruk sınırı değişmez.

```powershell
.\scripts\test_gpu_service.ps1 -Replay -Port 8091 -Output outputs/gpu-replay-test-v1
```

Gerçek RTX 3060 / FP16 koşusu geçti: demo analiz sayacı 4'ten 20'ye ilerlerken
aynı GPU'da ayrı worker dosyadan 16 nesne indeksledi, CLIP/Qdrant 4 sonuç döndürdü.
Kuyruk tepe değeri 1; stop sonrası durum `stopped`, eski JPEG yok.
Rapor: `outputs/gpu-replay-test-v1/report.json`, SHA256
`5ffbde123a79d827012c7e8630d31514734ce00f3474adec320c62359c952325`.
Tarayıcıda gerçek 768×576 kutulu görüntü görüldü ve sayaç ilerledi;
ekran görüntüsü aynı klasörde `browser-proof.png`.
Gerçek dosya decode/döngü/pacing/stop testi, HTTP kaynak türü ve yaşam döngüsü,
GPU izolasyonu ve web etiket testleri geçti.
`-Replay` olmadan test dosya demosunu reddeder; bu sonuç RTSP testi yerine geçmez.
