# Gerçek RTSP üzerinde canlı maske doğrulaması

2026-10-10, Windows Release, uygulama commit'i `e99f0e0`. YOLOv8n-seg ONNX
CPU inference'ı kullanıldı; capture/model mock değildir. Kaynak fiziksel kamera
değil, `pedestrians.mp4` dosyasının FFmpeg → MediaMTX → RTSP/TCP üzerinden
yeniden yayınıdır. Model, C++ pipeline ve canlı HTTP JPEG çıktısı birlikte çalıştı.

## Sonuç

İlk smoke başarılı oldu. Ardından test betiğine durdurmanın tamamlanması ve
durdurulmuş önizlemenin HTTP 204 vermesi kontrolü eklendi; bu sürüm de geçti.
İkinci koşunun raporundan:

| Durum | Bağlantı oturumu | Takip epoch'u | İşlenen kare | Önizleme |
| --- | ---: | ---: | ---: | --- |
| İlk görüntü | 1 | 1 | 1 | JPEG mevcut |
| Yayıncı kapatıldı | 1 | 1 | 7 | Temizlendi, HTTP 204 |
| Yayıncı yeniden açıldı | 2 | 2 | 8 | Yeni JPEG mevcut |
| Stop tamamlandı | 2 | 2 | 9 | Temizlendi, HTTP 204 |

- Son durumda 83 kare decode edildi, 73 kare atlandı; kuyruk tepe değeri **1**,
  kapasite **1** kaldı. Stop sırasında alınan/işlenmekte olan kare nedeniyle bu
  sayaçların toplamı birebir eşit olmak zorunda değildir.
- Dokuz tamamlanmış analizde ortalama **385,90 ms**; bu kısa smoke ölçümüdür,
  uzun süreli throughput veya p95 benchmark'ı değildir. Kamera gecikmesi değildir.
- Yeniden bağlantı anındaki önizleme decode-arrival yaşı **429,86 ms** idi.
- Canlı maskeleme sırasında CLIP metin araması başarıyla **4 sonuç** döndürdü.
  Bu kontrol eşzamanlı çalışmayı doğrular; arama kalitesini ölçmez.
- Yeniden bağlantı JPEG'inde **Epoch 2**, renkli kişi/araç maskeleri ve takip
  etiketleri görsel olarak incelendi. Bu görüntü maske AP veya kimlik doğruluğu
  ölçümü değildir. Bağlantı kesilince kimliklerin devam ettiği iddia edilmez.
- Arşiv bu testte kapalıydı; maskelerin arşive yazılması veya arşiv indeksleme
  bu testin kapsamı değildir. Sunucu/API ve üretilen JPEG doğrulandı;
  yeni bir tarayıcı etkileşim testi yapılmadı.

## Tekrar çalıştırma

Qdrant hazır olmalı. [Canlı video kılavuzundaki](live-video.md) relay'i açın.
Başka yayıncı aynı RTSP yolunu kullanmamalı; test betiği kendi FFmpeg yayıncısını
başlatır, bağlantı kesintisi için kapatır ve yeniden açar.

Mevcut 8090 servisini etkilememek için ayrı port, medya kökü ve SQLite kullanıldı:

```powershell
./scripts/start_service.ps1 -Port 8091 -ServerExecutable build/live-seg/Release/aegisvision_server.exe -MediaDirectory outputs/live-mask-rtsp-20261010/media -JobDatabase outputs/live-mask-rtsp-20261010/jobs.sqlite -LiveUrl rtsp://127.0.0.1:8554/pedestrians -SegmentationModel artifacts/models/yolov8n-seg/yolov8n-seg.onnx -LiveSegmentation
```

Yeni PowerShell terminalinde (her koşuda yeni çıktı dizini seçin):

```powershell
pwsh -NoProfile -File scripts/test_live_dashboard.ps1 -Port 8091 -ExpectSegmentation -Output outputs/live-mask-rtsp-new
```

Test canlı oturumu durdurur ve yalnızca kendi yayıncı süreçlerini kapatır. Ayrı
sunucu ve relay'i manuel açtıysanız test sonrasında kendi terminallerinde Ctrl+C
ile durdurun; mevcut Qdrant ve 8090 servisini kapatmanız gerekmez.

Yerel kanıtlar `outputs/live-mask-rtsp-20261010/verified/` altındadır:
`report.json`, `before.jpg`, `recovered.jpg`, yayıncı logları. Sunucu/relay logları
üst dizindedir. Bu üretilmiş dosyalar ve modeller Git'e dahil edilmez.

Girdi SHA256 değerleri:

- ONNX: `85c8cb2695f586a3b9cc0279a3fca8aaa4e233f5ed6dfee88b2ec7cad587495b`
- MP4: `ac89c3dff7977be26879f5cfe6d710cbd95ec16d196cdc161746c9df4f7a6588`

Önceki derlemede **35/35 CTest** geçti. Bu adımda C++/web kodu değişmedi;
gerçek RTSP smoke ve test betiğinin stop/cleanup davranışı doğrulandı.
