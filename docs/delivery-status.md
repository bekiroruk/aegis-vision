# Teslim durumu ve kalan işler

Güncelleme: 2026-10-15. Bu belge mevcut teslimin kısa listesidir; bütün ilan
maddelerinin tamamlandığı veya üretime hazır olduğu anlamına gelmez.

## Kullanılabilir demo

Hazır kurulumdaki panel: `http://127.0.0.1:8090`.
Kapandıysa repo içinden `./scripts/demo.ps1 -OpenBrowser` çalıştırılır.
Maske/video analizi, OCR, yerel VLM ve görsel/metin arama aynı paneldedir.
Model açıklamaları ve takip kimlikleri doğrulanmış gerçek etiketler değildir.

Servis başlatmadan dosya eksiklerini topluca kontrol et:

```powershell
.\scripts\demo.ps1 -CheckOnly | ConvertTo-Json
```

`FilesPresent` yalnız başlangıç dosyalarının varlığını belirtir. Boş/bozuk model,
eksik DLL, GPU veya port hazır demek değildir. Bu kontrol dosya oluşturmaz,
servis başlatmaz/durdurmaz; farklı çalışma klasöründen de kullanılabilir.

## Bu turda kapatılan küçük işler

- Linux CI derlemesini durduran VLM HTTP fixture generic-lambda tür belirsizliği
  giderildi. Windows'ta VLM taşıma/sınır/timeout/iptal testi geçti.
  `78fb968` için [GitHub çalışmasında](https://github.com/bekiroruk/aegis-vision/actions/runs/38696609083)
  5/5 grup başarılı: API, Windows/Linux temel, OpenCV Linux, arama/servis Linux.
  Son grup 20 tekrarlı arşiv kontrolünü de geçti. Bu sonuç sonraki teslim
  belgesi/ön kontrol commit'lerinin CI sonucunu otomatik olarak doğrulamaz.
- Demo dosya envanteri tek listede toplandı; eksikler topluca gösteriliyor.
  `-CheckOnly` Windows PowerShell 5.1 ve PowerShell 7'de boş/tam/eksik kurulum,
  dosya yerine klasör, farklı çalışma klasörü, çelişen seçenekler ve yan
  etkisizlik testlerini geçti. Windows CI'a eklendi.
- Demo servisleri yeniden başlatıldı. `/api/health`: `ready`, segmentation,
  video, OCR, Unicode ve VLM açık; kalıcı geçmiş açık, storage_error boş.
  Bu sağlık kontrolüdür; bütün modeller bu turda yeniden ölçülmedi.
- Web regresyonları geçti: video, arama seçimi, yenileme, kuyruk,
  OCR/VLM/maske, iptal ve güvenli URL kontrolleri.
- Gerçek maskeli/takipli videodan 20 saniyelik kısa demo hazırlandı:
  `outputs/delivery-20261015/aegis-vision-demo-20s.mp4`.
  H.264, 768×576, 200 kare, 20 saniye, 1.763.679 bayt; FFprobe ile doğrulandı.
  SHA256: `70496dc8c368377ea24bf1d14095f474c30e60c948a1b28853c33b2c50372e57`.
  Kaynak: `artifacts/media/analysis-output/v-17918018757271891-1/video.mp4`.
  Eski analiz çıktısının kısaltılmasıdır; yeni inference/canlı GPU kanıtı değildir.
  Video yerelde tutulur, Git'e yüklenmez; kaynağın lisansı ayrıca gözetilir.

## Kalan adımlar — kapanış koşulları

1. **Gerçek RTSP + GPU birleşik testi.** TensorRT analizi, eşzamanlı dosya
   işi, kesinti/yeniden bağlantı ve temiz stop tek gerçek yayında sınanmalı.
   CPU RTSP ve GPU dosya replay ayrı ayrı var; birleşik test yerine geçmez.
   Mevcut publisher engeli aşılmış sayılmıyor.
2. **Linux GPU Docker dağıtımı.** GPU servis imajı ve çalışan NVIDIA
   konteynerinde sağlık/inference/kapanış testi. Yerel Docker Linux motoru bu
   kontrolde erişilemiyordu; CPU CI başarısı GPU Docker kanıtı değildir.
3. **Çoklu kamera/Re-ID.** [32-ayarlı geliştirme kalibrasyonu](reid-calibration.md)
   tamamlandı; hiçbir ayar kalite koşullarını birlikte geçmedi, varsayılan korundu.
   Ayrı eğitim kimliklerinde temsil/metric-learning iyileştirmesi ve yeni son test;
   saat eşleme, gerçek kamera geçişleri ve canlı kimlik ilişkilendirmesi gerekiyor.
4. **INT8 ve diğer GPU modelleri.** INT8 engine üretimi var fakat ilk aday
   kalite/hız kapısını geçmedi. Ayrı veriyle yeni kalibrasyon/QAT ve ölçüm;
   segmentation/diğer modellerin TensorRT uyarlaması gerekiyor. FP16 detection var.
5. **Bağımsız kalite doğrulaması.** Detection/tracking/segmentation, arama,
   doğal Türkçe OCR, VLM ve geometrik eşleştirme için daha geniş bağımsız veri.
   Eğitim/değerlendirme araçları var; bütün alanlarda iyi kalite iddiası yok.
6. **Dağıtık kuyruk.** Tek-süreç C++/SQLite kuyruğunun ötesinde Redis worker,
   görev sahipliği, çökme kurtarma ve çift işlemeyi önleme testleri.
   Compose içinde Redis bulunması uygulamanın hazır olduğu anlamına gelmez.
7. **Temiz makine teslimi ve dağıtım incelemesi.** Sıfırdan kurulumun ayrı
   makinede denenmesi, model/veri/bağımlılık lisans envanteri ve dağıtım
   koşullarının incelenmesi, yük/operasyon kontrolleri. Kısa demo hazır;
   diğer teslim şartları açık. THIRD_PARTY_NOTICES şu an yalnız TrackEval
   bildirimini içerir; tam lisans denetimi değildir.
8. **Triton / Jetson.** Tercih sebebi olan ileri deployment için gerçek
   sunucu/edge ortamı ve uyumluluk/hız ölçümleri gerekiyor.

DINO/ViT, YOLO/DETR, Qdrant/FAISS, Redis/Celery ve GStreamer/DeepStream gibi
örneklerin her birini ayrıca uygulamak şart değildir. C++ tercihi korunur;
sırf isim tamamlamak için ikinci bir FastAPI uygulaması eklenmez. Kullanılan
alternatifler ve eksikler [ilan karşılaştırmasında](job-requirements.md) ayrıdır.
Mezuniyet ve üç yıllık profesyonel deneyim repo geliştirerek sağlanamaz.

Ayrıntılar: [demo](demo.md), [yol haritası](roadmap.md),
[GPU servis](gpu-service.md), [Re-ID karşılaştırması](reid-model-experiment.md).
