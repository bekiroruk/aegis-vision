# AegisVision yol haritası

Güncelleme: 2026-09-30. Aktif uygulama C++20'dir; Python yalnızca model hazırlama
ve eski prototip içindir. Aşağıdakiler mevcut özelliklerle hedefleri ayırır.

## Tamamlanan temel

- Modüler C++ port/adaptör yapısı, CMake, Windows/Linux CI ve deterministik testler.
- OpenCV görüntü okuma, kutu çizme, ORB eşleştirme ve RANSAC homografi.
- YOLOv8 ONNX ile gerçek piksel verisinden CPU detection.
- Yerel video okuma, IoU / iki aşamalı takip, ID etiketli video ve CSV/JSON raporları.
- Bellekte cosine similarity indeksi ve test amaçlı hash embedding.
- Gerçek CLIP görsel/metin embedding ve C++ Qdrant istemcisi; kalıcı görsel/kırpma indeksi,
  metin/görsel sorgu CLI'ı, Unicode tokenizer ve PyTorch referans testleri.
- Sürümlü TOML şemasıyla görüntü, video ve arama ayarları; mod, backend, yol ve eşik doğrulaması.
- COCO validation kırpmaları için toplu indeksleme ve Recall@K/Hit@K değerlendirme komutları.
- Redis/Qdrant Compose tanımları; **Redis istemcisi ve GPU deployment hazır değil**.

## Kalan işler — önerilen sıra

1. **Arama kalitesini genişlet.** CLIP/Qdrant akışı ve referans testleri için
   [arama kılavuzu](search.md); 64 COCO kırpmalı başlangıç ölçümü için
   [benchmark](search-benchmark.md). Sonraki işler: dizin/video indeksleme,
   daha büyük bağımsız test kümesi ve çok dilli model değerlendirmesi.
   DINO görsel tanıma için ayrı alternatiftir;
   tek başına metin araması sağlamaz.

2. **Ölçülebilir model ve tracking kalitesi.** Küçük, lisansı uygun etiketli veri kümesi;
   sabit train/validation/test ayrımı ve tekrarlanabilir değerlendirme komutları oluştur.
   Detection mAP, tracking IDF1/HOTA/ID switch ve arama Recall@K raporla. İhtiyaca göre
   fine-tuning, metric learning ve tam ByteTrack/Kalman + Re-ID ekle. Çoklu kamera için
   zaman eşleme ve kamera bazlı ID alanları gerekir. Kabul: baseline karşılaştırması
   aynı veri/sabit koşullarda; ID sayısı doğruluk metriği olarak kullanılmaz.

3. **Segmentation, OCR ve VLM.** Önce maskeler ve sonuç sözleşmesi/testleri, ardından
   OCR metin-kutu çıktısı ve ayrı VLM adaptörü ekle. Bunlar mevcut sistemde yoktur.
   Kabul: maske IoU, OCR hata oranı ve şemaya uygun VLM çıktısı ölçülür; timeout ve
   model hataları pipeline'ı belirsiz durumda bırakmaz. Otomatik etiketler gözden geçirilir.

4. **Canlı video ve servis katmanı.** RTSP/GStreamer kaynağı, reconnect, gerçek PTS,
   bounded queue/backpressure, cancellation ve C++ HTTP API/Redis worker geliştir.
   Kabul: bağlantı kesilmesi ve yük testleri; bellek sınırlı, görevler izlenebilir,
   tekrar denemeler aynı sonucu iki kez indekslemez. FastAPI/Celery şart değil;
   bu projede uygulama katmanını da C++ tutuyoruz.

5. **GPU optimizasyonu ve dağıtım.** TensorRT backend, FP16 eşdeğerlik testi,
   temsilî calibration verisiyle INT8, model sürümleme ve GPU Docker imajı ekle.
   Kabul: CPU/FP32 referansına karşı kalite farkı, warm-up sonrası p50/p95,
   throughput ve GPU bellek raporu. CUDA/GPU uyumu doğrulanmadan tamamlandı sayılmaz.

6. **Portföy teslimi ve ileri deployment.** Tek komutlu tekrar üretilebilir demo,
   API/OpenAPI dokümanı, mimari karar kayıtları, model/veri lisans notları ve kısa
   demo videosu hazırla. Triton/Jetson profilleri uygun donanım varsa son aşamadır;
   cihazda ölçülmeyen hız için iddiada bulunulmaz.

İş ilanındaki diğer adımlar için kalite ölçümü ve operasyon katmanları genişletilecek.
Tüm düzenlemeler bu bilgisayardaki repoda yapılır; derleme/test sonrası GitHub'a gönderilir.
