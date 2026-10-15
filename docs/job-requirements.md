# İş ilanına göre teslim kontrolü

2026-10-15. Bu tablo bir deneyim/işe uygunluk sertifikası değildir; repo içindeki
uygulama ve doğrulama kapsamını ayırır. "Var" üretim ortamında eksiksiz onay
anlamına gelmez. Üç yıl profesyonel deneyim, mezuniyet ve ekip çalışması gibi
nitelikler kod yazarak tamamlanamaz.

| İlan başlığı | Mevcut kanıt | Açık kalan kapsam |
|---|---|---|
| Uçtan uca CV pipeline | C++ detection → tracking → CLIP → Qdrant → HTTP/panel | Üretim yükü/operasyon doğrulaması |
| Detection, segmentation, tracking | Gerçek ONNX inference, maske/takip videoları ve etiketli ölçümler | Daha geniş bağımsız veri ve kalite iyileştirmesi |
| Eğitim/PyTorch | GPU segmentation fine-tuning ve native C++ karşılaştırması | Kaçırmaları artıran aday kabul edilmedi; eğitim kapsamını genişletme |
| Transformer/embedding | CLIP ViT görsel/metin encoder ve referans kontrolü | DINO ayrı alternatif; çok dilli arama ve metric-learning deneyi eksik |
| Çok modlu vektör arama | C++ CLIP/Qdrant indeksleme ve metin/görsel sorgu | Büyük bağımsız kalite kümesi |
| Geometrik eşleştirme | OpenCV ORB, matching, RANSAC homografi | Daha geniş gerçek sahne değerlendirmesi |
| OCR/VLM | Native adaptör, tek-kare HTTP işi/panel ve yerel model çalışması | Doğal Türkçe OCR ve VLM doğruluğu geniş ölçülmedi |
| TensorRT FP16 | Yerel engine, CPU karşılaştırması, GPU servis ve video demosu | Gerçek RTSP ile GPU birleşik test; diğer modellerin GPU uyarlaması |
| INT8 | Ayrık kalibrasyon ve native engine ölçümü | Kalite/hız kapısı geçmedi; kabul edilmiş INT8 modeli yok |
| GPU Docker deployment | Mevcut Dockerfile yalnız temel CPU demodur | GPU servis imajı ve gerçek konteyner testi eksik; yerel Docker motoru kapalı |
| Modüler/konfigürasyonlu mimari | C++ port/adaptör, TOML, testler, ADR'ler | Yeni backend'lerin aynı sözleşmelere eklenmesi |
| Asenkron kuyruk/API | Bounded C++ worker, SQLite kalıcılık/iptal, OpenAPI/CI kontrolü | Redis dağıtık worker/lease yok; FastAPI/Celery kullanılmıyor |
| C++ / Python / Linux / Git | Uygulama C++20, eğitim/araçlar Python, Git ve CI | Python uygulama deneyimi yerine C++ tercih edildi; Linux GPU deployment doğrulanmadı |
| Çoklu kamera/Re-ID | C++ tracklet/prototip; [x1.0 karşılaştırması](reid-model-experiment.md) ve [32-ayarlı geliştirme kalibrasyonu](reid-calibration.md); kabul kapısını geçen ayar yok | Temsil/metric-learning iyileştirmesi ve yeni son test, non-overlap geçiş ölçümü, saat eşleme ve canlı entegrasyon eksik |
| RTSP/GStreamer/DeepStream | RTSP/FFmpeg, reconnect/bounded queue; CPU maske testi | GPU+RTSP birleşik test açık; GStreamer/DeepStream yok |
| Triton/Jetson | Doğrulanmış deployment yok | Uygun ortam/donanım ve gerçek ölçüm gerekli |
| Teknik teslim | Kullanım belgeleri, ADR'ler, API/OpenAPI, hazır kurulum demosu, salt okunur dosya ön kontrolü ve 20 saniyelik yerel demo videosu | Sıfır makine kurulumu, tam model/veri/bağımlılık lisans incelemesi ve yük/operasyon doğrulaması |

## Uygulama sırası

Kısa ve güncel kapanış listesi: [teslim durumu](delivery-status.md).

1. Yayın erişilebilir olduğunda mevcut gerçek RTSP + GPU birleşik testini bitir.
   Dosya replay testi bunun yerine geçmez. Mevcut CPU RTSP testleri ayrı kanıttır.
2. Çalışan Linux GPU konteyner ortamında Docker dağıtımını geliştir ve ölç.
   Yalnız Dockerfile yazmak bu maddeyi kapatmaz.
3. Çoklu kamera/Re-ID ve dağıtık kuyruk gibi eksik uygulama bileşenlerini,
   kapsamı sınırlandırılmış ve test edilebilir adımlar halinde geliştir.
4. Kalite tarafında yeni bağımsız veriyle ölçüm; INT8 için ayrı geliştirme/holdout
   protokolü. Başarısız adaylar varsayılan yapılmaz.
5. Tekrarlanabilir teslim ve uygun donanımda ileri deployment.

Ayrıntılar ve kanıt bağlantıları: [yol haritası](roadmap.md),
[GPU servis](gpu-service.md), [OpenAPI](api.md). GStreamer/DeepStream,
FastAPI/Celery ve DINO ilandaki alternatif örneklerdir; mevcut C++/FFmpeg/CLIP
seçimlerinin yanında zorunlu ikinci uygulamalar diye sunulmaz. Buna karşılık
GPU Docker gibi açıkça istenen fakat doğrulanmamış yetenekler tamamlandı sayılmaz.
