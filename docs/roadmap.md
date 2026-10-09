# AegisVision yol haritası

Güncelleme: 2026-10-06. Aktif uygulama C++20'dir; Python yalnızca model/veri hazırlama,
bağımsız referans kontrolü ve eski prototip içindir. Aşağıdakiler mevcut özelliklerle hedefleri ayırır.

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
- Klasörden tam görsel ve örneklenmiş video karelerinden YOLO/CLIP nesne indeksleme;
  tekrar çalıştırmada sabit ID, dosya/kare/zaman/kutu metadata'sı.
- Redis/Qdrant Compose tanımları; **Redis istemcisi ve GPU deployment hazır değil**.
- Yerel C++ HTTP servisi, sınırlı asenkron iş kuyruğu, ilerleme/iptal ve tarayıcıda
  gerçek videoda metin arama, kutulu önizleme ve zamanına atlama; [servis kılavuzu](service.md).
- SQLite ile kalıcı iş geçmişi/kuyruk, ani kapanma sonrası sınırlı yeniden deneme,
  tek süreç sahipliği, bağlam doğrulaması ve disk yazım hatasında durma.
- RTSP/FFmpeg kaynağı, open/read timeout, sınırlı kare/byte kuyruğu, drop-oldest,
  reconnect/backoff, oturum ayrımı ve süre sınırlı YOLO/takip kaydı; [canlı video](live-video.md).
- Aynı analiz motoruyla tarayıcıdan canlı oturum başlatma/durdurma, kutulu JPEG önizleme,
  kesintide görüntüyü temizleme ve ayrı model instance'ıyla eşzamanlı dosya/metin araması.
- İsteğe bağlı analiz-kare arşivi: sınırlı segment/disk kotası, doğrulanmış MP4,
  sonlu SQLite encoder/CLIP/Qdrant işleri, canlı kapsamlı arama ve klip anına atlama.
  Kamera PTS/tam yayın FPS arşivi değildir; arrival/kare/epoch eşlemesi manifestte korunur.
- C++ category-aware bbox AP, CLEAR/IDF1 ve ham süre/p50/p95 raporu; 64 tam COCO
  sahnesi ve 179 karelik etiketli MOT15 videosunda aynı detection ile tracker
  karşılaştırması, resmi pycocotools/TrackEval kontrolü; [kalite ölçümü](model-quality.md).
- Deneysel C++ Kalman hareket tahmini ve aktif-önce Hungarian eşleştirmesi;
  önceki iki backend korunur. İki etiketli sahnede resmi kontrol tamamlandı;
  sonuçlar karışık olduğu için varsayılan yapılmadı. Ardışık yerel kareler içindir;
  canlı timestamp-aware dt/Re-ID yoktur.
- Ayrı merkez odaklı Kalman gate, dört yöntemli gerçek video karşılaştırması ve
  resmi referans doğrulaması. Campus boyut titreşimi azalırken Stadtmitte FP artışı
  raporlandı; geliştirme sahnelerinde gözlemdir, bağımsız hold-out sonucu değildir.
- Gerçek eğitimli OSNet x0.25 ile 512-boyut kişi görünüşü, native C++ ONNX inference,
  sınırlı high-score EMA prototipi ve isteğe bağlı Kalman association;
  [sözleşme, test ve kullanım](person-appearance.md). Canlı/çoklu kamera/global kimlik yoktur.
- Native C++ HOTA, 19 IoU eşiği ve resmi TrackEval kontrolü; model/config/eşik
  sabitlemeli proje düzeyinde validation/test ayrımı; [protokol](tracking-transfer.md).
- Modeli yeniden çalıştırmadan CLEAR kimlik değişimi olayları, ardışık/boşluk
  ayrımı ve eski rapora karşı kontrol; [hata analizi](tracking-switch-audit.md).
- İsteğe bağlı sınırlı Kalman aday karar kaydı; IoU/hareket/appearance reddi ile
  assignment kaybını ayırma; [geliştirme sahnesi bulguları](association-trace.md).
- Aktif appearance kapısını kaldıran opt-in altıncı takipçi ve iki geliştirme
  sahnesinde resmi kontrollü ablation; [sonuçlar](active-appearance-ablation.md).
  Ardışık kopmalar azalırken FP artışı görüldü; varsayılan yapılmadı.

## Kalan işler — önerilen sıra

Son tracking deneyi: [korumalı appearance esnetmesi](guarded-appearance-ablation.md).
250 gerçek karede resmi doğrulama tamamlandı; sonuçlar karışık, varsayılan değişmedi.
Gözlenmiş iki sahnede eşik denemeleri burada durduruldu. Linux arşiv testindeki
aralıklı durma, 20 tekrarlı CI kontrolünde `archive_unsupported_entry` olarak
yakalandı: geçici dosya yayını sırasında dizin taramasındaki giriş kaybolabiliyor.
Kaybolma için tüm kota taramasını en fazla üç kez yeniden başlatan düzeltme
eklendi; kısmi sayaç kabul edilmiyor. Uzak Linux tekrar testi doğrulaması bekleniyor.
Bu incelemede yerelde ayrıca manifest/mühür yayın yarışı yakalandı ve katalog
yalnız atomik `sealed.json` sonrasında listeleyecek şekilde düzeltildi; araya
okuma giren durum için deterministik test eklendi. Bu düzeltme, önceki Linux
zaman aşımının aynı nedenden kaynaklandığını tek başına kanıtlamaz.

1. **Arama kalitesini genişlet.** CLIP/Qdrant akışı ve referans testleri için
   [arama kılavuzu](search.md); 64 COCO kırpmalı başlangıç ölçümü için
   [benchmark](search-benchmark.md). Klasör/video akışı için [indeksleme](indexing.md).
   Sonraki işler: daha büyük bağımsız test kümesi ve çok dilli model değerlendirmesi.
   DINO görsel tanıma için ayrı alternatiftir;
   tek başına metin araması sağlamaz.

2. **Model ve tracking kalitesini geliştirme.** Küçük etiketli başlangıç ölçümü,
   AP/CLEAR/IDF1/ID switch ve CPU p50/p95 tamamlandı; [kapsam ve sonuçlar](model-quality.md).
   HOTA ve iki yeni sekans için sabit validation/test protokolü eklendi.
   Sonraki iş, daha geniş veri, hata analizi ve yalnız geliştirme/validation üzerinde
   kalibrasyondur; yeni ayarlar için dokunulmamış test verisi gerekir.
   Mevcut iki aşamalı tracker her sekans için daha iyi değildir.
   İhtiyaca göre
   fine-tuning, metric learning, tam ByteTrack ve appearance ablation/kalibrasyonunu genişlet. Mevcut
   Kalman backend resmi ByteTrack değildir; canlıda zaman-adımı desteği de gerekir. Çoklu kamera için
   zaman eşleme ve kamera bazlı ID alanları gerekir. Kabul: baseline karşılaştırması
   aynı veri/sabit koşullarda; ID sayısı doğruluk metriği olarak kullanılmaz.

3. **Segmentation, OCR ve VLM.** İlk C++ segmentation adaptörü, ayrı video CLI'ı
   ve ROI maske sözleşmesi eklendi; [durum ve kullanım](segmentation.md).
   Gerçek YOLOv8n-seg modeliyle yerel video inference, takip-ID bağlama ve yeniden
   kullanılabilir segmentation pipeline tamamlandı. Önceden görülmüş 64 COCO
   görüntüsünde resmi maske değerlendirmesi yapıldı (AP %45,47); kör test değildir.
   Sınırlı tek-kare HTTP `segment_frame` işi mevcut kuyruğa bağlandı; dashboard
   maske gösterimi ve canlı RTSP henüz dahil değil. Sırada dokunulmamış daha geniş
   veri ve canlı entegrasyon; ardından OCR metin-kutu çıktısı ve ayrı VLM
   adaptörü ekle. OCR/VLM mevcut sistemde yoktur.
   Kabul: maske IoU, OCR hata oranı ve şemaya uygun VLM çıktısı ölçülür; timeout ve
   model hataları pipeline'ı belirsiz durumda bırakmaz. Otomatik etiketler gözden geçirilir.

4. **Canlı video ve kalıcı servis katmanı.** Yerel C++ HTTP API, bounded queue ve
   cancellation ve SQLite job recovery tamamlandı. RTSP/FFmpeg kayıt CLI'ı,
   reconnect/backpressure, süre sınırlı canlı web ekranı ve kota sınırlı canlı nesne
   arşivi/indeksi tamamlandı. GStreamer adaptörü, kamera/stream PTS, tam FPS/sesli
   kayıt ve çoklu worker/Redis lease desteği geliştir. Mevcut arşiv yalnızca analiz
   karelerini kaydeder; kota dolunca capture yerine yalnızca arşiv durur.
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
