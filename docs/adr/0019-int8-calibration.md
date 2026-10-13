# 0019 — Yerel, ayrık veriyle INT8 kalibrasyonu

Tarih: 2026-10-13. Durum: uygulandı; üretime geçiş otomatik değildir.

## Karar

Native C++ YOLO backend'i, mevcut sabit TensorRT 10.13.3 SDK'sında
`IInt8EntropyCalibrator2` ile INT8-mixed engine oluşturabilir. FP16 fallback
açıktır; giriş/çıkış FP32, ortak letterbox/NMS CPU'dadır. Bu ayrı CLI desteğidir;
web servisi varsayılanını değiştirmez.

128 COCO val2017 görüntüsü, sekiz kategori başına 16 benzersiz sahne olacak
şekilde deterministik seçilir. Daha önce kullanılan 64 değerlendirme sahnesi
seçimden çıkarılır; hazırlayıcı byte hash ve çözülmüş piksel eşitliğini de reddeder.
Kalibrasyon sırası sabittir ve seçime model tahminleri katılmaz. Batch 1 statik
engine ile eşleşir. Bu veri gerçek üretim kameralarını veya 80 sınıfın tamamını
temsil ettiği iddiası taşımaz; ön eğitim verisinden bağımsızlık da iddia edilmez.

Kalibrasyon cache'i yeniden kullanılmaz: her build yeniden kalibre edilir.
Callback hataları `noexcept` sınırında yakalanır ve build sonrasında yeniden
fırlatılır; erken hata normal veri sonu sayılmaz. 128 batch'in tamamlanması ve
cache oluşması zorunludur. Model, veri ve kalibrasyon manifesti mutasyonu
reddedilir. Engine bundle'ında kalibrasyon manifesti/cache ve hash kayıtları
saklanır. Güvenilmeyen engine çalıştırmak bu kontrollerle güvenli hale gelmez.

## Ölçümden önce sabitlenen karar

Aynı model, ayarlar ve 64 sahnede ardışık FP16/INT8 koşuları; beş ısınma,
tek OpenCV thread. Native AP ayrıca resmi pycocotools ile doğrulanır.

- AP50 ve AP50:95 kaybı en çok 0,005 mutlak AP (0,5 yüzde puan).
- Sınıf başına AP50:95 kaybı en çok 0,02 (2 yüzde puan).
- FP16'ya göre p50 hızlanması en az 1,10×; p95 kötüleşmemeli.
- Kapı geçilse bile otomatik deployment yok; başarısızlık saklanmaz ve bu
  görülen test kümesine göre eşik/kalibrasyon seçimi değiştirilmez.

## Sürüm sınırı

Bu bilinçli olarak mevcut SDK'ya bağlı bir uygulamadır. NVIDIA'nın
[10.x quantization belgesi](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)
implicit kalibrasyonu ve FP32 kalibrasyon girdisini açıklar. API deprecated'dır;
[TensorRT 11 geçişi](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-c-api-patterns.html)
bu API'yi kaldırır. SDK 11 desteği için explicit Q/DQ model üretimi ve ayrı kalite
doğrulaması gerekir; bu değişiklikle 11 desteği sağlandığı iddia edilmez.
