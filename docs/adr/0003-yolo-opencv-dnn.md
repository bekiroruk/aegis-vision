# ADR-0003: İlk gerçek detector için OpenCV DNN

- Durum: Kabul edildi
- Tarih: 2026-09-30

Mevcut OpenCV 4.12 kurulumu ONNX okuyabilir. İlk YOLO detector CPU üzerinde OpenCV DNN
ile uygulanır; yol haritasındaki ONNX Runtime ayrı backend olarak ertelenir. TensorRT
ve GPU optimizasyonu bu aşamanın kapsamında değildir.

YOLOv8n'nin sabit 640×640, FP32, batch1, NMS'siz çıktısı açık sözleşme olarak seçildi.
Giriş hazırlama ve çıktı çözümleme ayrı fonksiyonlardır; model indirmeden test edilir.
Gerçek model isteğe bağlı entegrasyon testiyle kontrol edilir.

`Frame` artık `shared_ptr<const ImageBuffer>` tutabilir. Piksel verisi interleaved
8-bit BGR'dir, stride açıkça tanımlıdır. `image_frame` kopyalayarak sahipliği garanti
eder; çekirdek kütüphanede OpenCV türü yoktur. Detector tampon boyutunu doğrular.
Piksel içermeyen eski mock akışları çalışmaya devam eder.

Python yalnızca upstream modelin ONNX'e çevrilmesinde kullanılır. Uygulama ve inference
C++'tır. Büyük model dosyaları ve yerel örnek çıktıları Git dışında tutulur.
