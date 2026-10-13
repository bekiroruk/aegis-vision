# ADR-0018: Yerel TensorRT engine ve paylaşılan YOLO ön/son işleme

Tarih: 2026-10-13. Durum: ilk isteğe bağlı backend uygulandı; servis varsayılanı değil.

## Bağlam

GPU'da PyTorch eğitimi, C++ uygulamasında GPU inference yapıldığını göstermez.
TensorRT FP16 kazanımını mevcut CPU referansına karşı ayrı olarak ölçmek gerekir.
GPU SDK bağımlılığı mevcut CPU build'lerini ve çalışan servisi etkilememelidir.

## Karar

- `AEGISVISION_WITH_TENSORRT` varsayılan kapalıdır. Ayrı CMake modülü, PIMPL
  kullanan `IDetector` adaptörü, yerel engine builder ve kalite CLI'ı eklenir.
- İlk sözleşme YOLOv8 detection, `[1,3,640,640] → [1,84,8400]`, FP32 I/O'dur.
  Mevcut `prepare_yolo` / `decode_yolo` paylaşılır. Segmentasyon bu sözleşmeye
  zorla sokulmaz. INT8 başka bir kalibrasyon/kalite adımıdır.
- Güvenilir ONNX'ten cihaz üzerinde FP32 veya FP16-mixed engine oluşturulur.
  TF32 kapatılır. Hash/sürüm/GPU kayıtları olmadan plan yüklenmez; bunlar bir
  güvenlik imzası değildir. Güvenilmeyen TensorRT planları çalıştırılmaz.
- Worker kendi CUDA stream/context/tamponlarını taşır; senkron `detect` dışarıya
  mevcut sözleşmeyi korur. CUDA hatasından sonra instance yeniden yaratılır.
- İlk deneyde aynı model/manifest/konfigürasyonla CPU FP32, TRT FP32 ve TRT FP16
  ölçülür. Eşikler ve veri sonuçlara göre değiştirilmez; resmi COCO kontrolü yapılır.
- Detaylı protokol ve sınırlar [GPU kılavuzunda](../tensorrt.md) tutulur.

## Sonuçlar ve sınırlar

İlk sürüm Python'sız C++ inference sağlar fakat preprocessing/NMS CPU'da kalır.
Süre bu işleri, kopyaları ve senkronizasyonu kapsar. GPU bellek ölçümü süreç peak'i
değildir. Engine taşınabilir bir ONNX'in yerine geçmez; farklı GPU/SDK'da yeniden
oluşturulur. Geliştirme alt kümesindeki başarılı karşılaştırma otomatik yayın,
web paneli entegrasyonu, Linux GPU veya GPU Docker doğrulaması sayılmaz.
