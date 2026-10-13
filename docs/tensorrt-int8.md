# INT8 kalibrasyonu — native C++ YOLO

Bu adım, [FP16 backend'ini](tensorrt.md) INT8-mixed engine üretimiyle genişletir.
Windows / RTX 3060 / TensorRT 10.13.3.9 kapsamındadır. Panel/servis varsayılanı
değişmez. Segmentation INT8 veya GPU Docker doğrulaması değildir.

## Gerçek sonuç — 2026-10-13

**INT8 engine üretildi ve çalıştı; kalite ve hız kabul kapılarını geçemedi.**
FP16 adayı korunur. Bu ölçümde INT8'e geçiş önerilmiyor.

| Ölçüm | FP16-mixed | INT8-mixed |
|---|---:|---:|
| Bbox AP50:95 | %63,5736 | %58,0437 |
| Bbox AP50 | %83,1466 | %79,7311 |
| p50 detector süresi | 12,0393 ms | 11,7769 ms |
| p95 detector süresi | 31,5094 ms | 17,7208 ms |
| Ortalama süre | 15,0516 ms | 12,5365 ms |
| TP / FP / FN, score 0,25 ve IoU 0,50 | 105 / 16 / 54 | 96 / 6 / 63 |

AP50:95 kaybı **5,5300 yüzde puan**; izin verilen 0,5 puandan fazla.
En büyük sınıf kaybı zebra'da 12,2339 puan. p50 hızlanması **1,0223×**;
istenen 1,10× altında. p95 ve ortalama iyileşmiş olsa da tek ardışık koşu
kararlı bir hız üstünlüğü kanıtlamaz; kalite kaybı zaten tek başına reddetme
sebebidir. Bu sonuç genel olarak tüm INT8 yöntemlerinin başarısız olduğu
anlamına gelmez. Farklı veri/QDQ/QAT ayrı geliştirme ve görülmemiş test gerektirir;
bu 64 sahneye göre kalibrasyon veya eşik ayarı yapılmadı.

Native C++ AP, iki backend için resmi pycocotools ile `1e-6` içinde eşleşti.
Engine inspector 97 Int8, 20 Half, 24 Float **tensor çıkış tanımı** gösteriyor;
saf INT8 ağ veya 97 benzersiz layer iddiası değil. Engine hazırlığı **805,33
saniye** sürdü; inference sürelerinden ayrıdır. Kalibrasyonda 128 görüntünün
tamamı kullanıldı; testle çözülmüş piksel çakışması tekrar kontrolde de sıfırdı.

Yerel ölçümler: `outputs/trt-fp16-int8-baseline-v1`, `outputs/trt-int8-v1`;
her klasörde tahminler, süreler, rapor, önizleme ve resmi doğrulama kaydı var.
[Git deney kaydı](experiments/tensorrt-int8-v1.json) model/engine/kalibrasyon/veri
hash'lerini, kapı sonuçlarını ve sınıf farklarını içerir. FP16'nın buradaki
süreleri yeni koşuya aittir; ilk FP16 deneyinin süreleriyle karıştırılmamalı.

Doğrulama: 11 INT8 seçim/kapı testi, 8 mevcut FP16 karşılaştırma testi,
26 resmi referans girdi testi, mevcut CPU build'inde 46/46 CTest ve web
durum testleri geçti. Dört TensorRT CTest'e ek olarak gerçek INT8 engine
üzerinde tekrar çalıştırma, bozuk buffer ve değiştirilmiş kalibrasyon cache'i
kontrolleri geçti. Yanlış değerlendirme manifesti çıktı oluşturmadan reddedildi.
Uzak CI veya Linux GPU çalışması bu yerel sonuçların parçası değildir.

## Tekrar üretim

[SDK/derleme adımları](tensorrt.md#yerel-sdk-ve-derleme) tamamlanmış olmalı.
COCO annotation/evaluation manifesti ve YOLO ONNX önceki kalite adımından gelir.
Hazırlayıcı Python ortamında OpenCV ve NumPy kullanır; inference, kalibrasyon
callback'i ve engine oluşturma C++'tadır. Yeni klasör adları kullanın; mevcut
engine veya ölçümler üzerine yazılmaz.

```powershell
work/yolo-export/Scripts/python.exe scripts/prepare_int8_calibration.py artifacts/datasets/coco-search/annotations/instances_val2017.json artifacts/datasets/coco-search/quality-manifest.json artifacts/models/yolov8n.onnx artifacts/datasets/coco-int8-v2
cmake --build build/tensorrt --config Release --target aegisvision_trt_build aegisvision_trt_quality aegisvision_tensorrt_tests --parallel 4
build/tensorrt/Release/aegisvision_trt_build.exe artifacts/models/yolov8n.onnx artifacts/models/yolov8n-trt-int8-v1 int8 artifacts/datasets/coco-int8-v2/manifest.json
build/tensorrt/Release/aegisvision_trt_quality.exe artifacts/models/yolov8n-trt-fp16-v1 configs/evaluation.toml artifacts/datasets/coco-search/quality-manifest.json outputs/trt-fp16-int8-baseline-v1
build/tensorrt/Release/aegisvision_trt_quality.exe artifacts/models/yolov8n-trt-int8-v1 configs/evaluation.toml artifacts/datasets/coco-search/quality-manifest.json outputs/trt-int8-v1
foreach ($run in @('trt-fp16-int8-baseline-v1','trt-int8-v1')) {
  work/yolo-export/Scripts/python.exe scripts/verify_quality_reference.py coco artifacts/datasets/coco-search/quality-manifest.json "outputs/$run" artifacts/datasets/coco-search/annotations/instances_val2017.json
}
work/yolo-export/Scripts/python.exe scripts/compare_int8.py outputs/trt-fp16-int8-baseline-v1 outputs/trt-int8-v1 artifacts/models/yolov8n-trt-int8-v1 outputs/trt-int8-comparison-v1.json
work/yolo-export/Scripts/python.exe scripts/test_int8.py
build/tensorrt/Release/aegisvision_tensorrt_tests.exe artifacts/models/yolov8n-trt-int8-v1 artifacts/samples/bus.jpg
```

128 sahne, seçili sekiz kategori başına 16 benzersiz görüntü. Kalibrasyon/test
ayrımı ID, byte ve çözülmüş piksel eşitliğiyle hazırlıkta kontrol edilir.
Kaynak resmi COCO S3 bucket'ıdır; HTTPS sertifika kontrolü açıktır. İndirme
başarısız olduğunda yarım klasör tamamlandı sayılmaz; manifest son yazılır.

İki ölçüm GPU build bittikten sonra sırayla yapılır. Süre; CPU letterbox/NMS,
GPU'ya aktarım, GPU inference, geri aktarım ve senkronizasyonu içerir. Kamera FPS'i
değildir. Beş ısınma, aynı confidence/NMS, tek OpenCV thread kullanılır.

## Karar ve sınırlar

[ADR-0019](adr/0019-int8-calibration.md): Entropy2 / batch 1, her build için
taze kalibrasyon. Saf INT8 iddiası yoktur: TensorRT gerektiğinde FP16/FP32
kullanabilir. Inspector'da gerçek INT8 tensor çıkışları ayrıca kontrol edilir.
Kalibrasyon manifesti/cache/hash'leri engine bundle'ında saklanır; yükleyici
mutasyonu reddeder. Doğruluk CLI'ı değerlendirme manifestinin kalibrasyonda
hariç tutulan veriyle aynı olmasını zorunlu kılar.

Ölçümden önce sabit kapı: FP16'ya göre AP50/AP50:95 kaybı en çok **0,5 yüzde
puan**, sınıf kaybı en çok **2 puan**, p50 hızlanması en az **1,10×**, p95
kötüleşmemeli. `compare_int8.py` başarısız kalite kapısını rapora yazar;
rapor üretilebilmesi kabul kapısının geçildiği anlamına gelmez. Rapor veya
kalibrasyon kanıtı uyuşmazlığı ise hata ile durdurur. Otomatik deployment yok.

64 değerlendirme sahnesi daha önce gözlenmiştir; kör holdout değildir.
COCO ön eğitiminden bağımsızlık ve üretim kameralarında genelleme iddiası yoktur.
TensorRT 11 eski kalibrasyon API'sini kaldırır; 11 için explicit Q/DQ geçişi
ayrı iştir. SDK sürümünü kontrolsüz yükseltmeyin.
