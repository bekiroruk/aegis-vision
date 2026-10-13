# İsteğe bağlı native C++ TensorRT detection

Bu backend mevcut CPU servisini değiştirmez. `TensorRtYoloDetector`, aynı
`IDetector` arayüzünü ve CPU YOLO adaptöründeki letterbox/NMS kodunu kullanır.
İlk kapsam **YOLOv8n detection**, statik 640×640, batch 1, 80 COCO sınıfıdır.
Segmentation, CLIP/OCR/VLM GPU dönüşümü, INT8, web panelinde backend seçimi ve
GPU Docker bu adımın kapsamında tamamlanmış sayılmaz.

TensorRT 10.13.3.9 / CUDA 13.0 ile yerel engine oluşturulur. FP32 sürümünde
TF32 kapalıdır. FP16 seçeneği, TensorRT'ye FP16 katman kullanma izni verir;
her katmanın FP16 olduğu iddia edilmez. Model giriş/çıkışı FP32, görüntü
hazırlığı ve NMS CPU'dadır. Başka bir backend'e sessiz geri dönüş yoktur.

## Doğrulanmış sonuç — 2026-10-13

RTX 3060 Laptop 6 GB üzerinde iki engine gerçekten üretildi ve native C++ ile
64 etiketli görüntüde çalıştırıldı. Aşağıdaki süreler beş ısınma çağrısından
sonraki tek yerel koşuyu gösterir; üç backend sırayla ölçüldü.

| Backend | Bbox AP50:95 | p50 | p95 | Ortalama detector kapasitesi |
|---|---:|---:|---:|---:|
| OpenCV CPU FP32, 1 thread | %63,6516 | 478,22 ms | 597,95 ms | 2,10 görüntü/s |
| TensorRT GPU FP32 | %63,6516 | 11,94 ms | 22,57 ms | 71,67 görüntü/s |
| TensorRT GPU FP16-mixed | %63,5736 | 13,00 ms | 18,55 ms | 72,73 görüntü/s |

FP16, CPU referansına göre p50'de **36,78×** hızlı; AP50:95 kaybı **0,0780
yüzde puan**. En büyük sınıf kaybı giraffe için 1,4384 puan; sabit 2 puanlık
sınıf kapısı içinde. AP50 %83,0199 → %83,1466. Operating score 0,25 / IoU 0,50'de
TP/FP/FN her üçünde 105/16/54. COCO'nun ignore ettiği tahmin sayısı FP16'da 5→6.

**FP16'nın GPU FP32'ye göre ayrıca hız kazandırdığı kanıtlanmadı:** p50 biraz
daha kötü, ortalama benzer, p95 daha iyi. Bu tek koşudan kararlı sıralama veya
kernel hızlanması çıkarılmaz. CPU ön/son işleme, aktarım ve ölçüm değişkenliği
ayrı profilleme gerektirir. Her iki GPU seçeneği de CPU'ya karşı yerel hız/kalite
kapısını geçti; servis otomatik değiştirilmedi.

Üç native AP sonucu resmi pycocotools ile `1e-6` toleransında doğrulandı.
4 TensorRT CTest (gerçek GPU tekrar çalıştırma ve bozuk buffer dahil), 8
karşılaştırma testi, 26 mevcut referans testi geçti. Mevcut CPU build'indeki
46/46 CTest ve web testleri de geçti. Uzak CI/Linux GPU sonucu iddia edilmiyor.

FP32 engine hazırlığı 134,58 saniye, FP16 hazırlığı 447,74 saniye sürdü; bunlar
yukarıdaki inference sürelerine dahil değil. FP16 engine inspector raporunda
128 Half ve 26 Float **katman çıkış tanımı** var; saf FP16 ağ iddiası yoktur.
Derleme CUDA Toolkit 13.0.88 ile yapıldı. NVIDIA sürücüsü 616.92; çalışan
hibrit CUDA runtime/driver API'si 13040 (13.4) bildirdi ve engine manifestine
bu gerçek değer kaydedildi. SDK build sürümü 10.13.3.9'dur.

Yerel sonuçlar: `outputs/trt-cpu-v1`, `outputs/trt-fp32-v1`, `outputs/trt-fp16-v1`;
her birinde rapor, ham tahminler, süre CSV'si, önizleme ve resmi doğrulama kaydı.
Toplu kayıt `outputs/trt-comparison-v1.json`; Git'teki küçük
[deney kaydı](experiments/tensorrt-fp16-v1.json) model/engine/veri/sonuç hash'lerini taşır.

## Yerel SDK ve derleme

SDK kaynak bağlantısı [NVIDIA'nın v10.13.3 sürüm kaydındadır](https://github.com/NVIDIA/TensorRT/tree/v10.13.3).
Windows/CUDA13 ZIP boyutu 1.546.175.955 byte; bu çalışmada alınan SHA256:
`10ada15b14498e0ec8556cb19433129fd362a8e3debf75db06ccacd8659d9662`.
Bu, indirilen dosyanın kaydıdır; NVIDIA tarafından ayrıca yayımlanmış imza iddiası
değildir. SDK/model dosyaları Git'e girmez ve yeniden dağıtılmaz. Lisansları ayrıca
korunmalıdır; sistem CUDA kurulumu veya global PATH değiştirilmez.

Bu bilgisayarda SDK `artifacts/deps/tensorrt/sdk/TensorRT-10.13.3.9`, CUDA ise
`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0` konumundadır.
Başka bilgisayarda aynı ZIP resmi kaynaktan indirilip hash doğrulandıktan sonra
SDK çıkarılmalı; derleyici, CUDA, OpenCV ve mevcut arama bağımlılıkları hazır olmalıdır.
Bu komutlar sıfır makine için eksiksiz bağımlılık kurucusu değildir.

Repo kökünde PowerShell:

```powershell
$repo = (Get-Location).Path
cmake -S . -B build/tensorrt -G "Visual Studio 17 2022" -A x64 `
  -DAEGISVISION_WITH_OPENCV=ON -DAEGISVISION_WITH_SEARCH=ON -DAEGISVISION_WITH_TENSORRT=ON `
  "-DOpenCV_DIR=C:/opencv/opencv/build" `
  "-DTENSORRT_ROOT=$repo/artifacts/deps/tensorrt/sdk/TensorRT-10.13.3.9" `
  "-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0" `
  "-DONNXRUNTIME_ROOT=$repo/artifacts/deps/ort/onnxruntime-win-x64-1.23.2" `
  "-DICU_ROOT=$repo/artifacts/deps/icu"
cmake --build build/tensorrt --config Release --target aegisvision_trt_build aegisvision_trt_quality aegisvision_tensorrt_tests aegisvision_quality_cli --parallel 4
```

Backend/builder için `WITH_SEARCH` gerekmez; mevcut kalite değerlendirmesini
kullanan `aegisvision_trt_quality` hedefi arama/kalite bileşenleri açıkken eklenir.
TensorRT seçeneği kapalıyken CUDA/SDK gerekmez. Linux için CMake yolları vardır,
ancak bu ilk GPU uygulaması Windows'ta doğrulanmaktadır; Linux GPU doğrulaması
yapılmadan tamamlandı denmez.

## Engine oluşturma ve ölçüm

Engine klasörleri yeni olmalıdır. Yerel GPU için engine üretimi birkaç dakika
sürebilir; bu süre inference hızına dahil edilmez. Workspace sınırı 512 MiB'dir;
**toplam GPU bellek sınırı değildir**.

```powershell
build/tensorrt/Release/aegisvision_trt_build.exe artifacts/models/yolov8n.onnx artifacts/models/yolov8n-trt-fp32-v1 fp32
build/tensorrt/Release/aegisvision_trt_build.exe artifacts/models/yolov8n.onnx artifacts/models/yolov8n-trt-fp16-v1 fp16
build/tensorrt/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/coco-search/quality-manifest.json outputs/trt-cpu-v1
build/tensorrt/Release/aegisvision_trt_quality.exe artifacts/models/yolov8n-trt-fp32-v1 configs/evaluation.toml artifacts/datasets/coco-search/quality-manifest.json outputs/trt-fp32-v1
build/tensorrt/Release/aegisvision_trt_quality.exe artifacts/models/yolov8n-trt-fp16-v1 configs/evaluation.toml artifacts/datasets/coco-search/quality-manifest.json outputs/trt-fp16-v1
```

İki GPU koşusu CPU referansının ONNX hash'ini ve ayarlarını zorunlu olarak
eşler. Aynı 64 COCO sahnesi/seçili sekiz sınıf, confidence 0,001, NMS 0,45,
max detections 300, tek OpenCV thread kullanılır. Beş ısınma çağrısı ölçüm dışıdır.
CPU ve GPU koşuları arka arkaya çalıştırılmalı, aynı anda benchmark yapılmamalıdır.

`detector` süresi görüntü kopyalama, ön işleme, CPU→GPU aktarım, GPU çalıştırma,
GPU→CPU aktarım, CUDA senkronizasyonu ve NMS'yi kapsar. Yalnız kernel süresi diye
sunulmaz. `1000 / ortalama_ms` tek akış detector kapasitesidir, kamera uçtan uca
FPS'i değildir. CPU referansı tek thread'dir; tüm çekirdekler için optimize edilmiş
CPU'ya karşı hız iddiası değildir. GPU bellek alanları cihaz genelinde yükleme
öncesi/sonrası boş bellek ölçümüdür; izole süreç peak VRAM'i değildir.

Resmi COCO kontrolü ve sabit karar kapısı:

```powershell
foreach ($run in @('trt-cpu-v1','trt-fp32-v1','trt-fp16-v1')) {
  work/yolo-export/Scripts/python.exe scripts/verify_quality_reference.py coco artifacts/datasets/coco-search/quality-manifest.json "outputs/$run" artifacts/datasets/coco-search/annotations/instances_val2017.json
}
work/yolo-export/Scripts/python.exe scripts/compare_tensorrt.py outputs/trt-cpu-v1 outputs/trt-fp32-v1 outputs/trt-fp16-v1 outputs/trt-comparison-v1.json
work/yolo-export/Scripts/python.exe scripts/test_tensorrt_comparison.py
ctest --test-dir build/tensorrt -C Release -R 'tensorrt|trt_build' --output-on-failure
```

Gerçek GPU CTest'ini etkinleştirmek için configure sırasında
`-DAEGISVISION_TRT_BUNDLE=<mutlak-engine-klasörü>` ve
`-DAEGISVISION_YOLO_IMAGE=<mutlak-bus.jpg-yolu>` verilir. Bu fixture'lar yokken
yalnız GPU'suz giriş sözleşmesi/help kontrolleri çalışır; inference testi geçmiş sayılmaz.

Ölçümden önce sabitlenen sınırlar: CPU'ya göre AP50 ve AP50:95 kaybı en çok
0,5 yüzde puan; sınıf başına AP50:95 kaybı en çok 2 puan; p50 en az 2× hızlı ve
p95 CPU'dan kötü olmamalı. Karşılaştırma, resmi referans kontrolünün rapor/tahmin
hash'lerini doğrular. Geçen sonuç bile otomatik deployment yapmaz.
64 sahne daha önce gözlenmiş geliştirme verisidir; kör test veya genel COCO
garantisi değildir. INT8 kalibrasyonu için bu sonuçları kullanarak eşik seçilmez.

## Kaynak ve hata yönetimi

- Her worker kendi context, CUDA stream ve I/O tamponlarını tutar. `detect` senkron
  döner; aynı instance paralel thread'lerde kullanılmaz. İlk sürüm device 0 içindir.
- Çerçeve boyutu/stride, statik tensor boyutları, veri tipi, engine hash'i,
  sınıf sıralaması, GPU modeli/compute capability ve SDK/runtime sürümü denetlenir.
- CUDA hatasından sonra worker tekrar kullanılmaz. Hata yolunda da aktarım
  bitmeden host tamponları serbest bırakılmaz; kaynaklar RAII ile kapanır.
- Yalnız güvenilir yerel ONNX'ten üretilmiş engine yükleyin. TensorRT planları
  güvenilmeyen veri gibi güvenle çalıştırılamaz; hash kontrolü imza veya sandbox değildir.
- Engine başka GPU/SDK'ya taşınınca yeniden üretilmelidir. Manifest son yazılır;
  yarım engine klasörü geçerli sayılmaz. CPU ve açık servis varsayılanları korunur.

Kaynak: [resmi TensorRT Windows ZIP kurulumu](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/install-zip.html),
[10.13.3 sürüm notları](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html).
