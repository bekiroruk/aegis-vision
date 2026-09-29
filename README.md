# AegisVision

AegisVision, gerçek zamanlı video analizi ve multimodal nesne araması için geliştirilen
modüler bir **C++20 computer vision platformudur**. Hedefi; detection, tracking, Re-ID,
OCR/VLM, geometrik eşleştirme ve GPU inference bileşenlerini aynı üretim odaklı pipeline
içinde birleştirmektir.

> Durum: C++ çekirdeği ve OpenCV görüntü/hizalama aracı hazır. Gerçek YOLO/TensorRT ve Qdrant adaptörleri sonraki
> kilometre taşlarında eklenecek. İlk Python fikir doğrulaması `legacy/python` altında
> korunmaktadır; aktif geliştirme C++ tarafındadır.

## Mevcut özellikler

- Tür güvenli detection, track, frame ve search veri modelleri
- Değiştirilebilir model ve altyapı portları
- Sınıf duyarlı IoU tracker
- Cosine similarity tabanlı yerel vektör indeksi
- Homografi ile nokta dönüşümü
- OpenCV ile dosyadan görsel yükleme, kırpma ve verilen kutuları çizme
- ORB + Hamming eşleştirme, oran filtresi ve RANSAC ile kaynak → hedef homografisi
- Hizalanmış görsel, eşleşme görselleştirmesi ve sayısal hata raporu
- Detection → tracking → embedding → indexing pipeline'ı
- Harici bağımlılığı olmayan deterministik demo adaptörleri
- CMake/CTest, Windows ve Linux CI
- Redis ve Qdrant için Docker Compose altyapısı

Altyapı servislerini yerelde başlatmak için `docker compose up -d redis qdrant`,
container içindeki demoyu çalıştırmak için `docker compose --profile demo run --rm demo`
komutunu kullanabilirsiniz.

## Derleme

Visual Studio 2022 ile:

```powershell
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
./build/dev/Debug/aegisvision_demo.exe
```

Linux/macOS için:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/aegisvision_demo
```

## Mimari

Bu şema hedef mimaridir. Mevcut detector önceden verilen kutuları okur; hash embedding
yalnızca test içindir ve anlamsal arama sağlamaz. C++ `Frame` henüz gerçek piksel taşımaz;
OpenCV aracı ayrı bir adaptör katmanıdır. `configs/pipeline.toml` referans taslağıdır,
C++ programı henüz TOML okumaz. OCR, RTSP ve harici servis entegrasyonları planlanmıştır.

```text
RTSP / Image
     │
     ▼
Detector ──► Tracker / Re-ID ──► Embedder ──► Vector Store
     │                                │
     ├────────► OCR / VLM             └──────► Text & image search
     └────────► Geometry / Homography
```

Pipeline yalnızca arayüzlere bağımlıdır. YOLO/DETR, DINO/CLIP, TensorRT, Qdrant ve
PaddleOCR gibi teknoloji seçimleri adaptör olarak eklenir; çekirdek iş akışı değişmez.

## Yol haritası

1. [Tamamlandı] OpenCV görüntü yükleme, çizim ve feature matching
2. ONNX Runtime ile YOLO detector adaptörü
3. DINOv2/CLIP embedding ve Qdrant istemcisi
4. ByteTrack ve multi-camera Re-ID
5. TensorRT FP16/INT8 benchmark
6. RTSP/GStreamer ingest ve asenkron servis katmanı
7. Triton/Jetson deployment profilleri

Başarı metrikleri model ve sistem seviyesinde birlikte izlenecek: mAP, IDF1,
Recall@K, FPS, p50/p95 gecikme ve GPU bellek kullanımı.

## Gerçek görsellerle kullanım

OpenCV araçları isteğe bağlıdır; varsayılan çekirdek derlemesi harici bağımlılık istemez.
OpenCV 4 geliştirme paketi (C++ başlıkları, kütüphaneler ve Windows DLL'leri) gerekir.
Windows'ta OpenCV yolunu kendi kurulumunuza göre belirtin:

```powershell
cmake -S . -B build/opencv -G "Visual Studio 17 2022" -A x64 -DAEGISVISION_WITH_OPENCV=ON -DOpenCV_DIR="C:/opencv/opencv/build"
cmake --build build/opencv --config Release
ctest --test-dir build/opencv -C Release --output-on-failure
./build/opencv/Release/aegisvision_image.exe demo outputs/geometry-demo
./build/opencv/Release/aegisvision_image.exe inspect outputs/geometry-demo/source.png
./build/opencv/Release/aegisvision_image.exe annotate outputs/geometry-demo/source.png examples/boxes.tsv outputs/geometry-demo/annotated-from-file.png
./build/opencv/Release/aegisvision_image.exe align outputs/geometry-demo/source.png outputs/geometry-demo/target.png outputs/alignment
```

Bu bilgisayarda ayrıca Git'e eklenmeyen `CMakeUserPresets.json` oluşturulmuştur:
`cmake --preset opencv-local`, `cmake --build --preset opencv-local`, `ctest --preset opencv-local`.
Windows derlemesi gereken DLL'leri çalıştırılabilir dosyaların yanına kopyalar.
Linux'ta `libopencv-dev` kurulduktan sonra aynı CMake seçeneğiyle derleyebilirsiniz;
araç yolu `build/opencv/aegisvision_image` olur.

`align SOURCE TARGET OUT` için SOURCE ve TARGET yerine kendi fotoğraflarınızı verin.
Çıktı klasöründe `aligned.png`, `overlay.png`, `matches.png` ve `report.yml` oluşur.
Rapor homografi yönünü, aday/inlier sayısını, inlier oranını ve piksel cinsinden
yeniden izdüşüm RMSE'sini içerir. Görselleştirmede en fazla 80 inlier çizilir.
Aynı çıktı klasörü yeniden kullanılırsa bu dosyalar yenilenir.

`annotate` kutu dosyasını okur; otomatik nesne tespiti yapmaz. Satırlar
`x1 y1 x2 y2 score label` biçimindedir; boşluk içeren etiketler çift tırnakla yazılır.
Kutular görüntü sınırlarına kırpılır; boş veya tümüyle dışarıdaki kutular reddedilir.
Metin çizimi ASCII etiketler için tasarlanmıştır; dosya yollarında Türkçe karakter desteklenir.

Hizalama düzlemsel yüzeyler ve yaklaşık saf kamera dönüşü içindir. Parallax, hareketli
nesneler, tekrarlayan dokular veya az ayrıntı içeren görsellerde güvenilir olmayabilir.
Yetersiz eşleşmede program hata ve sıfırdan farklı çıkış kodu verir. Sentetik testte
bilinen dönüşümün dört kontrol noktasındaki hatanın 2 piksel altında olması beklenir;
bu, gerçek dünya doğruluk ölçümü değildir. Docker ve GitHub CI bu yerel aşamada çalıştırılmadı.

Yöntem referansları: [OpenCV özellik eşleştirme](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html),
[OpenCV homografi](https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html).

