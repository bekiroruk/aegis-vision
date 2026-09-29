# AegisVision

AegisVision, gerçek zamanlı video analizi ve multimodal nesne araması için geliştirilen
modüler bir **C++20 computer vision platformudur**. Hedefi; detection, tracking, Re-ID,
OCR/VLM, geometrik eşleştirme ve GPU inference bileşenlerini aynı üretim odaklı pipeline
içinde birleştirmektir.

> Durum: Çalışan C++ çekirdeği hazır. Gerçek YOLO/TensorRT ve Qdrant adaptörleri sonraki
> kilometre taşlarında eklenecek. İlk Python fikir doğrulaması `legacy/python` altında
> korunmaktadır; aktif geliştirme C++ tarafındadır.

## Mevcut özellikler

- Tür güvenli detection, track, frame ve search veri modelleri
- Değiştirilebilir model ve altyapı portları
- Sınıf duyarlı IoU tracker
- Cosine similarity tabanlı yerel vektör indeksi
- Homografi ile nokta dönüşümü
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

1. OpenCV görüntü yükleme, çizim ve feature matching
2. ONNX Runtime ile YOLO detector adaptörü
3. DINOv2/CLIP embedding ve Qdrant istemcisi
4. ByteTrack ve multi-camera Re-ID
5. TensorRT FP16/INT8 benchmark
6. RTSP/GStreamer ingest ve asenkron servis katmanı
7. Triton/Jetson deployment profilleri

Başarı metrikleri model ve sistem seviyesinde birlikte izlenecek: mAP, IDF1,
Recall@K, FPS, p50/p95 gecikme ve GPU bellek kullanımı.

