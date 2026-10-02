# AegisVision

AegisVision, gerçek zamanlı video analizi ve multimodal nesne araması için geliştirilen
modüler bir **C++20 computer vision platformudur**. Hedefi; detection, tracking, Re-ID,
OCR/VLM, geometrik eşleştirme ve GPU inference bileşenlerini aynı üretim odaklı pipeline
içinde birleştirmektir.

> Durum: C++ çekirdeği, OpenCV görüntü/hizalama, YOLOv8 tespiti, video takibi,
> CLIP + yerel Qdrant araması ve TOML konfigürasyonu hazır. İlk Python fikir doğrulaması `legacy/python` altında
> korunmaktadır; aktif geliştirme C++ tarafındadır.

## Mevcut özellikler

- Tür güvenli detection, track, frame ve search veri modelleri
- Değiştirilebilir model ve altyapı portları
- Sınıf duyarlı IoU tracker
- İsteğe bağlı iki aşamalı tracker: düşük güvenli tespitle takip sürdürme, doğrusal hareket tahmini ve Hungarian eşleştirme
- Deneysel yerel-video Kalman tracker: aktif-önce eşleştirme, kovaryans/Mahalanobis kapısı, sınırlı state ve hata sayaçları; iki sahnede karışık kalite sonuçları
- Cosine similarity tabanlı yerel vektör indeksi
- ONNX Runtime ile CLIP görsel/metin embedding, Unicode BPE tokenizer ve Qdrant'ta kalıcı arama
- COCO validation nesne kırpmalarını toplu indeksleme ve Recall@K/Hit@K raporu
- Klasör görsellerini ve örneklenmiş video karelerindeki YOLO nesnelerini CLIP/Qdrant ile indeksleme
- TOML dosyasıyla görüntü, video ve arama uygulamalarının model, takip ve servis ayarları
- Homografi ile nokta dönüşümü
- OpenCV ile dosyadan görsel yükleme, kırpma ve verilen kutuları çizme
- C++/OpenCV DNN ile YOLOv8 ONNX nesne tespiti, kutulu görsel ve JSON/TSV çıktısı
- Videoda YOLO + IoU veya iki aşamalı takip, ID etiketli AVI ve kare bazlı CSV raporu
- Etiketli tam görüntülerde C++ bbox AP; gerçek yaya videosunda CLEAR/IDF1,
  aynı tespitlerle iki tracker karşılaştırması ve bağımsız referans doğrulaması
- RTSP/FFmpeg canlı kaynak: timeout, reconnect, sınırlı drop-oldest kuyruğu ve oturum bazlı takip
- Tarayıcıdan süre sınırlı canlı analiz başlatma/durdurma, kutulu JPEG önizleme ve kesinti sayaçları
- İsteğe bağlı canlı analiz arşivi: kısa MP4 parçaları, disk/parça kotası, SQLite ile
  sonlu indeksleme ve CLIP/Qdrant üzerinden canlı kayıtta arama/klip anına atlama
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

Bu şema hedef mimaridir. `YoloDetector` gerçek piksel tamponu üzerinden inference yapar;
eski demo detector önceden verilen kutuları okur. Hash embedding yalnızca test içindir
ve anlamsal arama sağlamaz. OpenCV ayrı bir adaptör katmanıdır. C++ uygulamaları
`configs/*.toml` dosyalarını okuyabilir; RTSP kayıt/analiz CLI'ı hazırdır, OCR ve Redis entegrasyonları planlanmıştır.

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
2. [Tamamlandı] OpenCV DNN/ONNX ile YOLOv8 detector; ONNX Runtime alternatif backend olarak planlandı
3. [Tamamlandı] Yerel video pipeline'ı ve iki aşamalı tracking başlangıcı (tam ByteTrack değil)
4. [Tamamlandı: başlangıç sürümü] CLIP görsel/metin embedding ve yerel Qdrant arama; geniş ölçekli kalite değerlendirmesi bekliyor
5. [Konfigürasyon tamamlandı] Segmentation ve OCR/VLM adaptörleri
6. [Başlangıç ölçümü tamamlandı] Etiketli AP/CLEAR/IDF1 ve CPU p50/p95;
   geniş bağımsız test kümesi, HOTA, tam ByteTrack / Re-ID ve çoklu kamera bekliyor
7. [Yerel servis, RTSP CLI, canlı ekran ve sınırlı arşiv tamamlandı] C++ HTTP API,
   SQLite kurtarma, canlı/video arama, önizleme ve RTSP reconnect/backpressure;
   kamera PTS, tam FPS/sesli canlı kayıt ve Redis bekliyor
8. TensorRT FP16/INT8, GPU Docker ve Triton/Jetson deployment

Öncelikler, eksikler ve her adımın kabul ölçütleri: [ayrıntılı yol haritası](docs/roadmap.md).

Başarı metrikleri model ve sistem seviyesinde birlikte izlenecek: mAP, IDF1,
Recall@K, FPS, p50/p95 gecikme ve GPU bellek kullanımı.

## Gerçek görsellerle kullanım

**Çalışan video arama ekranı:** Yerel C++ servisini başlatma, gerçek kamera kaydını
indeksleme ve sonuca tıklayarak videonun ilgili anına gitme için [servis kılavuzu](docs/service.md).

**TOML ile çalıştırma:** Örnek dosyalar, doğrulama ve üç uygulamanın komutları için
[konfigürasyon kılavuzu](docs/configuration.md).

**Metin/görsel araması:** Derleme, model export, Qdrant başlatma, görsel/kırpma indeksleme
ve sorgu komutları için [CLIP + Qdrant kılavuzu](docs/search.md).

**Arama kalitesi:** 64 COCO validation kırpmasıyla tekrarlanabilir toplu indeksleme
ve Recall@K ölçümü için [benchmark kılavuzu](docs/search-benchmark.md).

**Model ve takip kalitesi:** 64 tam COCO sahnesinde detection AP ve 179 karelik
etiketli MOT15 videosunda başlangıç ölçümü; ek 71 karelik Campus ile IoU / iki aşamalı /
deneysel Kalman karşılaştırması, gerçek video
çıktısı, CPU gecikmesi ve resmi araçlarla kontrol için [kalite kılavuzu](docs/model-quality.md).
Küçük/yanlı başlangıç verisi resmi COCO/MOT leaderboard sonucu değildir.

**Klasör/video indeksleme:** Komutlar, tekrar çalıştırma ve kare/zaman metadata'sı
için [indeksleme kılavuzu](docs/indexing.md).

**Otomatik nesne tespiti:** Model hazırlama, çalıştırma ve çıktı açıklamaları için
[YOLO kılavuzu](docs/yolo.md).

**Video takibi:** Yerel videoda tespit, ID atama ve sonuç kaydetme için
[video kılavuzu](docs/video.md). Varsayılan IoU; `--tracker two-stage` ile hareket tahminli iki aşamalı takip seçilir. Tam ByteTrack/Re-ID henüz yoktur.

**Canlı RTSP:** Tarayıcıda canlı analiz için `./scripts/start_service.ps1 -LiveUrl rtsp://127.0.0.1:8554/pedestrians`.
Ekranda **Kaydet ve arşivde ara** ile yalnızca analiz edilen karelerden sınırlı MP4
parçaları üretip **Yalnızca canlı arşiv** kapsamında arayabilirsiniz. Klip zamanı
kamera zamanı değildir; kota dolunca arşiv durur ve önizleme devam eder. Önce yerel
yayın kurulmalıdır; üç terminalle kurulum, kayıt komutu ve gerçek yaya
videosuyla kesinti testi için [canlı video kılavuzu](docs/live-video.md).

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
bu, gerçek dünya doğruluk ölçümü değildir. Windows/Linux GitHub CI doğrulandı;
Docker/GPU deployment henüz doğrulanmadı. CI model indirmez; gerçek YOLO smoke testi yerel modelle ayrıca çalışır.

Yöntem referansları: [OpenCV özellik eşleştirme](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html),
[OpenCV homografi](https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html).

