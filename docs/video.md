# Video üzerinde YOLO ve nesne takibi

`aegisvision_video`, yerel bir video dosyasının her karesini YOLOv8 ile analiz eder,
sınıf duyarlı IoU, iki aşamalı veya aktif-önce Kalman takipçisiyle kimlik atar ve sonucu videoya çizer. Tek kaynak için
tek tracker kullanılır; videolar arasında takip durumu paylaşılmaz.

## Kullanım

OpenCV ve modeli hazırlamak için [YOLO kılavuzunu](yolo.md) izleyin.
TOML ile model/takip eşiklerini seçmek için [konfigürasyon kılavuzunu](configuration.md) izleyin.

```powershell
cmake --build --preset opencv-local
./build/opencv/Release/aegisvision_video.exe artifacts/models/yolov8n.onnx "C:/videolar/sahne.mp4" outputs/video-sonuc
```

İsteğe bağlı son argüman işlenecek en fazla kare sayısıdır:

```powershell
./build/opencv/Release/aegisvision_video.exe artifacts/models/yolov8n.onnx "C:/videolar/sahne.mp4" outputs/video-ilk60 60
```

Hareket tahmini ve düşük güvenli tespitleri kullanmak için:

```powershell
./build/opencv/Release/aegisvision_video.exe artifacts/models/yolov8n.onnx "C:/videolar/sahne.mp4" outputs/video-two-stage 60 --tracker two-stage
```

`--tracker iou` varsayılandır. Kare sınırı ve tracker seçeneği birbirinden bağımsızdır.

Çıktı klasörü yeni veya boş olmalıdır. Dosyalar:

- `tracked.avi`: kimlik ve sınıf etiketli MJPEG video; ses içermez.
- `preview.jpg`: ilk işlenen karenin görünümü.
- `tracks.csv`: sıfır tabanlı kare indeksi, zaman, track ID, sınıf, güven, kutu, yaş.
- `summary.json`: işlenen kareler, üretilen farklı ID sayısı, kaynak/işleme FPS, ortalama analiz süresi.

ID'ler her çağrıda 1'den başlar. Kaybolmuş bir track en fazla 20 boş kare tutulur;
bu arada çizilmez ve CSV'ye yazılmaz. Yeniden eşleşmek için aynı sınıf ve en az 0.30
IoU gerekir. Süresi dolan track geri geldiğinde yeni ID alır. IoU takipçisinde eşit
eşleşmelerde en küçük ID, eşit güvenlerde giriş sırası kullanılır.

## İki aşamalı takip

- YOLO güven filtresi 0.10'a indirilir (IoU modunda 0.35).
- Önce güveni en az 0.35 olan tespitler tüm mevcut track'lerle eşleştirilir.
- Ardından 0.10–0.35 arası tespitler yalnızca bir önceki karede aktif olup ilk aşamada
  eşleşmeyen track'leri sürdürebilir. Kayıp track'i geri getiremez, yeni ID başlatamaz.
- Eşleşmeyen güçlü tespit ancak güveni en az 0.50 ise yeni track başlatır.
- Her aşamada sınıf/IoU kapısı ve toplam IoU'yu maksimize eden Hungarian eşleştirme
  kullanılır. Amaç maksimum eşleşme adedi değil, maksimum toplam ağırlıktır.
- Merkez hareketi son ölçümden kestirilir; sonraki hız güncellemeleri %70 yeni,
  %30 önceki hızdır. Kayıp kare sayısı kadar ileri tahmin yapılır; kutu boyutu öngörülmez.

Sıralama deterministiktir (ID sırası ve kararlı güven sırası). Eşikler C++ API'sindeki
`VideoConfig` üzerinden değiştirilebilir; CLI henüz eşik seçenekleri sunmaz.
API'ye özel detector bağlarken detector filtresini `low_confidence` değerinin üstüne
çıkarmayın; aksi halde düşük güvenli eşleştirme adayları gelmez.
Rapor; düşük güvenli eşleşme, yeniden etkinleştirme, oluşturma ve silinme sayaçlarını içerir.

Bu tasarım [ByteTrack makalesindeki](https://arxiv.org/abs/2110.06864) düşük güvenli
tespitleri ikinci aşamada kullanma fikrinden esinlenir. **Tam ByteTrack uygulaması değildir:**
Kalman filtresi/kovaryansı, tentative track yaşam döngüsü, görünüş embedding'i ve Re-ID yoktur.

## Sınırlar ve ölçümler

### Deneysel, isteğe bağlı Kalman backend

```powershell
./build/search/Release/aegisvision_video.exe --config configs/video-kalman.toml artifacts/datasets/mot15-campus/TUD-Campus-raw.mp4 outputs/video-kalman
```

Konumsal komutta `--tracker kalman` de kullanılabilir. Aktif/yüksek güven,
aktif/düşük güven ve son olarak kayıp/kalan yüksek güven sırası, eski bir kayıp
kimliğin sürekli gözlenen track'e karışmasını azaltmayı amaçlar. Sekiz durumlu
kovaryanslı hareket tahmini, sınıf/IoU ve Mahalanobis kapısı kullanır. Yalnız
gözlenen kutular çizilir, kayıp track'in tahmin edilen kutusu sonuç yapılmaz.
256 state / 512 input detection sınırı ve sayısal reset/kapasite sayaçları vardır.
Varsayılan IoU ve eski iki aşamalı yöntem korunur; Re-ID veya resmi ByteTrack
değildir. RTSP'de bu backend kullanılamaz; hareket adımı bir decoded frame'dir.
[Tasarım kararı](adr/0013-active-first-kalman-tracking.md),
[etiketli karşılaştırma](model-quality.md).

İki gerçek sahnede sonuçlar karışıktır: Stadtmitte IDF1 %76.77 (IoU %77.20,
iki aşamalı %68.36), Campus %43.44 (IoU %59.92, iki aşamalı %68.90).
Bu nedenle genel kalite iyileştirmesi sayılmaz ve varsayılan yapılmaz.

### Merkez kapılı alternatif

```powershell
./build/search/Release/aegisvision_video.exe --config configs/video-kalman-center.toml artifacts/datasets/mot15-campus/TUD-Campus-raw.mp4 outputs/video-center
```

Konumsal karşılığı `--tracker kalman-center`'dır. Merkezin iki boyutlu marjinal
belirsizliğiyle eşleşme kapılanır; genişlik/yükseklik titreşimi Mahalanobis kararına
girmez. Tam dört ölçümlü Kalman correction, predicted-box IoU, sınıf ve yaşam
döngüsü korunur. Merkez kayması, örtüşme ve yakın kişilerde hatalar hâlâ mümkündür.
Eski Kalman seçeneği korunur; yeni seçenek deneysel ve yalnız yerel videodadır.
[Tasarım](adr/0014-center-only-motion-gating.md), [dört yöntemli ölçüm](model-quality.md).

Geliştirme koşusunda Campus IDF1 %43.44 → %67.39, ID switch 22 → 5 oldu.
Stadtmitte'de IDF1 %76.77 → %76.68, FP 46 → 71: daha geniş kabul bazı kısmi
kutuları da sürdürür. Bu iki sahne tasarımda kullanıldığı için bağımsız test değildir.

### Kişi görünüşü destekli alternatif

Search-enabled build ve gerçek OSNet bundle ile config-only `kalman-reid`
kullanılabilir: `configs/video-reid.toml`. Bu mod yalnız kişileri takip eder;
konum/sınıf/IoU kontrollerine cosine kapısı ve normalize görünüş prototipi ekler.
Native CLI bu modda bir OpenCV thread kullanır. Varsayılan yöntem değişmez;
görünüş desteği her sahnede daha iyi sonuç sağlamaz. RTSP/servis/çoklu kamera
desteği yoktur. Model, ayarlar, çıktı ve beşli karşılaştırma için
[kişi görünüşü kılavuzu](person-appearance.md).

Varsayılan greedy IoU referans takipçisinde hareket kestirimi yoktur. İki aşamalı
seçenekte basit hareket kestirimi vardır. Her iki modda hızlı hareket, örtüşme,
yön değişimi ve sınıf değişiminde kimlik parçalanması veya yanlış eşleşme olabilir.
`unique_track_ids` benzersiz kişi/nesne sayısı veya IDF1 ölçümü değildir.

Her kare sırayla işlenir. Kaynak FPS çıktı videosunda korunur; CPU işlemesi kaynak
hızından yavaş olabilir. `processing_fps` model yüklemesi ve ilk kare okumadan sonraki
analiz/çizim/kodlama/disk işini ölçer; `mean_analysis_ms` piksel kopyalama, YOLO ve
tracking (appearance modunda crop embedding dahil) ortalamasıdır. Tek yerel koşunun hızı kontrollü performans benchmark'ı değildir.
Yoğun sahnelerde Hungarian maliyeti ayrıca ölçülmelidir; gerçek zaman garantisi verilmez.

Zaman damgaları `frame_index / fps` üzerinden hesaplanır. Değişken FPS videoların
orijinal PTS değerleri korunmaz. Kaynak FPS geçersizse 25 FPS kullanılır ve raporda
belirtilir. OpenCV `read()` son kare ile bazı decode hatalarını ayıramadığı için bitiş
nedeni `end_of_stream_or_decode_stop` olarak yazılır; bu, dosyanın eksiksiz okunduğu
garantisi değildir. Açılmayan veya ilk karesi çözülemeyen dosya hata verir. Hata
halinde kısmi çıktılar klasörde kalabilir; başarı raporu yazılmaz.

Bu araç yerel dosyalar ve sabit, çift sayılı görüntü boyutları içindir. RTSP için
ayrı [canlı video akışı](live-video.md) vardır; GStreamer, çoklu kamera ve ses
aktarımı henüz yoktur. Giriş codec desteği
OpenCV kurulumuna bağlıdır; resmi Windows dağıtımındaki FFmpeg DLL'i video uygulamasının
yanına kopyalanır. Türkçe dosya yolları video backend'ine bağlıdır; sorun olursa ASCII
klasör yolu kullanın. Çıktı MJPEG/AVI'dir, MP4/H.264 değildir.

## Yerel doğrulama

`ctest --preset opencv-local` video okuma/yazma, frame sayısı/FPS, CSV, kısa kaybolmadan
sonra ID koruma, track silinmesi, sınıf izolasyonu, eşitliklerde deterministik karar,
kare limiti ve mevcut çıktıların korunmasını test eder. Testte model indirmeden
detector taklidi kullanılır; gerçek YOLO ayrıca yerel örnek video üzerinde çalıştırılır.

Bu bilgisayarda OpenCV `samples/data/vtest.avi` dosyasının ilk 60 karesi analiz edildi.
2026-09-30 tarihinde aynı YOLOv8n FP32 model ve OpenCV 4.12 CPU backend'iyle ardışık
çalıştırılan iki modun sonuçları:

| Ölçüm | IoU | İki aşamalı |
|---|---:|---:|
| İşlenen kare | 60 | 60 |
| Üretilen farklı ID | 15 | 10 |
| CSV takip satırı | 375 | 378 |
| Düşük güvenli eşleştirme | 0 | 11 |

Çıktılar yerelde `outputs/tracking-comparison/iou` ve `two-stage` altındadır;
videolar/model dosyaları Git'e eklenmez. Daha az ID tek başına daha iyi takip demek
değildir: doğum eşiği değişmiştir ve kaçırılan nesneler de ID sayısını azaltabilir.
Bu etiketli IDF1/HOTA değerlendirmesi veya yalnızca eşleştirme algoritmasını izole eden
bir karşılaştırma değildir. Doğru sonraki ölçüm etiketli klipte ID switch, IDF1 ve recall'dır.

Çekirdek testleri ayrıca 100 deterministik küçük matris için Hungarian sonucunu tüm
olasılıkları tarayan referansla karşılaştırır; zayıf tespit, kayıp track, sınıf izolasyonu,
süre dolması, hareketle boşluğu aşma ve hatalı giriş senaryolarını doğrular.

İsterseniz kurulu `bus.jpg` fotoğrafından deterministik bir test videosu da üretin:

```powershell
./build/opencv/Release/aegisvision_video_fixture.exe artifacts/samples/bus.jpg artifacts/samples/moving-bus.avi
./build/opencv/Release/aegisvision_video.exe artifacts/models/yolov8n.onnx artifacts/samples/moving-bus.avi outputs/moving-bus
```

Bu yardımcı fotoğrafı kaydırarak 30 kare üretir; gerçek hareketli sahne benchmark'ı değildir.

Kaynak: [OpenCV video okuma/yazma](https://opencv.org/reading-and-writing-videos-using-opencv/),
[OpenCV örnek video](https://github.com/opencv/opencv/blob/4.12.0/samples/data/vtest.avi).
