# Video üzerinde YOLO ve nesne takibi

`aegisvision_video`, yerel bir video dosyasının her karesini YOLOv8 ile analiz eder,
sınıf duyarlı IoU takipçisiyle kimlik atar ve sonucu videoya çizer. Tek kaynak için
tek tracker kullanılır; videolar arasında takip durumu paylaşılmaz.

## Kullanım

OpenCV ve modeli hazırlamak için [YOLO kılavuzunu](yolo.md) izleyin.

```powershell
cmake --build --preset opencv-local
./build/opencv/Release/aegisvision_video.exe artifacts/models/yolov8n.onnx "C:/videolar/sahne.mp4" outputs/video-sonuc
```

İsteğe bağlı son argüman işlenecek en fazla kare sayısıdır:

```powershell
./build/opencv/Release/aegisvision_video.exe artifacts/models/yolov8n.onnx "C:/videolar/sahne.mp4" outputs/video-ilk60 60
```

Çıktı klasörü yeni veya boş olmalıdır. Dosyalar:

- `tracked.avi`: kimlik ve sınıf etiketli MJPEG video; ses içermez.
- `preview.jpg`: ilk işlenen karenin görünümü.
- `tracks.csv`: sıfır tabanlı kare indeksi, zaman, track ID, sınıf, güven, kutu, yaş.
- `summary.json`: işlenen kareler, üretilen farklı ID sayısı, kaynak/işleme FPS, ortalama analiz süresi.

ID'ler her çağrıda 1'den başlar. Kaybolmuş bir track en fazla 20 boş kare tutulur;
bu arada çizilmez ve CSV'ye yazılmaz. Yeniden eşleşmek için aynı sınıf ve en az 0.30
IoU gerekir. Süresi dolan track geri geldiğinde yeni ID alır. Eşit eşleşmelerde en
küçük ID, eşit güvenlerde giriş sırası kullanılır.

## Sınırlar ve ölçümler

Bu, greedy IoU referans takipçisidir: hareket kestirimi, görünüş embedding'i, ByteTrack
ve Re-ID yoktur. Hızlı hareket, örtüşme ve sınıf değişiminde kimlik parçalanması olabilir.
`unique_track_ids` benzersiz kişi/nesne sayısı veya IDF1 ölçümü değildir.

Her kare sırayla işlenir. Kaynak FPS çıktı videosunda korunur; CPU işlemesi kaynak
hızından yavaş olabilir. `processing_fps` model yüklemesi ve ilk kare okumadan sonraki
analiz/çizim/kodlama/disk işini ölçer; `mean_analysis_ms` piksel kopyalama, YOLO ve
tracking ortalamasıdır. Bu koşuda eşzamanlı yerel derleme/test işi de yapılmıştır;
raporlanan hız kontrollü performans benchmark'ı değildir.

Zaman damgaları `frame_index / fps` üzerinden hesaplanır. Değişken FPS videoların
orijinal PTS değerleri korunmaz. Kaynak FPS geçersizse 25 FPS kullanılır ve raporda
belirtilir. OpenCV `read()` son kare ile bazı decode hatalarını ayıramadığı için bitiş
nedeni `end_of_stream_or_decode_stop` olarak yazılır; bu, dosyanın eksiksiz okunduğu
garantisi değildir. Açılmayan veya ilk karesi çözülemeyen dosya hata verir. Hata
halinde kısmi çıktılar klasörde kalabilir; başarı raporu yazılmaz.

Bu sürüm yerel dosyalar ve sabit, çift sayılı görüntü boyutları içindir. Kamera/RTSP,
GStreamer, çoklu kamera ve ses aktarımı sonraki aşamalardır. Giriş codec desteği
OpenCV kurulumuna bağlıdır; resmi Windows dağıtımındaki FFmpeg DLL'i video uygulamasının
yanına kopyalanır. Türkçe dosya yolları video backend'ine bağlıdır; sorun olursa ASCII
klasör yolu kullanın. Çıktı MJPEG/AVI'dir, MP4/H.264 değildir.

## Yerel doğrulama

`ctest --preset opencv-local` video okuma/yazma, frame sayısı/FPS, CSV, kısa kaybolmadan
sonra ID koruma, track silinmesi, sınıf izolasyonu, eşitliklerde deterministik karar,
kare limiti ve mevcut çıktıların korunmasını test eder. Testte model indirmeden
detector taklidi kullanılır; gerçek YOLO ayrıca yerel örnek video üzerinde çalıştırılır.

Bu bilgisayarda OpenCV `samples/data/vtest.avi` dosyasının ilk 60 karesi analiz edildi.
15 farklı track ID üretildi; 1, 4 ve 5 numaralı ID'ler 60 karenin tamamında görüldü.
Bu bir smoke testtir, etiketli tracking veri kümesi değerlendirmesi değildir.

İsterseniz kurulu `bus.jpg` fotoğrafından deterministik bir test videosu da üretin:

```powershell
./build/opencv/Release/aegisvision_video_fixture.exe artifacts/samples/bus.jpg artifacts/samples/moving-bus.avi
./build/opencv/Release/aegisvision_video.exe artifacts/models/yolov8n.onnx artifacts/samples/moving-bus.avi outputs/moving-bus
```

Bu yardımcı fotoğrafı kaydırarak 30 kare üretir; gerçek hareketli sahne benchmark'ı değildir.

Kaynak: [OpenCV video okuma/yazma](https://opencv.org/reading-and-writing-videos-using-opencv/),
[OpenCV örnek video](https://github.com/opencv/opencv/blob/4.12.0/samples/data/vtest.avi).
