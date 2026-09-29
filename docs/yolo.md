# C++ YOLOv8 nesne tespiti

`aegisvision_detect` gerçek ONNX modelini OpenCV DNN ile CPU üzerinde çalıştırır.
Runtime Python gerektirmez. İlk sürümün sabit sözleşmesi: YOLOv8 **detection**,
640×640 RGB float32, batch=1, COCO 80 sınıf, `[1,84,8400]` çıktı, model içinde NMS yok.
YOLOv5/segmentation/end-to-end modeller bu çıktıyla uyumlu kabul edilmez.

## Modeli bir kez hazırlama

Mevcut bilgisayarda model `artifacts/models/yolov8n.onnx` olarak hazırlandı.
Yeniden üretmek için ayrı bir Python ortamında:

```powershell
python -m venv work/yolo-export
./work/yolo-export/Scripts/python.exe -m pip install torch==2.9.1 ultralytics==8.3.252 onnx==1.19.1
./work/yolo-export/Scripts/python.exe scripts/export_yolo.py
```

Bu bilgisayardaki hazırlama ortamı zaten kurulu PyTorch/Ultralytics paketlerini kullanır
(`--system-site-packages`); ONNX bağımlılığı sadece bu yerel ortama kuruldu.
Linux'ta sanal ortamın Python yolu `work/yolo-export/bin/python` olur.
Script resmi YOLOv8n ağırlığını indirir, FP32/opset12/sabit boyut/batch1/`nms=False`
olarak dışa aktarır ve ONNX doğrulamasını çalıştırır. Sürümler, sınıf sırası ve SHA-256
değerleri `artifacts/models/model-manifest.json` dosyasında tutulur.
Model ve `.pt` dosyaları Git'e eklenmez.

## Çalıştırma

```powershell
cmake --preset opencv-local
cmake --build --preset opencv-local
./build/opencv/Release/aegisvision_detect.exe artifacts/models/yolov8n.onnx artifacts/samples/bus.jpg outputs/yolo-demo
```

Kendi görselinizi ikinci argüman olarak verin. Örneğin güven eşiğini 0.50 yapmak için:

```powershell
./build/opencv/Release/aegisvision_detect.exe artifacts/models/yolov8n.onnx "C:/fotograflar/sahne.jpg" outputs/tespit 0.50 0.45
```

Son iki argüman güven eşiği ve sınıf bazlı NMS IoU eşiğidir. Varsayılanlar 0.35 ve 0.45.
Çıktılar: `annotated.png`, `detections.tsv` ve `report.json`. Aynı çıktı klasörü yeniden
kullanılırsa bu dosyalar güncellenir. TSV, mevcut `aegisvision_image annotate` komutuyla
yeniden çizilebilir. Boş sonuç geçerlidir; bozuk model/görsel ve geçersiz eşikler hata verir.

Görüntü en-boy oranı korunarak ölçeklenir, ortalanır, 114 değerli kenarlık eklenir;
BGR→RGB, NCHW ve 1/255 normalizasyonu uygulanır. Kutular kenarlık çıkarıldıktan sonra
orijinal çözünürlüğe taşınır ve görüntü sınırlarına kırpılır. Sınıf olasılığına göre
süzülür; aynı sınıftaki örtüşen kutular elenir. En fazla 3000 aday NMS'ye, 300 sonuç
çıktıya girer. Tek karede takip ve sahte embedding aşamaları kapalıdır.

`YoloDetector`, `IDetector` arayüzünü uygular. `Frame::image` OpenCV'den bağımsız,
sahibi belirli bir BGR piksel tamponudur. `image_frame()` giriş görüntüsünü kopyalar.
Her worker kendi detector örneğini kullanmalıdır; tek OpenCV ağ örneği eşzamanlı çağrılmamalıdır.

## Doğrulama

```powershell
ctest --preset opencv-local
./build/opencv/Release/aegisvision_yolo_tests.exe artifacts/models/yolov8n.onnx artifacts/samples/bus.jpg
```

Model gerektirmeyen testler letterbox/RGB/ölçek, özgün koordinata dönüşüm, sınıf bazlı
NMS, kutu kırpma, NaN ve uyumsuz çıktı şeklini kapsar. İsteğe bağlı gerçek model testi
Ultralytics `bus.jpg` örneğinde en az üç kişi ve bir otobüs bekler. CTest'e eklemek için
`AEGISVISION_YOLO_MODEL` ve `AEGISVISION_YOLO_IMAGE` CMake ayarlarına mutlak yolları verin.
GitHub CI ağırlık indirmez; saf ön/son işleme testlerini çalıştırır.

Raporlanan süre model yüklemeyi ayrı, ilk analiz çağrısını (hazırlama + inference +
son işleme) ayrı ölçer. Disk okuma, sonuç çizme ve yazma analiz süresine dahil değildir.
Bu tek-görsel kontrolüdür; mAP, video FPS veya p95 benchmark değildir. GPU/TensorRT ve
ONNX Runtime backend'leri henüz eklenmedi.

## Kaynaklar

- [YOLOv8n ağırlığı](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)
- [ONNX dışa aktarma](https://docs.ultralytics.com/modes/export/)
- [OpenCV YOLO rehberi](https://docs.opencv.org/4.12.0/da/d9d/tutorial_dnn_yolo.html)
- [Ultralytics lisans bilgisi](https://www.ultralytics.com/license)
- Yerel demo fotoğrafı: kurulu Ultralytics paketinin `assets/bus.jpg` dosyası;
  [upstream kaynak](https://github.com/ultralytics/ultralytics/blob/v8.3.252/ultralytics/assets/bus.jpg).

Üçüncü taraf model ve örnek fotoğraf bu reponun kodundan ayrı kaynaklardır; lisansları
yukarıdaki upstream kaynaklarda tutulur. Repo bunlara yeni bir lisans atamaz.
