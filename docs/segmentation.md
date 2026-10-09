# C++ instance segmentation — ilk uygulama

`aegisvision_segment`, yerel videoyu YOLOv8n-seg ONNX modeliyle işler. Çalışma
zamanı C++/OpenCV CPU FP32'dir; Python yalnız bir defalık model hazırlığında kullanılır.
Detection/tracking CLI'larının varsayılanları değişmez. Bu ilk sürüm ayrı bir
video giriş noktasıdır; HTTP, RTSP, TOML ve takip-ID entegrasyonu henüz yoktur.

## Model ve çalıştırma

Mevcut export ortamında `ultralytics==8.3.252`, `onnx==1.19.1` kullanılır:

```powershell
work/yolo-export/Scripts/python.exe scripts/export_segmentation.py
cmake --build build/search --config Release --target aegisvision_segment aegisvision_segmentation_tests --parallel 1
build/search/Release/aegisvision_segment.exe artifacts/models/yolov8n-seg/yolov8n-seg.onnx artifacts/datasets/mot15-tud/TUD-Stadtmitte-raw.mp4 outputs/segmentation-tud-v1 179
```

Model yoksa hazırlık adımı resmi YOLOv8n-seg ağırlığını indirir; ağ erişimi gerekir.
Alternatif: resmi ağırlığı `artifacts/models/yolov8n-seg/yolov8n-seg.pt` konumuna
yerleştirip aynı export komutunu çalıştırın. Statik 640×640, batch 1, opset 12,
FP32, NMS'siz iki çıkış gerekir; detection ONNX bu komutta kullanılamaz.
Model manifesti kaynak adresi, sürüm ve SHA256 değerlerini kaydeder.
[Model lisans koşulları](https://www.ultralytics.com/license) ayrıca incelenmelidir.

## Çıktı sözleşmesi

- `segmented.avi`: sol orijinal, sağ renkli instance maskeleri; giriş FPS'i korunur.
- `preview.jpg`: aynı görünümün ilk karesi.
- `mask-N.png`: ilk karedeki her instance için sıkıştırılmamış anlamda ikili
  0/255 değerli, kayıpsız PNG **ROI** maskesi; konumu CSV'deki x/y'dir.
- `instances.csv`: sıfır tabanlı kare/instance, COCO sınıf numarası, confidence,
  ROI x/y/genişlik/yükseklik ve pozitif maske piksel sayısı. Instance sıra numarası
  kareler arasında takip kimliği değildir. Boş maskeler sıfır alanla korunur.
- `summary.json`: tamamlanan kare/instance sayıları ve çalışma ayarları.

Hedef dizin önceden var olmamalı. Hata halinde kısmi çıktı korunur; başarıyla
tamamlanmış bir çıktı için süreç çıkış kodu 0 ve summary gerekir. Video okuma
sonu OpenCV EOF davranışına dayanır; bozuk/eksik dosya için tam kare sayısı garantisi yoktur.

## Maske hesabı ve sınırlar

Çıkışlar `[1,4+C+32,N]` ve `[1,32,160,160]` olarak doğrulanır. Sınıf duyarlı
NMS sonrasında seçilen anchor'ın 32 katsayısı prototiplerle çarpılır. Letterbox
padding'i prototip düzleminden çıkarılır, logitler orijinal boyuta bilinear
ölçeklenir, orijinal kutu sınırında kırpılır ve logit > 0 (sigmoid > .5) ikili
maskeye dönüşür. ROI sınırları ceil ile yarı açık `[x1,x2)` piksel sözleşmesini
izler. Yuvarlama/interpolasyon ayrıntıları nedeniyle başka çalışma zamanlarıyla
birebir eşdeğerlik iddiası yoktur. Yaklaşım için
[sabit sürüm upstream maske işlemleri](https://github.com/ultralytics/ultralytics/blob/v8.3.252/ultralytics/utils/ops.py)
referans alındı; bu uygulamanın kendi sözleşmesi yukarıdadır.

Giriş kenarları en fazla 1920 piksel; çıktı en fazla 100 instance/kare.
Confidence .35, NMS IoU .45, maske eşiği .5. Aynı segmenter örneği eşzamanlı
worker'lar arasında paylaşılmaz. Bu CPU başlangıç sürümüdür; gerçek zamanlı hız
veya GPU optimizasyonu iddiası yoktur.

## Doğrulama durumu

Sentetik tensör testleri NMS sonrası katsayı eşleşmesini, padding geri dönüşünü,
pozitif/negatif logitleri, boş sonuçları, şekil/nonfinite reddini ve maske ROI
doğrulamasını kapsar. Bunlar etiketli veride model doğruluğu ölçümü değildir.

2026-10-09: Ağ engeli kalktıktan sonra resmi YOLOv8n-seg ağırlığı indirildi ve
ONNX export/checker kontrolü geçti. C++ uygulaması gerçek TUD-Stadtmitte videosunun
179 karesini işledi; toplam 1110 kare-içi instance üretti. Bu sayı benzersiz kişi
sayısı veya doğruluk metriği değildir. İlk karede altı kişi maskesi görsel olarak
kontrol edildi. Etiketli maske IoU değerlendirmesi **henüz yapılmadı**; MOT kutu
etiketleri piksel maskesi doğruluğunu ölçmek için yeterli değildir.

Yerel demo `outputs/segmentation-tud-v1/segmented.mp4`: 1280×480, 25 FPS,
179 kare, 7,16 saniye; solda orijinal, sağda C++ model maskeleri. İlk kare ROI
maskeleri, tüm karelerin alan raporu ve özet aynı dizinde bulunur. Model/veri ve
üretilen videolar Git'e dahil değildir. MP4, native AVI çıktısından dönüştürüldü:

```powershell
ffmpeg -n -i outputs/segmentation-tud-v1/segmented.avi -an -c:v libx264 -threads 2 -crf 20 -pix_fmt yuv420p -movflags +faststart outputs/segmentation-tud-v1/segmented.mp4
```

ONNX SHA256: `85c8cb2695f586a3b9cc0279a3fca8aaa4e233f5ed6dfee88b2ec7cad587495b`.
Ağırlık SHA256: `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`.
Kaynak/sürüm manifesti `artifacts/models/yolov8n-seg/manifest.json` içindedir.

Önceki kısıtlı oturumda derleme ve segmentation testleri geçmiş, tam paketin
yedi servis/arşiv/arama testi erişim veya bağlantı hatası vermişti. Bu işler
segmentation kapsamında değiştirilmedi. 2026-10-09 erişim kısıtı olmayan oturumda
Release derlemesi ve **34/34 yerel CTest** geçti (39,83 saniye).
