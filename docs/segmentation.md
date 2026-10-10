# C++ instance segmentation — ilk uygulama

`aegisvision_segment`, yerel videoyu YOLOv8n-seg ONNX modeliyle işler. Çalışma
zamanı C++/OpenCV CPU FP32'dir; Python model hazırlığı ve çevrimdışı resmi kalite
ölçümü içindir, uygulama inference'ında kullanılmaz. Detection/tracking CLI'larının
varsayılanları değişmez. Ayrı video giriş noktası ve yeniden kullanılabilir
`SegmentationPipeline` vardır. HTTP tek-kare işi aşağıdadır;
[canlı RTSP maske modu](live-video.md#canlı-piksel-maskeleri) da eklenmiştir.
Gerçek modelle RTSP kopma/yeniden bağlantı, maske önizlemesi ve stop doğrulaması
[2026-10-10'da geçti](live-mask-validation.md).
Segmentation için ayrı TOML modu henüz yoktur.

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
  Yeni sürümde son sütun `track_id` eklendi: takip kapalıysa 0; `--track` ile
  tracker'ın kimliği. Eski CSV sütunları ve sırası korunur.
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

## Maske + takip kimliği

```powershell
build/search/Release/aegisvision_segment.exe artifacts/models/yolov8n-seg/yolov8n-seg.onnx artifacts/datasets/mot15-tud/TUD-Stadtmitte-raw.mp4 outputs/segmentation-tracked-tud-v1 179 --track
```

Sınıf duyarlı mevcut IoU tracker kullanılır: eşik .30, en fazla 20 kaçırılmış
kare. Yeni `update_indexed` sözleşmesi her tespit için aynı giriş sırasında
kimlik döndürür; kutu benzerliğiyle ikinci bir maske eşleştirmesi yapılmaz.
Bu, aynı kutulu veya confidence sırası değişen tespitlerin karışmasını önler.
Eski `update` API'si kimlik sıralı çıktısını korur. Renk ve etiket `track_id`'ye
bağlıdır. Kaçırılmış karede tahmini maske üretilmez; yalnız gerçek tespitler çizilir.
Giriş boyutu değişirse hata verilir; yeni video yeni pipeline örneği gerektirir.
Kimlikler video-yereldir; çoklu kamera Re-ID veya video-object-segmentation modeli değildir.

## Kuyruk tabanlı HTTP tek-kare maskesi

[Yerel servis kurulumu](service.md) tamamlandıktan sonra, yeni bir iş veritabanıyla:

```powershell
./scripts/start_service.ps1 -SegmentationModel artifacts/models/yolov8n-seg/yolov8n-seg.onnx -JobDatabase artifacts/service/segmentation-jobs.sqlite
./scripts/test_segmentation_service.ps1 -Video pedestrians.mp4 -FrameIndex 0 -OutputDirectory outputs/segmentation-http-v1
```

Doğrudan sunucu komutunun sonuna `--segment-model MODEL.onnx` eklemek de mümkündür.
Model sunucuda bir kez yüklenir, tek dosya/arama worker'ından çağrılır. Model yolu
HTTP isteğinden alınmaz. Seçenek verilmezse eski iş akışı değişmez ve bu iş 400 ile
reddedilir; `/api/health` içinde `segmentation_enabled` yayınlanır. Model imzası
kalıcı iş bağlamına katılır; etkinleştirirken veya model değiştirirken ayrı DB seçin.

`POST /api/jobs`, `Content-Type: application/json`:

```json
{"type":"segment_frame","path":"pedestrians.mp4","frame_index":0}
```

202 yanıtındaki `id` ile `GET /api/jobs/{id}` izlenir; mevcut cancel uç noktası
kullanılır. Başarılı `result`, görüntü boyutu, sıfır tabanlı kare numarası, model
imzası ve `instances` döndürür. Her instance: `label`, `score`, xyxy `bbox`,
`mask_pixels`, tam görüntü koordinatlarında `segmentation: {size:[h,w],counts:[...]}`.
Counts, ilk sıfır koşusuyla başlayan COCO column-major RLE'dir; kutu/ROI maskesi
değildir. Boş tespit başarılı boş listedir. Tek kare bağımsız analiz edildiğinden
`tracking:false`; CLI'daki video-yerel ID'ler bu uç noktada üretilmez.

Yalnız medya kökündeki yerel videolar kabul edilir; yönetilen canlı arşiv bu işin
kapsamında değildir. Frame index 0..10000, görüntü kenarı en fazla 1920, en fazla
100 instance, toplam 100000 RLE koşusu ve 1 MiB result JSON sınırı vardır. Aşımda
maskeler sessizce kırpılmaz; iş başarısız olur. Decode sırayla yapılır, kareler
arasında iptal ve 30 saniyelik decode bütçesi kontrol edilir. OpenCV decode/forward
çağrısı ortasında kesme yoktur; bu bir hard timeout değildir. Model çağrısı dönüşünde
iptal tekrar kontrol edilir. Kaynak boyutu/mtime kabulde ve işin önce/sonrasında
kontrol edilir (kriptografik medya bütünlüğü garantisi değildir).

Mevcut bounded queue/backpressure, SQLite durum kaydı, hata sonrası worker'ın
devamı ve aynı-origin kontrolleri kullanılır. Tam video HTTP export'u henüz
dahil değildir. Canlı RTSP maskeleri ayrı canlı worker'da çalışır. Test betiği gerçek HTTP sonucunun RLE
kapsamını/alanını denetler, `job.json` ve `preview.jpg` çıktısını bilgisayarda saklar.

### Web panelinde kullanım

Yerel video seçin, **Piksel maskesi analizi** bölümünde kare numarasını girip
**Seçili kareyi analiz et** düğmesine basın. Durum/iptal mevcut İşler panelindedir.
Tamamlanınca maske görüntüsü, nesne sınıfları, skorlar ve piksel alanları gösterilir.
Önceki başarılı işlerde **Maskeyi göster** ile saklanan sonuç açılabilir; eski
önizlemesiz işler için yeniden analiz istenir. Model kapalıysa veya yönetilen
arşiv seçiliyse gönderim kapalıdır. Kaynak değişince eski önizleme temizlenir;
geciken gönderim yanıtı yeni kaynağın maskesi olarak gösterilmez.

Worker aynı analiz karesinden C++ `paint_masks` + JPEG üretir. `preview_data_url`
bu görüntüyü sonuçla birlikte saklar; tarayıcı FPS/zaman tahminiyle başka bir kare
seçmez ve model yeniden çağrılmaz. Görsel en uzun kenarı 960'a küçültülebilir;
RLE/alan/koordinatlar orijinal çözünürlükte kalır. JPEG 512 KiB, tüm JSON (JPEG
dahil) 1 MiB sınırındadır. Önizleme bir gösterimdir; kayıpsız maske RLE'dir.
Tarayıcı yalnız sınırlı JPEG data URL kabul eder, sınıf/metinleri HTML olarak işlemez.

Panel otomasyon doğrulaması: Release derlemesi ve 35/35 CTest geçti (36,66 saniye).
HTTP testindeki base64 JPEG çözülüp gerçek OpenCV decode boyutları doğrulandı.
`node tests/test_web_state.cjs` gönderim, yetenek kapalı durumu, kaynak değişimi,
geciken yanıt, model hatası, geçmiş sonuç açma ve güvensiz önizleme URL reddini
kapsar.

2026-10-10 HTTP doğrulaması: Release derlemesi ve 35/35 CTest geçti (40,48 saniye).
Servis testi gerçek loopback HTTP ve video decode kullanır; segmenter bu testte
deterministik fixture'dır. RLE içeriği, geçersiz parametreler, olmayan kare,
model hatası, sonuç kotası, kuyrukta iptal, değiştirilmiş girdi, kapalı yetenek ve
yeniden başlatma sonrası maskelerin korunması kontrol edildi.

2026-10-10 gerçek servis ve tarayıcı smoke doğrulaması da tamamlandı. Qdrant 1.12.5,
kalıcı iş kuyruğu ve resmi YOLOv8n-seg modeliyle `pedestrians.mp4` kare 0 işlendi:
768x576 kaynakta 6 maske (3 person, 2 car, 1 truck) üretildi. Web paneli işi
tamamlandı gösterdi; maske JPEG'i ve sınıf/skor/alan kartları tarayıcıda görsel
olarak kontrol edildi. Aynı istek `scripts/test_segmentation_service.ps1` ile
tekrar doğrulandı; COCO RLE kapsamı/alanı geçti ve yerel, Git dışı
`outputs/segmentation-http-v1/job.json` ile `preview.jpg` üretildi. Bu smoke testi
tek video/tek karedir; genel model doğruluğu iddiası değildir.

## Etiketli maske değerlendirmesi

Mevcut, önceden seçilmiş 64 COCO validation görüntüsü ve 8 kategori kullanılır.
Bu görüntüler daha önce detection/arama geliştirmesinde görüldü; yeni kör test
veya tam COCO sonucu değildir. Eşikler demo ile aynıdır, sonuç sonrası ayarlanmaz.
İlk adım native C++ inference ve tam görüntü koordinatlı column-major COCO RLE
export'udur. İkinci adım resmi `pycocotools==2.0.11` ile çevrimdışı ölçümdür:

```powershell
cmake --build build/search --config Release --target aegisvision_segment_quality --parallel 2
build/search/Release/aegisvision_segment_quality.exe artifacts/models/yolov8n-seg/yolov8n-seg.onnx artifacts/datasets/coco-search/quality-manifest.json outputs/segmentation-quality-coco-v1
work/yolo-export/Scripts/python.exe scripts/evaluate_segmentation.py artifacts/datasets/coco-search/quality-manifest.json artifacts/datasets/coco-search/annotations/instances_val2017.json artifacts/models/yolov8n-seg/yolov8n-seg.onnx outputs/segmentation-quality-coco-v1/mask-predictions.json outputs/segmentation-quality-coco-v1/metrics.json
```

Kalite CLI'ı search build'indeki mevcut JSON/hash altyapısını kullanır. Referans
paketi varsayılan olarak `artifacts/deps/quality-reference` içindedir; başka bir
kurulum için `--reference-root` verilebilir. Tüm seçilmiş görüntüler, model,
manifest ve annotation SHA256 kontrol edilir; eksik kare/görüntü, değişen veri,
hatalı RLE veya yanlış sınıf eşlemesi kabul edilmez. Ground truth inference'a girmez.

Ölçüm [resmi COCOeval segmentation protokolünü](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
kullanır: IoU .50:.05:.95, maxDets 1/10/100, crowd/ignore kuralları. Modelin .35
confidence sınırı korunur; bu yüzden üretici benchmark'ıyla karşılaştırılabilir
bir düşük-eşik AP iddiası değildir. TP/FP/FN, IoU .50'de ayrıca raporlanır.
Eşleşmiş maskelerin ortalama IoU'su yalnız TP'leri kapsar; kaçırılan ve yanlış
nesneleri gizlememek için AP ve TP/FP/FN ile birlikte okunmalıdır.

## Doğrulama durumu

Sentetik tensör testleri NMS sonrası katsayı eşleşmesini, padding geri dönüşünü,
pozitif/negatif logitleri, boş sonuçları, şekil/nonfinite reddini ve maske ROI
doğrulamasını kapsar. Bunlar etiketli veride model doğruluğu ölçümü değildir.

2026-10-09: Ağ engeli kalktıktan sonra resmi YOLOv8n-seg ağırlığı indirildi ve
ONNX export/checker kontrolü geçti. C++ uygulaması gerçek TUD-Stadtmitte videosunun
179 karesini işledi; toplam 1110 kare-içi instance üretti. Bu sayı benzersiz kişi
sayısı veya doğruluk metriği değildir. İlk karede altı kişi maskesi görsel olarak
kontrol edildi. MOT kutu etiketleri piksel maskesi doğruluğunu ölçmek için yeterli
değildir; aşağıdaki ayrı COCO ölçümü gerçek piksel etiketlerini kullanır.

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

2026-10-10: Native kalite CLI'ı 64 görüntüde 242 maske üretti; seçili 8 sınıfa
ait 109 tahmin resmi `pycocotools 2.0.11` ile değerlendirildi. Sonuçlar:

| Ölçüm | Sonuç |
| --- | ---: |
| Maske AP (.50:.95) | %45,47 |
| AP50 / AP75 | %66,73 / %50,88 |
| IoU .50: TP / FP / FN | 97 / 11 / 62 |
| Ignore edilen tahmin | 1 |
| Yalnız eşleşen TP maskelerinde ortalama IoU | %80,87 |

Küçük nesne AP'si %0; bu başlangıç modelinin bu veri ve sabit .35 confidence
eşiğindeki zayıflığıdır. Sonuç sonrası eşik ayarı yapılmadı. Tam rapor ve girdilerin
SHA256 değerleri `outputs/segmentation-quality-coco-v1/metrics.json` içindedir.
Bu, daha önce görülmüş geliştirme seçkisidir; bağımsız genelleme kanıtı değildir.

Takipli demo `outputs/segmentation-tracked-tud-v1/segmented.mp4`:
179 kare, 1110 kare-içi instance. CSV'nin önceki dokuz sütununun tamamı takip
kapalı demo ile birebir aynı; ID eklenmesi kutuları, skorları veya maske alanını
değiştirmedi. Her karede ID'ler pozitif ve benzersiz; aynı ID sınıf değiştirmiyor.
İlk kare maskeleri ve kimlik etiketleri görsel olarak kontrol edildi. Bunlar
ilişkilendirme tutarlılık kontrolleridir, ground-truth ID doğruluğu ölçümü değildir;
özellikle örtüşmelerde basit IoU tracker kimlik değiştirebilir.

2026-10-10 doğrulaması: Release derlemesi, **35/35 CTest** (44,50 saniye) ve
**9/9 değerlendirme sözleşme testi** geçti. Maske testleri sıra değişimi,
aynı kutulu farklı instance'lar, kısa kayıp/kimlik sonlandırma, takip kapalı modu
ve tam görüntü column-major RLE sözleşmesini de kapsar.

```powershell
ffmpeg -n -i outputs/segmentation-tracked-tud-v1/segmented.avi -an -c:v libx264 -threads 2 -crf 20 -pix_fmt yuv420p -movflags +faststart outputs/segmentation-tracked-tud-v1/segmented.mp4
```
