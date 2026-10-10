# C++ OCR başlangıç adaptörü

2026-10-10: `IOcr` / `PpocrCrnn` ve `aegisvision_ocr` eklendi. PP-OCRv3
İngilizce detection modeli dört köşeli metin bölgelerini bulur; perspektif
düzeltmesinden sonra CRNN EN modeli yazıyı okur. Inference tamamen C++/OpenCV DNN
CPU FP32'dir. Python, Tesseract servisi veya harici OCR API'si kullanılmaz.

Yerel görüntü CLI'ına ek olarak **`ocr_frame` HTTP işi ve web paneli** vardır.
`IOcr` ayrık metin/kutu/skor sözleşmesini korur. RTSP OCR, TOML seçimi ve
`AnalysisPipeline` içindeki `ITextExtractor` portuna bağlantı henüz yoktur.
VLM mevcut değildir.

## Çalıştırma

Proje kökünde, modelleri sabit commit ve SHA256 ile hazırlayın:

```powershell
./scripts/prepare_ocr.ps1 -WithSample
```

Yerelde hazırlanmış executable ile (çıktı dizini daha önce var olmamalı):

```powershell
./build/live-seg/Release/aegisvision_ocr.exe artifacts/models/ocr-en/text_detection_en_ppocrv3_2023may.onnx artifacts/models/ocr-en/text_recognition_CRNN_EN_2021sep.onnx artifacts/media/ocr/text_det_test2.jpg outputs/ocr-canon-new
```

`result.json` özgün görüntü boyutunu, süreyi, metni, iki ayrı skoru ve polygon'u
verir. `preview.jpg` metin kutuları ve okunan yazıları gösterir. Görüntüde yazı
bulunmazsa boş `regions` geçerli bir başarıdır; uydurma metin doldurulmaz.

Arama/Qdrant bağımlılığı olmadan temiz derleme de yapılabilir:

```powershell
cmake -S . -B build/ocr -DAEGISVISION_WITH_OPENCV=ON -DOpenCV_DIR=C:/opencv/opencv/build
cmake --build build/ocr --config Release --parallel 2
ctest --test-dir build/ocr -C Release --output-on-failure
```

Bu örnekte OpenCV'nin kurulu olduğu yolu kendi bilgisayarınıza göre ayarlayın.
Yeni derlemenin CLI yolu `build/ocr/Release/aegisvision_ocr.exe` olur. Yerel test
OpenCV 4.12.0 ile yapıldı; diğer sürümlerde gerçek ONNX importer uyumu ayrıca
doğrulanmalıdır. Eski 4.x sürümlerindeki skaler preprocessing API'siyle derleme
uyumu için kanal normalizasyonu adaptör içinde açıkça uygulanır.

## Sözleşme ve sınırlar

- Tanınan alfabe **`0123456789abcdefghijklmnopqrstuvwxyz`**. Büyük harfler küçük
  harf döner; Türkçe karakterler, noktalama ve çok dilli okuma desteklenmez.
  Desteklenmeyen dil otomatik algılanmaz; başka yazı sistemleri yanlış ASCII
  metne dönüşebilir. Belge doğrulama veya plaka okuma ürünü olarak sunulmamalıdır.
- Polygon sırası **sol-alt, sol-üst, sağ-üst, sağ-alt**; orijinal görüntü piksel
  koordinatları, floating point. Çıktı yüksek detection skorundan düşüğe sıralıdır;
  belge/paragraf okuma sırası iddiası yoktur.
- Detection: 736×736'e resize, BGR kanal sırası; mean
  `[123.675,116.28,103.53]`, std `[.229,.224,.225]` ve 255 normalizasyonu.
  Binary threshold .3, polygon threshold .5, unclip 2.0, en çok 200 aday.
- Recognition: perspektif düzeltmeyle 100×32, grayscale, `(pixel-127.5)/127.5`.
  `[T,1,37]` logitlerinden greedy CTC; boş etiket 0, ardışık tekrarlar birleştirilir,
  boş etiketle ayrılan tekrarlar korunur.
- `recognition_confidence`, CTC'nin metne eklediği karakter adımlarındaki softmax
  değerlerinin aritmetik ortalamasıdır. **Kalibre edilmiş doğruluk olasılığı
  değildir.** Detection skoru ile çarpılarak tek bir doğruluk skoru yapılmaz.
- BGR8, en fazla 1920 piksel kenar; CLI sıkıştırılmış dosya limiti 32 MiB.
  Boyut kontrolü decode sonrasındadır; toplam süreç belleğine OS kotası değildir.
  Varsayılan en fazla 64 sonuç (API ayarı 1–100); kalan düşük skorlu bölgeler
  işlenmez. Her model dosyası en fazla 128 MiB olabilir.
- Varsayılan 10 saniyelik **kooperatif** deadline ve opsiyonel atomik iptal
  kontrolü DNN çağrılarından önce/sonra uygulanır. Devam eden inference zorla
  kesilemez; model yükleme süresi bu deadline'a dahil değildir. Hata/iptalde
  kısmi sonuç dönmez. Modeller ardışık kullanımlıdır; worker'lar arası paylaşılmaz.
- ASCII edit-distance yardımcısı en fazla 1024 byte metin kabul eder. CER,
  edit distance / referans karakter sayısıdır; boş referansa bölme yapılmaz.
  Unicode CER ölçümü olarak kullanılmaz.

## Gerçek çıktı ve test kapsamı

OpenCV extra'nın `text_det_test2.jpg` fotoğrafında görünen **Canon** yazısı,
herhangi bir fine-tuning/eşik denemesi yapılmadan `canon` olarak okundu:

- Bir metin bölgesi; detection skoru **0,994585**, recognition skoru **0,999804**.
- Tek CLI çağrısında detection + recognition **290,67 ms** (model yükleme ve
  çıktı dosyalarını yazma hariç). Bu p50/p95 veya yük testi değildir.
- Küçük harfe normalize edilmiş referansa karşı **CER = 0/5 = 0**. Tek kelimelik
  smoke kontrolüdür; genel OCR doğruluğu veya bağımsız benchmark sonucu değildir.
- Yerel sonuçlar: `outputs/ocr-canon-v1/result.json` ve `preview.jpg`.

Deterministik testler CTC blank/tekrar davranışı, NaN/Inf/şekil reddi, perspektif
düzeltmesi, geçersiz dörtgen, edit distance, model/ayar hatalarını kapsar. Opsiyonel
gerçek model testi ayrıca Canon referansını, boş beyaz görüntüde boş sonucu ve
işlem öncesi iptali kontrol eder. Modeller CI sırasında kendiliğinden indirilmez.
Gerçek model testi için CMake'e `AEGISVISION_OCR_BUNDLE` ve
`AEGISVISION_OCR_TEST_IMAGE` mutlak yolları verilir.

Bu çalışma ağacında Release derlemesi ve gerçek OCR/CLIP referanslarını içeren
**38/38 CTest** geçti. Ayrı OCR veri kümesi, Türkçe/multilingual model, video
karelerinden geniş kalite ölçümü henüz tamamlanmadı.

## HTTP kuyruğu ve web paneli

Qdrant açıkken, proje kökünden yeni derlemeyi OCR modelleriyle başlatın:

```powershell
./scripts/start_service.ps1 -ServerExecutable build/live-seg/Release/aegisvision_server.exe -OcrModels artifacts/models/ocr-en -JobDatabase artifacts/service/ocr-jobs.sqlite
```

8090'da başka sunucu varsa önce kendi terminalinde Ctrl+C ile durdurun veya
`-Port 8091` kullanın. Mevcut SQLite dosyalarını silmeyin: OCR'ın açılıp kapanması
veya model içeriklerinin değişmesi persistence bağlamını değiştirir; eski işlerin
farklı modelle sessizce yürütülmesi reddedilir. Ayrı veritabanı bu nedenle önerilir.
`-SegmentationModel` ve `-LiveSegmentation` seçenekleriyle birlikte kullanılabilir.

Panelde video seçin, **Karedeki yazıları oku** bölümünde 0 tabanlı kare numarası
girin ve **Yazıları oku** düğmesine basın. Metin listesi ile kutulu JPEG gösterilir.
İş listesinden iptal edilebilir; tamamlanan işe **Yazıları göster** ile dönülür.
Video değişince eski önizleme temizlenir. Modeller kapalıysa düğme devre dışıdır.

API isteği:

```json
{"type":"ocr_frame","path":"ocr-canon.mp4","frame_index":0}
```

`POST /api/jobs` 202 döner; `GET /api/jobs/{id}` ile izlenir.
`POST /api/jobs/{id}/cancel` iptal ister. Model ve sonuçlar tek kuyruk worker'ında
çalışır; SQLite tamamlanmış metin/polygon/önizlemeyi saklar. Canlı ayrı worker'a
OCR eklenmez. `GET /api/health` içindeki `ocr_enabled` yeteneği bildirir.

- Yalnız medya kökündeki yerel video; yol kaçışı ve yönetilen canlı arşiv reddedilir.
  HTTP üzerinden keyfî resim yolu, model veya dil seçimi kabul edilmez.
- Kare 0–10000; doğru indeks için baştan ardışık decode. Decode 30 saniye kooperatif
  deadline; modelin ayrıca 10 saniyelik kooperatif limiti vardır. Decode/inference
  çağrısı zorla kesilemez. İptalde kısmi sonuç yayımlanmaz.
- Kabul anındaki dosya boyutu/mtime, işlem başı ve sonuç öncesi yeniden kontrol
  edilir. Bu içerik hash'i veya kötü niyetli eşzamanlı dosya değişimine karşı
  atomik snapshot garantisi değildir.
- En fazla 64 bölge; metin/finite skor/konveks geometri doğrulanır. Preview en
  fazla 960 piksel kenar ve 512 KiB JPEG; JSON sonuç en fazla 1 MiB'dir.
- HTTP sonucundaki polygon orijinal kare koordinatıdır; küçültülmüş JPEG'e
  çizim sunucuda yapılır. Metinler tarayıcıda HTML olarak yorumlanmaz.

Tekrar üretilebilir HTTP kontrolü:

```powershell
./scripts/test_ocr_service.ps1 -Port 8091 -Video ocr-canon.mp4 -ExpectedText canon -Output outputs/ocr-http-new
```

Bu komut sunucunun medya kökünde `ocr-canon.mp4` ister; yalnız kendi OCR işini
oluşturur. Beklenen kelime denetimi isteğe bağlıdır. İş JSON'u ve JPEG yerelde
kaydedilir; hazır sunucuları başlatmaz veya durdurmaz.

### 10 Ekim 2026 uçtan uca doğrulaması

Release derlemesi, **38/38 CTest (57,17 saniye)** ve web durum testleri geçti.
OCR ile segmentation aynı sunucuda açılarak 8091 portunda gerçek HTTP işi
çalıştırıldı. Canon örneğinden üretilen bir saniyelik, 10 FPS videonun ilk
karesinde `canon` bulundu: detection **0,993406**, recognition **0,999435**.
Yerel kanıtlar `outputs/ocr-http-20261010/smoke/job.json` ve `preview.jpg`.
Tarayıcıdaki **Yazıları oku** düğmesiyle ayrıca yeni bir iş gönderildi;
tamamlanan iş, metin listesi ve kutulu önizleme panelde doğrulandı.

Test videosu sabit fotoğrafın tekrarıdır; doğal video veya genel OCR kalite
benchmark'ı değildir. Örnek hazırsa, sunucunun medya kökünde henüz bulunmayan
bir hedef dosyaya şu komutla benzer bir fixture oluşturulabilir:

```powershell
ffmpeg -hide_banner -loglevel error -n -loop 1 -i artifacts/media/ocr/text_det_test2.jpg -t 1 -r 10 -an -c:v libx264 -threads 2 -pix_fmt yuv420p artifacts/media/ocr-canon.mp4
```

HTTP smoke betiğine dosyanın sunucunun medya köküne göre yolunu verin.

## Kaynaklar ve lisans

Çoklu görüntü, kaçırılan/fazladan bölge ve karakter hatası raporu için
[C++ OCR değerlendirme aracını](ocr-evaluation.md) kullanın. İki örneklik kontrol
genel doğruluk benchmark'ı değildir; küçük Welcome kırpımında tespit başarısızdır.

- [PP-OCRv3 / OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/text_detection_ppocr)
- [CRNN EN / OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/text_recognition_crnn)
- [OpenCV metin detection/recognition API](https://docs.opencv.org/4.10.0/d4/d43/tutorial_dnn_text_spotting.html)
- [Sabit örnek fotoğraf kaynağı](https://github.com/opencv/opencv_extra/blob/22b4a7bc7cf5e4ddd3a0426eafc88bae28d3dfc3/testdata/dnn/text_det_test2.jpg)

İki model klasörü Apache-2.0 lisanslıdır; hazırlama betiği lisansları model
dosyalarının yanına indirir. Model commit'i, URL ve SHA256 değerleri
`artifacts/models/ocr-en/manifest.json` içindedir. Örnek fotoğrafın SHA256 değeri
`a8e1910727ff5043e65f3039cdb22dbf23509dfd73293c81805fabca327fadc0`.
Fotoğraf yalnız upstream test örneği olarak yerelde indirilir; bu repoda yeniden
dağıtılmaz, modelin Apache lisansı fotoğrafa genellenmez. Ticari veri dağıtımı
için kaynak/görsel hakları ayrıca değerlendirilmelidir.
