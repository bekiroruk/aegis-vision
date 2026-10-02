# Detection ve tracking kalite başlangıç ölçümü

Bu aşama modelin **kaç ID ürettiğini** değil, etiketli nesneleri ne kadar doğru
bulup takip ettiğini ölçer. Değerlendirme, model çalıştırma ve iki takipçi C++20
içindedir. Python yalnızca yerel veri hazırlama ve bağımsız referans kontrolünde
kullanılır; servis için Python inference gerektirmez.

## Veri kapsamı ve sınırlar

Detection, mevcut CLIP arama ölçümünde seçilen 64 COCO val2017 sahnesinin **tam
görüntülerini** kullanır. Sekiz sınıf: bicycle, motorcycle, bus, train, cat, dog,
zebra, giraffe. Seçim büyük nesne kırpmalarından geldiği için taraflıdır. Hazırlayıcı,
seçilen sahnelerde bu sekiz sınıfın bütün kutularını, küçük nesne ve crowd dahil,
yeniden resmi annotation dosyasından alır; yalnızca 64 hedef kırpma değerlendirilmez.
Sekiz sınıflı küçük alt küme sonucu resmi 80 sınıflı COCO skoru değildir.

Tracking, resmi MOT15 **TUD-Stadtmitte eğitim sekansını** kullanır: 179 kare,
640×480, 25 FPS. Küçük resmi raw MP4 önizlemesi tekrar kodlanmıştır; challenge'ın
orijinal JPEG kareleriyle aynı piksel dosyaları değildir. Sonuçlar leaderboard
karşılaştırması değildir. MOT15 yedinci alan değerlendirme bayrağıdır; son üç alan
dünya koordinatıdır, MOT16 sınıf/görünürlük alanları değildir. Bayrak 0 satırlar
dahil edilmez. Resmi 1-based xywh koordinatlar 0-based xyxy'ye çevrilir, görüntü
dışındaki kutular kırpılmaz. İhraç edilen MOT tahmin dosyaları yeniden 1-based'dir.

İki veri kümesi de keşif amaçlı başlangıç ölçümüdür. Bu aşamada eğitim/fine-tuning
yapılmaz, model veya eşikler sonuçlara bakılarak seçilmez. COCO validation ve MOT
training verisi bağımsız bir hold-out test kümesi diye sunulmaz. Sonraki model
geliştirmesinde ayrı ve sızıntısız train/validation/test bölünmesi gereklidir.

Veri dosyaları `artifacts/`, ölçüm çıktıları `outputs/` altında yereldir ve Git'e
girmez. COCO görsellerinin kaynak bazlı lisansları korunmalıdır. MOT materyalleri
tarihsel olarak CC BY-NC-SA 3.0 koşulları yayımlamıştır; güncel lisans ve atıf
şartları kullanımdan önce doğrulanmalıdır. Bu veriyle ticari kullanım hakkı veya
yeniden dağıtım izni verilmez. Lisansın güncel kök sayfasına erişilememesi bir
izin anlamına gelmez.

## Tekrarlanabilir hazırlık ve çalıştırma

Önce mevcut COCO tam sahneleri ve resmi annotation dosyası gerekir. Hazırlayıcı
selection annotation SHA256 eşleşmesini kontrol eder. Bütün görüntü/video,
annotation ve manifest hash'leri ölçüm raporunda korunur. Çakışan mevcut çıktılar
sessizce değiştirilmez; yeni bir çıktı dizini/yolu seçilir.

```powershell
python scripts/test_quality_data.py
python scripts/prepare_quality_data.py coco artifacts/datasets/coco-search/annotations/instances_val2017.json artifacts/datasets/coco-search/manifest.json artifacts/datasets/coco-search/quality-manifest.json
python scripts/prepare_quality_data.py mot artifacts/datasets/mot15-tud --download

cmake --build build/search --config Release --parallel 3
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/coco-search/quality-manifest.json outputs/quality-coco-verified
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-mot-verified
```

MOT indirmesi yalnızca belirtilen iki HTTPS varlığını, sabit küçük byte
limitlerini ve 2026-10-02'de alınan içeriklerin sabit SHA256 kimliklerini kullanır.
Aynı boyutta değiştirilmiş bir yerel dosya da reddedilir. ZIP topluca çıkarılmaz; sadece sekansın `gt.txt` ve
`seqinfo.ini` kayıtları okunur. OpenCV çalışma anında boyut/FPS ve eksiksiz kare
sayısını ayrıca doğrular. Hazırlayıcı testleri ağ ya da model gerektirmez.

## Detection protokolü

CPU / FP32 / OpenCV DNN YOLOv8n, 640 giriş, NMS IoU 0.45 ve 0.001 skor tabanı
kullanılır. Düşük skor tabanı AP eğrisini tutmak içindir; canlı servis varsayılan
eşiği değildir. IoU .50:.05:.95, 101 recall noktası, category-aware bbox matching,
all-area ve görüntü/sınıf başına en fazla 100 tahmin ile AP50 ve AP50:95 ölçülür.
Crowd eşleştirmeleri normal nesne eşleştirmelerinden sonra ve prediction-area
paydasıyla değerlendirilir; crowd eşleşmeleri TP/FP olarak sayılmaz. Ayrıca skor
0.25 ve IoU 0.5 noktasında precision/recall, TP/FP/FN raporlanır.

Sonuç, pycocotools'un resmi `COCOeval` uygulamasıyla aynı image/category ID,
aynı orijinal annotation ve aynı dışa aktarılan tahminlerden bağımsız kontrol
edilir. Ana uygulama pycocotools'a bağlı değildir. Hiç geçerli GT olmayan metrik
alanları 0 başarı skoru yerine `null` olarak raporlanır.

## Tracking karşılaştırması

IoU takipçisi ve mevcut iki aşamalı takipçi **aynı karelere ve tek YOLO inference
sonucuna** bakar; durumları ayrıdır ve GT takipçilere verilmez. IoU: skor ≥0.35;
iki aşamalı: low ≥0.10, high ≥0.35, new track ≥0.50. Her ikisi match IoU 0.30,
20 missed-frame sınırıyla çalışır. İki aşamalı uygulama tam ByteTrack/Kalman veya
Re-ID sistemi değildir.

Değerlendirme IoU 0.5'te CLEAR TP/FP/FN, ID switch, MOTA/MOTP ve global identity
eşleştirmesiyle IDTP/IDFP/IDFN, IDF1 raporlar. Bu kapsam HOTA içermez. Global
identity eşleştirmesi yalnızca CLEAR seçilmiş eşleşmelerinden hesaplanmaz;
eşik üstündeki bütün olası GT/prediction örtüşmeleri kullanılır. Reference
TrackEval `CLEAR` ve `Identity` uygulamalarının hesaplarıyla ayrıca doğrulanır.

`comparison.avi` iki takipçiyi gerçek video karelerinde yan yana gösterir:
sarı kutular GT, yeşil kutular takip tahminleridir. `tracking-frames.json` aynı
GT/tahminleri reference kontrolüne, `iou-mot.txt` ve `two-stage-mot.txt` dış
araçlara sunar. Bu çıktılar üretilmiş demo görseli değil gerçek inference kaydıdır.

## Performans kapsamı

Model yüklemesi ve varsayılan beş warmup çalışması latency örneklerinden ayrıdır.
OpenCV bir thread ile çalışır. Detector süresi sahip olunan frame kopyasını,
preprocessing, forward ve NMS'yi içerir. CSV ham kare sürelerini; JSON ortalama,
p50/p95, min/max ve örnek sayısını verir. Yüzdelikler nearest-rank yöntemiyle
hesaplanır. Görüntü okuma, video decode ve takipçi süreleri ayrı alanlardır.

Loop süresi çizim, karşılaştırma AVI yazım çağrıları ve CSV/MOT biçimlendirmesini
de içerir; model yükleme, warmup, metrik hesapları, checksum kontrolü, son flush,
AVI kapatma ve JSON serileştirmesini içermez. Video decode örneklerinde warmup
için ilk okunan kare yoktur. Bu tek süreç, kısa sekans ve CPU koşullarındaki
sonuç canlı RTSP FPS garantisi, GPU/TensorRT ölçümü veya ürün SLA'sı değildir.
Donanım, build, model/config/data hash'leri ve arka plan yükü rapora eklenmelidir.

## Bağımsız resmi kontrol

CLI mevcut [search derlemesinde](search.md) bulunur; çalıştırırken CLIP veya Qdrant
kullanmaz. Core metrik testleri ise OpenCV olmadan da Windows/Linux'ta çalışır.
Referans kontrolü geliştirme içindir, C++ binary'nin runtime bağımlılığı değildir.
NumPy/SciPy bulunan ayrı bir Python ortamında resmi bağımlılıkları sabitleyin:

```powershell
python -m pip install --no-deps --target artifacts/deps/quality-reference pycocotools==2.0.11
git clone https://github.com/JonathonLuiten/TrackEval.git artifacts/deps/quality-reference/TrackEval
git -C artifacts/deps/quality-reference/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
python scripts/verify_quality_reference.py coco artifacts/datasets/coco-search/quality-manifest.json outputs/quality-coco-verified artifacts/datasets/coco-search/annotations/instances_val2017.json
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-mot-verified
```

Bu bilgisayarda `python` yerine NumPy/SciPy içeren
`work/yolo-export/Scripts/python.exe` kullanıldı. Başarılı kontrol
`reference-validation.json` üretir; AP ve oranlar için mutlak tolerans 1e-6,
bütün sayımlar için tam eşitlik gerekir. Manifest, ham orijinal GT ve dışa
aktarılan tahmin hash'leri doğrulama kaydına bağlanır. TrackEval'deki eski
`np.int` yazımı için Python `int` alias'ı sağlanır; metrik algoritması değiştirilmez.
COCO'nun `linspace` recall/IoU grid yuvarlaması C++'ta korunur ve regresyonla test
edilir; 100 GT/70 doğru tahminde AP 70/101 olur, 71/101 değil.

## Bu bilgisayarda doğrulanan baseline — 2026-10-02

Release / MSVC 14.44, OpenCV 4.12.0; Windows 11 Pro 10.0.26300,
Intel i7-12650H (10 core / 16 logical), yaklaşık 15.7 GiB RAM. GPU kullanılmadı.
Model YOLOv8n ONNX, CPU/FP32, OpenCV bir thread, beş warmup. Eğitim veya eşik
ayarı yapılmadı. İşletim sistemi/arka plan yükü ve güç modu sabitlenmedi; MOT
koşusu sırasında kısa bir COCO referans kontrolü de çalıştı. Süreler bu yerel
koşunun gözlemidir; kontrollü performans laboratuvarı sonucu değildir.

Detection verisi 64 tam sahne / 8 sınıf / 159 normal + 2 crowd kutudur.

| Ölçüm | Sonuç |
|---|---:|
| AP50 | %83.02 |
| AP50:95 | %63.65 |
| Precision / Recall (skor .25, IoU .5) | %86.78 / %66.04 |
| TP / FP / FN / ignored | 105 / 16 / 54 / 5 |
| Detector p50 / p95 | 429.72 / 485.34 ms |

Tracking verisi 179 kare / 10 kimlik / 1156 GT kutusudur. Aynı YOLO çıktısı
iki takipçiye uygulanmıştır; hiçbir kare atlanmamıştır.

| Ölçüm | IoU | İki aşamalı |
|---|---:|---:|
| IDF1 | %77.20 | %68.36 |
| MOTA | %79.67 | %77.25 |
| TP / FP / FN | 981 / 47 / 175 | 979 / 74 / 177 |
| ID switch | 13 | 12 |
| IDTP / IDFP / IDFN | 843 / 185 / 313 | 755 / 298 / 401 |
| Tracker p50 / p95 | 0.0103 / 0.0161 ms | 0.0253 / 0.0399 ms |

Ortak detector p50/p95 **442.54 / 658.78 ms**; iki tracker ve çizim/yazım içeren
döngü yaklaşık **2.07 FPS**. Çıktı videosunun 25 FPS oynatılması inference'ın
25 FPS çalıştığı anlamına gelmez. Çok düşük AP skor tabanı (.001) da bu koşunun
maliyetinin parçasıdır; canlı servis varsayılanı daha yüksek eşik kullanır.

Bu sekans üzerinde iki aşamalı yöntem ID switch'i bir azaltırken IDF1 ve MOTA'da
daha kötüdür. Daha gelişmiş eşleştirme otomatik olarak daha iyi kalite değildir.
Sonraki çalışma örtüşme/kaybolma/yeniden eşleşme hatalarını incelemek ve ayrı
validation verisi üzerinde Kalman/Re-ID seçeneklerini değerlendirmektir;
aynı sekansın skoruna bakıp parametre seçmek bağımsız iyileşme kanıtı değildir.

Resmi **pycocotools 2.0.11** ve **TrackEval
12c8791b303e0a0b50f753af204249e622d0281a** ile toplam/per-class AP, bütün tracking
sayımları ve oranlar kontrol edildi; iki `reference-validation.json` başarılıdır.
AP/oran toleransı 1e-6, sayımlar bire bir eşittir.

Yerel çıktılar:

- `outputs/quality-coco-verified/`: `report.json`, `predictions.json`, `timings.csv`.
- `outputs/quality-mot-verified/`: `report.json`, `tracking-frames.json`,
  `iou-mot.txt`, `two-stage-mot.txt`, `timings.csv`, `comparison.avi`.
- `comparison.mp4`: 1280×480 H.264, 25 FPS, **179 kare / 7.16 saniye**;
  sarı GT ve yeşil takip kutuları, solda IoU / sağda iki aşamalı yöntem.

MP4 dönüşümü yeniden yapılacaksa var olan dosya korunur:

```powershell
ffmpeg -nostdin -n -i outputs/quality-mot-verified/comparison.avi -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p -movflags +faststart outputs/quality-mot-verified/comparison.mp4
```

Dosya kimlikleri (SHA256):

```text
model:          0ce2a36db63861d17a81866856ee37283a1e18eeb062369352267f3404239781
config:         9d341c5075a566ccfa50898f6bd502808e485e84da36641c4a33d8ba38428111
COCO manifest:  b68d404fc44c1ffe133250b80a5848a2272ace28ab69e8ef33e9f1a933235a1b
COCO labels:    e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f
MOT manifest:   adc32ea13e1338d083968d2278a87487136f764beb13b1374a24ab533c009187
MOT video:      057efff329eb73f3434649f9b21b37d0d3ca7de8f194524140161e2d13a6ae33
MOT labels GT:  009b3ef8df68c963fd8104350083fd6bc9798b6b435858b99dbd1385cfbde873
```

## Protokol kaynakları

- [Resmi COCO veri ve değerlendirme](https://cocodataset.org/#detection-eval).
- [Resmi COCOeval uygulaması](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
- [Resmi MOT15 sekans listesi](https://motchallenge.net/data/MOT15/).
- [MOTChallenge benchmark makalesi; 1-based koordinatlar ve GT consideration alanı](https://arxiv.org/abs/1504.01942).
- [TrackEval CLEAR](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py)
  ve [Identity](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py).
