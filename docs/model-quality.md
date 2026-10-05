# Detection ve tracking kalite başlangıç ölçümü

Bu aşama modelin **kaç ID ürettiğini** değil, etiketli nesneleri ne kadar doğru
bulup takip ettiğini ölçer. Değerlendirme, model çalıştırma ve takipçiler C++20
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
eşleştirmesiyle IDTP/IDFP/IDFN, IDF1 raporlar. Bu ilk ölçüm HOTA içermez;
2026-10-05 eklemesi için [HOTA ve transfer protokolü](tracking-transfer.md). Global
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

## Deneysel Kalman karşılaştırması — 2026-10-02

Kalite CLI'ya isteğe bağlı `--kalman` eklendi. Bayrak yoksa önceki iki takipçi ve
iki panel korunur. Bayrak varsa aktif-önce Kalman üçüncü takipçi/panel olarak
çalışır; her karede **tek YOLO sonucu**, ayrı takipçi durumları ve aynı GT kullanılır.
`tracking-frames.json` ham person tespitlerini de içerir; GT takipçilere verilmez.
Bu yeni backend **deneyseldir**, varsayılan IoU değiştirilmedi.

Hareket durumu `(cx,cy,w,h,vx,vy,vw,vh)`, zaman adımı bir ardışık decoded frame'dir.
Sınıf/IoU eşleştirmesine squared Mahalanobis kapısı `13.2767` eklenir. Aktif/high,
aktif/low, kayıp/high sırası kullanılır. Önceki low .10, high .35, new .50, IoU .30
ve 20 missed-frame eşikleri değiştirilmedi; veriye göre eşik araması yapılmadı.
Kalman ve eşleştirme politikası birlikte değiştiğinden fark yalnız filtreye
bağlanamaz. Resmi ByteTrack veya Re-ID değildir; canlı RTSP/drop-oldest için
timestamp-aware zaman adımı henüz yoktur. Ayrıntı: [ADR-0013](adr/0013-active-first-kalman-tracking.md).

İkinci sahne **TUD-Campus**, 71 kare / 640×480 / 25 FPS / 8 kimlik / 359 GT kutusu.
Bu da MOT15 **training** sekansının resmi yeniden kodlanmış MP4 önizlemesidir;
bağımsız hold-out veya challenge leaderboard verisi değildir. Veri kullanımının
ve lisansın yukarıdaki sınırları iki sekans için de geçerlidir.

| Sahne / metrik | IoU | İki aşamalı | Kalman aktif-önce |
|---|---:|---:|---:|
| Stadtmitte IDF1 | %77.20 | %68.36 | %76.77 |
| Stadtmitte MOTA | %79.67 | %77.25 | %79.41 |
| Stadtmitte ID switch | 13 | 12 | 12 |
| Stadtmitte TP / FP / FN | 981 / 47 / 175 | 979 / 74 / 177 | 976 / 46 / 180 |
| Stadtmitte IDTP / IDFP / IDFN | 843 / 185 / 313 | 755 / 298 / 401 | 836 / 186 / 320 |
| Campus IDF1 | %59.92 | %68.90 | %43.44 |
| Campus MOTA | %58.50 | %60.45 | %57.66 |
| Campus ID switch | 8 | 4 | 22 |
| Campus TP / FP / FN | 300 / 82 / 59 | 304 / 83 / 55 | 301 / 72 / 58 |
| Campus IDTP / IDFP / IDFN | 222 / 160 / 137 | 257 / 130 / 102 | 159 / 214 / 200 |

Stadtmitte'de iki aşamalı yönteme göre IDF1 **+8.41 yüzde puan**, fakat basit IoU
hâlâ daha yüksektir. Önceki aktif/kayıp kimlik karışması kare 97'de yeni backend'de
görülmez: GT2 kimlik 3'ü 120 kare korur. Campus'te ciddi regresyon vardır:
GT3 sürekli doğru tespit edilmişken kimlik 3 → 11 → 3 → 11 değişir; iki aşamalı
yöntem aynı GT için 63 kare kimlik 3'ü korur. Yeni backend tüm sahneler için
iyileştirme olarak sunulmaz ve varsayılan yapılmaz. Sonraki çalışma kutu şekli/
hareket gürültüsü, kayıp-aktif geçişleri ve appearance/Re-ID'yi ayrı validation
verisiyle incelemektir; bu iki sahnenin skorunu yükseltmek için eşik ayarlanmadı.

Campus kare 18'de GT3 için predicted width 95.07 px, observed width 133.73 px;
IoU .699 olmasına rağmen squared Mahalanobis 22.13, 13.2767 kapısını aşar.
Gait/örtüşmeyle geniş-dar kutu oynaması sabit genişlik gürültüsü modeline uymadığı
için yeni veya eski kimlikler arasında parçalanma oluşur. Üretilen gözlemlerle
bağımsız NumPy filtre rekonstrüksiyonu 82 gate rejection'ın tamamını yeniden
hesapladı. Bu, matriste bulunan bir uygulama hatası veya %99 gerçek eşleşme
garantisi değildir; nominal chi-square oranı kalibre Gaussian innovation gerektirir.

Her iki yeni raporda üç takipçinin CLEAR/Identity sonuçları resmi, değiştirilmemiş
TrackEval `12c8791b303e0a0b50f753af204249e622d0281a` ile geçti: oranlarda 1e-6,
sayımlarda tam eşitlik. Stadtmitte IoU ve iki aşamalı raporlarının 18 metrik alanı
eski başlangıç koşusuyla bire bir aynıdır. Gate rejection sayısı Stadtmitte 35,
Campus 82; numerical reset ve capacity rejection ikisinde de 0. Gate sayısı
reddedilen aday çiftleri sayar, kaçırılan gerçek kişi sayısı değildir.

Aynı CPU/model/config ve beş warmup kullanıldı. İki inference koşusu sırayla
çalıştı; referans kontrolü ve derleme inference bittikten sonra yapıldı. Güç
modu/arka plan yükü sabitlenmedi; bu kontrollü performans laboratuvarı sonucu değildir.

| Süre / hız | Stadtmitte | Campus |
|---|---:|---:|
| Detector p50 / p95 | 449.01 / 491.03 ms | 439.39 / 474.65 ms |
| IoU p50 / p95 | 0.0113 / 0.0170 ms | 0.0120 / 0.0293 ms |
| İki aşamalı p50 / p95 | 0.0251 / 0.0382 ms | 0.0272 / 0.0459 ms |
| Kalman p50 / p95 | 0.0458 / 0.0669 ms | 0.0574 / 0.0822 ms |
| Üç takipçi, çizim/yazım dahil loop FPS | 2.13 | 2.24 |

MP4'ler **1920×480 H.264**, soldan sağa IoU / iki aşamalı / Kalman; sarı GT,
yeşil takip kutuları. Stadtmitte **179 kare / 7.16 saniye**, Campus **71 kare /
2.84 saniye**, 25 FPS oynatma. Her ikisinde bütün kareler çözüldü; oynatma hızı
inference hızı değildir. Yerel çıktılar `outputs/quality-tud-kalman/` ve
`outputs/quality-campus-kalman/`: `report.json`, `tracking-frames.json`, üç
`*-mot.txt`, `timings.csv`, `reference-validation.json`, `comparison.avi/mp4`.

Yeni çıktı dizinleri seçerek tekrar üretim:

```powershell
python scripts/test_quality_data.py
python scripts/test_quality_reference.py
python scripts/prepare_quality_data.py mot artifacts/datasets/mot15-campus --sequence TUD-Campus --download
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-kalman --kalman
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-kalman --kalman
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-kalman
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-kalman
ffmpeg -nostdin -n -i outputs/quality-tud-kalman/comparison.avi -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p -movflags +faststart outputs/quality-tud-kalman/comparison.mp4
ffmpeg -nostdin -n -i outputs/quality-campus-kalman/comparison.avi -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p -movflags +faststart outputs/quality-campus-kalman/comparison.mp4
```

Model, evaluation config ve Stadtmitte hash'leri yukarıdakiyle aynı. Ek SHA256:

```text
Campus manifest: d880b78c006938ddcbf13cabd00cba61b6dda3f6d6c9efe6c7e5e6eccbb0ff25
Campus video:    95590324a7fcd27c6a5babf7e69f763be5709f788ac1a32c986a4431aee14eae
Campus GT:       6ea5c56dffa72db2d286bf3c4593465583bfe43e9ecaa110001ccce2c4d10e39
```

## Merkez kapılı Kalman geliştirme ölçümü — 2026-10-02

Campus'teki kutu boyutu titreşiminden hareketle ayrı `CenterOnly` seçeneği eklendi.
Merkezin marjinal `(cx,cy)` innovation'ı iki boyutlu kapılanır: nominal chi-square(2)
%99 eşiği **9.2103**. Eski FullBox kapısı dört boyutta 13.2767 kalır. Tam dört
ölçümlü correction, predicted-box IoU .30, sınıf, low/high/new ve lifetime
değişmez. İki eşik farklı serbestlik derecelerine aittir; yalnız sayısal eşiği
büyütüp küçültme deneyi değildir. [ADR-0014](adr/0014-center-only-motion-gating.md).

**Bu bir geliştirme koşusudur:** tasarım aynı Campus verisindeki önceki hata
analizinden seçildi. İki MOT15 training sekansını yeniden kullanmak bağımsız
validation veya hold-out kanıtı değildir. Alternatif/eşik ilk çalıştırmadan önce
sabitlendi; grid search, eşik taraması, fine-tuning veya GT'nin tracker'a verilmesi
yoktur. Genel başarı iddiası için farklı ve sızıntısız veri gerekir.

`--kalman --kalman-center` ile aynı karelerde tek inference ve dört ayrı tracker
durumu kullanıldı. Her sekans bütün kareleriyle değerlendirildi.

| Sahne / metrik | IoU | İki aşamalı | Kalman FullBox | Kalman Center |
|---|---:|---:|---:|---:|
| Stadtmitte IDF1 | %77.20 | %68.36 | %76.77 | %76.68 |
| Stadtmitte MOTA | %79.67 | %77.25 | %79.41 | %77.42 |
| Stadtmitte ID switch | 13 | 12 | 12 | 11 |
| Stadtmitte TP / FP / FN | 981 / 47 / 175 | 979 / 74 / 177 | 976 / 46 / 180 | 977 / 71 / 179 |
| Stadtmitte IDTP / IDFP / IDFN | 843 / 185 / 313 | 755 / 298 / 401 | 836 / 186 / 320 | 845 / 203 / 311 |
| Campus IDF1 | %59.92 | %68.90 | %43.44 | %67.39 |
| Campus MOTA | %58.50 | %60.45 | %57.66 | %61.84 |
| Campus ID switch | 8 | 4 | 22 | 5 |
| Campus TP / FP / FN | 300 / 82 / 59 | 304 / 83 / 55 | 301 / 72 / 58 | 302 / 75 / 57 |
| Campus IDTP / IDFP / IDFN | 222 / 160 / 137 | 257 / 130 / 102 | 159 / 214 / 200 | 248 / 129 / 111 |

Campus FullBox'a göre IDF1 **+23.95 yüzde puan**, ID switch **22 → 5**;
GT3 artık 1–63. karelerde kesintisiz kimlik 3'tedir. Yine de iki aşamalı yöntemin
IDF1'i %68.90 ile daha yüksektir. Stadtmitte'de Center IDF1 FullBox'a göre **-0.09
yüzde puan**, MOTA **-1.99 yüzde puan**; daha az ID switch her zaman daha yüksek
kalite demek değildir. Varsayılan IoU ve eski Kalman config'i değiştirilmedi.

Stadtmitte ek 25 FP'nin 21'i .10–.35 güven aralığındadır; bunlar detector'ın
gerçek gözlemleridir, çizilen hayali filtre prediction'ları değildir. On iki kutu
track 4'ün sağ kenar devamı (kare 63–74), yedisi track 2'nin kenar devamıdır
(90–96). Örneğin kare 64 skor .218 kutusunun GT IoU'su 0; daha önce ilişkili
kişinin annotation'ı kare 62'de biter. Kare 51 skor .197 kısmi kutunun maksimum
GT IoU'su .254'tür. Bu protokolde GT ile IoU .5'e ulaşmadıkları için FP sayılırlar;
görüntüde gerçekten kişi bulunmadığı iddia edilmez. Merkez kapısı boyut filtresinin
elediği bazı düşük güvenli kısmi/kenar kutularını da sürdürür. Appearance/Re-ID,
merkez/boyut gürültüsü ve yeni bağımsız veri sonraki çalışmalardır.

İki `reference-validation.json` **resmi TrackEval ile passed**: dört takipçinin
CLEAR/Identity sayımları tam eşit, oranlarda tolerans 1e-6. Eski IoU / iki aşamalı /
FullBox yöntemlerinin 18 metrik alanı ve FullBox diagnostic sayımları önceki
üçlü koşularla bire bir aynı. Center gate rejection Stadtmitte 28 / Campus 58;
iki koşuda numerical reset ve capacity rejection 0.

Aynı CPU/FP32 YOLO/config/data hash'leri ve beş warmup kullanıldı. Campus başlangıcı
yerel derlemenin son işleriyle kısmen örtüştü; güç modu ve arka plan yükü
sabitlenmedi. Süreler gözlemdir, eski koşulara karşı hız iyileşmesi kanıtı değildir.
Detector p50/p95 Stadtmitte **536.47 / 616.23 ms**, Campus **391.62 / 597.88 ms**;
Center tracker **0.0466 / 0.0635 ms**, **0.0395 / 0.0632 ms**. Dört tracker,
çizim/yazım dahil loop hızları **1.77 / 2.10 FPS**. 25 FPS MP4 oynatma inference
hızı değildir.

Yerel `outputs/quality-tud-center/` ve `outputs/quality-campus-center/` altında
rapor, ham tespit/kare JSON'u, dört MOT tahmin dosyası, CSV, resmi doğrulama ve
AVI/MP4 bulunur. MP4 **2560×480 H.264**, soldan sağa IoU / iki aşamalı / FullBox /
Center; sarı GT, yeşil takip kutuları. Stadtmitte **179 kare / 7.16 saniye**,
Campus **71 kare / 2.84 saniye**, 25 FPS; decode sayıları ayrıca doğrulandı.

Tekrar üretim (var olan sonuçları değiştirmemek için yeni çıktı dizini seçin):

```powershell
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-center --kalman --kalman-center
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-center --kalman --kalman-center
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-center
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-center
ffmpeg -nostdin -n -i outputs/quality-tud-center/comparison.avi -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p -movflags +faststart outputs/quality-tud-center/comparison.mp4
ffmpeg -nostdin -n -i outputs/quality-campus-center/comparison.avi -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p -movflags +faststart outputs/quality-campus-center/comparison.mp4
```

Yalnız `--kalman-center` üç panel üretir; eski `--kalman` zorunlu değildir.
Yerel video CLI için `configs/video-kalman-center.toml` veya `--tracker kalman-center`.
İki Kalman modu da RTSP'de desteklenmez; timestamp-aware zaman adımı henüz yoktur.

## OSNet kişi görünüşü ve beşli karşılaştırma — 2026-10-02

Gerçek eğitimli OSNet x0.25 / MSMT17-combineall, C++ OpenCV DNN ile kişilerin
512-boyut özelliklerini çıkarır. Yeni `kalman_reid` CenterOnly hareket takibine
cosine distance ≤ .20, IoU/cosine eşit reward ve high-score-only .90 EMA ekler.
Model/protokol, lisans ve tekrar üretim için [kişi görünüşü](person-appearance.md),
[ADR-0015](adr/0015-person-appearance-association.md).
Bu değerler iki ölçümden önce donduruldu; sonuçlara göre yeniden ayarlanmadı.

| Sekans | Yöntem | IDF1 | ID switch | FP | FN | MOTA |
|---|---|---:|---:|---:|---:|---:|
| Stadtmitte | CenterOnly | %76.68 | 11 | 71 | 179 | %77.42 |
| Stadtmitte | CenterOnly + OSNet | %75.98 | 19 | 47 | 182 | %78.55 |
| Campus | CenterOnly | %67.39 | 5 | 75 | 57 | %61.84 |
| Campus | CenterOnly + OSNet | %64.29 | 10 | 69 | 59 | %61.56 |

**Yanlış kutular azaldı, kimlik sürekliliği geriledi. Genel takip iyileşmesi yok;
appearance deneysel kalır ve varsayılan IoU değiştirilmez.** Stadtmitte
IDTP 845→827, TP 977→974; Campus IDTP 248→234, TP 302→300. IDF1 kaybı,
yalnız toplam ID sayısına bakılarak gizlenmez. İki geliştirme sekansı bağımsız
hold-out değildir; model, eşikler veya kullanım önerisi buradan genellenmez.

Kare çıktısı incelemesi Campus GT 7'nin CenterOnly track 11 ile 25–71 arası
tek kimlikte kaldığını, OSNet modunda 14→17→18 olarak parçalandığını gösterir
(geçişler 27 ve 29). Aynı ham tespitler görüntü soluna kırpılmıştır: kutu
genişlikleri kare 25/27/29'da 39.65/49.97/65.79, skorları .855/.865/.872.
Crop değişimi gözlenir; per-pair cosine log'u olmadığı için bu geçişlerin
belirli bir mesafe reddinden kaynaklandığı kanıtlanmaz. GT 3'ün track 3 ile 63
gözlemi korunur. Stadtmitte GT 2'nin CenterOnly ile düzelmiş 120 gözlemli track 3
sürekliliği de korunur; başka kimliklerde ek parçalanma vardır.

Campus 42 / Stadtmitte 36 `appearance_rejections`, diğer kapıları geçen aday
**çiftlerinin** cosine kapısından elenmesidir; bunlar 42/36 kaçırılan kişi demek
değildir. Appearance match 343/988, high-score prototype update 328/973,
low-score match 15/15; numerical reset ve capacity rejection iki koşuda 0.

Resmi TrackEval `CLEAR`/`Identity` iki `reference-validation.json` için **passed**:
beş takipçinin sayımları tam eşit, oran toleransı 1e-6. Eski dört takipçinin 18
metrik alanı, her karedeki tahminleri, GT ve ham detection listeleri önceki
`quality-*-center` koşularıyla bire bir aynı. Golden doğrulamada bir sentetik
renk örneği ve iki gerçek kişi crop'u için C++ input blob farkı 0; normalize
özelliklerin PyTorch/OpenCV referansına maksimum farkı 4.992e-7. Bu, inference
uygulama eşitliğidir; kişi tanıma doğruluğu ölçümü değildir.

CPU FP32, bir OpenCV thread, önceki YOLO/config/veri hash'leri ve beş detector
warmup korunur. Model ONNX SHA256
`32d0f46f48f7a6dd783dc17e4715ad262ad95195ff8496f6c8a1f95cdba730b5`,
bundle manifest SHA256
`e723b7fa52a3c3f1f7b4da154440e5512edd6a4032dc788a96a1ca6235223d66`.
Model load/probe detector ve appearance örneklerine dahil değildir; ayrı crop
warmup'u yoktur, ilk gerçek crop dahildir.

| Sekans | YOLO p50/p95 ms | Appearance p50/p95 ms/kare | Crop çağrısı | Yeni tracker p50/p95 ms | Beşli loop FPS |
|---|---:|---:|---:|---:|---:|
| Stadtmitte | 298.97 / 820.16 | 79.93 / 245.54 | 1.328 | .0986 / .3024 | 1.72 |
| Campus | 748.63 / 852.37 | 263.32 / 360.15 | 676 | .2697 / .4988 | .92 |

Appearance süresi frame/observation kopyası, bütün low-person crop'ları,
preprocessing, crop başına forward ve L2'yi içerir; tracker süresi bundan ayrıdır.
Loop bütün beş takipçi, embedding, çizim ve MJPEG/CSV yazımını içerir. Koşular
ardışık, derleme/diğer model inference işleriyle örtüşmeden çalıştırıldı; güç
modu, CPU frekansı/ısı ve dış arka plan yükü kontrol edilmedi. Bu değişken süreler
kontrollü hız karşılaştırması değildir; 25 FPS MP4 oynatımı analiz FPS'i değildir.

Yerel `outputs/quality-tud-reid/` ve `outputs/quality-campus-reid/`: rapor, ham
detection/frame JSON, beş ayrı MOT dosyası, CSV, resmi doğrulama, AVI/MP4.
Beşli H.264 MP4 **3200×480 / 25 FPS**, Stadtmitte 179 kare / 7.16 s,
Campus 71 kare / 2.84 s; decode sayıları ayrıca doğrulandı. Soldan sağa
IoU / iki aşamalı / FullBox / CenterOnly / OSNet. Kaynak kareler ve gerçek
inference kullanılır; sarı GT yalnız değerlendirme/çizim içindir.
Native `aegisvision_video --config configs/video-reid.toml` yolu da gerçek
YOLO/OSNet ile ilk üç karede decode, takip, AVI/CSV ve başarılı JSON üretimiyle
kontrol edildi; bu küçük smoke test doğruluk benchmark'ı değildir.

```powershell
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-reid --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-reid --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-reid
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-reid
```

Var olan sonuçları korumak için yeni dizin seçin. Ayrı validation/test verisi ve
HOTA, [2026-10-05 protokolünde](tracking-transfer.md) eklendi. Mesafe/görünüş
ağırlığı kalibrasyonu ve appearance ablation sonraki iş; bu iki sekansa bakıp
eşiği değiştirerek aynı veriyi bağımsız test diye sunmayın.

## Protokol kaynakları

- [Resmi COCO veri ve değerlendirme](https://cocodataset.org/#detection-eval).
- [Resmi COCOeval uygulaması](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
- [Resmi MOT15 sekans listesi](https://motchallenge.net/data/MOT15/).
- [MOTChallenge benchmark makalesi; 1-based koordinatlar ve GT consideration alanı](https://arxiv.org/abs/1504.01942).
- [TrackEval CLEAR](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py)
  ve [Identity](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py).
