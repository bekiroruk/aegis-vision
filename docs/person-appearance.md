# OSNet ile yerel kişi görünüşü eşleştirme

Bu bileşen, gerçek kişi görüntülerinden **512 boyutlu özellik** çıkarır ve aynı
videodaki Kalman takipçisinin eşleştirmesine ekler. Uygulama ve inference C++20 /
OpenCV DNN CPU FP32'dır. Python yalnızca sabit modelin hazırlanması ve bağımsız
referans üretimi içindir; çalışma anında Python, PyTorch veya Torchreid gerekmez.
Eski takipçiler ve varsayılan IoU yöntemi korunur.

## Model ve sözleşme

[Yazarın model deposundaki](https://huggingface.co/kaiyangzhou/osnet/tree/a5c5cc037c24235cda3b21085b93ad77c9616224)
**OSNet x0.25, MSMT17-combineall** ağı kullanılır. ImageNet sınıflandırıcı ağı veya
CLIP embedding'i kişi Re-ID modeli diye sunulmaz. Hazırlayıcı yalnızca sabit,
incelenmiş [MIT mimari dosyasını](https://github.com/KaiyangZhou/deep-person-reid/blob/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/torchreid/models/osnet.py)
indirir; bütün Torchreid paketini çalıştırmaz. Ağ sınıfları AST ile seçilir,
indirici ve eski checkpoint yükleme yardımcıları dışarıda kalır.

Checkpoint `torch.load(weights_only=True, map_location="cpu")` ile yüklenir.
565 backbone tensorunun ad, boyut ve türleri bire bir kontrol edilir; yalnızca
bilinen 4.101 kimlikli eğitim classifier başlığı kullanılmaz. Ağırlıklar eğitimli
özellik katmanına yüklenir; rastgele ağırlıklı bir modelle demo yapılmaz.

- HF revision: `a5c5cc037c24235cda3b21085b93ad77c9616224`.
- Checkpoint: `osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`.
- Checkpoint: 9.336.983 byte, SHA256 `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`.
- Mimari revision: `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`, kaynak SHA256
  `c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae`.
- Yerel ONNX: 891.011 byte, opset 13, SHA256
  `32d0f46f48f7a6dd783dc17e4715ad262ad95195ff8496f6c8a1f95cdba730b5`.

Giriş `images`, float32 `[1,3,256,128]`; çıkış `features`, float32 `[1,512]`.
Kişi kutusu görüntü sınırına kırpılır, 128×256'ya OpenCV `INTER_LINEAR` ile
ölçeklenir, BGR→RGB dönüşür, 255'e bölünür. Kanal mean `[.485,.456,.406]`, std
`[.229,.224,.225]`; çıktı L2-normalize edilir. Upstream PIL dönüşümleriyle bit
düzeyinde eşitlik iddiası yoktur; bu projenin dönüşümü açık ve testlidir.

Native adaptör boyut/stride/taşma, finite kutu ve kişi sınıfı, manifestin bütün
alanları, sabit kaynak kimliği, dosya boyutu ve SHA256 kontrolü yapar. Model ve
manifest bundle dışına yönlenemez. Sıfır giriş probe'u isimli çıkış sözleşmesini
doğrular. Gerçek feature sıfır normlu veya non-finite ise işlem durur. Model
instance'ı thread-safe değildir; eşzamanlı worker için ayrı instance gerekir.

## Takip kararı

`kalman-reid` bağımsız deneysel seçenektir. Sabit başlangıç ayarları:

- CenterOnly Mahalanobis gate 9.2103, IoU ≥ .30, aynı sınıf.
- Low .10, high .35, yeni takip .50; kayıp takip ömrü 20 kare.
- Cosine distance ≤ .20; reward `.5 * IoU + .5 * cosine_similarity`.
- Her kimlik için bir L2-normalize prototip; high-score eşleşmelerde
  `normalize(.9 * eski + .1 * yeni)` güncellemesi. Low-score eşleşme prototipi
  değiştirmez. Doğum, ilk gözlemin normalize feature'ını kaydeder.
- Sıra: aktif/high → kalan aktif/low → kayıp/kalan high.

Görünüş, sınıf/IoU/hareket kapılarını geçemeyen bir adayın eşleşmesini sağlamaz.
Low gözlem yeni ID doğuramaz. Yalnızca gerçekten gözlenen kutular çıktıya girer;
tahmin edilen konumlar görünür kişi diye çizilmez. En fazla 256 canlı state /
512 detection, 512 boyutlu tek prototip; sınırsız gallery veya global kimlik yoktur.
Hatalı feature girişi tracker state/sayaçlarını değiştirmeden reddedilir.

Bu yaklaşım **tam DeepSORT değildir**: gallery, yaş sıralı cascade, ayrı motion
fallback ve geniş çaplı yeniden tanıma uygulanmaz. `.20` başlangıç mesafesi
[DeepSORT CLI'ın bilinen başlangıç değeridir](https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py);
bu OSNet ve bu sahneler için kalibre edildiği iddia edilmez. Tek kişi prototipi,
crop bozulması/örtülme/ışık değişimiyle eşleşmeyi reddedebilir. Parametreler
benchmark sonuçlarına bakılarak bu koşuda ayarlanmaz.

## Hazırlama ve çalıştırma

Önce [etiketli yerel video hazırlığını](model-quality.md) tamamlayın. Mevcut model
hazırlama ortamında PyTorch, ONNX, NumPy ve OpenCV gerekir. Yerel doğrulamada
torch 2.9.1, ONNX 1.19.1, NumPy 2.2.6, OpenCV 4.12.0 kullanıldı.

```powershell
python scripts/test_reid_export.py
python scripts/export_reid.py artifacts/models/osnet-x0-25-msmt17 --reference-video artifacts/datasets/mot15-tud/TUD-Stadtmitte-raw.mp4 --reference-manifest artifacts/datasets/mot15-tud/quality-manifest.json
cmake -S . -B build/search
cmake --build build/search --config Release --parallel 3
build/search/Release/aegisvision_reid_tests.exe --bundle artifacts/models/osnet-x0-25-msmt17
build/search/Release/aegisvision_video.exe --config configs/video-reid.toml artifacts/datasets/mot15-tud/TUD-Stadtmitte-raw.mp4 outputs/person-appearance-demo
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-reid --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-reid
```

Komutlardaki video adı manifestteki gerçek dosya adıyla eşleşmelidir. Bundle ve
çıktılar yeni dizin olmalı; mevcut dosyalar otomatik değiştirilmez. Model
hazırlığından sonra üretim komutları yalnız C++ çalıştırır. Search-enabled build
OSNet adaptörünü de derler; OpenCV-only video build `kalman-reid` talebini açık
hatayla reddeder, sessizce geometrik takipçiye dönmez. Search derleme bağımlılıkları
için [arama kılavuzu](search.md). Qdrant/CLIP modelini çalıştırmak gerekli değildir.

`--reid` tek başına üçüncü paneli ekler; iki Kalman flag'iyle toplam beş panel:
IoU / iki aşamalı / FullBox / CenterOnly / OSNet appearance. Sarı kutular GT,
yeşil kutular gerçek takip gözlemleri. Her kare tek YOLO çalışması paylaşır.
Yalnız yeni takipçi low-person detection kopyalarına embedding ekler; önceki
dört takipçi ve ham tespit ihracı değişmez. GT yalnız metrik/çizim girdisidir.
Exporter golden fixture'ları GT'den seçse de bu kutular inference veya eşleştirme
benchmarkına verilmez.

`appearance_ms` kopya, crop, preprocessing, her crop için forward ve L2'yi içerir;
tracker sürelerinden ayrıdır. Constructor/model probe'u dışarıdadır; ayrı crop
warmup'u yoktur ve ilk gerçek crop süreye dahildir. CSV her kare crop sayısını,
rapor toplam çağrıyı ve model/manifest hash'ini korur. Ek inference maliyeti var;
25 FPS video oynatma, 25 FPS analiz anlamına gelmez.

## Kapsam, lisans ve sonraki değerlendirme

Bu özellik kişilerin kim olduğunu belirlemez, yüz tanıma sağlamaz ve kameralar
arasında kalıcı ID garantilemez. Feature'lar sadece bellek içindeki sınırlı takip
state'inde tutulur; Re-ID vektörleri Qdrant'a veya ayrı bir profile yazılmaz.
Yetkili kamera/veri, veri minimizasyonu ve uygun kullanım koşulları kullanıcı
sorumluluğundadır.

[MIT kaynak lisansı](https://github.com/KaiyangZhou/deep-person-reid/blob/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/LICENSE)
ve yazarın model kartındaki MIT kaydı, upstream eğitim veri setinin bütün kullanım
haklarını otomatik sağlamaz. MSMT17 ve MOT veri şartları ayrıca değerlendirilir;
ticari kullanım/yeniden dağıtım izni varsayılmaz. Lisans metni indirilen bundle'da
korunur; ağırlıklar, veri ve görüntüler Git'e gönderilmez.

Golden fixture eşitliği Re-ID doğruluğu ölçümü değildir. MOT15 sahneleri önceki
geliştirme adımlarında incelendi; bunlar bağımsız hold-out test değildir. Sonraki
adım ayrı validation/test sekanslarıyla eşikleri dondurmak, HOTA ve appearance
ablation'ları ölçmek; ardından timestamp-aware canlı destek ve çoklu kamera
değerlendirmesi. Şu anda RTSP/servis backend'ine Re-ID açılmaz.
