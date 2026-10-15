# Re-ID: gerçek GPU metric-learning deneyi

2026-10-15. Eğitim ve native C++ karşılaştırması tamamlandı; aday kabul edilmedi.
OSNet x1.0'ın ana ağı sabit tutuldu, yalnız bias içermeyen 512×512 doğrusal
katman (262.144 parametre) eğitildi. Tam ağ fine-tuning veya başarılı canlı
çoklu-kamera takip teslimi değildir. Servis varsayılanları değişmedi.

## Eğitim ve veri ayrımı

- MOT15 eğitim bölümündeki TUD-Stadtmitte: 232 kırpma, 10 sekans-yerel kimlik.
- TUD-Campus: 73 kırpma, 7 sekans-yerel kimlik. Toplam 305 kırpma;
  sekanslar arasında 17 ayrı gerçek kişi olduğu varsayılmaz.
- Her beşinci kare, en az 20×64 piksel; kimlik başına en az dört gözlem.
  Her batch aynı sekanstan dört kimlik × dört ayrı gözlem içerir.
  Pozitif/negatif çiftler yalnız o sekansın GT kimliklerinden çıkarılır.
- Birim uzunluklu özelliklerde batch-hard Euclidean triplet loss, margin 0,2;
  identity matrisinden sapmaya 0,001 ağırlıklı kareler toplamı cezası.
  Adam, öğrenme oranı 0,001, seed 17, 200 güncelleme. Yalnız son checkpoint;
  erken durdurma veya hiperparametre araması yok.
- RTX 3060 Laptop GPU / PyTorch 2.9.1+cu130 üzerinde gerçekten çalıştı.
  Ağırlık değişimi L2 = 4,03315; ana ağın bütün tensorları bit düzeyinde sabit kaldı.
  Eğitim raporu süre alanı özellik çıkarma + eğitimi kapsar, ONNX export'u kapsamaz.
- WILDTRACK eğitim kodunda okunmaz. MOT15 TUD daha önce tracking geliştirmesi
  ve referans kontrollerinde kullanılmıştır; yeni/kör veri diye sunulmaz.
  ETH validation ve PETS son test bu eğitimde kullanılmaz.

Kayıp formülü: [PyTorch TripletMarginLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.TripletMarginLoss.html).
Veri koşulları: [resmî MOT15 sayfası](https://motchallenge.net/data/MOT15/)
CC BY-NC-SA 3.0 ve kaynak veri koşullarını belirtir. Veri, kırpmalar ve öğrenilmiş
model yerel araştırma çıktısı olarak tutulur, repoya yüklenmez. Upstream kaynak
kodunun MIT bildirimi eğitim verisinin koşullarını ortadan kaldırmaz; dağıtım ve
ticari kullanım için ayrıca inceleme gerekir.

## Sabit geliştirme karşılaştırması

Eğitim politikası eğitimden önce, model hash'ini içeren seçim politikası adayın
geliştirme inference'ından önce yazıldı. Aynı WILDTRACK C3→C4 geliştirme verisi:
39 kare, 304 sorgu, 94 karşılığı bulunan sorgu, 210 karşılığı olmayan sorgu.
GT kişi kutuları kullanılır; detection/tracking uçtan uca ölçümü değildir.
Single-view özellik, benzerlik 0,85 ve belirsizlik farkı 0,05 sabit tutuldu.

| Ölçüm | OSNet x1.0 başlangıç | Eğitilen doğrusal katman |
|---|---:|---:|
| Doğru ilk aday / karşılığı olan sorgu | 66/94 | 64/94 |
| Rank-1 | %70,21 | %68,09 |
| Doğru / yanlış kabul | 0 / 0 | 0 / 0 |
| Karşılık recall | %0 | %0 |

Fark **−2,13 yüzde puanı**; gerekli +2 puan artışı geçmedi.
Sıfır yanlış kabul başarı değildir: hiçbir doğru eşleşme de kabul edilmedi.
Seçici `candidate_selected=false`, `production_promoted=false` döndürdü.
Eğitim kaybı kalite artışı sayılmadı; sonuç görüldükten sonra ayarlar değiştirilmedi.
Yeni eşik taraması veya son test yapılmadı. Önceden görülmüş C5/C6 yeni kör test
olarak tekrar kullanılamaz. Sonraki kalite çalışması için daha çeşitli ayrı eğitim
verisi ve dokunulmamış yeni test gerekir; bu küçük deney yeterli olmadı.

## Yerel kanıtlar ve tekrar üretme

- `artifacts/models/osnet-x1-0-metric-tud-v1/training-report.json`: 200 gerçek
  güncelleme, parametre değişimi, GPU ve kaynaklar.
- Aynı klasörde `training-data.json`: sekans/kare/kimlik/kutu ve kırpma hash'leri;
  `metric-head.f32`: 1.048.576 bayt, little-endian FP32 512×512 ağırlık.
- `model.onnx` + v2 `manifest.json`: öğrenilmiş katman ONNX'e birleşiktir;
  native C++ çalışma zamanı Python'a ihtiyaç duymaz. v1 modeller desteklenir.
  Model hash'i özellik uzayını ayırır; eski ve yeni embedding'ler karıştırılmaz.
- `outputs/reid-training-dev-v1/report.json`: native C++ geliştirme ölçümü.
- `outputs/reid-training-selection-v1/selection.json`: otomatik ret kararı.

Native C++ üç referans girdide PyTorch/OpenCV ile karşılaştırıldı:
en büyük embedding farkı 0,0000012815; preprocessing farkı sıfır.
Referans kişi kırpmaları eğitim kaynağındandır; doğruluk testi değil uygulama
uyumluluğu testidir. Bozuk v2 provenance/lisans/hash sözleşmeleri de reddedilir.

Yerel doğrulama: 10 eğitim protokolü/sampler/loss testi, 15 export testi,
23 model seçim testi ve 13 kalibrasyon seçim testi geçti. Temel C++ paketi
7/7, hedefli Re-ID/crossview paketi 2/2 geçti; eski x1.0 ve eğitilen modelin
üçer native referans kontrolü ayrıca geçti. Hafif CI eğitim testinde Torch
kurulu değilse yalnız üç Torch testi atlanır; gerçek GPU eğitimi CI işi değildir.

Hazır yerel eğitim ortamında, repo kökünden; çıktı klasörleri yeni olmalıdır:

```powershell
.\work\yolo-export\Scripts\python.exe scripts/train_reid_metric.py configs/reid-training.json artifacts/models/osnet-x1-0-msmt17 artifacts/models/osnet-x1-0-metric-tud-rerun
.\build\tensorrt\Release\aegisvision_reid_tests.exe --bundle artifacts/models/osnet-x1-0-metric-tud-rerun
```

CUDA zorunludur; CPU'ya sessiz geçiş yoktur. Kaynak video/manifest, checkpoint,
mimari kaynak ve lisans hash'leri doğrulanır. Seed sabittir ancak farklı GPU,
kütüphane veya export sürümlerinde bit düzeyinde aynı çıktı garanti edilmez.
Tekrar eğitim çıktısı için yeni model hash'ini **değerlendirmeden önce** ayrı
bir seçim politikasına kaydetmek gerekir; kayıtlı politika yalnız aşağıdaki
orijinal aday içindir. Kaydedilmiş adayın ölçüm komutları:

```powershell
.\build\tensorrt\Release\aegisvision_crossview.exe artifacts/datasets/wildtrack-flip-dev-v2/manifest.json artifacts/models/osnet-x1-0-metric-tud-v1 outputs/reid-training-dev-rerun
python scripts/select_reid_flip.py configs/reid-training-selection.json outputs/reid-model-dev-candidate-v1/report.json outputs/reid-training-dev-rerun/report.json outputs/reid-training-selection-rerun
```

SHA256 kayıtları:

```text
Eğitim politikası: 9c047d284b468487f99a3dd6cc22816fc0808ac14bf1de896be4e69af04ce71b
Seçim politikası: 6f117148c0cb8434c644536b126fe63a98b24d2f9854d2ba6e8f6611eab3c989
Başlangıç ONNX: 5bf5321c31d064507acaec3aefecf3381b24673123c61403ceab491f49f9dc6a
Eğitilen ONNX: 2a20ac28be34142eb9807bb417293ea1d0c0700cec592f54ee7a61cda50ef319
Eğitim gözlemleri: d9ec80e07bd5a9fe4fcdad60578d4517b4e7f41a66d7cd58bcc1c02cd4913138
Öğrenilen katman: ed8bd2f6fb2fb1d0c77fe42572814f47a2332c2ad512d600352c85cd5164e9ff
Geliştirme raporu: c687caf7edae67cd3063baa7b30d30f088ff53572d333c0e6168cb720e4e668f
Seçim sonucu: f568acab859dc18a16e80cea66f1aef1b1adf074ee5ccd9e0d8c1f557d1fb348
```
