# HOTA ve yeni sekanslarda sabit ayarlı değerlendirme

Protokol tarihi: 2026-10-05. Önceki geliştirme sahneleri TUD-Campus ve
TUD-Stadtmitte'dir. Bu turda model/parametre seçimi yapılmadan, bütün kareleriyle
ETH-Sunnyday (validation) ve PETS09-S2L1 (test) ayrılmıştır. Model, detector config,
beş takipçi ve eşikleri [sabit protokol](../configs/tracking-transfer-v1.json)
kaydeder. Runner farklı model/config hash'i, eksik takipçi, değiştirilmiş eşik
veya yanlış sekans/split'i inference ve çıktı oluşturmadan reddeder.

Bu ayrım proje düzeyindedir. İki sekans da halka açık MOT15 **training** verisidir;
resmi gizli test, geniş bağımsız genelleme kanıtı veya ön eğitim veri çakışmasının
olmadığı garantisi değildir. Bu tek koşudan sonra sahneler gözlenmiş sayılır;
gelecekte eşik ayarı yapılırsa yeni, dokunulmamış bir test grubu gerekir.

## Veri ve seçim

| Kullanım | Sekans | Kare | Boyut | FPS | GT kutusu / kimlik |
|---|---|---:|---|---:|---:|
| Validation | ETH-Sunnyday | 354 | 640×480 | 14 | 1.858 / 30 |
| Test | PETS09-S2L1 | 795 | 768×576 | 7 | 4.476 / 19 |

ETH hareketli platform görüntüsüdür; PETS yukarıdan yürüyen kişileri gösterir.
Seçim süre, kamera ve çözünürlük metadata'sıyla, inference sonuçlarından önce
yapıldı. Kare/kişi seçimi, veri artırma veya test üzerinde eşik araması yoktur.
Başlangıç adayı KITTI-17 sonuç üretilmeden elendi: resmi MP4 612×184, GT metadata
1224×370'tir. Yarım yükseklik 185 olmalıydı; 184'e geçişin resize/crop işlemi
doğrulanamadığından kutular varsayımla ölçeklenmedi. Bu karar protokolde kayıtlıdır.

Kullanılan videolar resmi tekrar kodlanmış raw MP4 önizlemeleridir. Orijinal
challenge JPEG dosyalarıyla piksel eşitliği veya leaderboard karşılaştırılabilirliği
iddia edilmez. Kullanılan iki videoda decode boyutu, FPS ve bütün kare sayısı
metadata ile doğrulanır. MOT15 flag=0 satırları dışarıda kalır: ETH 43, PETS 174.
1-based xywh → 0-based xyxy dönüşümü korunur; GT görüntü sınırına kırpılmaz.

- ETH video: 8.750.726 byte; SHA256 `1323d5c68c19a4f8acebcce9b6dcce58467d8a84bcec59e9c376930150499a17`.
- PETS video: 21.357.352 byte; SHA256 `e03a8d3ae953c640a2eb26bdeff8502111e8c5ac11e529f4dc67931dd548557e`.
- Etiket ZIP'i: önceki sabit `b48ee31a720a3dae558df4303b82924bd862067138bc5376e1c86aadb82f782a`.

Hazırlayıcı yalnızca sabit boyut/hash'li bu varlıkları indirir, gereken ZIP
kayıtlarını sınırlı boyutla okur ve mevcut farklı dosyayı değiştirmez. Manifest,
tam protokolü ve protokol dosyasının SHA256 değerini içerir. Veri/model/çıktı
Git dışında kalır; önceki [veri kullanım notları](model-quality.md) geçerlidir.
[Resmi sekans metadata'sı](https://motchallenge.net/data/MOT15/).

## Native HOTA hesabı

HOTA, nesne bulma doğruluğu (DetA) ile kimlik eşleştirme doğruluğunu (AssA)
birlikte ölçer. C++ çekirdek 19 IoU eşiğinde `.05:.05:.95` hesap yapar:
`HOTA_alpha = sqrt(DetA_alpha * AssA_alpha)`. Raporlanan HOTA, bu 19 HOTA
değerinin aritmetik ortalamasıdır; ortalama DetA/AssA'nın geometrik ortalaması
alınmaz. LocA, eşleşen kutuların ortalama IoU'sudur; TP olmayan eşikte resmi
uygulama geleneğiyle 1 olur, başarı kanıtı sayılmaz. Hiç GT yoksa üst seviye
ortalamalar `null`, GT varken hiç prediction yoksa HOTA 0'dır.

Önce her karedeki bütün IoU'larla normalize eşleşme olasılıkları biriktirilir.
Sekans genelindeki kimlik uyumu ile IoU çarpılarak kare başına tek maksimum
ağırlık eşleştirmesi yapılır. Sonra aynı eşleşmeler her alpha'da filtrelenir;
eşik başına yeniden eşleştirme yapılmaz ve CLEAR'ın continuity eşleşmeleri
kullanılmaz. AssA, kimlik çifti Jaccard değerlerini eşleşen gözlem sayısıyla
ağırlıklandırır. Sekansın tamamına bakan bu işlem yalnız **offline metrik**
hesabıdır; takipçilere GT veya gelecek kare verilmez.

Kaynaklar: [HOTA makalesi](https://arxiv.org/abs/2009.07736),
[sabit TrackEval uygulaması](https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/metrics/hota.py),
[atıf/lisans](../THIRD_PARTY_NOTICES.md).

Çekirdek mevcut frame/nesne/kimlik/toplam işlem sınırlarını paylaşır. En fazla
19 milyon uint32 kimlik çifti sayacı vardır; bütün sekans IoU matrisleri bellekte
tutulmaz. Giriş validation'ı ağır matris/assignment işinden önce tamamlanır.
Bağımsız Python aracı resmi TrackEval ile her takipçinin bütün 19 eşiğinde
TP/FP/FN, HOTA/DetA/AssA/LocA ve ortalamaları denetler; tolerans 1e-6'dır.
Eski HOTA içermeyen raporların CLEAR/Identity kontrolü korunur. Yeni raporda
HOTA varsa bütün seçili takipçilerde bulunması zorunludur. NumPy'nin kaldırdığı
`np.int`/`np.float` adları eski Python türlerine alias edilir; metrik kodu değişmez.

## Ölçülen sonuçlar — 2026-10-05

Skorlar yüzde olarak gösterilir; FP/FN/IDSW, CLEAR IoU .50 sayılarıdır.
HOTA/DetA/AssA ise 19 eşik ortalamasıdır. Bunların TP/FP/FN değerleri aynı
eşleştirmeden gelmez. Sekanslar ayrı raporlanır; basit sekans ortalaması bir
pooled HOTA sonucu diye sunulmaz.

Bu turda bilgisayarın duvar saati/işlem süresi arasında uzun kesintiler gözlendi.
Ham timing dosyaları korunur, fakat bu koşuların FPS/p50/p95 değerleri kontrollü
performans karşılaştırması olarak kullanılmaz. Sıralı dosya decode'u bütün
kareleri işler; gerçek zamanlı kaynakta olduğu gibi zaman aşımına bağlı kare
atlama yoktur. Kalite sonuçları ayrı değerlendirilir.

### ETH-Sunnyday — validation

| Takipçi | HOTA | DetA | AssA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| IoU | 61,60 | 54,67 | 69,57 | 77,21 | 9 | 682 | 147 |
| İki aşamalı | 63,35 | 54,86 | 73,28 | 79,71 | 3 | 675 | 153 |
| Kalman tam kutu | 55,60 | 55,64 | 55,83 | 64,33 | 32 | 619 | 163 |
| Kalman merkez | 61,01 | 54,74 | 68,18 | 73,36 | 17 | 674 | 156 |
| Kalman + OSNet | 57,83 | 56,66 | 59,26 | 65,80 | 34 | 556 | 175 |

En yüksek HOTA/IDF1 iki aşamalı yöntemdedir. OSNet eklenmesi merkez yöntemine
göre FP'yi 118 azaltırken FN'yi 19, kimlik değişimini 17 artırdı; HOTA 3,18
yüzde puan düştü. DetA artışı AssA düşüşünü karşılamadı. Bu fark eşleştirme
politikasının etkisini gösterir; tek başına encoder kalitesini izole etmez.
194 appearance rejection aday-çift sayısıdır, 194 kişi veya kare değildir.
Varsayılan IoU ve deneysel OSNet statüsü değiştirilmedi.

Beş takipçinin CLEAR/Identity değerleri ve HOTA'nın bütün 19 eşiği resmi
TrackEval `12c8791b303e0a0b50f753af204249e622d0281a` ile 1e-6 toleransta geçti.
Çıktı: `outputs/quality-transfer-eth-v1`. Yerel rapor SHA256:
`2dda941d7a76d26cd53f5e05ff111104a661b6d1597a128f7e8748e5b5746bb2`;
ham takip kareleri: `0ca93164c0bd5bbc2a10edd24316add5f4603c48d3b599e729640061efe13b5e`.

### PETS09-S2L1 — test

| Takipçi | HOTA | DetA | AssA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| IoU | 37,70 | 60,99 | 23,44 | 43,00 | 87 | 562 | 402 |
| İki aşamalı | 46,15 | 61,88 | 34,50 | 56,44 | 48 | 535 | 387 |
| Kalman tam kutu | 34,81 | 61,86 | 19,75 | 37,02 | 395 | 504 | 402 |
| Kalman merkez | 41,89 | 61,79 | 28,51 | 48,38 | 126 | 536 | 394 |
| Kalman + OSNet | 41,82 | 61,72 | 28,45 | 47,51 | 220 | 496 | 418 |

Bu sekansta da en yüksek HOTA/IDF1 iki aşamalı yöntemdedir. Merkeze OSNet
eklenince FP 40 azalırken FN 24, IDSW 94 artmıştır. HOTA farkı yalnız -0,07
yüzde puandır. HOTA ve CLEAR IDSW farklı eşleştirme/protokol kullanır;
bu sayıların paralel değişmesi gerekmez. Tek metrik diğerinin yerine geçmez.
378 appearance rejection
aday-çift kaydedildi; kapasite reddi veya sayısal reset yoktur.

Beş takipçinin CLEAR/Identity ve 19 eşikte HOTA değerleri aynı resmi referansla
1e-6 toleransta geçti. Çıktı: `outputs/quality-transfer-pets-v1`.
Rapor SHA256: `3d4202e60b80c67970a89c8c937fbbfe899b04aed79479a66e41d386528e8fbd`;
ham takip kareleri: `4e5d149b5e7a3aeab306b9d88db9475f3e92befa56615abec20ec2857d277be4`.

İki yeni sekans toplam 1.149 kare / 6.334 GT gözlemidir. İki aşamalı yöntemin
bu iki sahnede üstün olması genel üstünlük kanıtı değildir. Sonraki çalışma:
geliştirme/validation üzerinde kayıp-aktif geçişi, hareket kapısı ve appearance
eşiği için kontrollü ablation; ardından yeni, dokunulmamış test sekansları.
Bu turda hiçbir takip eşiği değiştirilmedi.

### Çıktılar ve doğrulama

Her çıktı dizininde `comparison.mp4` beş paneli (IoU, iki aşamalı, Kalman,
merkez, OSNet) gösterir. `appearance.mp4` aynı çalışmanın son panelidir;
ayrı inference veya seçilmiş başarılı kareler değildir. Sarı GT ve takipçi
kutuları birlikte gösterilir. Kaynak FPS korunur; video hızı analiz hızı değildir.

- Protokol SHA256: `19e70de3c264067fa33a43996220ac922afb11895f730cd644a4cd753ba201b1`.
- ETH manifest SHA256: `5494bbf4391426a6cf0249a6f31f2131dc9ab5a3b1f507a903c169f698748466`.
- PETS manifest SHA256: `0cc6057d8590f31dc86f3280579f57331ef018d13a3a1ee991aec5b30306e690`.
- Yerel doğrulama: 32 CTest, 15 veri hazırlama + 24 referans kontrolü + 14
  model export yardımcı testi ve web durum regresyonu geçti. Gerçek iki
  sekansın resmi kontrol sonuçları kendi `reference-validation.json` dosyasındadır.
- Model, veri ve çıktı dosyaları Git dışında; kaynak, testler ve bu özet Git içindedir.

## Tekrar üretme

Model hazırlığı için [OSNet kılavuzu](person-appearance.md). Proje uygulaması
C++'dır; Python yalnız hazırlık ve bağımsız kontrol için kullanılır.

```powershell
python scripts/prepare_quality_data.py mot artifacts/datasets/mot15-transfer-eth-v1 --sequence ETH-Sunnyday --download
python scripts/prepare_quality_data.py mot artifacts/datasets/mot15-transfer-pets-v1 --sequence PETS09-S2L1 --download
cmake --build build/search --config Release --parallel 3
ctest --test-dir build/search -C Release --output-on-failure
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-transfer-eth-v1/quality-manifest.json outputs/quality-transfer-eth-v1 --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-transfer-eth-v1/quality-manifest.json outputs/quality-transfer-eth-v1
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-transfer-pets-v1/quality-manifest.json outputs/quality-transfer-pets-v1 --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-transfer-pets-v1/quality-manifest.json outputs/quality-transfer-pets-v1
```

Çıktı dizini yeni/boş olmalıdır. Sabit profil beş takipçiyi gerektirir. Her kare
tek YOLO sonucunu paylaşır; yalnız OSNet yolu crop embedding ekler. Gerçek
5-panel AVI, ham tahmin JSON'u, MOT dosyaları, süre CSV'si ve metrik raporu
üretilir. HOTA/CLEAR/Identity hesapları latency ve loop FPS ölçümünün dışındadır.
Bir OpenCV thread / CPU FP32 ve beş detector warmup kullanılır; appearance
zamanı ayrı kaydedilir. Video oynatma FPS'i modelin analiz hızı değildir.
