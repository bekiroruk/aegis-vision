# Kişi görünümü iyileştirme deneyi: yatay çevirme ortalaması

Tarih: 2026-10-15. Mevcut OSNet x0.25 modelinin ağırlıkları değiştirilmeden,
normal kişi görüntüsü ve yatay çevrilmiş görüntünün özellikleri birleştirilir.
Bu **inference varyantıdır**, fine-tuning veya metric-learning eğitimi değildir.
Mevcut video/servis/tracklet varsayılanı korunur; otomatik üretim terfisi yoktur.

## Sonuç: aday seçilmedi

39 eşzamanlı C3/C4 çiftinde 304 sorgu; karşılığı olan 94 sorgu ve toplam
398 özellik çıkarımı. Aynı veride karşılaştırma:

| Ölçüm | Normal | Normal + yatay çevirme |
|---|---:|---:|
| Doğru ilk aday | 59 / 94 | 60 / 94 |
| Rank-1 | %62,77 | %63,83 |
| Kabul edilen doğru eşleşme | 0 | 0 |
| Kabul edilen yanlış eşleşme | 0 | 0 |
| Kaba ortalama özellik süresi | 21,16 ms | 40,53 ms |

Fark **+1,06 yüzde puan**, yani yalnız **bir ek doğru ilk aday**; önceden
belirlenen +2 puan kapısını geçmedi. Gözlenen işlem maliyeti yaklaşık 1,92×.
Bu küçük fark genellenebilir bir kazanç olarak sunulmaz. Varsayılan `single`
korundu; son kontrol C5/C6 verisi indirilmedi/ölçülmedi. Cosine `.85` eşiği
değişmedi; her iki varyantın kabul recall'ı sıfır, precision'ı tanımsızdır.

Yerel kanıtlar:
`outputs/reid-flip-dev-baseline-v1/report.json`,
`outputs/reid-flip-dev-candidate-v1/report.json`,
`outputs/reid-flip-selection-v1/selection.json`.
Geliştirme manifesti SHA256:
`bdd3583c4db7b8d7c4038cdfa4f59dfcf6c649811b523fb795e5617b247d785c`.
Ölçüm öncesi politika SHA256:
`58c32b129255b60abafeeaf8985759895eedac01e2320a6cfeaeb11fc11b41aa`.

**Sonraki aday:** daha güçlü Re-ID ağırlığı veya ayrı kimliklerle metric-learning
fine-tuning. Bu deney ağırlıkları eğitmedi; mevcut çoklu-kamera kalite açığını
kapatmış sayılmaz. Geliştirme kümesi artık görülmüştür, gelecek aday seçiminde
kullanılabilir fakat kör test diye raporlanamaz.

## Ön kontrol ve aday

Eski model üç referans görüntüde yeniden kontrol edildi: PyTorch referansıyla
minimum cosine 1, maksimum embedding farkı 4,992e-7, giriş tensoru farkı 0.
Bu sınırlı kontrol belirgin kanal/normalizasyon hatasına işaret etmedi; tek
başına bütün saha hatalarının yokluğunu kanıtlamaz.

`ReIdAugmentation::HorizontalFlipMean`, resize/normalizasyon sonrası NCHW
tensorunun yalnız genişlik eksenini ters çevirir. Normal ve çevrilmiş giriş
ayrı çalıştırılır. Her feature L2-normalize edilir, eşit ağırlıklı ortalama
yeniden normalize edilir. Sıfır norm, sonlu olmayan değer, yanlış shape hata
verir. Yeni `space_id` farklıdır: eski ve yeni vektörler aynı galeriye sessizce
karıştırılamaz. Normal modun model uzayı ve varsayılan davranışı değişmez.

Flip/geri-flip, kanal/row sınırları ve eşit ağırlıklı ortalama unit testleri
eklendi. Gerçek model testinde iki ayrı görüntü kırpmasından hesaplanan
ortalama ile yeni adaptör karşılaştırılır. İki inference nedeniyle maliyet
artar; sonuç raporu kaba ortalama feature süresini de kaydeder. Bu süre resmi
warm-up/p50/p95 performans benchmark'ı değildir.

## Veriyi ayırma ve önceden tanımlanan karar

Kaynak yine resmî WILDTRACK; [önceki teşhis](crossview-evaluation.md) C1→C2
sonuçları değiştirilmedi. Protokol `configs/reid-flip-experiment.json`:

- Geliştirme: C3→C4, çift `personID`, 60..1960 arasında 50 adımla **39 kare**.
- Son kontrol: C5→C6, tek `personID`, 60..1860 arasında 200 adımla 10 kare.
- Bu deneyde kamera çiftleri ve kişi kimliği kümeleri ayrıdır. Bazı kişiler
  önceki C1/C2 teşhisinde görülmüş olabilir; dış veri veya tamamen yeni kişi
  genellemesi iddiası yoktur. Aynı kayıt ve tekrar eden kişiler bağımlılık yaratır.
- İlk hazırlama yalnız 10 geliştirme karesi içeriyordu. Etiket sayımı ortak
  görüşün küçük olduğunu gösterince, **model karşılaştırmasından önce** 39 kareye
  genişletildi. Başarı eşiği değiştirilmedi.
- Native C++ `single` ile `flip_mean` aynı kare/kutu/kişi kümesinde karşılaştırılır.
- Eşleştirme cosine `.85`, margin `.05` sabit. Eşik taraması yapılmaz.
- Aday seçimi: en az 30 karşılıklı sorgu, Rank-1'de en az **2 yüzde puan** artış,
  yanlış kabullerde artış olmaması. Hiç kabul olmaması %100 precision değildir.
- Yalnız geliştirme raporları seçim aracına girebilir. Holdout raporları,
  farklı veri/model hash'i veya farklı sorgu sayısı reddedilir. Seçim kaydı
  politika ve rapor SHA256'larını tutar; mevcut çıktı üzerine yazılmaz.
- Geliştirme kapısını geçmeyen aday için holdout açılmaz. Geçen aday da sırf
  bu nedenle servise alınmaz; ayrıca son kontrol ve kabul kalitesi gerekir.

Geliştirme galerisinde çok az ortak görünüm olabilir. Bu nedenle buradaki
Rank-1 eski C1/C2 %25,52 ile doğrudan kıyaslanamaz; yalnız **aynı protokoldeki
baseline ve aday farkı** anlamlıdır. GT kutular kullanılır; detection, tracklet
prototipi, temporal handoff veya uçtan uca canlı doğruluğu ölçülmez.

## Tekrar üretme

```powershell
python scripts/prepare_crossview_data.py artifacts/datasets/flip-dev-new --download --profile development
cmake --build build/tensorrt --config Release --target aegisvision_crossview aegisvision_reid_tests --parallel 4
build/tensorrt/Release/aegisvision_crossview.exe artifacts/datasets/flip-dev-new/manifest.json artifacts/models/osnet-x0-25-msmt17 outputs/flip-baseline-new
build/tensorrt/Release/aegisvision_crossview.exe artifacts/datasets/flip-dev-new/manifest.json artifacts/models/osnet-x0-25-msmt17 outputs/flip-candidate-new --flip-mean
python scripts/select_reid_flip.py configs/reid-flip-experiment.json outputs/flip-baseline-new/report.json outputs/flip-candidate-new/report.json outputs/flip-selection-new
```

Python yalnız hazırlama/seçim kontrolü yapar. Özellik çıkarımı ve eşleştirme
C++'tır. İndirme Range/CRC/SHA kontrolleri ve bütçe sınırları önceki hazırlayıcıyla
aynıdır. Yeni dizinler kullanılır; görüntüler/vektörler Git'e gönderilmez.

```powershell
python scripts/test_crossview_data.py
python scripts/test_crossview_cli.py build/tensorrt/Release/aegisvision_crossview.exe
python scripts/test_reid_flip_selection.py
build/tensorrt/Release/aegisvision_reid_tests.exe --bundle artifacts/models/osnet-x0-25-msmt17
```

Yerel kontroller: hazırlama 12/12, CLI giriş sözleşmesi 12/12, seçim politikası
10/10, tracklet aktarımı 13/13, temel CTest 7/7 ve seçili CTest 7/7 geçti.
Gerçek model flip-ortalama ve eski golden referans testleri de geçti.
Eski C1/C2 manifesti yeni binary ile yeniden çalıştırıldı; bütün sorgu skorları
ve kararları eski raporla birebir aynı. Varsayılan embedding uzayı da değişmedi.
Kontrol çıktısı: `outputs/reid-single-regression-v1/report.json`.
Uzak CI sonucu henüz bu yerel testlerden çıkarılamaz.
