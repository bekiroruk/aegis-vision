# Re-ID geliştirme verisi eşik kalibrasyonu

Tarih: 2026-10-15. OSNet x1.0 ile C3→C4, 39 eşzamanlı kare çiftinde
304 sorgu (94 karşılıklı, 210 karşılıksız). Önceki model deneyinde görülen
geliştirme verisidir; bağımsız başarı veya canlı takip ölçümü değildir.

## Sonuç: bu taramada uygun eşik bulunamadı

| Cosine / margin | Doğru kabul | Yanlış kabul | Karşılıksız kişide yanlış kabul | Kabul doğruluğu | Karşılıklı sorgu recall |
|---|---:|---:|---:|---:|---:|
| .85 / .05 (eski) | 0 | 0 | 0 | Tanımsız | %0 |
| .50 / .02 | 37 | 9 | 5 | %80,43 | %39,36 |
| .50 / .05 | 32 | 4 | 2 | %88,89 | %34,04 |
| .50 / .10 | 19 | 1 | 1 | %95,00 | %20,21 |
| .65 / .05 | 7 | 0 | 0 | %100,00 | %7,45 |

32 kombinasyonun **hiçbiri** bütün koşulları geçmedi. `.50/.10` doğruluk/recall
koşulunu sağlasa da 20 doğru kabul alt sınırının altında ve karşılıksız bir
kişiyi yanlış kabul ediyor. `.65` eşiklerinde yanlış kabul görülmedi fakat
yalnız 7 doğru kabul var. Bu yedi örnekteki %100, genel güvenilirlik garantisi
değildir. Başarı koşulları sonuca göre gevşetilmedi, yeni aday seçilmedi.

Eski `.85/.05` ayarında 286 sorgu benzerlik eşiğinden, 18 sorgu boş galeriden
reddediliyor. Eşik sorunun bir parçası; ancak yalnız eşiği düşürmek bu deneyde
istenen doğruluk/yakalama dengesini sağlamadı. Rank-1 bütün ayarlarda 66/94;
eşik değiştirmek temsilin sıralama doğruluğunu değiştirmez.

**Varsayılan model/eşikler, servis, indeks ve veritabanları değiştirilmedi.**
C5/C6 son kontrol görüntüleri okunmadı. Bu önceki sonuçları silmez veya C5/C6'yı
yeniden kör test yapmaz. Sonraki kalite adımı ayrı eğitim kimlikleri/verisiyle
temsil/metric-learning iyileştirmesi; ardından yeni, dokunulmamış son testtir.
Bu dar 32-ayarlı tarama, mümkün olan bütün modellerin/eşiklerin yetersiz olduğunu
kanıtlamaz; veri azdır ve tekrarlanan kişiler bağımsız örnekler değildir.

## Ölçümden önce sabitlenen politika

`configs/reid-calibration.json`, C++ taramasından önce kaydedildi.

- Cosine: `.50, .55, .60, .65, .70, .75, .80, .85`.
- Margin: `.02, .05, .10, .15`. Her iki yönde mutual-best ve margin korunur.
- En az 30 karşılıklı ve 30 karşılıksız sorgu; en az 20 doğru kabul.
- Kabul precision ≥ %95, counterpart recall ≥ %20; en fazla 1 yanlış kabul.
- Karşılığı olmayan sorgularda yanlış kabul sayısı 0 olmalı.
- Uygun adaylar arasında en çok doğru kabul, en az yanlış kabul, daha yüksek
  cosine, sonra daha yüksek margin tercih edilir. Hiç uygun aday yoksa seçim `null`.
- Otomatik üretim terfisi yoktur. Uygun aday çıksaydı bile yeni son test gerekirdi.

Politika SHA256: `e1618f09975fb17fdba3e8165805ec00021cddfe9125b56bbea4944fc584bc22`.
Model SHA256: `5bf5321c31d064507acaec3aefecf3381b24673123c61403ceab491f49f9dc6a`.
Veri SHA256: `bdd3583c4db7b8d7c4038cdfa4f59dfcf6c649811b523fb795e5617b247d785c`.

## Uygulama ve tekrar üretme

C++ `--calibrate-dev POLICY.json`, yalnız sabit C3/C4 geliştirme profiline
izin verir. Diagnostic/holdout, farklı veri/model hash'i, bozuk/sırasız/tekrarlı
grid ve üretim terfisi isteği reddedilir. Grid eksenleri en fazla 8 değer;
policy en fazla 64 KiB. Aynı 398 özellik bir kez hesaplanır; her ayarda mevcut
native eşleştirici çağrılır. Kimlikler eşleştirmeyi seçmez, yalnız skorlamada
kullanılır. Kişi embedding'leri dosyaya yazılmaz. Python yalnız sonucun seçim
kurallarını denetler, model inference yapmaz.

```powershell
cmake --build build/tensorrt --config Release --target aegisvision_crossview --parallel 4
build/tensorrt/Release/aegisvision_crossview.exe artifacts/datasets/wildtrack-flip-dev-v2/manifest.json artifacts/models/osnet-x1-0-msmt17 outputs/reid-calibration-new --calibrate-dev configs/reid-calibration.json
python scripts/select_reid_thresholds.py configs/reid-calibration.json outputs/reid-calibration-new/calibration.json outputs/reid-calibration-selection-new
```

Her çıktı dizini yeni olmalıdır. `report.json` eski `.85/.05` davranışının
raporudur; `calibration.json` ayrı olarak 32 kombinasyonu, karar nedenlerini,
model/veri/politika ve baseline rapor hash'lerini kaydeder. Kısmi JSON önce
`.part` olarak yazılır; tamamlanınca yeniden adlandırılır.

Seçici bütün grid hücrelerini, sabit sorgu/Rank-1 sayılarını, metric tutarlılığını,
karar sayılarının toplamını ve hash eşliğini denetler. Precision/recall oranlarını
rapordaki hazır oranlardan değil tamsayı sayımlardan hesaplar; sıfır payda `null`.
Holdout veya eksik/tekrarlı hücreler seçim girdisi olamaz.

Yerel kanıtlar:

- `outputs/reid-calibration-dev-v1/report.json`: eski ayarın regresyon raporu.
- `outputs/reid-calibration-dev-v1/calibration.json`, SHA256 `274580d58f4d4c676f2f7b1e251aad13325d0a213a882e64b9f010d24386be35`.
- `outputs/reid-calibration-selection-v1/selection.json`: aday yok, üretim terfisi yok.

Bu çalışma RTSP/GPU veya GPU Docker engelini çözmez; onlar ayrı açık maddelerdir.

Yerel doğrulama: CLI giriş/sınır testleri 18/18, eşik seçici 13/13, önceki model
seçici 23/23, temel CTest 7/7 ve seçili CTest 7/7 geçti. Eski x1.0 raporundaki
304 sorgunun skorları/kararları ve embedding uzayı yeniden üretilen baseline ile
birebir aynı. CI'a eşik-seçici testleri eklendi; bu commit'in uzak CI sonucu
yerel kontrollerden çıkarılamaz.
