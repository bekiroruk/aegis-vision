# OCR kalite ölçümü (C++)

`aegisvision_ocr_evaluate`, etiketlenmiş yerel görüntülerde **tespit + okuma**
zincirini ölçer. Model eğitmez, eşik aramaz, başarılı örnekleri seçip hataları atmaz.
Mevcut İngilizce modelin sınırlarını görünür kılar; Türkçe desteği eklemez.

## Çalıştırma

Proje kökünde:

```powershell
./scripts/prepare_ocr.ps1 -WithSample
cmake --build build/live-seg --config Release --target aegisvision_ocr_evaluate
./build/live-seg/Release/aegisvision_ocr_evaluate.exe --validate-manifest configs/ocr-evaluation-smoke.toml
./build/live-seg/Release/aegisvision_ocr_evaluate.exe artifacts/models/ocr-en configs/ocr-evaluation-smoke.toml outputs/ocr-evaluation-new
```

Çıktı klasörü yeni olmalıdır. `report.json` her görüntünün referans ve tahmin
kutularını, metinlerini, sayaçlarını ve sürelerini içerir. Görüntü/model hatası
olursa komut başarısız olur; kısmi veri kümesi başarı raporu oluşturulmaz.
Sıfır çıkış kodu yalnız çalıştırmanın tamamlandığı anlamına gelir, kalite eşiği değildir.
`--validate-manifest` yalnız etiket şemasını doğrular; resimleri/modeli açmaz.

## Etiket sözleşmesi

Örnek manifest: `configs/ocr-evaluation-smoke.toml`. Görüntü yolları manifest
klasörüne göredir; bu yerel CLI kullanıcı tarafından verilen dosyaları okuyabilir,
HTTP servisinin medya-kökü güvenlik sınırı değildir.

- 1–1000 görüntü, görüntü başına en fazla 64 bölge; negatif örnekte `regions=[]`.
- Benzersiz `id` ve kanonik görüntü yolu; açık `width`/`height` (en fazla 1920).
- Görüntü dosyası en fazla 32 MiB; manifest en fazla 1 MiB.
- Kutu `[x,y,width,height]`, sürekli piksel koordinatı; pozitif alan, resim içinde.
- Referans metni boş olamaz; 1–256 karakter, yalnız `a-z` ve `0-9`.
  Küçük harfe çevirme etiketleme sırasında açıkça yapılır. Araç Türkçe karakter,
  boşluk veya noktalama işaretini sessizce atmaz; desteklenmeyen etiketi reddeder.
- Desteklenen veri kümesinde tüm hedef sözcükleri etiketleyin. Ignore-region
  sözleşmesi yoktur; karma alfabeli sahneleri seçerek etiketlemek yanıltıcıdır.

## Metrikler

Tahmin dörtgeninin eksenlere paralel dış kutusu kullanılır; **polygon IoU değildir**.
IoU ≥ 0,5 için önce en çok sayıda bire bir eşleşme, ardından en yüksek toplam IoU
seçilir. Eşleştirmede metin kullanılmaz. Yinelenen tahminler fazlalık sayılır.

- `detection_precision/recall`: eşleşen kutular / tahminler veya referanslar.
- `exact_precision/recall`: hem konumu eşleşen hem metni birebir doğru bölgeler.
- `matched_cer`: **yalnız konumu eşleşen** bölgelerde toplam Levenshtein mesafesi /
  referans karakterleri. Tek başına bütün pipeline doğruluğu değildir.
- `spatial_character_error_ratio`: eşleşenlerin edit mesafesine, kaçırılan referansın
  tüm karakterleri ve eşleşmeyen tahminin tüm karakterleri eklenir; tüm referans
  karakterlerine bölünür. Bu projeye özgü, konuma bağlı bir hata oranıdır;
  standart sayfa-transkript CER/WER veya resmi OCR benchmark metriği değildir.
  Birden büyük olabilir. Boş metinli fazladan kutular detection precision'da
  cezalandırılır, karakter oranına karakter eklemez.
- Paydası sıfır olan oran raporda bulunmaz; sıfır hata/başarı uydurulmaz.
  Sayaçlar yine yazılır; toplam oranlar görüntü oranlarının ortalaması değildir.

Süreler tek CPU geçişidir; warm-up yoktur, model yükleme/resim decode hariçtir.
Bu çıktı gecikme/throughput benchmark'ı değildir. Model sürümü ve SHA256 için
hazırlama betiğinin doğruladığı `artifacts/models/ocr-en/manifest.json` saklanmalıdır.

## 10 Ekim 2026 kontrolü

Manuel yaklaşık kutulu **iki önceden incelenmiş upstream örnek** kullanıldı.
Canon geliştirmede zaten görülmüştü. Welcome bir metin tanıma kırpımıdır;
burada yalnız recognizer değil, bütün detector + recognizer çalıştırıldı.
Model/eşik bu sonuçlara göre değiştirilmedi.

| Örnek | Sonuç | Kaçırılan | Fazladan |
| --- | --- | --- | --- |
| Canon, 1296×864 fotoğraf | `canon` doğru | 0 | 0 |
| Welcome, 66×18 kırpım | Metin bölgesi bulunamadı | 1 | 0 |

Toplam: exact recall **1/2**, detection precision **1/1**. Eşleşen CER **0/5**
olmasına rağmen konuma bağlı karakter hatası **7/12 = %58,33**; bu fark kaçırılan
metni ayrıca saymanın neden gerekli olduğunu gösterir. Yerel rapor:
`outputs/ocr-evaluation-20261010-final/report.json`.

Yeni C++ hedefleri Release olarak derlendi; **41/41 CTest** geçti (59,12 saniye).
Metrik testleri eşleşme sırası, yinelenen tahmin, eksik/fazla bölge, boş tahmin,
IoU sınırı ve hatalı kutu/alfabeyi kapsar. Manifest testleri negatif örnek,
Türkçe etiket reddi, bozuk kutu, eksik alan ve yinelenen kimliği denetler.
Var olan raporun üzerine yazma isteği reddedildi ve dosya hash'i değişmedi.

Bu iki örnek bağımsız veya geniş bir kalite ölçümü değildir. Küçük metinler için
farklı ön işleme/model seçenekleri ayrı validation verisinde karşılaştırılmalı;
son test kümesi bu seçimlerden önce dondurulmalıdır.

Welcome kaynağı [sabit OpenCV extra revizyonu](https://github.com/opencv/opencv_extra/blob/22b4a7bc7cf5e4ddd3a0426eafc88bae28d3dfc3/testdata/dnn/text_rec_test.png),
SHA256 `d2f60c7c0423eb4254c2b53cc4da49dabcd244e0b4d3c2d1e6a53d009fb12885`.
Resimler repoda yeniden dağıtılmaz; [OCR kaynak/lisans notları](ocr.md) geçerlidir.

## Sonraki adım

Türkçe karakterleri gerçekten üretebilen ayrı bir recognizer/dictionary ve buna
uygun Unicode karakter ölçümü gerekir. Mevcut CTC çıktısına birkaç karakter
eklemek veya `ü` → `u` dönüştürmek Türkçe desteği değildir. Geniş, lisansı açık
etiketli veri kümesi, Türkçe backend ve VLM entegrasyonu hâlâ tamamlanmadı.
