# Türkçe / çok dilli OCR — C++ CPU

İsteğe bağlı `PpocrMultilingual` adaptörü mevcut PP-OCRv3 detector'ünü **resmî
PP-OCRv6 tiny** recognizer ile birleştirir. ONNX Runtime C++ CPU kullanılır;
çalışma zamanında Python, PaddlePaddle kurulumu veya ağ bağlantısı gerekmez.
Eski CRNN EN backend varsayılan davranışını korur. Çok dilli seçenek için
`AEGISVISION_WITH_OPENCV=ON` ve `AEGISVISION_WITH_SEARCH=ON` derlemesi gerekir
(bu projede ONNX Runtime bağımlılığı search seçeneği altında kuruluyor).

## Hazırlama ve çalıştırma

Proje kökünden:

```powershell
./scripts/prepare_ocr_multilingual.ps1
cmake --build build/live-seg --config Release
./scripts/create_turkish_ocr_fixture.ps1
./build/live-seg/Release/aegisvision_ocr_multi.exe artifacts/models/ocr-multi-v6 outputs/ocr-turkish-fixture/turkish.png outputs/ocr-turkish-new
```

Fixture üretimi Windows/System.Drawing ve Arial kullanır. Çıktı klasörleri yeni
olmalıdır; mevcut dosyalar silinmez. Fixture daha önce üretildiyse o satırı
atlayın. Bu **üretilmiş tipografi testi**, doğal sahne veya bağımsız benchmark değildir.
CLI `result.json` içinde UTF-8 metin/polygon/skor, `preview.jpg` içinde numaralı
kutular üretir. OpenCV Hershey fontu Türkçe çizmediği için metin JPEG'e bozuk
karakterlerle basılmaz; kutu numarası JSON/panel listesindeki sıraya karşılık gelir.

Web paneli (Qdrant açık olmalı; 8091 boş olmalı):

```powershell
./scripts/start_service.ps1 -Port 8091 -ServerExecutable build/live-seg/Release/aegisvision_server.exe -OcrMultilingual artifacts/models/ocr-multi-v6 -JobDatabase artifacts/service/ocr-multi-jobs.sqlite
```

`-OcrModels` ve `-OcrMultilingual` aynı anda verilemez. `-SegmentationModel` ile
birlikte kullanılabilir. Model değişimi persistence bağlamını değiştirir;
eski veritabanını silmeyin, ayrı `-JobDatabase` kullanın. `ocr_frame` istek şeması
değişmez; keyfî HTTP model/dil seçimi yoktur. `/api/health` `ocr_unicode=true`
bildirir. Panel etkin backend'e göre dil bilgisini gösterir.

## Model / sözlük doğrulaması

- [Resmî model kartı](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/tree/2612ab37152ae0a677521bae4e1e3d4fb4cf7c30): Apache-2.0.
- Sürüm: `2612ab37152ae0a677521bae4e1e3d4fb4cf7c30`.
- Recognizer SHA256: `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6`.
- Sözlüğü içeren `inference.yml` SHA256: `66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1`.
- Detector aynı sabit [OpenCV Zoo modeli](ocr.md) ve lisansıdır.

Hazırlama betiği sürümü sabitler; C++ adaptörü üç dosyanın hash'ini ayrıca
doğrular. Sözlük için genel YAML parser kullanılmaz: yalnız bu doğrulanmış
dosyanın tek-karakter scalar listesi okunur. CTC blank=0, ardından sözlük ve
sonda boşluk gelir. Sözlük, modelin çıktı sınıf sayısıyla eşleşmek zorundadır.
`ÇçĞğİıÖöŞşÜü` değiştirilmeden korunur. Latin adı tek başına yeterli değildir;
eski PP-OCRv3 Latin sözlüğü Türkçe harflerin tamamını içermediği için seçilmedi.

Ön işleme: BGR, 48 piksel yükseklik, perspektif düzeltme sırasında en-boy oranını
koruma, 320–3200 genişliğe sıfır dolgulu normalize tensor, `(pixel/127.5)-1`.
Geniş satırlar 3200 ile sınırlanır; açı sınıflandırıcı/180° döndürme yoktur.
[PaddleX ön/son işleme kaynağı](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/models/text_recognition/processors.py)
referans alınmıştır; framework çıktılarına karşı geniş sayısal eşdeğerlik testi
henüz yapılmadı. Bu adaptör tüm PaddleOCR pipeline'ının kopyası değildir.

CPU tek-thread inference; 64 bölge, 1920 piksel kenar ve 10 saniye kooperatif
deadline. İptal/deadline kontrolü model çağrıları arasında yapılır, çalışan ONNX
çağrısı zorla kesilmez. Hata/iptalde kısmi sonuç yayımlanmaz. CLI metni en fazla
1024 bayt; HTTP bölge metni mevcut güvenlik sınırı olan 256 bayt ile sınırlıdır.
HTTP geçersiz UTF-8, kontrol/bidi yönlendirme karakterlerini reddeder; panel
metni `textContent` ile gösterir. Skor ortalama seçilen CTC olasılığıdır, gerçek
doğruluk için kalibre edilmiş bir güven değildir.

## Unicode ölçümü

```powershell
./build/live-seg/Release/aegisvision_ocr_evaluate.exe --multilingual artifacts/models/ocr-multi-v6 configs/ocr-evaluation-turkish-smoke.toml outputs/ocr-turkish-evaluation-new
```

Etiket/metin UTF-8 olarak doğrulanır. Mesafe bayt değil **Unicode kod noktası**
üzerinden hesaplanır; büyük/küçük harf ve boşluklar anlamlıdır. NFC/NFD
normalizasyonu veya Türkçe harfleri ASCII'ye çevirme yoktur: birleşik/ayrışık
karakter yazımı ayrı hata sayılabilir. Grapheme-cluster metriği değildir.
`--multilingual --validate-manifest FILE` model çalıştırmadan Unicode etiketleri
doğrular. Eski bayrak kullanılmayan İngilizce ölçüm aynen korunur.
Konuma bağlı eşleştirme ve eksik/fazla kutu ölçümü [aynı sözleşmeyi](ocr-evaluation.md)
kullanır; yalnız eşleşen CER tüm pipeline kalitesi olarak sunulmaz.

## İlk gözlem ve sınırlar

1000×300 Arial fixture'ın ilk C++ çalıştırmasında:

| Referans | Tahmin |
| --- | --- |
| Türkçe İstanbul | Türkçe İstanbul |
| Çığ ŞÖLEN ıİ öü 123 | Çiğ ŞÖLENiÍ öü 123 |

Türkçe karakter çıktısı çalışıyor, ancak ikinci satırda noktalı/noktasız harf
ve boşluk hataları var. Sözlük desteği kusursuz okuma garantisi değildir.
İlk ölçüm yaklaşık 350 ms; model yükleme hariç tek geçiştir, hız benchmark'ı değildir.
Yerel ilk kanıt: `outputs/ocr-turkish-v1/result.json` ve `preview.jpg`.

Konuma göre eşleşen iki satırın Unicode karakter hatası **4/34 = %11,76**;
iki bölge de bulundu, yalnız bir satır tamamen doğru. Bu yalnız sentetik
kontroldür. Rapor: `outputs/ocr-turkish-evaluation-v1/report.json`.

### Teslim doğrulaması — 11 Ekim 2026

Release derlemesi tamamlandı; **44/44 CTest** geçti (49,47 saniye).
`node tests/test_web_state.cjs` ve `git diff --check` de geçti.

Gerçek model CLI testi ve Türkçe fixture testi çalıştırıldı. Servis testleri
UTF-8 HTTP yanıtını, hatalı UTF-8/kontrol karakteri reddini ve yeniden açılışta
Türkçe metnin SQLite'dan bozulmadan dönmesini denetler; bu HTTP testleri test
modeli kullanır. Web durum testleri çok dilli yetenek göstergesini de kapsar.
Gerçek çok dilli modelin ayrı sunucu + tarayıcı doğrulaması henüz tamamlanmadı:
son oturumda arka plan sunucusu başlatma işlemi araç politikası tarafından reddedildi.
Panelin açık olduğu varsayılmamalıdır; yukarıdaki başlatma komutunu kullanın.

Geniş, lisansı açık doğal Türkçe veri kümesi, gerçek video koşullarında kalite,
rotasyon/uzun metin testleri ve GPU hız ölçümü tamamlanmadı. Sonraki bağımsız
bileşen VLM sahne analizi; OCR kalite genişletmesi ayrı açık iştir.
