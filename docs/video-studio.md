# Video Studio — portföy demosu

18 Ekim 2026. Teslim kapsamı: bilgisayarda video seçme, gerçek YOLOv8-seg çıkarımı,
renkli maskeler, video-içi takip kimlikleri, kaynak/sonuç oynatma, MP4 ve CSV indirme.
Fiziksel kamera gerekmez. Analiz C++ / OpenCV DNN üzerinde CPU ile çalışır.

## Aç ve göster

1. Proje klasöründeki `DEMOYU-AC.cmd` dosyasına çift tıkla.
2. Açılan panelde `demo-TUD-Stadtmitte...` veya `demo-KITTI-17...` videosunu seç.
3. Mevcut başarılı sonuç sağda otomatik açılır. Yoksa **Videoyu analiz et** düğmesine bas.
4. Kaynak ve sonuç ayrı oynatıcılardır. Sunumda senkron karşılaştırma için aşağıdaki
   dışa aktarma aracının tek parça karşılaştırma videosunu kullan.
5. **MP4 indir** ve **Takip raporunu indir (CSV)** ile sonucu sakla.

CPU analizi video süresinden uzun sürebilir. İlerleme panelde gösterilir; iş iptal
edilebilir. Arama için indeksleme, maske videosu oluşturmanın ön koşulu değildir.
OCR, sahne açıklaması, arama ve iş geçmişi alt taraftaki diğer araçlar bölümündedir.

## Başka videolar

`VIDEO-EKLE.cmd` dosyasına çift tıkla, dosya penceresinden videonu seç.
Orijinal dosya değiştirilmez. İlk 15 saniye sessiz H.264, 10 FPS ve en fazla
960×540 olarak medya kütüphanesine eklenir. Yeni video seçili halde panel açılır.
Tarayıcıya sürükle-bırak yükleme uygulanmadı; dosya penceresi yerel hazırlayıcıdır.

Başka bir bölüm için PowerShell'de proje klasöründen:

```powershell
.\scripts\import_video.ps1 -VideoPath "D:\Videolar\ornek.mp4" -StartSeconds 30 -Seconds 20
```

Süre 1–60 saniye; varsayılan 15 saniyedir. Bu hazırlayıcı en fazla 600 kare üretir.
Kaynağın sonunda istenenden kısa klip oluşabilir; başlangıçta kare yoksa hata verir.
Çıktı doğrulanmadan kütüphaneye yayımlanmaz. `-NoOpen` yalnız hazırlama yapar.
Elle eklenen dosyaları **Listeyi yenile** düğmesiyle listeye alabilirsin.

## Bu teslimde gerçekten işlenen örnekler

| Kaynak | İçerik | İşlenen video | Sonuç |
|---|---|---|---|
| MOT15 / TUD-Stadtmitte | Sokakta yürüyen insanlar | 7,2 sn, 72 kare, 720×540 | 447 kare-içi maske, 26 takip ID'si |
| MOT15 / KITTI-17 | Hareketli araç kamerası | 12 sn, 120 kare, 960×288 | 832 kare-içi maske, 36 takip ID'si |

İş kimlikleri: `1791700823339-16` ve `1791700823339-15`.
İkisi de bu teslimde yeni çıkarımla tamamlandı; eski sonucun yeniden adlandırılması
veya yapay görüntü animasyonu değildir. Sayılar doğruluk puanı değildir.
Takip ID'leri benzersiz kişi/araç sayısı değildir; örtüşme ve hareketle değişebilir.

Veriler önceden yerelde bulunan [MOT15](https://motchallenge.net/data/MOT15/)
örnek videolarından hazırlanmıştır. Kaynak URL ve hash'leri ilgili
`artifacts/datasets/mot15-*/quality-manifest.json` dosyalarında bulunur.
Veriler ve türetilmiş videolar Git'e yüklenmez. Harici yayın/yeniden dağıtım için
orijinal veri setinin kullanım koşullarını kontrol et; kendi izinli videolarını tercih et.

## Sunum çıktısı

```powershell
.\scripts\export_demo.ps1 -JobIds '1791700823339-16','1791700823339-15'
```

`outputs/presentation-TARIH-SAAT/` altında her örnek için:

- `*-comparison.mp4`: 1280×450, başlıklı, senkron kaynak/sonuç karşılaştırması.
- `*-source.mp4` ve `*-analysis.mp4`: kaynak demo kopyası ve model çıktısı.
- `*-tracks.csv`: kare bazında takip raporu.
- `*-preview.jpg`: sunumda kullanılabilecek gerçek sonuç karesi.
- `manifest.json`: sonuç metrikleri, MP4 kare/boyut kontrolü ve SHA256.

Çıktılar servis kapalıyken de video oynatıcıda açılır. Başka iş kimlikleriyle kendi
sonuçlarını aynı biçimde dışa aktarabilirsin. Araç hazır demo servisini ve FFmpeg'i kullanır.

## 60 saniyelik anlatım akışı

Önce insan videosunun karşılaştırmasını göster: modelin nesne sınırlarını maskelediğini,
takip numaralarının kareler arasında ilişki kurduğunu anlat. Ardından araç kamerası
örneğine geç: aynı pipeline'ın farklı bir sahnede çalıştığını göster. Son olarak
dosya seçim penceresini, MP4 indirmeyi ve CSV raporunu göster.

"Yerel C++ video analiz demosu" de; "kusursuz kimlik takibi", "gerçek zamanlı GPU
segmentation", "üretime hazır tüm ilan maddeleri" deme. Fiziksel kamera, çoklu-kamera
kalitesi ve ileri deployment araştırmaları bu demo tesliminden ayrı tutulmuştur.

Kod/test kontrolleri: `node tests/test_web_state.cjs`, `node tests/test_redis_web.cjs`.
İçe aktarma iki gerçek videoda; dışa aktarma kare sayısı/boyut/codec ve tarayıcı sonucu
ile kontrol edilir. Sıfır bilgisayarda otomatik kurulum değildir; [kurulum](demo.md).
