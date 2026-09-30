# Klasör ve video arama indeksi

`aegisvision_search_cli`, CLIP modelini komut başına bir kez yükleyerek klasördeki
görselleri veya videonun seçilmiş karelerindeki YOLO nesne kırpmalarını Qdrant'a
ekler. Arama derlemesi ve yerel Qdrant kurulumu için [arama kılavuzu](search.md).

## Klasör

```powershell
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml init
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml index-directory artifacts/samples
# Alt klasörler dahil:
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml index-directory artifacts/datasets/coco-search/images --recursive
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml text "a photo of a bus" 5
```

Varsayılan tarama yalnızca verilen klasörü kapsar. `--recursive` alt klasörleri
ekler; sembolik bağlantılar izlenmez. JPEG, PNG, BMP, TIFF, WebP ve PPM uzantıları
(büyük/küçük harf fark etmeden) seçilir. Diğer dosyalar atlanır. Desteklenen
uzantıya sahip bozuk bir görsel hata oluşturur; hata yolu ve tamamlanan kayıt
sayısını bildirir. Boş klasör veya 10.000'den fazla görsel, indeks yazılmadan
reddedilir. Görseller sıralı ve tek tek işlenir; tüm pikseller bellekte tutulmaz.

Her kayıt tam görsel embedding'i içerir. Nesne tespiti bu komutta yapılmaz.
ID, mutlak dosya yolundan üretilir: aynı yoldaki değişmiş görsel tekrar
indekslendiğinde mevcut kayıt güncellenir. Metadata, dosya yolu, SHA256 ve tam
görsel kutusunu içerir. Silinen/taşınan dosyaların eski kayıtları otomatik silinmez.

## Video nesne kırpmaları

```powershell
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml index-video configs/image.toml artifacts/samples/moving-bus.avi 10 30
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml text "a photo of a bus" 10
```

Sözdizimi: `index-video DETECTOR.toml VIDEO [STRIDE [MAX_FRAMES]]`.
Arama ayarları `search.toml` içindeki CLIP/Qdrant bölümünden; YOLO modeli,
confidence ve NMS eşikleri ise **image modundaki** `DETECTOR.toml` dosyasından
okunur. `configs/image.toml` hazır örnektir. Bu komut takipçi çalıştırmaz.

Örnekte 30 kare okunur, 0/10/20 numaralı karelerde YOLO çalışır ve bulunan her
nesnenin kırpması CLIP ile indekslenir. `MAX_FRAMES` işlenen toplam kareyi sınırlar;
örneklenen kare sayısı değildir. Parametreler verilmezse stride 30 olur ve dosyanın
tamamı okunur. Stride ve açıkça verilen kare sınırı pozitif olmalıdır. Tespit
olmayan bir video sıfır kayıtla başarıyla tamamlanır.

Arama sonuçlarının metadata alanında şunlar bulunur:

- `path`, `frame_index`, `timestamp_ms`: kaynağı ve kareyi bulmak için.
- `bbox`, `label`, `confidence`: nesne kutusu ve YOLO çıktısı.
- `sha256`, `source_id`, `detector_signature`: video/model/ayar kimliği.
- `source_fps`, `used_fallback_fps`, `timestamp_basis`: zaman hesaplama yöntemi.

Zaman, `frame_index / source_fps` ile hesaplanan CFR tahminidir; gerçek container
PTS değeri değildir. Geçerli FPS bulunamazsa 25 FPS kullanılır ve bu metadata'da
belirtilir. Değişken FPS ve RTSP akışları bu komutun hedefi değildir. Decoder durması
ile normal dosya sonu kesin ayrıştırılamadığı için özet bunu
`end_of_stream_or_decode_stop` olarak bildirir.

Başlangıçta videonun tamamı SHA256 için okunur; ardından kareler sıralı çözülür.
Bellekte bir kare ve onun tespitleri tutulur. Video ID'leri içerik hash'i, YOLO
model/ayar hash'i, kare numarası ve sıralanmış tespit numarasından türetilir.
Aynı video/model/ayarlarla tekrar çalıştırma aynı kayıtları günceller. Stride
değiştiğinde ortak kareler çoğalmaz. İçeriği aynı video farklı dosya yolundan
indekslendiğinde mevcut kayıtların yolu güncellenir. İçerik veya detector ayarı
değişirse yeni bir kaynak kimliği oluşur.

## Yeniden çalıştırma ve doğrulama

İşlemler upsert yapar. Hata durumunda önceden tamamlanan yazımlar kalır; aynı
komut yeniden çalıştırılabilir. Daha kısa bir aralıkla yeniden indeksleme veya
model değiştirme, önceki kayıtları silmez. Temiz bir karşılaştırma için yeni
koleksiyon kullanın. `init` mevcut koleksiyonu sıfırlamaz.

Windows'ta gerçek CLIP/YOLO modelleri, örnek görsel/video ve Qdrant hazırken:

```powershell
./scripts/smoke_indexing.ps1 -Collection aegis_indexing_demo -Output outputs/indexing-demo
```

Bu geliştirme scripti yeni bir koleksiyon ister. Klasör ve videoyu iki kez
indeksleyip tekrarların kayıt sayısını artırmadığını doğrular; otobüs metin
sorgusunu çalıştırır ve JSON raporu yazar. Koleksiyon ve çıktılar korunur.
Otomatik C++ testi ayrıca Unicode yollarını, alt klasörleri, dosya limitini,
örnekleme zamanlarını, boş/bozuk girdileri, sırası değişen tespitleri ve store
hatalarını model indirmeden test eder. Arama kalite ölçümü için
[COCO benchmark'ı](search-benchmark.md) ayrı tutulur.

2026-09-30 yerel çalıştırması: `artifacts/samples` içinden 1 görsel ve 30 karelik
hareketli otobüs fotoğrafı videosunun 0/10/20 karelerinden 15 nesne kaydı üretildi.
Her işlem iki kez çalıştırıldı; koleksiyon 16 kayıtta kaldı. Otobüs metin sorgusunun
ilk üç sonucu video içindeki otobüs kırpmalarıydı (0, 2000, 1000 ms sırasıyla).
Rapor `outputs/indexing-demo/report.json` konumunda. Bu video bir işlev testidir;
gerçek video arama kalitesi benchmark'ı değildir.
