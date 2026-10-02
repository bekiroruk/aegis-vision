# TOML ile uygulama ayarları

OpenCV'li C++ derlemesi [toml++ 3.4.0](https://github.com/marzer/tomlplusplus)
ile `version = 1` şemasını okur. İlk derlemede kütüphane sabit arşiv/SHA256 ile
indirilir; bağımlılıksız çekirdek derlemesi etkilenmez.

Hazır uygulama örnekleri:

- `configs/image.toml`: YOLO ile tek fotoğraf tespiti.
- `configs/pipeline.toml`: YOLO ve iki aşamalı video takibi.
- `configs/video-kalman.toml`: ardışık yerel karelerde deneysel aktif-önce Kalman takibi.
- `configs/video-kalman-center.toml`: aynı takipçiyle boyut titreşimine duyarsız merkez hareket kapısı.
- `configs/search.toml`: CLIP ile yerel Qdrant araması.
- `configs/stream.toml`: süre sınırı, timeout/reconnect ve sınırlı RTSP kare kuyruğu.
- `configs/live-preview.toml`: aynı stream şemasıyla web önizlemesi için 180 saniyelik profil;
  CLI kayıt profili 45 saniyede kalır. Her iki profilde open deadline 8000 ms'dir.

Model yolları TOML dosyasının bulunduğu klasöre göre çözülür; çalıştırdığınız
terminalin dizinine göre değişmez. Örneklerdeki `artifacts/models` dosyaları
yerelde hazırlanmalıdır. ONNX modelin gerçek giriş/çıkış sözleşmesi uygulama
modeli yüklerken, CLIP manifest/SHA256 kontrolü arama komutunda doğrulanır.

```powershell
cmake --build build/search --config Release
./build/search/Release/aegisvision_config.exe validate configs/image.toml
./build/search/Release/aegisvision_config.exe validate configs/pipeline.toml
./build/search/Release/aegisvision_config.exe validate configs/search.toml
./build/search/Release/aegisvision_config.exe validate configs/live-preview.toml
```

`validate`, şema/alan türü/değer aralığı, desteklenen backend ve yerel model
dosyalarının varlığını kontrol eder; modeli çalıştırmaz ve Qdrant'a bağlanmaz.
Yanlış alan, eksik dosya, uyumsuz vektör boyutu veya eşik için sıfırdan farklı
çıkış kodu ve hatalı anahtar/alan bilgisini döndürür.

## Fotoğraf ve video

```powershell
./build/search/Release/aegisvision_detect.exe --config configs/image.toml artifacts/samples/bus.jpg outputs/config-image
./build/search/Release/aegisvision_video.exe --config configs/pipeline.toml artifacts/samples/moving-bus.avi outputs/config-video 12
```

Videonun sondaki `12` argümanı TOML'deki `[video].max_frames` değerini bu çağrı
için değiştirir. Kaldırılırsa örnekteki 60 kare sınırı kullanılır. Çıktı klasörü
yeni veya boş olmalıdır. Eski konumsal komutlar çalışmaya devam eder.

`[detector].confidence_threshold`, görüntü modunda YOLO minimum güvenidir.
Videoda IoU takipçisi için aynı eşiktir. İki aşamalı takipte yüksek güven
eşleştirme eşiğidir; YOLO filtresi `[tracking].low_confidence` değerine indirilir.
Yeni track açma eşiği `[tracking].new_track_confidence` değeridir. Şema
`low_confidence < confidence_threshold <= new_track_confidence` gerektirir.
`[tracking].backend` değeri `iou`, `two-stage` veya `kalman` olabilir; `iou` seçildiğinde
düşük/yeni güven alanları kullanılmaz ve verilirse hata oluşur.
Kalman aynı low/high/new eşiklerini kullanır; yalnız `kalman` için opsiyonel
`gating_mode` vardır: `full-box` varsayılan, `center` yalnız konumu kapılar.
`mahalanobis_gate` varsayılanı moda göre 13.2767 (4B) veya 9.2103 (2B), aralığı
(0,100]; açık verilirse moda bağlı varsayılanı değiştirir. Boyut correction'ı
ve IoU kapısı iki modda da korunur. Detector kapasitesi
en fazla 512 olmalıdır. Kalman `stream` modunda reddedilir: drop-oldest ve
değişken zaman adımı için timestamp-aware motion modeli henüz yoktur.

## CLIP araması

Qdrant `127.0.0.1:6333` üzerinde çalışırken:

```powershell
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml init
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml index bus artifacts/samples/bus.jpg
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml text "a photo of a bus" 5
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml image artifacts/samples/bus.jpg 5
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml index-directory artifacts/samples
./build/search/Release/aegisvision_search_cli.exe --config configs/search.toml index-video configs/image.toml artifacts/samples/moving-bus.avi 10 30
```

`[embedding].dimension` ve `[vector_store].dimension` 512 olmalıdır. Yalnızca
`clip_onnx` ve yerel `qdrant` desteklenir. Koleksiyon adı, port ve timeout
dosyadan okunur. `init` var olan koleksiyonu silmez. Arama verilerini yönetme
ayrıntıları için [arama kılavuzu](search.md).

`index-video`, arama konfigürasyonuna ek olarak image modunda bir detector
konfigürasyonu alır. Bu akışta takipçi çalıştırılmaz; kare aralığı ve maksimum
kare sayısı komut argümanlarıdır. [İndeksleme ayrıntıları](indexing.md).

Şema belirtilmeyen anahtarları ve moda uymayan bölümleri reddeder. `runtime`
yalnızca `cpu`, `fp32` ve moda göre `opencv-dnn`/`onnxruntime` değerlerini kabul
eder. `stream` modu `[tracking]` ve `[stream]` kullanır; `[video]` kabul etmez.
Diğer modlar `[stream]` kabul etmez. Ayar aralıkları ve komut için [RTSP kılavuzu](live-video.md).
TensorRT, OCR ve segmentation bu şemanın desteklenen seçenekleri
değildir; gelecek bir şema sürümünde eklenecekler.

Bu bilgisayarda üç örnek dosya uygulamalarla çalıştırıldı: 5 fotoğraf tespiti,
12 karelik video ve otobüsü ilk sıraya koyan metin sorgusu elde edildi. Bunlar
model kalitesi değerlendirmesi değil, konfigürasyonun uygulandığını gösteren
yerel smoke testleridir.
