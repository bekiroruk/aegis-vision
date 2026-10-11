# Yerel VLM sahne açıklaması — ilk C++ adaptörü

`aegisvision_scene`, fotoğrafı yerel llama.cpp üzerinden SmolVLM-500M'e gönderir
ve kısa İngilizce açıklamayı `scene.json` olarak kaydeder. İstek, doğrulama ve
çıktı C++20'dir; Python veya bulut inference kullanılmaz. CLI yanında kalıcı
`scene_frame` HTTP işi ve dashboard entegrasyonu vardır. Servis ayarları
`-AnalysisConfig configs/analysis.toml` ile de verilebilir; [şema](configuration.md).

## Windows'ta çalıştırma

Komutları repo kökünde çalıştır. Mevcut `build/live-seg` yapılandırması
`AEGISVISION_WITH_SEARCH=ON` olmalıdır; bu ilk adaptör mevcut OpenCV/HTTP/JSON
bağımlılıklarını kullanır. Sıfırdan bağımlılık kurulumu için [arama kılavuzu](search.md).

```powershell
.\scripts\prepare_vlm.ps1 -WithRuntime
cmake --build build/live-seg --config Release --target aegisvision_scene --parallel 2
.\scripts\start_vlm.ps1
```

Son komut açık kalır, `127.0.0.1:8092` üzerinde yalnız CPU ile çalışır.
İkinci terminali de repo kökünde aç:

```powershell
.\build\live-seg\Release\aegisvision_scene.exe artifacts/media/ocr/text_det_test2.jpg outputs/my-scene-01
```

Görüntü önceki OCR demosundan gelir; yerine kendi fotoğrafını verebilirsin.
Çıktı klasörü daha önce bulunmamalıdır; mevcut sonuçlar üzerine yazılmaz.
Port değiştirmek için sunucuda `-Port 8093`, CLI'da son argüman `8093` kullan.
Sunucuyu ilk terminalde Ctrl+C ile durdur. Qdrant ve ana servis bu CLI için gerekmez.

## Gerçek çalıştırma sonucu

2026-10-11, Windows, CPU, 4 thread, tek slot, sabit Q8_0 model ile
`artifacts/media/ocr/text_det_test2.jpg` üzerinde C++ CLI başarıyla çalıştı.
Yerel kayıt: `outputs/vlm-canon-v1/scene.json` (Git'e eklenmez).

```json
{
  "backend": "local-smolvlm-500m",
  "elapsed_ms": 5138.1188,
  "schema_version": 1,
  "summary": "Black and white Canon camera head on a reflective surface.",
  "verified": false,
  "warning": "Generated interpretation; may hallucinate. Not detection/OCR ground truth."
}
```

Fotoğraftaki nesne lens kapağıdır; modelin **camera head** ifadesi yanlıştır.
Bu sonuç inference bağlantısının çalıştığını gösterir, model doğruluğunu kanıtlamaz.
5,14 saniye tek çalıştırmanın süresidir; performans benchmark'ı değildir.
İkinci yerel deneme, `artifacts/samples/bus.jpg` için 5,67 saniyede
`A blue bus with cero emissions written on it.` üretti
(`outputs/vlm-bus-v1/scene.json`). Otobüs/rengi uygundur; yazıyı birebir aktarmıyor
(fotoğrafta `cero emisiones`), yayalardan söz etmiyor. Kapsamlı sahne/OCR analizi değildir.
Doğrudan modelden nesne listeli JSON isteme denemelerinde tekrarlar/yanlış nesneler
görüldüğünden ilk sürüm yalnız kısa açıklama üretir. JSON şemasını uygulama kurar;
modelin metni komut, araç çağrısı veya doğrulanmış detection/OCR etiketi değildir.

## Sınırlar ve güvenlik

- İstekler yalnız sabit `127.0.0.1` adresine gider; yönlendirme takip edilmez.
  Fotoğraf JPEG/base64 olarak gönderilir; harici görüntü URL'si veya serbest prompt yoktur.
- BGR8 giriş, kenar başına en fazla 1920 piksel; uzun kenar 768'e küçültülür,
  JPEG kalite 85, en fazla 512 KiB. CLI dosya sınırı 32 MiB'dir. Piksel sınırı
  decode sonrasında uygulanır; güvenilmeyen yüklemeler için decode izolasyonu yoktur.
- Varsayılan toplam istemci süresi 60 saniye; adaptör 100–120000 ms aralığını kabul eder.
  İptal/deadline bağlantıyı keser; sunucu hesaplamasının anında bittiği garanti edilmez.
- Yanıt en fazla 64 KiB, JSON derinliği en fazla 16, tek tamamlanmış seçim,
  beklenen model alias'ı, en fazla 512 byte UTF-8 açıklama. İç kontrol/bidi
  karakterleri, araç çağrıları ve `finish_reason=length` reddedilir.
- Başlatıcı loopback'e bağlanır, web UI/agent kapalıdır, CORS yalnız kendi origin'idir.
  API anahtarı yoktur: güvenilen yerel prototiptir, portu ağ erişimine açma.
  Alias kontrolü model kimliğinin kriptografik kanıtı değildir; başlatıcı model
  dosyalarını SHA256 ile kontrol eder, adaptör yerel servise güvenir.
- Görüntüdeki talimatları izlememesi istenir; bu prompt enjeksiyonuna karşı
  garanti değildir. Metin hiçbir zaman çalıştırılmaz; gelecekteki panel metni
  HTML olarak yorumlamamalıdır. Otomatik etiketler insan kontrolü gerektirir.

## Sabitlenmiş bileşenler

[GGUF model deposu](https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/tree/72e986006ef53e37cdd3f6d4241c90b0f01df376),
revision `72e986006ef53e37cdd3f6d4241c90b0f01df376`, Apache-2.0;
[orijinal model kartı](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct).

| Dosya | SHA256 |
|---|---|
| SmolVLM-500M-Instruct-Q8_0.gguf | `9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4` |
| mmproj-SmolVLM-500M-Instruct-Q8_0.gguf | `d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150` |
| llama-b11552-bin-win-cpu-x64.zip | `c00741179811ff903b515eae28bdeef1f47af6e81c49920c4e68b62954e056d6` |

Runtime: [llama.cpp b11552 Windows CPU](https://github.com/ggml-org/llama.cpp/releases/tag/b11552),
MIT lisanslı proje; paket içindeki üçüncü taraf lisanslarını koru.
İndirme betiği SHA256 uyuşmazlığında durur, mevcut hatalı dosyayı üzerine yazmaz.
Runtime arşivi indirmede doğrulanır; çıkarılmış executable her açılışta tekrar
hash doğrulamasından geçmez. Model, runtime ve sonuç dosyaları Git dışındadır.

## Kalıcı kuyruk ve web paneli

Önce `scripts/start_vlm.ps1` ile modeli başlat. Ayrı terminalde repo kökünden:

```powershell
.\scripts\start_service.ps1 -ServerExecutable build/live-seg/Release/aegisvision_server.exe -VlmPort 8092 -SegmentationModel artifacts/models/yolov8n-seg/yolov8n-seg.onnx -OcrMultilingual artifacts/models/ocr-multi-v6 -JobDatabase artifacts/service/vlm-demo-jobs.sqlite
```

Ana servis için Qdrant açık olmalıdır. [Panelde](http://127.0.0.1:8090) video seç,
“Sahneyi açıkla” bölümüne git ve kare numarasını gönder. Açıklama ve gerçekten
analiz edilen kare birlikte gösterilir. İş listesinden iptal edebilir veya geçmişteki
tamamlanmış açıklamayı tekrar açabilirsin. Sayfa yenilenirse “Açıklamayı göster”i kullan.

API: `POST /api/jobs`, JSON gövdesi:

```json
{"type":"scene_frame","path":"pedestrians.mp4","frame_index":0}
```

Kare indeksi 0–10000, medya kökü içindeki mevcut video ile sınırlıdır; harici URL,
prompt veya yönetilen canlı arşiv klibi kabul edilmez. Kaynak değişikliği hata verir.
Decode için 30 saniyelik döngü sınırı, inference için 60 saniyelik istemci sınırı vardır;
OpenCV'nin tek bir bloklanan decode çağrısını zorla kesme garantisi yoktur.
`vlm_enabled` yapılandırılmış adaptörü bildirir; ayrı model sunucusunun anlık sağlık
garantisi değildir. Model sunucusu yoksa iş `failed` olur, diğer işlere devam edilir.

İşler mevcut tek worker üzerinden sırayla çalışır. SQLite sonucu ve JPEG önizlemeyi
saklar; tamamlanan iş restart sonrasında tekrar inference yapmaz. Model ayarları
değiştiğinde yeni veritabanı seç; eski veritabanlarını silme. Bağlam imzası adaptör
sürümü ve portu içerir; yerel sunucunun model değişimini tespit etmez.

2026-10-11 gerçek tarayıcı kontrolünde `pedestrians.mp4`, kare 0 için iş
`1791700823339-1` tamamlandı. 768×576 önizleme ve
`A street with people walking and a tripod in the foreground.` metni panelde görüldü.
Bu, bağlantı ve panel doğrulamasıdır; caption doğruluğu benchmark'ı değildir.

## Doğrulama ve kalan işler

`aegisvision_vlm` testi gerçek modeli indirmeden yerel HTTP fixture ile sözleşme,
HTTP hata/yanıt boyutu, timeout, iptal ve iptal sonrası toparlanmayı sınar.
`aegisvision_scene_help` CLI girişini kontrol eder. Gerçek fotoğraf denemesi
yukarıda ayrıca kaydedilmiştir; otomatik testler model doğruluğunu ölçmez.
Sunucu durdurulduktan sonra CLI hata kodu 1 verdi ve yeni çıktı klasörü
oluşturmadı; başarısız inference başarı kaydı olarak yayınlanmaz.

Servis testi kuyrukta başarılı sonuç, model hatası, iptal, hatadan toparlanma,
kapalı yetenek, geçersiz istek ve SQLite yeniden açılışında sonuçların korunmasını
doğrular. Web durum testleri güvenli metin, kaynak değiştirme, geçmişten açma ve
gecikmiş cevapları kapsar. Değişiklikten sonra ilgili 3 CTest ve web testleri geçti.
Daha güçlü model ve bağımsız etiketli sahne değerlendirmesi bekliyor.
