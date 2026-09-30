# Yerel C++ video arama servisi

YOLO → nesne kırpması → CLIP → Qdrant zinciri artık uzun ömürlü bir C++ HTTP
servisi üzerinden çalışır. Tarayıcıda video seçilir, indeksleme başlatılır,
ilerleme/iptal izlenir ve metinle bulunan sonuca tıklanarak kaydın ilgili anı açılır.
Python veya Node.js çalışma zamanında gerekmez. Bu **yerel geliştirme servisi**dir;
internete açık, çok kullanıcılı production deployment değildir.

## Başlatma

Önce [arama kurulumu](search.md) ve [YOLO modeli](yolo.md) hazırlanmış olmalı.
Arama derlemesi `aegisvision_server` hedefini de üretir. Proje kökünde iki terminal:

```powershell
# Terminal 1 — zaten çalışıyorsa yeniden başlatmayın
./artifacts/deps/qdrant/qdrant.exe --config-path configs/qdrant-local.yaml
```

```powershell
# Terminal 2 — yalnızca artifacts/media içindeki videoları erişime açar
./scripts/start_service.ps1
```

Tarayıcı: <http://127.0.0.1:8090>. Başlatıcı eksik model/derleme veya çalışmayan
Qdrant için açıklayıcı hata verir; paket kurmaz, veri silmez, arka plan süreci açmaz.
Terminal açık kalmalıdır; kapatmak için Ctrl+C. Özel klasör/port:

```powershell
./scripts/start_service.ps1 -MediaDirectory C:/videos -Port 8091
```

Doğrudan Windows komutu (Linux'ta exe uzantısını ve Release dizinini kaldırın):

```powershell
./build/search/Release/aegisvision_server.exe configs/service-search.toml configs/image.toml artifacts/media web 8090
```

CLIP ve YOLO başlangıçta bir kez yüklenir. `service-search.toml` ayrı
`aegis_service` koleksiyonunu kullanır; yoksa oluşturur, mevcut verileri sıfırlamaz.
İngilizce sorgularla başlayın: `a person walking on the street`.

## Gerçek kamera kaydı

Bu bilgisayarda OpenCV ile gelen
[`vtest.avi`](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi)
yerel olarak tarayıcı uyumlu H.264 MP4'e dönüştürüldü. Kayıt gerçek yaya hareketi
içerir; fotoğraftan oluşturulmuş animasyon değildir. Kaynak: OpenCV örnek verisi;
yeniden dağıtım için veri kaynağının lisans koşullarını ayrıca inceleyin. Videoyu
ve model dosyalarını repoya eklemiyoruz. Yerel dosya: `artifacts/media/pedestrians.mp4`.

Aynı dosya bilgisayarınızda varsa yeniden üretmek için (FFmpeg gerekir):

```powershell
New-Item -ItemType Directory -Force artifacts/media
ffmpeg -n -i C:/opencv/opencv/sources/samples/data/vtest.avi -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p -movflags +faststart -an artifacts/media/pedestrians.mp4
```

Kayıt 768×576, 10 FPS, 795 kare ve 79,5 saniyedir. Varsayılan stride 15 ile
53 karede tespit yapılır. Nesne sayısı **benzersiz kişi sayısı değildir**;
aynı kişi farklı karelerde ayrı kırpma kaydı oluşturabilir. Bu servis Re-ID yapmaz.
Arama skoru cosine benzerliğidir; başarı olasılığı veya etiketli doğruluk metriği değildir.

Yerel entegrasyon doğrulamasında bu kayıt 53 örnek kareden **410 nesne kaydı**
üretti. Metin sorgusu 8 sonuç döndürdü; tarayıcıda 25,5 saniyelik sonuca tıklama
videoyu o andan oynattı. Yeniden indekslemeyi iptal etme denemesinde 13 tamamlanan
yazım raporlandı ve koleksiyon 410 kayıtta kaldı. Bunlar bu dosya/model/ayarların
çalışma kontrolüdür, detection veya arama doğruluğu değerlendirmesi değildir.

## API

Tüm POST istekleri `Content-Type: application/json` gerektirir.

| İstek | Sonuç |
| --- | --- |
| `GET /api/health` | Süreç hazır, kuyruk/geçmiş sınırları; anlık Qdrant sağlık testi değildir |
| `GET /api/media` | Medya kökündeki video dosyaları |
| `POST /api/jobs` | `202` ve iş ID'si; dolu kuyrukta `429` |
| `GET /api/jobs` | En yeni önce olmak üzere tutulan işler |
| `GET /api/jobs/{id}` | Durum, ilerleme, sonuç veya hata |
| `POST /api/jobs/{id}/cancel` | Bekleyen işi kaldırır veya çalışan işten iptal ister |
| `GET /api/preview/{search_job_id}/{result_index}.jpg` | Kutusu çizilmiş gerçek sonuç karesi |
| `GET /media/{relative_path}` | Byte-range destekli video; tarayıcı codec desteği gerekir |

İndeksleme ve arama gövdeleri:

```json
{"type":"index_video","path":"pedestrians.mp4","stride":15,"max_frames":0}
```

```json
{"type":"search","query":"a person walking on the street","limit":8}
```

Durumlar: `queued`, `running`, `succeeded`, `failed`, `cancelled`.
`stride`: 1–10000; `max_frames`: 0–1000000 (0 = tüm kayıt); `limit`: 1–20.
Yollar medya köküne göre verilmelidir. Mutlak yollar, kökten kaçış ve dışarıya
işaret eden symlink'ler reddedilir. UI yalnızca kökün doğrudan altındaki videoları listeler.

## Kuyruk, tutarlılık ve sınırlar

- Bir inference worker; en fazla 8 bekleyen iş ve 128 işlik bellek içi geçmiş.
  Eski tamamlanmış işler kapasitede atılır. Arama da indekslemenin arkasında bekler.
- İş geçmişi yeniden başlatmada kaybolur; Qdrant verileri diskte kalır. Kalıcı
  iş kuyruğu, Redis, otomatik retry ve crash sonrası job recovery henüz yoktur.
- İptal işbirlikçidir: sürmekte olan model veya Qdrant çağrısı bitince kontrol edilir.
  Tamamlanan yazımlar korunur; rollback yoktur. Aynı dosya/model/ayarlarla tekrar
  çalıştırma sabit ID'leri günceller, kayıtları çoğaltmaz. Ayar/dosya değişikliği yeni
  ID oluşturabilir; eski kayıtlar otomatik temizlenmez.
- HTTP istekleri 8 KiB ile sınırlıdır; dört HTTP worker ve 16 bekleyen bağlantı.
  Video akışı 64 KiB parçalarla okunur; tüm dosya belleğe alınmaz.
- Yalnızca `127.0.0.1` dinlenir. Host/Origin kontrolleri, dosya yolu sınırlaması ve
  içerik güvenlik politikası vardır; kimlik doğrulama/TLS yoktur. Dış ağa açmayın.
- Kaynak dosyaları işlem boyunca değiştirmeyin. Video yükleme endpoint'i yoktur;
  kendi MP4 dosyanızı medya klasörüne koyup sayfayı yenileyin.
- Kare zamanı `frame_index / FPS` üzerinden hesaplanır; VFR/RTSP gerçek PTS desteği
  henüz yoktur. Önizleme arşivden yeniden çözülür, indeksleme anındaki dosya korunmalıdır.

## Test

`ctest --test-dir build/search -C Release --output-on-failure` kuyruğun taşmasını,
bekleyen/çalışan iş iptalini, kısmi ilerlemeyi, hata sonrası devam etmeyi, geçmiş
budamayı, HTTP doğrulamasını, yol sınırlamasını, tekrar indekslemeyi, JPEG önizlemeyi
ve video byte-range isteklerini test eder. Servis testi sahte modellerle deterministiktir;
gerçek YOLO/CLIP/Qdrant kamera denemesi ayrı entegrasyon doğrulamasıdır.
