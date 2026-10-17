# Windows'ta tek komutla demo

Bu bilgisayardaki hazır model/derleme kurulumu için repo kökünde:

```powershell
.\scripts\demo.ps1 -OpenBrowser
```

Başlatıcı Qdrant (6333), yerel VLM (8092) ve AegisVision (8090) servislerini
kontrol eder. Hazır olanları kullanır, eksik süreçleri gizli pencerelerde başlatır.
Panel: http://127.0.0.1:8090. Komut başka dizinden tam dosya yolu ile de çağrılabilir.

1. `pedestrians.mp4` videosunu seç.
2. Üstteki “İlk kareyi analiz et” ile nesne maskelerini gör; indeksleme gerekmez.
   Hareketli çıktı için **Maskeli video + nesne takibi → Videoyu analiz et** seç.
   Varsayılan 0 tüm videoyu işler (en fazla 1.000 kare); örneğin 150 girersen
   yalnız ilk 150 kare işlenir. Tamamlanınca oynat, **MP4 indir** veya takip CSV'sini indir.
   İşlem sürerken **Analizi iptal et** kullanılabilir. Bu işlem de indeksleme gerektirmez.
3. “Sahneyi açıkla” ile seçili karenin yerel model yorumunu al.
4. “Yazıları oku” ile Türkçe destekli OCR çalıştır. Yazı olmayan karede boş sonuç normaldir.
5. Yalnız arama için “Arama için indeksle”yi kullan; ardından İngilizce bir arama metni gönder. Varsayılan sınır ilk 150 karedir; tüm video için 0 gir. Önceden indekslenmiş videoyu tekrar indekslemek gerekmez.

Demo tek işçiyle çalışır: uzun indeksleme sırasında maske/OCR/VLM ve arama işleri
sırada bekler. Üstteki durum satırı çalışan işi ve bekleyen iş sayısını gösterir;
gerekirse İşler bölümünden iptal et. Sayfa yenilenince başlangıç videosunun son
tek-kare maske, OCR, VLM ve maskeli video işi otomatik açılır (bitmemişse takip edilir). Başka bir sonuç
için İşler bölümündeki göster düğmelerini kullan. Kaynak tanımlanmayan canlı
kamera paneli ve devre dışı/boş arşiv paneli gösterilmez; eski arşiv kayıtları korunur.

Arayüzdeki kare numarası kullanılır; videoyu duraklattığın an otomatik seçilmez.
Model çıktıları hatalı olabilir. Canlı RTSP kaynağı bu demo profilinde etkin değildir;
[canlı video kılavuzundaki](live-video.md) ayrı başlatma komutunu kullan.

Durdurmak için:

```powershell
.\scripts\demo.ps1 -Stop
```

Yalnız bu başlatıcının açtığı ve kayıtlı süreç kimliği, başlangıç zamanı ve executable
yolu eşleşen süreçler kapatılır. Önceden açık servisler korunur. Veritabanı,
videolar ve modeller silinmez. Devam eden işleri önce panelden bitir/iptal et;
durdurma süreç sonlandırmadır, yarıda kalan işler sonraki açılışta mevcut kuyruk
kurtarma kurallarına tabidir. Birden fazla başlatıcı aynı anda çalışamaz.

Günlükler ve süreç kayıtları `artifacts/service/demo-launcher` altındadır.
Yeni süreç başlatıldığında ilgili log yenilenir. Başlatma hatasında yalnız bu
çağrıda açılan süreçler kapatılır. Port başka uygulamadaysa üzerine yazılmaz.
Hazırlık kontrolü bileşen başına en fazla 120 saniye bekler.

## Redis görüntü bölümü — hazır yerel kurulum

Docker Desktop açık ve motoru hazırken, bu bilgisayarda oluşturulmuş Redis
konteynerini ve tam demo + ayrı C++ görüntü işçisini başlat:

```powershell
docker start aegis-redis-local
.\scripts\demo.ps1 -RedisImages -OpenBrowser
```

Ana panelin fotoğraf bölümü veya `http://127.0.0.1:8090/redis.html` üzerinden
`bus.jpg` → **Fotoğrafı analiz et**. Yerel örnek, `artifacts/samples/bus.jpg`
dosyasının `artifacts/media/bus.jpg` kopyasıdır. Başlatıcı fotoğraf kopyalamaz.
Video/OCR/VLM hâlâ aynı panelde mevcut SQLite iş akışını kullanır.

`-RedisImages`, Redis ile derlenmiş `build/search/Release/aegisvision_server.exe`
ve `build/redis-cv/Release/aegisvision_redis_worker.exe` kullanır. Redis adresi
`127.0.0.1:16379`, kuyruk `desktop-images-v1`, medya kökü `artifacts/media`.
Başlatıcı Docker/Redis kurmaz veya başlatmaz. Mod değiştirirken önce
`./scripts/demo.ps1 -Stop` kullan; başka süreçlerin portunu ele geçirmez.
İşçi bağlantı/sahiplik hatasında güvenli biçimde çıkar. Bağlantı düzeldikten
sonra aynı başlatma komutu eksik işçiyi yeniden açar; çalışan servisleri korur.

`-Stop` kayıtlı C++ işçisini de kapatır, Docker/Redis'i kapatmaz. Redis'i ayrıca
durdurmak için `docker stop aegis-redis-local`; veri volume'da korunur.
Ön kontrol `./scripts/demo.ps1 -RedisImages -CheckOnly` ile yapılabilir.
Bu kurulum yeni klonda kendiliğinden oluşmaz; ayrıntılar [Redis belgesinde](redis-queue.md).

### Yerel gerçek analiz kanıtı — 2026-10-17

Docker Engine 29.0.1 + loopback Redis + ana Windows C++ servisi + ayrı worker
zinciri tarayıcıdan doğrulandı. `img-c5c2fe82-df4c-46aa-8745-b4c0a9da7414`
işi `succeeded`, attempts=1; 4 kişi ve 1 otobüs, kutulu önizleme. Sayfa
yenilemesi aynı sonucu korudu. Bu yalnız Linux CI fixture kanıtı değildir.
Yerel `outputs/local-redis-20261017/job.json` ve `result.jpg` Git'e yüklenmez.
Tek ölçüm inference 400,05 ms; benchmark veya GPU kanıtı değildir.
Servis sağlığı segmentation/video/OCR/Unicode/VLM açık ve storage_error boş;
bu turda bu modellerin hepsi tekrar çalıştırılmadı.

Başlatıcının yeniden çağrılması mevcut süreçleri çoğaltmadı; kapalı işçi yeniden
açıldı. Ön kontrol testleri Windows PowerShell 5.1 ve PowerShell 7'de geçti.
`73fa5b1` için [CI koşusunun](https://github.com/bekiroruk/aegis-vision/actions/runs/39009437068)
7/7 grubu başarılı; bu sonuç Docker güncellemesinin tamamlandığı anlamına gelmez.

### Docker yerinde güncelleme — 2026-10-17

Kullanıcı onayıyla Docker Desktop 4.53.0 → 4.95.0.242831 güncellendi.
Resmî kurulum dosyasının SHA256 değeri
`73431db38491f770543880366228a23c993f84d1e8e045f56d83b8a98940cf76`
ve geçerli Docker Inc imzası doğrulandı; kurulum çıkış kodu 0.
Linux Engine 29.8.2 açıldı. Mevcut `aegis-redis-local` konteyneri ve
`aegis-redis-local-data` volume'u korundu; Redis PING → PONG.
Yukarıdaki eski iş aynı ID, `succeeded` ve attempts=1 ile geri geldi.
Ana servisin sağlık kontrolü `ready`, `storage_error` boş döndü.

Demo ve Redis kontrollü durdurulduktan sonra Docker tamamen kapatılıp
yeniden açıldı; socket başlangıç hatası bu denemede tekrarlamadı ve Redis
yeniden PONG döndürdü. Panelden yeni `img-46f69b54-6d8f-4359-a7fc-773620b7c9f4`
işi gönderildi: `succeeded`, attempts=1, 4 kişi/1 otobüs; tek inference ölçümü
398,06 ms. Yerel ekran kanıtı `outputs/local-redis-20261017/panel-after-update.jpg`.
Bu tek yeniden açılış testidir; Windows yeniden başlatma
ve uzun süreli kararlılık testi değildir. Fabrika ayarlarına dönüş,
imaj/konteyner/volume silme yapılmadı; önceki geçici klasör yedekleri korundu.
Bu kontrol Linux GPU inference veya GPU konteyner teslimi yerine geçmez.

## Kurulum sınırı

Servisleri açmadan bütün eksik başlangıç dosyalarını görmek için:

```powershell
.\scripts\demo.ps1 -CheckOnly | ConvertTo-Json
```

Bu salt okunur envanterdir; model içeriği, DLL, port veya GPU doğrulaması
yapmaz. `FilesPresent: true` tek başına uygulama çalışır demek değildir.
`-Stop` / `-OpenBrowser` ile birlikte kullanılamaz. Regresyon kontrolü:
`./scripts/test_demo_preflight.ps1`.

Bu komut kurulum/indirme aracı değildir. Eksik dosyada hangi dosyanın gerektiğini
yazar. İlk kez klonlanan repoda derleme ve modeller ayrıca hazırlanmalıdır:

- [Servis ve CLIP/Qdrant](service.md), [arama bağımlılıkları](search.md)
- [Segmentasyon modeli](segmentation.md), [Türkçe OCR](ocr-turkish.md)
- [Yerel VLM ve runtime](vlm.md)

`build/live-seg/Release/aegisvision_server.exe`, sabit demo portları, mevcut
`configs/analysis.toml` profili ve `artifacts/service/vlm-demo-jobs.sqlite`
kullanılır. Profilde VLM portunu değiştirirsen ayrı servis başlatıcılarını kullan.
Program/model sürümü otomatik yükseltilmez. Çalışan süreç yeniden kullanıldığında
yetenekleri ve executable yolu doğrulanır; medya kökü/model içeriğinin aynı olduğu
kanıtlanmaz. Özel ayarlı servis açıkken kendi başlatma komutunu tercih et.

## Yerel doğrulama — 2026-10-11

PowerShell 7 ve Windows PowerShell ile hazır servislerin yeniden kullanımı geçti.
VLM ve ana servis kapalıyken başlatıldı; önceden açık Qdrant korundu.
Gerçek `scene_frame` işi `1791700823339-3` başarıyla sonuçlandı. `-Stop` yalnız
başlatıcının açtığı iki süreci kapattı; yeniden açılışta sonuç `attempts=1` ile
korundu. Qdrant'ın bu başlatıcı üzerinden sıfırdan açılışı ve hata sırasında
otomatik geri alma bu kontrolde ayrıca denenmedi.

## Görünür sonuç doğrulaması — 2026-10-12

Tarayıcıda sayfa açılışında mevcut maske/OCR/VLM sonuçlarının otomatik geri geldiği
doğrulandı. “İlk kareyi analiz et” gerçek servis üzerinde `1791700823339-10` işini
oluşturdu ve 768×576 kare üzerinde 6 maske gösterdi. Önceki tam video indekslemesi
795 kare okudu, 53 kare örnekledi ve 410 nesne kaydı oluşturdu (benzersiz nesne
sayısı değildir). Arama 8 eşleşme döndürdü; 3 eski arşiv eşleşmesinin medya dosyası
bu profilin medya kökünde olmadığı için oynatma düğmeleri devre dışıydı.
OCR'ın tek karakterlik çıktısı güvenilir metin okuma kanıtı sayılmadı.
`node tests/test_web_state.cjs` yenileme, kuyruk açıklaması, hızlı analiz,
kaynak değişimi ve mevcut güvenli çıktı regresyonlarıyla geçti.

## Tam video panel doğrulaması — 2026-10-12

Panelde “Videoyu analiz et” ile `1791700823339-11` işi gönderildi. Gerçek
YOLOv8n-seg + IoU pipeline'ı `pedestrians.mp4` videosunun 795 karesini işledi.
Çıktı H.264 MP4, 768×576, 10 FPS, 79,5 saniye ve 17.137.901 bayt.
7.034 kare-içi maske ve 96 takip ID'si üretildi; bunlar doğruluk metriği veya
benzersiz kişi sayısı değildir. Basit IoU takipçisi kopabilir, model yanlış sınıf
üretebilir. Çıktı `artifacts/media/analysis-output/v-17918018757271891-1/` altında.

Sayfa yeniden açıldığında sonuç geri geldi; tarayıcıda `paused=false`,
`readyState=4`, süre 79,5 saniye ve doğru video boyutlarıyla oynatma doğrulandı.
İleri sarma kontrolü çalıştı. “MP4 indir” dosyayı bilgisayarın Downloads klasörüne
kaydetti; indirilen dosyanın SHA256 değeri yerel çıktıyla aynı:
`70bb298cf2d10733dc8ba1b55293c01feacc6dd7509559ea4c26542075ab2f14`.
FFprobe dosyanın 795 karesinin tamamını saydı. 46/46 CTest geçti; ek sınır ve
iptal testlerinden sonra servis/encoder testleri tekrar geçti. Web regresyonu
gönderim, ilerleme, MP4 URL'si, indirme, iptal, hata, sayfa yenileme, kaynak değişimi
ve güvensiz ID reddini kapsadı. Bu doğrulama Windows CPU üzerindedir.
