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
3. “Sahneyi açıkla” ile seçili karenin yerel model yorumunu al.
4. “Yazıları oku” ile Türkçe destekli OCR çalıştır. Yazı olmayan karede boş sonuç normaldir.
5. Yalnız arama için “Arama için indeksle”yi kullan; ardından İngilizce bir arama metni gönder. Varsayılan sınır ilk 150 karedir; tüm video için 0 gir. Önceden indekslenmiş videoyu tekrar indekslemek gerekmez.

Demo tek işçiyle çalışır: uzun indeksleme sırasında maske/OCR/VLM ve arama işleri
sırada bekler. Üstteki durum satırı çalışan işi ve bekleyen iş sayısını gösterir;
gerekirse İşler bölümünden iptal et. Sayfa yenilenince başlangıç videosunun son
maske, OCR ve VLM işi otomatik açılır (bitmemişse takip edilir). Başka bir sonuç
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

## Kurulum sınırı

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
