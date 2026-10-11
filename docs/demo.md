# Windows'ta tek komutla demo

Bu bilgisayardaki hazır model/derleme kurulumu için repo kökünde:

```powershell
.\scripts\demo.ps1 -OpenBrowser
```

Başlatıcı Qdrant (6333), yerel VLM (8092) ve AegisVision (8090) servislerini
kontrol eder. Hazır olanları kullanır, eksik süreçleri gizli pencerelerde başlatır.
Panel: http://127.0.0.1:8090. Komut başka dizinden tam dosya yolu ile de çağrılabilir.

1. `pedestrians.mp4` videosunu seç.
2. “Seçili kareyi analiz et” ile nesne maskelerini gör.
3. “Sahneyi açıkla” ile seçili karenin yerel model yorumunu al.
4. “Yazıları oku” ile Türkçe destekli OCR çalıştır. Yazı olmayan karede boş sonuç normaldir.
5. Arama için videoyu indeksle; ardından İngilizce bir arama metni gönder.

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
