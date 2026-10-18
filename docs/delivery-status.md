# Teslim durumu

Güncelleme: 2026-10-18. Bu belge mevcut teslimin kısa listesidir; bütün ilan
maddelerinin tamamlandığı veya üretime hazır olduğu anlamına gelmez.

## Kullanılabilir demo

18 Ekim kapanış kapsamı: **Video Studio / yerel video portföy demosu**.
Kullanıcı fiziksel kamera testini kapsamdan çıkardı. Aşağıdaki ileri araştırma ve
üretim başlıkları bu demonun teslimini bloke etmez; tamamlanmış da sayılmaz.
Yeni kullanım ve iki farklı gerçek videonun sonuçları: [Video Studio](video-studio.md).
`DEMOYU-AC.cmd` paneli, `VIDEO-EKLE.cmd` kendi video seçim penceresini açar.

Hazır kurulumdaki panel: `http://127.0.0.1:8090`.
Kapandıysa repo içinden `./scripts/demo.ps1 -OpenBrowser` çalıştırılır.
Maske/video analizi, OCR, yerel VLM ve görsel/metin arama aynı paneldedir.
Model açıklamaları ve takip kimlikleri doğrulanmış gerçek etiketler değildir.

Servis başlatmadan dosya eksiklerini topluca kontrol et:

```powershell
.\scripts\demo.ps1 -CheckOnly | ConvertTo-Json
```

`FilesPresent` yalnız başlangıç dosyalarının varlığını belirtir. Boş/bozuk model,
eksik DLL, GPU veya port hazır demek değildir. Bu kontrol dosya oluşturmaz,
servis başlatmaz/durdurmaz; farklı çalışma klasöründen de kullanılabilir.

## Bu turda kapatılan küçük işler

- Linux CI derlemesini durduran VLM HTTP fixture generic-lambda tür belirsizliği
  giderildi. Windows'ta VLM taşıma/sınır/timeout/iptal testi geçti.
  `78fb968` için [GitHub çalışmasında](https://github.com/bekiroruk/aegis-vision/actions/runs/38696609083)
  5/5 grup başarılı: API, Windows/Linux temel, OpenCV Linux, arama/servis Linux.
  Son grup 20 tekrarlı arşiv kontrolünü de geçti. Bu sonuç sonraki teslim
  belgesi/ön kontrol commit'lerinin CI sonucunu otomatik olarak doğrulamaz.
- Demo dosya envanteri tek listede toplandı; eksikler topluca gösteriliyor.
  `-CheckOnly` Windows PowerShell 5.1 ve PowerShell 7'de boş/tam/eksik kurulum,
  dosya yerine klasör, farklı çalışma klasörü, çelişen seçenekler ve yan
  etkisizlik testlerini geçti. Windows CI'a eklendi.
- Demo servisleri yeniden başlatıldı. `/api/health`: `ready`, segmentation,
  video, OCR, Unicode ve VLM açık; kalıcı geçmiş açık, storage_error boş.
  Bu sağlık kontrolüdür; bütün modeller bu turda yeniden ölçülmedi.
- Web regresyonları geçti: video, arama seçimi, yenileme, kuyruk,
  OCR/VLM/maske, iptal ve güvenli URL kontrolleri.
- Gerçek maskeli/takipli videodan 20 saniyelik kısa demo hazırlandı:
  `outputs/delivery-20261015/aegis-vision-demo-20s.mp4`.
  H.264, 768×576, 200 kare, 20 saniye, 1.763.679 bayt; FFprobe ile doğrulandı.
  SHA256: `70496dc8c368377ea24bf1d14095f474c30e60c948a1b28853c33b2c50372e57`.
  Kaynak: `artifacts/media/analysis-output/v-17918018757271891-1/video.mp4`.
  Eski analiz çıktısının kısaltılmasıdır; yeni inference/canlı GPU kanıtı değildir.
  Video yerelde tutulur, Git'e yüklenmez; kaynağın lisansı ayrıca gözetilir.

## Demo dışında kalan ileri çalışmalar — isteğe bağlı ayrı kapsam

1. **Gerçek RTSP + GPU birleşik testi.** TensorRT analizi, eşzamanlı dosya
   işi, kesinti/yeniden bağlantı ve temiz stop tek gerçek yayında sınanmalı.
   CPU RTSP ve GPU dosya replay ayrı ayrı var; birleşik test yerine geçmez.
   Mevcut publisher engeli aşılmış sayılmıyor.
2. **Linux GPU Docker dağıtımı — yerel işlevsel kabul geçti.**
   [18 Ekim gerçek GPU testi](gpu-container.md) Docker Desktop/WSL2 Linux amd64
   konteynerinde imaj derleme, RTX 3060'a özel FP16 engine, video indeksleme,
   CLIP/Qdrant arama ve JPEG önizlemeyi doğruladı. Aktif iş sırasında temiz stop
   1,417 saniye; aynı ID attempts=2 ile tamamlandı, eski arama sonucu korundu.
   Servis root değil, kök/model/engine salt okunur; izole test host portu açmaz.
   Bu artık yalnız paket/CPU Redis kanıtı değildir. Kalan: native Linux sunucu
   ağı/izinleri, bağımsız kalite ve yük testleri, güvenlik/lisans/üretim imajı
   incelemesi. Engine'in üretim onayı verilmedi; bütün dağıtım işi bitmiş değildir.
3. **Çoklu kamera/Re-ID.** [32-ayarlı geliştirme kalibrasyonu](reid-calibration.md)
   tamamlandı; hiçbir ayar kalite koşullarını birlikte geçmedi, varsayılan korundu.
   [İlk GPU metric-learning deneyi](reid-training.md) tamamlandı: 305 ayrı eğitim
   kırpması, sabit OSNet + doğrusal katman; geliştirme Rank-1 %70,21→%68,09,
   aday reddedildi. Daha çeşitli eğitimle kalite iyileştirmesi ve yeni son test;
   saat eşleme, gerçek kamera geçişleri ve canlı kimlik ilişkilendirmesi gerekiyor.
4. **INT8 ve diğer GPU modelleri.** INT8 engine üretimi var fakat ilk aday
   kalite/hız kapısını geçmedi. Ayrı veriyle yeni kalibrasyon/QAT ve ölçüm;
   segmentation/diğer modellerin TensorRT uyarlaması gerekiyor. FP16 detection var.
5. **Bağımsız kalite doğrulaması.** Detection/tracking/segmentation, arama,
   doğal Türkçe OCR, VLM ve geometrik eşleştirme için daha geniş bağımsız veri.
   Eğitim/değerlendirme araçları var; bütün alanlarda iyi kalite iddiası yok.
6. **Dağıtık kuyruk.** [C++ Redis sahiplik/kurtarma çekirdeği](redis-queue.md)
   eklendi; lease, token kontrollü sonuç kabulü, iptal, retry ve kapasite sınırları.
   Linux'ta gerçek Redis ile sekiz süreç yarışması ve AOF restart/kurtarma geçti.
   Gerçek YOLO görüntü worker'ı, otomatik heartbeat, iptal/sahiplik kaybında
   sonuç bastırma ve model/medya hash kontrolü eklendi. `69a6f31` için
   [7/7 CI grubu](https://github.com/bekiroruk/aegis-vision/actions/runs/38897763793)
   geçti; gerçek Redis → C++ YOLO → sonuç zinciri 4 kişi/1 otobüs ile doğrulandı.
   Linux görüntü worker'ı OpenCV 4.12 kullanır; eski 4.6 importer uyumsuzdur.
   İsteğe bağlı görüntü HTTP/panel bağlantısı da doğrulandı: `2ff7fd0` için
   [7/7 CI grubu](https://github.com/bekiroruk/aegis-vision/actions/runs/38935643135)
   başarılı. Gerçek Chromium → HTTP → Redis → ayrı C++ YOLO worker → kutulu
   sonuç, sayfa yenileme ve iptal geçti. Bu birleşik test hafif HTTP fixture'ında;
   ana Windows servisinin derlemesi ve ayrı servis regresyon testi de geçti.
   Bu ilk koşu Windows'ta Redis çalıştırıldığı anlamına gelmiyordu.
   Süreli worker sinyali ve panelde otomatik yenilenen varlık durumu eklendi;
   `ced9aba` için [gerçek Redis ve tarayıcı testleri](https://github.com/bekiroruk/aegis-vision/actions/runs/38941310334)
   geçti: boşta sinyalin yenilenmesi, gerçek inference ve kapanınca süre dolumu.
   Aynı koşunun 7/7 CI grubu başarılı.
   Bu model hazır/boş kapasite garantisi değildir. 17 Ekim'de Docker'ın bozuk
   geçici socket klasörleri silinmeden yedek adlara taşınınca motor açıldı.
   [Yerel tam servis + Redis + C++ worker + tarayıcı](demo.md) gerçek fotoğrafta
   4 kişi/1 otobüs üretti; sayfa yenilemesi sonucu korudu. Başlatıcıya
   `-RedisImages` eklendi. Docker 4.53.0 hatası sonraki açılışta tekrarladı;
   geçici klasör düzeltmesi kalıcı çözüm sayılmaz. Kullanıcı onaylı yerinde
   4.95.0 güncellemesi sonrasında Engine 29.8.2 ve Redis açıldı; eski iş korundu.
   Bir kontrollü Docker kapatma/açma testinde socket hatası tekrarlamadı.
   Windows yeniden başlatma/uzun süreli kararlılık henüz sınanmadı.
   Fabrika ayarlarına dönüş, imaj/konteyner/volume silme yapılmadı.
   Diğer analiz türleri ve dosya/Qdrant yan etkilerinin korunması hâlâ eksik.
   Mevcut panel SQLite kullanmaya devam eder; yeni worker JSON sonucu Redis'e yazar.
7. **Temiz makine teslimi ve dağıtım incelemesi.** Sıfırdan kurulumun ayrı
   makinede denenmesi, model/veri/bağımlılık lisans envanteri ve dağıtım
   koşullarının incelenmesi, yük/operasyon kontrolleri. Kısa demo hazır;
   diğer teslim şartları açık. THIRD_PARTY_NOTICES TrackEval ve hiredis
   bildirimlerini içerir; tam lisans denetimi değildir.
8. **Triton / Jetson.** Tercih sebebi olan ileri deployment için gerçek
   sunucu/edge ortamı ve uyumluluk/hız ölçümleri gerekiyor.

DINO/ViT, YOLO/DETR, Qdrant/FAISS, Redis/Celery ve GStreamer/DeepStream gibi
örneklerin her birini ayrıca uygulamak şart değildir. C++ tercihi korunur;
sırf isim tamamlamak için ikinci bir FastAPI uygulaması eklenmez. Kullanılan
alternatifler ve eksikler [ilan karşılaştırmasında](job-requirements.md) ayrıdır.
Mezuniyet ve üç yıllık profesyonel deneyim repo geliştirerek sağlanamaz.

Ayrıntılar: [demo](demo.md), [yol haritası](roadmap.md),
[GPU servis](gpu-service.md), [Re-ID karşılaştırması](reid-model-experiment.md).
