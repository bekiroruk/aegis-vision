# ADR-0011: Sınırlı canlı analiz arşivi ve sonlu indeksleme işleri

Durum: Kabul edildi. Tarih: 2026-10-02.

## Bağlam

Canlı worker düşük gecikmeli, drop-oldest önizleme için tasarlandı. RTSP kaynağını
sonsuz SQLite işine dönüştürmek tek dosya/arama worker'ını süresiz bloke eder;
her kayda ayrı CLIP yüklemek belleği ve concurrency'yi kontrolsüz büyütür. Kullanıcı
canlıda görülen nesneyi metinle arayıp o kayda dönmek istiyor. Öte yandan kamera
PTS henüz bulunmuyor; analiz karelerinin sıkıştırılmış oynatma zamanı gerçek yayın
zamanıyla karıştırılmamalı.

## Karar

- Arşiv API'de açık opt-in (`archive: true`); gönderilmezse yalnızca önizleme çalışır.
  UI özelliğin kullanılabilirliğini ve sınırlarını gösterir. URL/encoder yolu
  sunucu başlangıcında seçilir, HTTP istemcisi komut veya kaynak URL'si göndermez.
- Canlı worker analiz edilen kareleri küçük MJPEG parçalarına yazar. Parça 10 saniye
  arrival aralığı, 100 kare, source session/format sınırı veya stop ile kapanır.
  Oturumda 4, küresel olarak 8 parça ve 256 MiB disk sınırı; kota dolunca yalnızca
  arşiv durur. Eski kullanıcı kayıtları otomatik silinmez. Medya kökündeki arşiv
  için ayrıca native tek-süreç sahiplik kilidi alınır; farklı SQLite DB seçmek
  aynı arşive ikinci yazıcı başlatma yetkisi vermez. Sahip süreç kapanınca işletim
  sistemi kilidi serbest bırakır; kilit dosyasının kalması aktif sahiplik değildir.
  Her slotta raw, encoded output ve manifestler için 32 MiB rezervasyon ayrılır.
- Sonlu indeksleme işi mevcut SQLite kuyruğunda kabul edilir. Mevcut worker önce
  güvenilir FFmpeg executable'ını shell olmadan sabit argv ile başlatır, 30 saniye
  deadline ve iptalde yalnızca sahip olduğu alt süreci sonlandırır. Ham/MP4 dosya
  ayrı ayrı 12 MiB sınırındadır. Encoder `-fs` seçeneği ayrıca bir önlem olsa da
  advisory'dir; geçici MP4 kısa süre bu sınırı aşabilir. Final MP4 boyutu ve decode
  kare sayısı doğrulanmadan yayımlanmaz;
  raw/staging dosyaları HTTP medya erişiminden çıkarılır.
- Aynı worker/model ile YOLO/CLIP/Qdrant indekslenir; varsayılan stride 10'dur.
  Kuyruk doluysa parça pending kalır. Otomatik sonsuz retry/backlog yoktur; özel
  retry API'si running/queued işi tekilleştirir ve yine kuyruk sınırına tabidir.
  Kabul edilmiş işler standart crash-recovery kurallarıyla kurtarılır.
- Manifest, kapalı parça ve kare/arrival/source sequence/session/tracking epoch
  eşlemesini saklar. Yeniden açılışta kapalı manifestler listelenir; canlı capture
  auto-resume etmez ve kabul edilmemiş pending işler kendiliğinden gönderilmez.
  Tamamlanma kaydı iş geçmişi budansa da başarılı parça durumunu korur.
- Kapalı ham video ve manifest SHA256 mühürleriyle sabitlenir; iş kabulünde ve
  çalıştırmada içerik doğrulanır. Katalog yenilemesi MP4 için ucuz boyut/mtime
  kontrolü kullanır; medya/önizleme erişimi ise yalnızca istenen klibin SHA256
  içeriğini de kontrol eder. Aynı boyut/mtime ile değiştirilen MP4 sunulmaz.
  Bu kontrol yerel dosyaları işlem sırasında değiştirmeme gereğini veya filesystem
  erişim kontrolünü ortadan kaldırmaz; harici yazıcılara karşı atomik snapshot değildir.
  Katalog kabul noktası atomik yayımlanan `sealed.json` dosyasıdır; ondan önce
  yazılan `manifest.json` tek başına bir parçayı görünür yapmaz. Böylece kayıt
  sürerken HTTP okuması henüz mühürlenmemiş parçayı bozuk olarak etiketlemez.
  Mühür mevcutken bozuk/eksik manifest yine başarısız parça olarak görünür.
- Arama payload'ındaki `origin: live_archive` ve live session/source metadata'sı
  ayrı canlı kapsam filtresi sağlar. Sabit parça/kare/model ID'leri yeniden
  indekslemede kayıtları günceller. Genel `index_video` arşiv alt ağacına kabul
  edilmez; özgün metadata özel iş yolunda korunur.
- Tarayıcı tek son JPEG önizleme akışını değiştirmez. Arşiv listesi iki saniyede
  bir, tek in-flight istekle güncellenir; kaydı açma hazır MP4'e bağlıdır. Metinler
  `textContent` ile işlenir; metadata HTML olarak yorumlanmaz.

## Sonuçlar ve sınırlar

Ek CLIP modeli veya sınırsız canlı iş oluşmaz. Arşiv kayıt/encoder hatası önizlemeyi
durmak zorunda bırakmaz; state/error UI'da görünür. Arama finite encoder/index
işlerinin arkasında bekleyebilir, CPU/RAM yarışması sürer; hard realtime değildir.

Bu tam kamera kaydı değildir. Yalnızca analiz kareleri 10 FPS CFR klibe dönüşür;
`timestamp_ms` klip kare/FPS konumudur. Manifestteki monotonic arrival, kamera
PTS veya duvar saati senkronizasyon garantisi değildir. Ses, GStreamer, tam FPS
kayıt, otomatik retention, çoklu worker ve fiziksel kamera doğrulaması ayrı işlerdir.
Yerel servis kimlik doğrulama/TLS içermez; yalnızca loopback'te kullanılmalıdır.
Disk rezervasyonu uygulama kabul kontrolüdür, OS filesystem kotası değildir;
dış süreç yazımları, decoder/model dosyaları ve geçici encoder taşması için mutlak
disk kullanım garantisi sağlamaz.
