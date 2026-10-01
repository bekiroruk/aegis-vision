# ADR 0010 — Süre sınırlı, tek worker'lı canlı HTTP önizleme

Tarih: 2026-10-01. Durum: Kabul edildi.

## Bağlam

RTSP CLI reconnect/drop-oldest özelliklerine sahipti, ancak tarayıcıda yalnızca
dosya araması vardı. Sonsuz canlı işi kalıcı dosya kuyruğuna eklemek CLIP aramasını
bloke eder; HTTP istemcisi başına decoder/model açmak kaynak kullanımını büyütür.
Decoder bağlantısı koptuğunda son JPEG'i göstermeye devam etmek de yanıltıcıdır.

## Karar

- CLI ve servis ortak `vision::analyze_stream` motorunu kullanır. CLI sink dosyaya,
  servis sink yalnızca son kutulu JPEG'e yazar. Capture RAII ile durur/birleştirilir.
- Bir aktif, süre sınırlı, nonpersistent canlı oturum ve ona özel detector instance'ı.
  SQLite dosya/arama kuyruğu değişmez; canlı oturum yeniden başlatmada replay edilmez.
- Kaynaklar servis başlangıcında allowlist'te belirlenir. HTTP yalnızca source ID
  kabul eder; URL/kimlik bilgisi istemciden alınmaz veya API'de döndürülmez.
- Model tek worker'da çalışır. Son JPEG değişmez shared buffer'dır; en fazla
  960×720 / 2 MiB. Capture kuyruğu ayrıca kare/byte sınırlarını korur. Model ve
  FFmpeg iç tamponlarının toplam RAM'i bu sınırlarla garanti edilmez.
- Kısa JPEG GET istekleri yaklaşık 2 FPS; uzun MJPEG bağlantısı yoktur. Disconnect,
  source session değişimi, stop veya 2 saniyeden eski decode arrival önizlemeyi
  geçersiz kılar. Tarayıcı ayrıca süre dolunca görüntüyü temizler.
- Start/worker join bir lifecycle mutex, snapshot ayrı mutex ile korunur. İkinci
  aktif start `409`; stop atomic cancellation isteği ve `202` verir. Mevcut model
  yüklemesi/inference/open/read zorla kesilmez. Shutdown worker'ı birleştirir.
- Yalnızca geçerli oturum ve son terminal özeti tutulur; yeni ID eski ID'yi
  geçersiz kılar. Süre sonunda en az bir kareli oturum completed; boş/hatalı kaynak failed.

## Sonuçlar ve doğrulama

Dosya/metin araması canlı analiz sırasında çalışabilir; CPU ve ek model RAM'i
paylaşılır, throughput garantisi yoktur. Bu sürüm canlı arşiv, sürekli nesne
indeksleme, fiziksel kamera PTS, Re-ID, ses veya tam FPS video sunmaz.

Deterministik HTTP testi source allowlist, paralel start, ayrı model, stale JPEG,
disconnect/recovery/epoch, stop, hata, süre sınırı ve cleanup'ı doğrular. Ayrı
Windows entegrasyonu gerçek yaya MP4'ünü FFmpeg/MediaMTX RTSP olarak yayınlar,
publisher'ı kesip geri getirir ve eşzamanlı gerçek CLIP/Qdrant sorgusunu tamamlar.
Yerel görüntüler ve rapor outputs altındadır; model/video repo dışında tutulur.
