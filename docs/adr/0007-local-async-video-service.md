# ADR 0007 — Yerel asenkron C++ video arama servisi

Tarih: 2026-10-01 — Kabul edildi.

## Bağlam

CLI modelleri her çağrıda yeniden yükler ve video sonuçlarını etkileşimli göstermeyi
zorlaştırır. Amaç mevcut C++ YOLO/CLIP/Qdrant akışını gerçek video üzerinde
kullanılabilir hale getirmek; henüz dağıtık veya internete açık servis tasarlamak değil.

## Karar

- Mevcut sabitlenmiş cpp-httplib bağımlılığıyla loopback HTTP API ve statik web ekranı.
- Model/store erişimini tek worker'a ait tutan sınırlı bellek içi kuyruk. HTTP
  thread'leri inference çalıştırmaz; doğrulama, job kontrolü ve medya sunar.
- İndeksleme ve arama için aynı asenkron iş sözleşmesi. Kuyruk doluysa 429;
  durum/ilerleme polling ile okunur. Modeller süreçte bir kez yüklenir.
- İşbirlikçi iptal; tamamlanan Qdrant upsert'leri korunur. Sabit kayıt ID'leri
  değişmeyen girdilerde tekrar çalıştırmayı güvenli kılar; rollback yapılmaz.
- Video yalnızca yapılandırılmış medya kökünden sunulur. Dosyalar bounded parçalarla
  aktarılır, HTTP Range ile tarayıcı zaman atlaması desteklenir. Arama önizlemesi
  kaynak videodaki kareyi çözer ve indekslenen nesnenin kutusunu çizer.

## Sonuçlar ve alternatifler

Model thread-safety varsayımı ve fazladan Python/FastAPI çalışma zamanı gerekmez.
Uzun video sırasında aramalar da bekler. CPU paralellik/batching performansı henüz
optimize edilmedi. Aynı iş kuyruğu kullanımı bilinçli başlangıç kararıdır.

Redis/ayrı worker süreçleri ilk sürümde eklenmedi: iş geçmişi process-local ve
crash recovery yoktur. Gelecek adım kalıcı kuyruk, sahiplik/lease, retry ve yeniden
başlatmada kurtarma sözleşmesidir. Qdrant'ın kalıcı olması işleri kalıcı yapmaz.

RTSP, gerçek PTS, Re-ID, auth/TLS ve GPU deployment ayrı aşamalardır. API dış ağa
açılamaz; Host/Origin kontrolleri kimlik doğrulamanın yerine geçmez. Ayrıntılı
sınırlar ve testler [servis kılavuzunda](../service.md).
