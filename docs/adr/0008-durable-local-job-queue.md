# ADR 0008 — SQLite ile kalıcı yerel iş kuyruğu

Tarih: 2026-10-01 — Kabul edildi. ADR 0007'nin process-local geçmiş kararını günceller.

## Bağlam ve karar

Video indeksleme uzun sürer; süreç kapanınca kabul edilmiş işin kaybolmaması gerekir.
Tek bilgisayar/tek worker aşamasında ayrı Redis kurulumu yerine SQLite 3.53.4
WAL/FULL kullanıyoruz. Kaynak ve SHA3-256 sabitlenir; runtime'a yeni servis eklenmez.

HTTP 202, çalışan duruma geçiş, sonuç ve iptal isteği commit sonrası yayınlanır.
İlerleme en fazla saniyede bir checkpoint edilir. Kabul ve geçmiş budama aynı
transaction'dadır. Hata halinde transaction geri alınır ve kuyruk yeni iş almayı
bırakır; sağlık yanıtı 503 olur. Varsayılan DB sayfa kotası kontrolsüz büyümeyi sınırlar.

SQLite EXCLUSIVE sahipliği aynı veritabanını ikinci worker'ın açmasını engeller.
Kuyruk kimliği artan kalıcı sayaçla üretilir; restart ID'leri tekrar kullanmaz.
Bağlam: medya kökü + YOLO imzası + CLIP embedding alanı + Qdrant hedefi. Uyuşmazlık
başlatmayı durdurur; sessizce başka model/koleksiyona yeniden iş gönderilmez.

## Kurtarma sözleşmesi

Bekleyen işler aynı sırada çalışır. Çalışırken kesilmiş iş aynı ID ile baştan tekrar
işlenir; en fazla üç kurtarma denemesi yapılır. Kullanıcı iptalini ve tamamlanmış
sonuçları tekrar çalıştırmayız. Video boyut/değişiklik zamanı kabul anında kaydedilir;
çalıştırma öncesinde kontrol edilir. Dosyalar iş boyunca sabit kalmalıdır.

Qdrant ve SQLite arasında dağıtık transaction yoktur. Qdrant yazımı ile iş commit'i
arasında süreç kapanabilir; aynı girdi/modelde deterministik point ID'leri tekrarın
kayıt sayısını artırmasını engeller. Sözleşme at-least-once'dur. Kareden devam,
çoklu worker lease'leri ve Redis sonraki kapsamdır. Generic JobQueue handler'ları
kurtarma için tekrar çalıştırılabilir olmalıdır; yan etkileri buna göre tasarlanmalıdır.

## Doğrulama

Deterministik testler checkpoint, iptal, retry sınırı, ID, budama, bağlam, tek
sahiplik ve SQLITE_FULL sonrasında kurtarmayı kapsar. Ayrı süreç `_Exit` testleri
SQLite kapanışını atlayarak WAL kurtarmasını doğrular. HTTP servis yeniden açıldığında
eski arama sonucu ve JPEG önizlemesi korunur. Gerçek kamera kaydında ayrıca süreç
sonlandırma/yeniden başlatma ve Qdrant kayıt sayısı kontrol edilir.

Referans: [SQLite WAL ve EXCLUSIVE](https://sqlite.org/wal.html),
[locking mode](https://sqlite.org/pragma.html#pragma_locking_mode).
