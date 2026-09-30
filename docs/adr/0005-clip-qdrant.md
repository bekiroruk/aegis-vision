# ADR 0005: CLIP ve yerel Qdrant adaptörleri

- Tarih: 2026-09-30
- Durum: Kabul edildi

## Karar

Ortak görsel/metin uzayı için CLIP ViT-B/32, CPU inference için ONNX Runtime
seçildi. Statik batch=1, 224×224 görsel / 77 token sözleşmesi kullanılır.
İki encoder ayrı ONNX dosyasıdır. Unicode normalizasyon/regex için ICU,
byte-level BPE için modelin vocab/merges verilerini okuyan C++ uygulama kullanılır.
Model export ve referans üretimi Python'dadır; uygulama C++'tır.

`IEmbedder` ve `IVectorStore` korunur. Qdrant adapter yalnızca yerel HTTP ve
Cosine/adsız vektör koleksiyonunu destekler. nlohmann/json ve cpp-httplib bağımlılıkları
yalnızca isteğe bağlı arama derlemesinde açılır. SHA256 için PicoSHA2 kullanılır.
Görseller sunucuya gönderilmez; vektör ve yerel yol/kutu metadata'sı kaydedilir.

## Veri uyumluluğu

Model dosyaları SHA256 ile kontrol edilir; model/tokenizer hash'leri ve preprocessing
sürümü vektör alanının kimliğini oluşturur. Sorgular alan kimliğine göre filtrelenir.
ID'ler alan ve kullanıcı item ID'sinden deterministik UUIDv8 olarak türetilir;
tekrar indeksleme idempotenttir. Mevcut koleksiyon silinmez veya yeniden yaratılmaz;
uyumsuz boyut/mesafe bulunduğunda işlem hata verir.

## Ödünleşimler

OpenCV bicubic orijinal PIL antialias preprocessing'inden farklıdır. Bu fark açıkça
sürümlenir; PyTorch referansı aynı piksel dönüşümüyle üretilir. Tek CLI çağrısında
model yükleme/hash maliyeti yüksektir. Sonraki servis aşamasında bir kez yükleme ve
batch gerekir. İngilizce ağırlıklı CLIP, çok dilli arama başarısı vaadi değildir.
TLS/auth, uzak servisler ve üretim güvenlik sertleştirmesi kapsam dışıdır.

Alternatifler: DINO yalnızca görsel embedding için uygundur; metin encoder'ı olmadan
ortak arama uzayı sağlamaz. OpenCV DNN yerine ONNX Runtime ayrı token/int64 girişleri
ve transformer grafikleri için seçildi. Tam HuggingFace tokenizer runtime bağımlılığı
yerine sabit CLIP sözleşmesi ve referans token testleri kullanıldı.
