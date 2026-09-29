# ADR-0001: Model ve altyapı bağımlılıklarını portlarla ayırma

- **Durum:** Kabul edildi
- **Tarih:** 2026-09-22

## Bağlam

Araştırma aşamasında PyTorch modelleri sık değişirken üretimde TensorRT/Triton ve
harici Qdrant/Redis servisleri kullanılır. Bu detayların pipeline'a sızması testleri
yavaşlatır ve deployment seçeneklerini birbirine bağlar.

## Karar

Pipeline yalnızca C++ `IDetector`, `ITracker`, `IEmbedder`, `ITextExtractor` ve
`IVectorStore` sözleşmelerini bilir. Yerel referans adaptörleri deterministik ve hafiftir.
Gerçek model ve altyapı entegrasyonları ayrı adaptörler olarak eklenir.

## Sonuçlar

- Unit testler GPU ve ağ bağlantısı olmadan çalışır.
- PyTorch modeli TensorRT engine ile pipeline değiştirilmeden yer değiştirebilir.
- Port sözleşmelerinin sürüm yönetimi gerekir.
- Adaptör seviyesinde ayrıca entegrasyon ve performans testleri tutulmalıdır.

