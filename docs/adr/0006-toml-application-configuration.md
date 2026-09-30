# ADR 0006: Sürümlü TOML uygulama konfigürasyonu

- Tarih: 2026-09-30
- Durum: Kabul edildi

## Karar

`toml++` ile `version = 1` şeması uygulanır. `pipeline.mode` image, video veya
search seçer. Her mod yalnızca çalışan C++ adaptörlerini kabul eder: YOLO ONNX,
IoU/iki aşamalı takip, CLIP ONNX ve yerel Qdrant. Model yolları TOML dosyasına
göre çözülür. Eksik/yanlış türde alan, bilinmeyen anahtar, desteklenmeyen backend,
geçersiz eşik ve yanlış embedding boyutu ilk aşamada hata üretir.

Görüntü/video uygulamaları `IDetector` arkasında yapılandırılmış YOLO nesnesi
oluşturur; video takipçi seçimini mevcut `VideoConfig` ve `process_video` yapar.
Arama CLI'ı yapılandırılmış CLIP paketi ve Qdrant istemcisini kullanır.
Eski konumsal CLI arayüzleri korunur. `aegisvision_config validate` ağ çağrısı
ve model inference yapmadan ayarları denetler.

## Gerekçe ve sınırlar

Önceki `configs/pipeline.toml`, `mock` backend ve çalışmayan özellik bayrakları
içeriyordu. Yeni şema yalnızca mevcut uygulamaların sağlayabildiği görevleri
tanımlar; `enable_ocr` gibi henüz uygulanmayan seçenekleri kabul etmez.
Genişleme yeni sürüm ve açık dönüşüm gerektirir. `validate` yalnızca dosya
varlığı ve şema kontrolüdür; bozuk ONNX dosyası model yüklenirken, Qdrant
erişilemiyorsa arama komutunda belirlenir.
