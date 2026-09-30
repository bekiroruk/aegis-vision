# ADR 0004: İsteğe bağlı iki aşamalı takip

- Tarih: 2026-09-30
- Durum: Kabul edildi

## Bağlam

Greedy IoU başlangıcı düşük güvenli tespitleri atıyor ve hareketli nesnelerin kısa
kaybolmalarında kimlik parçalayabiliyor. Çekirdeğin OpenCV/CUDA bağımlılığı olmadan
test edilebilir kalması ve eski baseline'ın korunması isteniyor.

## Karar

`ITracker` arkasında saf C++ `TwoStageTracker` eklendi. Önce yüksek güvenli tespitler,
sonra yalnızca aktif/eşleşmemiş track'ler için düşük güvenli tespitler eşleştirilir.
Yeni kimlik için ayrı daha yüksek eşik kullanılır. Sınıf duyarlı IoU kapısından sonra
Hungarian toplam ağırlığı maksimize eder; eşleşmeme serbesttir. Basit merkez hızı
tahmini kısa kaybolmaları destekler. CLI `--tracker two-stage` ile seçer;
varsayılan `iou` değişmez.

## Sonuçlar ve sınırlar

Düşük güvenli gözlemler mevcut kimlikleri sürdürebilir, ancak görünüş bilgisi olmadan
örtüşen nesnelerde doğru kimlik garantisi yoktur. Doğum eşiği recall'ı azaltabilir.
Eşleştirme yoğun sahnelerde pahalıdır ve ayrıca ölçülmelidir. Kalman kovaryansı,
tentative track yönetimi ve Re-ID yoktur; bu **tam ByteTrack değildir**.

Referans fikir: [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864).
Yöntem uygulaması çekirdekte bağımsızdır. Doğrulama; exhaustive assignment referansı,
sentetik yaşam döngüsü testleri ve gerçek video smoke karşılaştırmasını içerir.
Etiketli IDF1/HOTA değerlendirmesi sonraki kilometre taşıdır.
