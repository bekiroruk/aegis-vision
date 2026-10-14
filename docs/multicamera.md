# Çoklu kamera: çevrimdışı takip-parçası ilişkilendirme

İlk kapsam (2026-10-14): native C++20 `associate_cameras` ve JSON alan
`aegisvision_multicamera` CLI'ı. Aynı kişiye ait olabilecek **kapanmış takip
parçalarını** açık kamera geçişleri ve kişi özellikleriyle ilişkilendirir.
Canlı iki-kamera capture, saat senkronizasyonu, otomatik tracklet çıkarımı veya
gerçek çoklu-kamera doğruluk değerlendirmesi bu adımda tamamlanmadı.
Mevcut tek-video/HTTP/GPU demo davranışı değiştirilmedi.

## Giriş sözleşmesi

- En fazla 512 kapanmış parça, 2–16 kamera; en fazla 240 yönlü geçiş.
- Yerel anahtar `(camera, epoch, local_id)`; epoch ve ID pozitif olmalı.
  Kamera reconnect sonrası ID sayacı sıfırlanıyorsa epoch artırılmalı.
  Aynı anahtarın iki kez verilmesi reddedilir. Farklı kamerada aynı yerel ID
  aynı kişi demek değildir.
- `start_ms` / `end_ms` negatif olmayan, aynı senkronize zaman eksenindeki
  milisaniyelerdir. Kamera PTS'si veya iki bağımsız decode-arrival saati doğrudan
  kullanılamaz. CLI `time_basis=synchronized_ms` beyanını ister; bu beyan saat
  doğruluğunu ölçmez veya garanti etmez. Saat offset/drift çözümü upstream iştir.
- Her parça tek, sonlu, sıfır olmayan **512 boyutlu kişi prototipi** taşır.
  Bileşen L2-normalize eder; giriş belleğini değiştirmez. Tüm parçaların
  `embedding_space` değeri config ile aynı olmalıdır. Gerçek OSNet adaptöründe
  `ReIdEmbedder::space_id()` kullanılır; CLIP vektörü kişi Re-ID yerine geçmez.
  Boyut/etiket kontrolü tek başına vektörün gerçekten o modelden geldiğini kanıtlamaz.
- Kamera adları 1–64 ASCII harf/rakam/alt çizgi/tire. Geçiş `from → to` yönlüdür;
  ters yön otomatik açılmaz. Min gap en az 1 ms, max gap en fazla 24 saattir.
  Örtüşen görüş alanları ve aynı kameradaki fragman birleştirme kapsam dışıdır.

## Karar

1. Kamera geçişi tanımlı olmalı; kaynak parça bitişi hedef başlangıcından önce
   olmalı ve aradaki süre kapalı `[min_gap_ms, max_gap_ms]` aralığında olmalı.
2. Uygun her çiftin normalize cosine benzerliği hesaplanır.
3. Hedefin en iyi önceki parçası ve kaynağın en iyi sonraki parçası birbirini
   seçmeli. **Her iki yönde** en iyi/ikinci en iyi skor farkı margin'i geçmeli
   (eşit sınır kabul edilir); minimum benzerlik de sağlanmalı.
4. Kabul edilmiş bağlar zaman yönünde zincir oluşturur. Her parçanın en fazla
   bir öncülü ve ardılı vardır; zincirde eşzamanlı görünüm veya döngü oluşmaz.
   Kabul edilmeyen hedef yeni bir kimlik hipotezi alır; ikinci tercihe zorlanmaz.

Varsayılan similarity `.85`, margin `.05` **kalibre edilmemiş başlangıç
değerleridir**. Skor olasılık değildir. Bu muhafazakâr mutual-best yaklaşımı
maksimum ağırlıklı atama/optimal global çözüm iddiası taşımaz; gerçek aynı kişiyi
ayırabilir veya benzer kişileri yanlış bağlayabilir. Prototip güncellenmez;
association sonucu model/özellik eğitimine geri beslenmez.

Global ID'ler yalnız aynı giriş kümesi içinde deterministiktir; giriş sırası
değişse bile aynı anahtar aynı ID'yi alır. Giriş kümesi değişirse numaralar
değişebilir. Kalıcı kimlik, biyometrik doğrulama veya benzersiz kişi sayısı değildir.

## CLI

JSON bağımlılığı mevcut search build'inde olduğundan executable orada derlenir;
komut çalışırken CLIP/Qdrant/OpenCV modeli açmaz, GPU veya Python istemez.
Çekirdek algoritma ve CTest ise bağımlılıksız temel build'de de bulunur.

```powershell
cmake --build build/tensorrt --config Release --target aegisvision_multicamera --parallel 4
build/tensorrt/Release/aegisvision_multicamera.exe input.json outputs/new-association
```

Giriş üst alanları `version:1`, `time_basis:"synchronized_ms"`, `config`,
`tracklets`. Config alanlarının tümü zorunlu:

```json
{
  "cameras": ["entrance", "corridor"],
  "embedding_space": "MODEL_AND_PREPROCESSING_SPACE_ID",
  "min_similarity": 0.85,
  "ambiguity_margin": 0.05,
  "transitions": [
    {"from":"entrance","to":"corridor","min_gap_ms":100,"max_gap_ms":2000}
  ]
}
```

Her tracklet `camera`, `epoch`, `local_id`, `start_ms`, `end_ms`,
`embedding_space`, `embedding` alanlarını taşır. `embedding` tam 512 sayı
olmalıdır. Yukarıdaki örnek yalnız config parçasıdır; gerçek model kimliği ve
gerçek tracklet özellikleri yerine placeholder gönderilmemelidir.
Giriş en fazla 16 MiB / 16 JSON derinliği. Bilinmeyen alan, kesirli kimlik/zaman,
yanlış boyut/uzay ve geçersiz config reddedilir. Çıktı dizini yeni olmalıdır;
mevcut rapor üzerine yazılmaz. Raporda özellik vektörleri kopyalanmaz.

Çıktı `report.json`: her anahtar için `global_id`, `predecessor`,
`best_similarity`, `reason`; toplam `accepted_handoffs` ve `identity_hypotheses`.
Nedenler: `matched`, `no_candidate`, `below_threshold`, `ambiguous_incoming`,
`ambiguous_outgoing`, `not_mutual_best`. `verified_person_identities=false`.

## Doğrulama ve açık işler

```powershell
ctest --test-dir build/dev -C Release --output-on-failure
python scripts/test_multicamera_cli.py build/tensorrt/Release/aegisvision_multicamera.exe
# İsteğe bağlı, açıkça sentetik ve yeni dizinde saklanan çalıştırılabilir örnek:
python scripts/test_multicamera_cli.py build/tensorrt/Release/aegisvision_multicamera.exe --example-directory outputs/multicamera-synthetic-v1
```

Yerel temel CTest 6/6 geçti. Yeni test; üç-kamera zinciri, giriş permütasyonu,
çift yönlü belirsizlik, çakışan zaman, yön/travel gate sınırı, kamera/epoch ID
ayrımı, boyut/NaN/uzay/kapasite ve normalize etme kontrollerini içerir.
CLI testinde sentetik dört parça → iki geçiş/iki hipotez; belirsiz özelliklerle
sıfır geçiş. Geçersiz input çıktı oluşturmaz; mevcut rapor korunur.
Yerel örnek `outputs/multicamera-synthetic-v1/synthetic-result/report.json`.
Bu sayılar **kamera/model doğruluk metriği değildir**. CI'a core ve CLI testleri
eklendi; bu belge uzak CI koşusunun başarılı olduğunu iddia etmez.

Sıradaki entegrasyon: gerçek videodan kapanmış tracklet/prototip çıkarımı,
iki kamera için doğrulanmış ortak zaman ekseni, etiketli kamera-geçiş verisiyle
eşiklerin geliştirme kümesinde ayarı ve ayrık test ölçümü. Sonra canlı capture,
durable kimlik kataloğu ve UI entegrasyonu. Mevcut OSNet'in tek-kamera testleri
çoklu-kamera kalitesinin kanıtı olarak kullanılmaz.
