# C++ CLIP + Qdrant arama

Bu modül gerçek CLIP ViT-B/32 görsel ve metin encoder'larını ONNX Runtime CPU üzerinde
çalıştırır. C++ inference, Unicode BPE tokenizer, nesne kırpma ve Qdrant REST istemcisi
birlikte çalışır. Python uygulamanın çalışma zamanında gerekmez; sadece model export
ve geliştirme değerlendirmesi için kullanılır.
Model, koleksiyon ve bağlantı ayarlarını TOML dosyasından vermek için
[konfigürasyon kılavuzuna](configuration.md) bakın.

## Yerel Windows kurulumu

OpenCV geliştirme kurulumu için [YOLO kılavuzu](yolo.md). Ardından proje kökünde:

```powershell
./scripts/setup_search.ps1 -WithQdrant
cmake -S . -B build/search -G "Visual Studio 17 2022" -A x64 -DAEGISVISION_WITH_OPENCV=ON -DAEGISVISION_WITH_SEARCH=ON -DOpenCV_DIR=C:/opencv/opencv/build -DONNXRUNTIME_ROOT="$PWD/artifacts/deps/ort/onnxruntime-win-x64-1.23.2" -DICU_ROOT="$PWD/artifacts/deps/icu"
cmake --build build/search --config Release --parallel 4
ctest --test-dir build/search -C Release --output-on-failure
```

ONNX Runtime 1.23.2, ICU 77.1 ve isteğe bağlı Qdrant 1.12.5 yerel `artifacts/deps`
altına indirilir. Arşiv SHA256 değerleri kontrol edilir; sistem PATH'i değiştirilmez.
CMake ilk arama derlemesinde sabit commit/SHA256 ile nlohmann/json 3.11.3,
cpp-httplib 0.18.3 ve PicoSHA2 1.0.1 kaynaklarını indirir. Varsayılan çekirdek
derlemesi bu bağımlılıklara ihtiyaç duymaz. Bu sürümler test edilen başlangıç
sürümleridir; dış ağa servis açmadan önce güvenlik/güncelleme değerlendirmesi gerekir.

Linux için `libopencv-dev`, `libicu-dev` ve ONNX Runtime Linux x64 C++ dağıtımı gerekir.
`ONNXRUNTIME_ROOT` dağıtımın `include` ve `lib` klasörlerinin üst dizinidir;
`ICU_ROOT` Linux'ta gerekli değildir. CI arama modülünü Linux'ta ayrıca derler.

## Model hazırlama

Mevcut yerel export ortamında PyTorch/OpenCV hazırdır. Yeni ortamda bunları da kurun.
Sabit model: `openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (MIT).

```powershell
./work/yolo-export/Scripts/python.exe -m pip install transformers==4.57.1 onnxruntime==1.23.2 onnx==1.19.1
./work/yolo-export/Scripts/python.exe scripts/export_clip.py artifacts/models/clip-vit-b32 --images artifacts/samples/bus.jpg C:/opencv/opencv/sources/samples/data/fruits.jpg C:/opencv/opencv/sources/samples/data/baboon.jpg
```

Tamamlanmış paket üzerine export yapılmaz. ONNX dosyaları toplam yaklaşık 605 MB'dır;
model indirme cache'i nedeniyle daha fazla disk alanı gerekir. Paket; `vision.onnx`,
`text.onnx`, `tokenizer.json`, `manifest.json` ve test referanslarını içerir.
Başlatmada SHA256, tensor adı/türü/şekli ve embedding boyutu doğrulanır. Hash doğrulama
ve model yükleme her CLI çağrısında tekrar yapılır; kısa demo çağrılarında başlangıç
maliyeti belirgindir. Toplu komutlar modeli süreç içinde yeniden kullanır;
uzun ömürlü [C++ servis](service.md) modelleri başlangıçta bir kez yükler.
Tensor batch optimizasyonu sonraki aşamadır.

## Qdrant başlatma

Docker çalışıyorsa `docker compose up -d qdrant`. Alternatif Windows yerel sunucusu:

```powershell
./artifacts/deps/qdrant/qdrant.exe --config-path configs/qdrant-local.yaml
```

Bu komutu proje kökünde ayrı terminalde çalıştırın. Sunucu `127.0.0.1:6333` dinler,
veriler `artifacts/qdrant/storage` altındadır. Aynı portta iki sunucu başlatmayın.
Yerel ikili ve Docker farklı depolama alanları kullanır. Yerel sürümün web dashboard
dosyaları yoktur; REST araması bundan etkilenmez. Veriler Git'e eklenmez.

## Görsel ekleme, metin ve görsel sorgusu

```powershell
./build/search/Release/aegisvision_search_cli.exe artifacts/models/clip-vit-b32 6333 aegis_clip init
./build/search/Release/aegisvision_search_cli.exe artifacts/models/clip-vit-b32 6333 aegis_clip index bus artifacts/samples/bus.jpg
./build/search/Release/aegisvision_search_cli.exe artifacts/models/clip-vit-b32 6333 aegis_clip text "a photo of a bus" 5
./build/search/Release/aegisvision_search_cli.exe artifacts/models/clip-vit-b32 6333 aegis_clip image artifacts/samples/bus.jpg 5
```

Nesne kırpması için `index ID IMAGE x1 y1 x2 y2` kullanın; kutu görüntünün içinde
olmalıdır. `ClipEmbedder`, `IEmbedder` arayüzünü uyguladığı için mevcut detection →
embedding → indexing pipeline'ına da bağlanabilir. Klasör taraması ve videoda
YOLO kırpma indeksleme komutları için [indeksleme kılavuzu](indexing.md).

`init` eksik koleksiyonu oluşturur; mevcut koleksiyonu silmez veya sıfırlamaz.
Var olan koleksiyon 512 boyutlu, adsız Cosine vektörleri kullanmalıdır. Aynı model
alanındaki aynı ID'yi yeniden indekslemek o kaydın vektör/metadata'sını günceller.
Sonuçlar JSON olarak ID, cosine skoru, yerel dosya yolu ve kırpma kutusunu içerir.
Görsel dosyaları Qdrant'a yüklenmez; yalnızca embedding ve metadata yazılır.

## Sözleşmeler ve sınırlar

- Görsel: BGR → kısa kenar 224 / OpenCV bicubic → merkez 224×224 crop → RGB →
  CLIP mean/std → NCHW float32. Nesne kutusu önce kırpılır. OpenCV bicubic, orijinal
  PIL antialias resize ile bit düzeyinde aynı değildir; kendi sürümlenmiş preprocessing
  sözleşmemiz ve ona göre PyTorch referansımız vardır. Farklı preprocessing ile
  üretilmiş vektörler aynı alana karıştırılmamalıdır.
- Metin: NFC, Unicode lowercase, CLIP regex ve byte-level BPE. BOS/EOS ve attention
  mask uygulanır; padding EOS'tur. Boş/bozuk UTF-8, özel token içeren ve 75 içerik
  token'ını aşan sorgular reddedilir; sessiz kesme yapılmaz.
- Türkçe karakterler doğru token'lanır, ancak CLIP'in Türkçe anlamsal başarısı garanti
  edilmez. Başlangıç demosunda İngilizce sorgular tercih edilir. Çok dilli model ayrı iş.
- İki encoder da 512 boyutlu L2-normalize vektör döndürür. Cosine skoru olasılık değildir.
- Vektör alanı; model/tokenizer dosya hash'leri ve preprocessing sürümünden türetilir.
  Her sorgu `space_id` filtresi taşır. UUIDv8 ID, SHA256(namespace, alan, item ID)
  üzerinden deterministiktir. Farklı model alanındaki aynı ID diğer kaydı ezmez.
- HTTP istemcisi yalnızca localhost içindir; TLS/auth/uzak Qdrant desteği henüz yoktur.
  Timeout, HTTP hata ve boyut kontrolü vardır. Bileşen örneklerini thread'ler arasında
  paylaşmayın. Koleksiyon/payload'a elle müdahale bu sözleşmeleri bozabilir.

## Doğrulama

```powershell
cmake -S . -B build/search -DAEGISVISION_CLIP_BUNDLE="$PWD/artifacts/models/clip-vit-b32"
ctest --test-dir build/search -C Release --output-on-failure
./build/search/Release/aegisvision_search_tests.exe --tokenizer artifacts/models/clip-vit-b32/tokenizer.json tests/fixtures/clip_tokens.json
./build/search/Release/aegisvision_search_tests.exe --qdrant-write 6333 aegis_test
# Qdrant'ı aynı storage ile yeniden başlatın; yalnızca okuma ile kontrol edin:
./build/search/Release/aegisvision_search_tests.exe --qdrant-read 6333 aegis_test
```

Gerçek model referans testi: 12 metin, 3 görüntü; PyTorch'a göre her embedding için
cosine > 0.9999 koşulu. Tokenizer testi Türkçe, emoji, sayı, contraction, boşluk ve
birleşik Unicode örneklerini kapsar. HTTP testleri hata, dimension ve metadata
sözleşmesini; gerçek Qdrant testi idempotent upsert, alan izolasyonu ve diskten
geri okumayı doğrular. CI büyük model ağırlıklarını indirmez; golden tokenizer ve
gerçek Qdrant restart testi çalışır. Tam encoder referans testi yereldir.

Üç görselli, önceden etiketlenmiş smoke değerlendirmesi:

```powershell
./work/yolo-export/Scripts/python.exe scripts/evaluate_search.py build/search/Release/aegisvision_search_cli.exe artifacts/models/clip-vit-b32 yeni_demo outputs/yeni-arama/evaluation.json
```

Bu script yeni/ayrı koleksiyon ister, koleksiyon silmez. Kayıtları C++ CLI ile ekler;
metinden arama Recall@1 ve aynı görseli sorgulayan self-retrieval raporunu kaydeder.
Üç kolay örnekteki sonuç genel kalite benchmark'ı değildir. Daha büyük ve benzer
sınıflı başlangıç değerlendirmesi için aşağıdaki COCO kırpma ölçümüne bakın.

2026-09-30 yerel sonucu: üç metin sorgusunda 3/3 Recall@1; üç aynı-görsel sorgusunda
3/3 self-retrieval. Metin cosine skorları otobüs 0.2744, meyve 0.3053, babun 0.3004.
12 CTest testi geçti. Yerel Qdrant süreci yeniden başlatıldıktan sonra hem sentetik
entegrasyon kayıtları hem gerçek CLIP otobüs sorgusu tekrar okundu. Windows'ta süreç
tam kapanmadan tekrar başlatmak WAL dosya kilidi hatasına yol açabilir; eski sürecin
çıkışını bekleyin, veri klasörünü silmeyin. Değerlendirme JSON'u yerelde
`outputs/search-demo/evaluation.json` altındadır.

Sekiz sınıf × sekiz COCO validation kırpması için toplu indeksleme ve gerçek
Recall@K değerlendirmesi: [COCO arama benchmark'ı](search-benchmark.md).

Kaynaklar: [CLIP](https://github.com/openai/CLIP),
[ONNX Runtime C++](https://onnxruntime.ai/docs/get-started/with-cpp.html),
[Qdrant points](https://qdrant.tech/documentation/concepts/points/).
