# Linux GPU servis paketi

2026-10-18. `Dockerfile.gpu` imajı Docker Desktop 4.95.0 / WSL2 üzerinde
Linux amd64 için başarıyla derlendi. Konteynerden RTX 3060 Laptop GPU'ya erişim
ve gerçek FP16 video indeksleme, arama, kontrollü stop/restart ve aynı-ID iş
kurtarma doğrulandı. Bu, native Linux sunucuda veya üretimde onay değildir.

## Kapsam ve sınırlar

- Linux x86_64, NVIDIA GPU/uyumlu sürücü, NVIDIA Container Toolkit ve Compose V2
  gerekir. Bu tarif native Linux içindir; mevcut Windows demo komutu değişmez.
- [NVIDIA TensorRT 25.09](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/)
  tabanı TensorRT 10.13.3.9 / CUDA 13.0.1 sağlar; C++ derlemesi SDK sürümünü
  ayrıca kontrol eder. ONNX Runtime 1.23.2 arşivi SHA256 ile doğrulanır.
  NVIDIA tabanı doğrulanan `sha256:2bc1b502a8ec29fbb1642aff6bcf2392bab11e59adf923c726b2b33e8e5d6d6b`
  digest'ine sabitlendi. apt paketleri/Qdrant etiketi ayrıca sabitlenmiş değildir;
  üretim tesliminden önce güvenlik taraması ve lisans incelemesi gerekir.
- GPU'da YOLO FP16 detection, CPU'da CLIP, Qdrant arama ve SQLite iş geçmişi.
  Segmentation/OCR/VLM veya RTSP bu ilk konteyner profiline dahil değildir.
- Host network bilinçli seçildi: uygulamanın yalnız `127.0.0.1` dinlemesi ve
  Host/Origin kontrolleri değiştirilmez. Public API/auth eklenmez. Host network
  ağ izolasyonunu azaltır; yalnız güvenilen tek-kullanıcılı Linux makinesi içindir.
  Panel `127.0.0.1:8093`, ayrı Qdrant `127.0.0.1:16333/16334` kullanır.
- Model ve servis engine mount'ları salt okunur, servis root olmayan kullanıcıyla
  ve salt okunur kök dosya sistemiyle çalışır. State/media yazılabilir; yollar
  eksikse Docker bunları sessizce oluşturmaz. Docker socket veya tüm repo bağlanmaz.
- Windows'ta hazırlanmış engine'i kopyalama: aynı Linux konteyneri ve hedef GPU'da
  yeni FP16 bundle oluştur. Builder var olan bundle'ı üzerine yazmaz.
  C++ çalışma zamanı sürüm/GPU/model hash kontrollerini yapar; CPU fallback yoktur.
- Python yalnız dosya ön kontrolü/sağlık için kullanılır, ardından C++ sunucuyu
  `exec` eder. Sağlık kontrolü `ready` + `tensorrt` + `fp16-mixed` + `cuda:0`
  ister. Bu, etiketli veriyle doğruluk veya GPU inference testi yerine geçmez.
- Sunucuya [SIGINT/SIGTERM köprüsü](service.md#işletim-sistemi-sinyaliyle-kapanış)
  eklendi: yeni iş kabulünü durdurur, aktif işi işbirlikçi olarak keser ve
  kalıcı işi aynı ID ile yeniden oynatılmak üzere saklar. `init` sinyali iletir.
  Bütün işleri bitirerek drain etme veya 30 saniyede kapanma garantisi değildir;
  yerel izole GPU konteynerindeki ölçüm aşağıdadır; tüm yükler için garanti değildir.

## Hazır modelleri olan Linux makinesinde

Repo kökünden; aşağıdakiler Windows PowerShell komutları değildir. Kaynak dosyalar:
`artifacts/models/yolov8n.onnx`, tam `artifacts/models/clip-vit-b32` bundle'ı,
`artifacts/media` içinde test videosu. Modeller imaja veya Git'e dahil edilmez.

```bash
export AEGIS_UID="$(id -u)"
export AEGIS_GID="$(id -g)"
mkdir -p artifacts/container/engines artifacts/container/state artifacts/media
docker compose -f compose.gpu.yaml config --quiet
docker compose -f compose.gpu.yaml build service
docker compose -f compose.gpu.yaml run --rm --no-deps engine-build
docker compose -f compose.gpu.yaml run --rm --no-deps service --check
docker compose -f compose.gpu.yaml up -d qdrant-gpu service
docker compose -f compose.gpu.yaml ps
curl --fail http://127.0.0.1:8093/api/health
```

Engine üretimi birkaç dakika sürebilir. Çalıştıran kullanıcı root olmamalı;
state/media/engine dizinleri bu kullanıcıya yazılabilir olmalıdır. Paket hatada
izinleri genişletmez, root'a yükselmez, modeli indirmez, CPU'ya geçmez.
Qdrant için başlangıçta en fazla 15 bağlantı denemesi yapılır.

Panel: `http://127.0.0.1:8093`. Test videosunu indeksle, tamamlanan işte GPU
motorunu ve nesne kayıtlarını kontrol et; metin aramasıyla sonuç/video aç.
İlgili API istekleri ve sonuç şemaları [API kılavuzunda](api.md).

Kapatma (veri silmez):

```bash
docker compose -f compose.gpu.yaml down
```

`down -v` kullanma: ayrı Qdrant volume'undaki kayıtları siler. CPU demosunun
6333/8090 servisleri ve verileri bu profilin dışında kalır. 8093/16333/16334
başka bir uygulamadaysa onu otomatik durdurma; önce port çakışmasını çöz.

## Windows Docker Desktop / WSL2 için izole kabul testi

`compose.gpu.isolated.yaml` yalnız test içindir. Host networking açtırmaz,
host portu yayınlamaz. Servis ve Qdrant özel bir ağ alanında aynı loopback'i
paylaşır; HTTP kontrolü `docker compose exec` üzerinden yapılır. Windows'taki
8090 paneli/Redis ve yerel Qdrant ile veri paylaşmaz. Bu profilde tarayıcıdan
`localhost:8093` açılmaz; normal Linux host-network tarifi yukarıdadır.

Mevcut model dosyaları ve Docker GPU erişimi gereklidir. PowerShell'de repo kökünden:

```powershell
New-Item -ItemType Directory -Force artifacts/container/engines, artifacts/container/isolated-media, artifacts/container/isolated-state
# Yalnız hedefte yoksa örnek videoyu kopyala; mevcut dosyayı değiştirme.
if (-not (Test-Path artifacts/container/isolated-media/pedestrians.mp4)) {
    Copy-Item artifacts/media/pedestrians.mp4 artifacts/container/isolated-media/pedestrians.mp4
}
docker compose -f compose.gpu.yaml build service
# Yalnız ilk kurulum: mevcut linux-fp16 bundle'ını üzerine yazmaz.
docker compose -f compose.gpu.yaml -f compose.gpu.isolated.yaml run --rm --no-deps engine-build
docker compose -f compose.gpu.yaml -f compose.gpu.isolated.yaml up -d qdrant-gpu service
python scripts/test_gpu_container_runtime.py --output outputs/gpu-container-new-run --restart-test
```

Her koşuda yeni çıktı klasörü seç. Test gerçek video indeksleme ve arama işleri
oluşturur, GPU engine kimliğini ve JPEG önizlemeyi doğrular. `--restart-test`
yalnız bu test profilindeki servisi, kendi işi çalışırken kontrollü durdurup
yeniden başlatır; aynı-ID iş tekrarını ve eski arama sonucunu kontrol eder.
Başka bekleyen/çalışan iş varsa durdurmayı reddeder. Hata halinde rapor ve
veriler korunur; raporun `status` alanı `passed` değilse test başarılı sayılmaz.
Test sonrası servisler açık kalır. Yalnız bu profili durdurmak için:

```powershell
docker compose -f compose.gpu.yaml -f compose.gpu.isolated.yaml stop
```

Bu komut veri/volume silmez. Native Linux'ta bind dizinlerinin UID/GID izinleri
ayrıca hazırlanmalıdır; Windows bind mount izin davranışı Linux izin kanıtı değildir.

## Gerçek GPU kabul sonucu — 2026-10-18

`python scripts/test_gpu_container_runtime.py --output outputs/gpu-container-20261018 --restart-test`
başarıyla tamamlandı (`status=passed`). Rapor ve `preview.jpg` yalnız bilgisayardadır.

- Docker Desktop 4.95.0 / WSL2, Linux amd64, RTX 3060 Laptop 6 GB.
  Bu test ayrı bir native Linux işletim sistemi kurulumu değildir.
- İmaj kimliği: `sha256:177ecd0e54f748c66c35b85d131e048024fe0b8573b6392f2566a5a510722719`.
  C++ kaynakları `878d58a`; yeni test profili/sürücü araçları `80d2a6c`.
- Linux'ta üretilen engine SHA256:
  `c8b44c9b84e0bb9e9d73be20815b57c006de78b2c2e5c64e86ed95289c59308c`.
  TensorRT 10.13.3, FP16-mixed, TF32 kapalı; üretim 313,14 saniye.
- Sağlık `ready`, `tensorrt`, `cuda:0`, `fp16-mixed`; kalıcı SQLite açık,
  depolama hatası boş. YOLO GPU'da; CLIP CPU/ONNX Runtime'dadır.
- İlk indeksleme: 30 kare okundu, stride=10 ile 3 kare örneklendi, 16 nesne
  kaydı. Metin araması 4 sonuç ve geçerli JPEG önizleme döndürdü; tüm eşleşmeler
  aynı GPU engine imzasını taşıdı. Bu sayılar benzersiz kişi veya doğruluk metriği değildir.
- `1792337853088-3` işi çalışırken 11 kayıt ilerlemesi gözlendi. Servis stop
  süresi 1,417 saniye, exit=0, OOM=false; zorla SIGKILL başarısı gibi sunulmadı.
  Yeniden açılışta aynı ID attempts=2 ile başarılı oldu: 90 kare, stride=3,
  204 kare-içi kayıt. Önceki başarılı arama sonucu değişmeden geri geldi.
- Çalışan serviste UID=1000, salt okunur kök/model/engine mount'ları doğrulandı.
  Servis test sonunda `healthy`; host portu yayınlanmadı, Windows demosuna dokunulmadı.

Engine manifest'indeki `deployment_approved=false` korunur. Bu işlevsel kabul
kalite eşiği, hız benchmark'ı, uzun süreli yük, native Linux bind izinleri,
güvenlik/lisans taraması veya bağımsız veriyle doğruluk testi yerine geçmez.

## Regresyon kontrolleri / kalan kabul koşulları

- Docker Compose 2.40.3 ile yapılandırma ayrıştırıldı; daemon gerekmez.
- 9 test: tam/boş dosyalar, eksik/yazılamayan mount, bozuk/INT8 manifest,
  doğru GPU health alanları, yan etkisiz `--check`, zorunlu GPU argümanları,
  Qdrant hatasında sınırlı tekrar ve CPU'ya geçmeme. CI'a eklendi.
- 3 Compose sözleşme testi: native/izole ağ ayrımı, yayınlanmayan portlar,
  ayrı state/media, root olmayan kullanıcı, salt okunur modeller/kök ve GPU isteği.
- 5 runtime-test sözleşme kontrolü: CPU/INT8/eksik engine kimliği reddi,
  kalıcılık/depolama sağlığı, ilgisiz işe dokunmama ve başarılı işin yanlışlıkla
  "aktifken kesildi" sayılmaması. Bu 17 kontrol yerelde geçti; GPU testinden ayrıdır.
- Bu model bağımsız kontroller gerçek GPU kabul testinden ayrıdır.
  Kalanlar: native Linux host-network tarifi/izinleri, model kalitesi, uzun yük,
  güvenlik taraması, lisanslar ve üretim imajını küçültme/sürümleme.

Hazır Windows demosunu kullanmak için [demo kılavuzu](demo.md); güncel genel
liste için [teslim durumu](delivery-status.md).
