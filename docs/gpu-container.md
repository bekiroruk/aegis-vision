# Linux GPU servis paketi — doğrulama bekleyen aday

2026-10-15. `Dockerfile.gpu`, `compose.gpu.yaml`, konteyner konfigürasyonları ve
başlangıç kontrolleri eklendi. **İmaj henüz burada derlenmedi veya GPU ile
çalıştırılmadı.** Yerel Docker Linux motorunun bağlantısı yoktu. Bu nedenle
GPU Docker deployment maddesi tamamlanmış değildir.

## Kapsam ve sınırlar

- Linux x86_64, NVIDIA GPU/uyumlu sürücü, NVIDIA Container Toolkit ve Compose V2
  gerekir. Bu tarif native Linux içindir; mevcut Windows demo komutu değişmez.
- [NVIDIA TensorRT 25.09](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/)
  tabanı TensorRT 10.13.3.9 / CUDA 13.0.1 sağlar; C++ derlemesi SDK sürümünü
  ayrıca kontrol eder. ONNX Runtime 1.23.2 arşivi SHA256 ile doğrulanır.
  İmaj etiketi digest sabitlemesi değildir; üretim tesliminden önce kullanılan
  digest, güvenlik taraması ve lisans incelemesi kaydedilmelidir.
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
- Mevcut sunucuda SIGTERM ile aktif işleri drain etme garantisi yoktur. `init`
  sinyali iletir; aktif işte kapanış/kurtarma gerçek konteyner testinde ölçülmelidir.

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

## Gerçekte yapılan kontroller / kalan kabul koşulları

- Docker Compose 2.40.3 ile yapılandırma ayrıştırıldı; daemon gerekmez.
- 9 test: tam/boş dosyalar, eksik/yazılamayan mount, bozuk/INT8 manifest,
  doğru GPU health alanları, yan etkisiz `--check`, zorunlu GPU argümanları,
  Qdrant hatasında sınırlı tekrar ve CPU'ya geçmeme. CI'a eklendi.
- Bu kontroller imajın derlendiğini veya Linux GPU inference'ının çalıştığını
  göstermez. İmaj derlemesi, gerçek GPU engine üretimi/inference, model kalitesi,
  arama, aktif işte stop/restart/SQLite kurtarma ve izin kontrolleri hâlâ bekliyor.

Hazır Windows demosunu kullanmak için [demo kılavuzu](demo.md); güncel genel
liste için [teslim durumu](delivery-status.md).
