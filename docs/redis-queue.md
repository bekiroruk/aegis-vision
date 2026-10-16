# Redis iş sahipliği ve kurtarma çekirdeği

2026-10-16. İsteğe bağlı C++ `RedisLeaseQueue` ve native komut aracı.
**Gerçek YOLO görüntü analizi yapan bağımsız C++ worker eklendi.**
Panelin JobQueue backend'i henüz değildir. Mevcut HTTP/SQLite servisi değişmez;
Redis worker ayrı süreç olarak iş alır ve sonuç JSON'unu token kontrolüyle yayımlar.

## Sözleşme

- `submit`, `claim`, `renew`, `complete`, `fail`, `cancel`, `get` işlemleri.
  Aynı ID/payload tekrar gönderilince yeni iş oluşturulmaz; farklı payload
  reddedilir. Context model/veri sözleşmesini ayırmak için çağıran tarafından
  verilmelidir; aynı kuyrukta farklı context reddedilir.
- Claim atomik olarak yalnız bir sahip seçer. Renew/complete/fail, owner,
  artan token ve sürenin geçerli olmasını ister. Eski worker'ın geç gelen
  sonucu yeni sahibin sonucunun üzerine yazamaz.
- Saat Redis TIME; çekirdek lease 100–300.000 ms. Yeni worker 3.000–300.000 ms
  kullanır, ayrı thread'de lease/3 aralıkla otomatik renew yapar. Claim süresi dolmuş işleri geri alır;
  toplam üç claim sonrasında başarısız sayar. Ayrı zamanlanmış reaper yoktur.
- İptal aktif token'ın sonuç kabulünü engeller, model çağrısını zorla kesmez.
  Terminal işler tekrar alınmaz. **128 toplam iş** sınırı terminal geçmişi de
  kapsar; TTL/silme/eviction henüz yoktur. Payload 8 KiB, sonuç 64 KiB.
- Bağlantı/komut timeout'u bir saniye. Kaybolan cevaptan sonra yazı otomatik
  tekrarlanmaz: işlem olmuş olabilir. Submit aynı ID ile, complete ise get ile
  uzlaştırılabilir. Cevabı kaybolan claim lease süresiyle kurtarılır.

Dört açık anahtar kullanılır: `aegis:{QUEUE}:jobs/pending/leases/meta`.
Atomiklik [Redis EVAL](https://redis.io/docs/latest/commands/eval/) ve
[Lua sözleşmesine](https://redis.io/docs/latest/develop/programmability/eval-intro/)
dayanır. Script atomikliği runtime hatalarında transaction rollback garantisi
değildir. Yeterli bellek/noeviction ve yalnız bu protokolle erişilen namespace
gerekir. Bozuk anahtar türü veya eksik context açık hata verir.

## Derleme / kullanım

Core build varsayılanı değişmez. Hiredis 1.3.0 ve JSON arşivleri hash kontrollü:

```powershell
cmake -S . -B build/redis -DAEGISVISION_WITH_REDIS=ON
cmake --build build/redis --config Release --target aegisvision_redis_queue aegisvision_redis_tests
ctest --test-dir build/redis -C Release -R aegisvision_redis_validation --output-on-failure
```

Ayrı ve güvenilen Redis loopback 16379'da çalışıyorsa:

```powershell
'{"op":"submit","id":"example-1","payload":{"type":"fixture"}}' | .\build\redis\Release\aegisvision_redis_queue.exe 16379 example-queue context-v1
'{"op":"claim","owner":"worker-a","lease_ms":30000}' | .\build\redis\Release\aegisvision_redis_queue.exe 16379 example-queue context-v1
'{"op":"get","id":"example-1"}' | .\build\redis\Release\aegisvision_redis_queue.exe 16379 example-queue context-v1
```

Claim'in döndürdüğü token complete/renew için kullanılmalıdır. `null`, boş
kuyruk/bilinmeyen iş/geçersiz sahiplik anlamındadır. Bunlar görüntü analizi değil,
koordinasyon komutlarıdır. Model/medya dosyaları Redis'e yüklenmez.

## Test ve sınırlar

Windows native derleme ve bağlantısız kontrat testi geçti. Ayrı `redis-linux`
CI grubu gerçek Redis 7.4 üzerinde sekiz entegrasyon testi çalıştırır: sekiz
native süreçle tek claim, idempotency/context, lease devri/eski token reddi,
renew, iptal, retry, kapasite ve 64-bit sayı/Unicode korunması. Payload/sonuç
Lua'nın double sayı dönüşümüne sokulmadan opaque JSON olarak korunur.
Ayrıca AOF `appendfsync always` açık gerçek
Redis restart'ında aktif/bekleyen işlerin, iptalin ve tamamlanmış sonucun
korunması sınanır. Yerel makinede Redis yoktur; gerçek sonuç CI'dan doğrulanır.
Fixture payload kullanılır; CV/GPU veya Redis failover testi değildir.

**Doğrulanan koşu:** `66e72d1` için [Linux Redis grubu](https://github.com/bekiroruk/aegis-vision/actions/runs/38865988386/job/116858823943)
başarılı: native derleme/kontrat, sekiz gerçek Redis entegrasyon testi ve
AOF restart/kurtarma geçti. Bu sonuç diğer CI gruplarının veya gelecekteki
worker/panel entegrasyonunun geçtiği anlamına gelmez. Test konteyneri koşunun
sonunda kaldırıldı; bilgisayardaki mevcut servis/veriler değiştirilmedi.

Sonraki parça HTTP kuyruğuna opsiyonel bağlantıdır. Dosya sonuçlarının token
kontrollü yayını ve Qdrant yan etkilerinin idempotent yönetimi ayrıca gerekir.
Yeni worker bunlara yazmaz; yalnız JSON detection sonucu üretir.
Lease eski worker'ın hesaplamasını engellemez;
yalnız bu kuyruğa sonucunu kabul ettirmez. **Exactly-once / çift inference
engeli iddiası yoktur.**

Tek standalone Redis, yalnız güvenilen local istemciler: TLS/ACL, uzaktaki
sunucu, Cluster/Sentinel/failover desteklenmez. Redis veri kaybı/yedekten dönüş
token sayacını geri alabilir; bu durumda güvence yoktur. Redis sunucu lisansı
ayrıca incelenmelidir; hiredis bildirimi THIRD_PARTY_NOTICES içindedir.

## Gerçek görüntü worker'ı

`AEGISVISION_WITH_REDIS=ON` ve `AEGISVISION_WITH_OPENCV=ON` birlikte açılınca
`aegisvision_redis_worker` derlenir. Model: mevcut COCO YOLOv8 detect ONNX;
**OpenCV >= 4.12.0 gerektirir.** İlk Linux birleşik koşusu dağıtımın OpenCV
4.6 ONNX `Add` importer hatasıyla durdu; model veya kuyruk başarısı sayılmadı.
CI bu yüzden hash kontrollü 4.12.0 kaynağını özel bir dizine derleyip cache'ler;
sistem OpenCV'si veya mevcut diğer test grupları değiştirilmez. Bu özel build
yalnız görüntü testi içindir, FFmpeg/GStreamer açık video dağıtımı değildir.
CPU/OpenCV, 640 giriş, confidence 0.35, NMS 0.45, en fazla 100 detection.
Model işçi başına ilk geçerli işte bir kez yüklenir. Python yalnız CI'da model
export/test sürücüsüdür; kuyruk ve inference native C++'tır.

```powershell
cmake -S . -B build/redis-cv -DAEGISVISION_WITH_REDIS=ON -DAEGISVISION_WITH_OPENCV=ON -DOpenCV_DIR=C:/opencv/opencv/build
cmake --build build/redis-cv --config Release --target aegisvision_redis_worker aegisvision_redis_queue aegisvision_redis_worker_tests

# Redis gerektirmeden gerçek modeli ve referans resmi dene:
python scripts/test_redis_image.py build/redis-cv/Release/aegisvision_redis_worker.exe artifacts/models/yolov8n.onnx artifacts/samples --output outputs/redis-worker/local-image.json
```

Aşağıdaki örnek **ayrı Redis sunucusunun 16379'da çalışmasını gerektirir**.
Proje kökünde, terminal geçmişinde henüz kullanılmamış bir kuyruk adı seç:

```powershell
$worker = '.\build\redis-cv\Release\aegisvision_redis_worker.exe'
$queueCli = '.\build\redis-cv\Release\aegisvision_redis_queue.exe'
$request = (& $worker request artifacts/models/yolov8n.onnx artifacts/samples bus.jpg) | ConvertFrom-Json
$queueName = 'images-' + [Guid]::NewGuid().ToString('N')
@{op='submit'; id='bus-1'; payload=$request.payload} | ConvertTo-Json -Depth 8 -Compress | & $queueCli 16379 $queueName $request.context
& $worker run 16379 $queueName worker-a artifacts/models/yolov8n.onnx artifacts/samples --once
'{"op":"get","id":"bus-1"}' | & $queueCli 16379 $queueName $request.context
```

`--once` çıkarılırsa worker boş kuyruğu 250 ms aralıkla kontrol eder; Ctrl+C veya
SIGTERM yeni iş almayı durdurur. Aktif işin sonucu yayımlanmaz; başka worker
lease dolduktan sonra alabilir. OpenCV inference zorla kesilemez. İptal en geç
bir sonraki heartbeat'te görülür, heartbeat'ten önce biten işte de Redis'in
atomik complete kontrolü iptal edilmiş/eski token'lı sonucu reddeder.
Heartbeat bağlantı hatasında worker sahipliğini kaybetmiş sayıp sonuç yazmaz
ve çıkar. Complete cevabı kaybolursa otomatik yeniden yazmaz; `get` ile kontrol
edilmelidir. `analysis_failed` terminal hatası stderr'de ayrıntı verir.

Context modelin SHA256'sı, canonical medya kökü, OpenCV sürümü, COCO etiketleri
ve inference ayarlarından hesaplanır. Worker hash'lediği aynı model byte'larını
yükler. Payload `type=detect_image`, köke göre `image` ve görüntü `sha256`
alanlarıyla sınırlıdır. Inference aynı hash kontrolünden geçen byte'ları decode
eder. Model/sürüm/kök değiştiğinde aynı kuyruğa sessizce devam edilmez.

Dosyalar güvenilen yerel kullanıcıya aittir: absolute/parent/drive/ADS yolları
ve kök dışına çıkan symlink reddedilir; encoded görüntü 16 MiB, decode sonrası
16 megapiksel sınırı vardır. Bu bir codec sandbox'ı değildir; kötü niyetli
görüntüde decode öncesi bellek koruması veya eşzamanlı symlink değiştirmeye
karşı işletim sistemi seviyesinde koruma iddiası yoktur. Immutable medya kökü
ve güvenilen model gerekir. Context kökü içerdiğinden ilk sürüm aynı makinede
çoklu süreç içindir; farklı makineler arası ortak storage mapping henüz yoktur.

### Worker doğrulaması

- `aegisvision_redis_worker_tests`: sonuç/hata/boyut sınırı, otomatik heartbeat,
  iptal/sahiplik kaybı, bağlantı hatası, shutdown ve belirsiz yazıyı tekrarlamama.
- Aynı test binary'sine Redis portu verilirse gerçek Redis üzerinde ilk lease
  süresinden uzun iş, rakip claim reddi ve aktif iptal kontrol edilir.
- `scripts/test_redis_image.py`: gerçek bus.jpg + YOLO inference, beklenen en
  az üç kişi/bir otobüs, kutu sınırları, SHA, güvenli yollar. `--queue` verilince
  submit → native worker → Redis `succeeded` sonucu, terminal işin tekrar
  alınmaması, bozuk SHA/yol/tür reddi ve bekleyen iş iptali de sınanır.
- `redis-image-linux` CI grubu public ağırlık/fotoğrafı SHA256 ile doğrular,
  modeli export eder ve gerçek Redis+YOLO birleşik testini çalıştırır. Kanıt
  JSON'u `redis-image-evidence` artifact'ına kaydedilir. Test eklenmesi başarılı
  koşu iddiası değildir; son koşunun sonucu ayrıca kontrol edilmelidir.

Bu bağlantı segmentation/video/OCR/VLM, web paneli, GPU worker veya performans
yük testini kapsamaz. Bunlar mevcut HTTP servisinde ayrı çalışmaya devam eder.

### Doğrulanan gerçek worker koşusu — 2026-10-16

`69a6f31` kodu için [CI koşusunun](https://github.com/bekiroruk/aegis-vision/actions/runs/38897763793)
**7/7 grubu başarılı**. [Redis + gerçek YOLO grubu](https://github.com/bekiroruk/aegis-vision/actions/runs/38897763793/job/116962259367)
`redis_verified=true`, 5 detection (4 kişi, 1 otobüs) üretti. Kuyruktaki iş
`succeeded`, attempts=1; terminal replay, SHA/yol/tür reddi ve iptal testleri geçti.
[JSON kanıt artifact'ı](https://github.com/bekiroruk/aegis-vision/actions/runs/38897763793/artifacts/12045037486)
aynı koşuda saklandı. Artifact saklama süresi GitHub ayarlarıyla sınırlıdır.

Ayrı Redis grubu otomatik heartbeat ile ilk lease süresinden uzun işin
tamamlandığını, rakip claim reddini ve aktif iptalde sonuç yayımlanmadığını
gerçek sunucuda doğruladı. Önceki sekiz kuyruk testi ve AOF restart testi de geçti.

Windows OpenCV 4.12 üzerinde 4/4 ilgili CTest, gerçek YOLO referans testi,
ayrı Redis-only build'de 2/2 test ve temel build'de 7/7 test geçti. Yerel gerçek
görüntü raporu `outputs/redis-worker/local-image.json`: 4 kişi/1 otobüs,
tek ölçüm inference 210.19 ms (benchmark değildir). Bu rapordaki
`redis_verified=false` doğrudur: Windows'ta Redis yok; birleşik Redis kanıtı
yukarıdaki Linux koşusudur. Yerel model/fotoğraf/rapor Git'e yüklenmedi.
