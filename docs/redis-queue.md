# Redis iş sahipliği ve kurtarma çekirdeği

2026-10-16. İsteğe bağlı C++ `RedisLeaseQueue` ve native komut aracı.
**Henüz panelin JobQueue backend'i veya CV işi yürüten dağıtık worker değildir.**
Mevcut HTTP/SQLite servisi değişmez; bu parça süreçler arası iş alma, sahiplik
yenileme, süre aşımı ve sonuç kabul protokolüdür.

## Sözleşme

- `submit`, `claim`, `renew`, `complete`, `fail`, `cancel`, `get` işlemleri.
  Aynı ID/payload tekrar gönderilince yeni iş oluşturulmaz; farklı payload
  reddedilir. Context model/veri sözleşmesini ayırmak için çağıran tarafından
  verilmelidir; aynı kuyrukta farklı context reddedilir.
- Claim atomik olarak yalnız bir sahip seçer. Renew/complete/fail, owner,
  artan token ve sürenin geçerli olmasını ister. Eski worker'ın geç gelen
  sonucu yeni sahibin sonucunun üzerine yazamaz.
- Saat Redis TIME; lease 100–300.000 ms. Worker süre dolmadan renew yapmalıdır;
  otomatik heartbeat döngüsü henüz yoktur. Claim süresi dolmuş işleri geri alır;
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

Sonraki parça model yüklü C++ worker, heartbeat/iptal döngüsü ve HTTP kuyruğuna
opsiyonel bağlantıdır. Media/model context'inin otomatik hash'lenmesi, dosya
sonuçlarının token kontrollü yayını ve Qdrant yan etkilerinin idempotent
yönetimi ayrıca gerekir. Lease eski worker'ın hesaplamasını engellemez;
yalnız bu kuyruğa sonucunu kabul ettirmez. **Exactly-once / çift inference
engeli iddiası yoktur.**

Tek standalone Redis, yalnız güvenilen local istemciler: TLS/ACL, uzaktaki
sunucu, Cluster/Sentinel/failover desteklenmez. Redis veri kaybı/yedekten dönüş
token sayacını geri alabilir; bu durumda güvence yoktur. Redis sunucu lisansı
ayrıca incelenmelidir; hiredis bildirimi THIRD_PARTY_NOTICES içindedir.
