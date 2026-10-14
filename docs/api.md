# C++ HTTP API sözleşmesi

Sürüm: 0.1.0. Makine tarafından okunabilir [OpenAPI JSON](../web/openapi.json),
[OpenAPI 3.1.1 standardını](https://spec.openapis.org/oas/v3.1.1.html) kullanır.
Çalışan serviste aynı dosya `/openapi.json` adresindedir. API istemcilerine
aktarılabilir; ayrı Swagger sunucusu/CDN, Python veya Node runtime gerekmez.
Bu belge yeni inference özelliği veya uzak erişim açmaz.

## Bağlantı ve güvenlik

Varsayılan CPU adresi `http://127.0.0.1:8090`, mevcut GPU demo adresi
`http://127.0.0.1:8091`. JSON içindeki server adresi aynı origin'i kullanır.
Servis yalnız loopback dinler; kimlik doğrulama/TLS yoktur. İnternete açmayın.
`Host` localhost/127.0.0.1 ve servis portuyla eşleşmelidir. `Origin` varsa
`http://Host` ile aynı olmalıdır; aksi durumda 403. CORS gevşetilmez.
Tüm POST istekleri `Content-Type: application/json` ister (415); gövde en fazla
8192 byte (413). Framework kaynaklı 413/416 yanıtları JSON olmak zorunda değildir.

`GET /api/health` yetenekleri bildirir: segmentation/OCR/VLM kapalıysa ilgili
iş 400 alır. Bu çağrı Qdrant/GPU/VLM'nin anlık uçtan uca sağlık testi değildir.
Depolama hatasında 503 ve `status=storage_error` döner. CPU/GPU etiketi yalnız
nesne tespitine aittir; diğer modellerin GPU'da çalıştığını göstermez.

## İş gönderme ve sonucu alma

PowerShell örneği (çalıştırmak yeni bir arama işi oluşturur):

```powershell
$base = 'http://127.0.0.1:8091'
$job = Invoke-RestMethod "$base/api/jobs" -Method Post -ContentType 'application/json' -Body '{"type":"search","query":"a person walking on the street","limit":4}'
Invoke-RestMethod "$base/api/jobs/$($job.id)"
```

202 yalnız kabul edildi demektir. `queued` → `running` → `succeeded`, `failed`
veya `cancelled` durumunu takip edin. Bir saniyelik polling yeterlidir; örnekteki
tek GET, tamamlanmayı beklemez. Dolu kuyruk 429 ve `Retry-After: 2` döndürür.
İptal: `POST /api/jobs/{id}/cancel`, JSON `{}`; işlemde tamamlanan yazımlar geri
alınmaz. Normal iş/model hataları otomatik tekrar edilmez. 404, geçmişten
budanmış veya bulunmayan iş de olabilir. Kalıcı işlerin yeniden deneme davranışı
[servis kılavuzundadır](service.md#kuyruk-tutarlılık-ve-sınırlar).

| İş tipi | Gerekli alanlar | İsteğe bağlı alanlar |
|---|---|---|
| `index_video` | `type`, `path` | `stride`: 1–10000, varsayılan 15; `max_frames`: 0–1000000 |
| `search` | `type`, `query` | `limit`: 1–20, varsayılan 8; `scope`: all/live |
| `segment_frame` | `type`, `path` | `frame_index`: 0–10000, varsayılan 0 |
| `ocr_frame` | `type`, `path` | `frame_index`: 0–10000, varsayılan 0 |
| `scene_frame` | `type`, `path` | `frame_index`: 0–10000, varsayılan 0 |
| `segment_video` | `type`, `path` | `max_frames`: 0–1000 |

`max_frames=0` tüm dosya anlamına gelir; segment_video'nun ayrıca 1000 karelik
sert sınırı vardır. `path`, medya kökü içindeki mevcut dosyadır; upload endpoint'i
yoktur. Kaynak arşivse ayrı `/api/live/archive/index` kullanılır. Dosya/model
değişikliği ve feature availability gibi kontroller JSON Schema dışında runtime'da
yapılır. `query` sınırı **1024 UTF-8 byte**; standart maxLength karakter saydığı
için bu ek byte kuralı şemada açıklama ve `x-max-utf8-bytes` ile belirtilir.

İstemci bilinmeyen alan gönderemez. Kabul edilmiş işin `request` alanında ise
servisin eklediği dosya boyutu/zaman damgası bulunabilir. Bunu kopyalayıp yeni
istek diye göndermeyin. `index_live_archive` iç iş türüdür; `/api/jobs` kabul etmez.
Başarılı sonuç şemaları iş türüne göre bağlıdır; sonuç metinleri/metadata
güvenilmeyen veri olarak gösterilmeli, HTML veya komut olarak yürütülmemelidir.

## Görüntü, video ve oturumlar

- Arama önizlemesi: `/api/preview/{job_id}/{result_index}.jpg`, sıfır tabanlı index.
- Segment video çıktıları: `/api/jobs/{id}/video.mp4`, `tracks.csv`, `preview.jpg`.
  Yalnız başarılı `segment_video` işine aittir; `?download=1` indirmeyi ister.
- `/media/{path}` göreli medya yoludur. Her yol segmentini URL-encode edin;
  byte range ile oynatma desteklenir (200/206/416).
- Tek kare maske/OCR/VLM sonucu JSON içinde `preview_data_url` taşır. Maske
  COCO sıkıştırılmamış sütun-major RLE; OCR polygon sırası BL,TL,TR,BR.
- Canlı kaynakları `/api/live/sources` ile alın; start gövdesi
  `{"source_id":"local-replay","archive":false}`. Kaynak URL/path'i gönderilmez.
  `source_kind=file_replay` yerel demo, `rtsp` ağ kaynağıdır. Kaynağın tanımlı
  olması bağlantının başarılı olduğunu kanıtlamaz.
- `/api/live` önce `session:null`, sonra son oturumu döndürür. Stop gövdesi
  kesinlikle `{}` olmalıdır. Oturumlar kalıcı kuyruk işi değildir.
- Canlı JPEG 204 ise görüntüyü temizleyin: kare yok, eski, durdurulmuş veya
  bağlantı kopmuş olabilir. Önceki JPEG'i yeni görüntü gibi tutmayın.
  `X-Decode-Age-Ms` kamera PTS/ağ gecikmesi değildir.

## Sözleşmeyi güncelleme ve test

Kaynak `scripts/openapi_contract.py`; `web/openapi.json` mekanik üretilir.
Uygulama çalışırken bu Python dosyasına veya doğrulama paketlerine ihtiyaç yoktur.

```powershell
python -m venv work/api-contract
.\work\api-contract\Scripts\python.exe -m pip install -r tests/requirements-api.txt
python scripts/openapi_contract.py --write
.\work\api-contract\Scripts\python.exe scripts/test_openapi.py
# Çalışan yerel servise yalnız GET; iş/oturum başlatmaz:
.\work\api-contract\Scripts\python.exe scripts/test_openapi.py --base-url http://127.0.0.1:8091
```

Testler OpenAPI biçimini, üretimin güncelliğini, C++ route kapsamını, sayı
sınırlarını, geçerli/geçersiz istekleri ve replay/RTSP ayrımını kontrol eder.
2026-10-14'te 7 test ve GPU servisinden 11 gerçek yanıt/sonuç şema kontrolü geçti.
Bu, tüm model sonuçlarının veya tüm hata kodlarının canlıda test edildiği anlamına
gelmez; model bağımsız C++ HTTP testleri ayrıca korunur. Yeni route/sayı sınırı
değişikliği testi bozar; diğer anlamsal değişiklikler hâlâ kod incelemesi gerektirir.
