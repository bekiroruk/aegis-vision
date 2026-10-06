# Kalman aday eşleştirme karar kaydı

`--trace`, kalite aracında seçili Kalman takipçileri için isteğe bağlı C++ karar
kaydı açar. Normal kullanım kapalıdır; eşikler, sıralama, Hungarian ağırlıkları
ve model çıktıları değişmez. GT karar kaydına veya takipçiye verilmez.

```powershell
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-trace-v1 --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17 --trace
```

Yeni çıktı klasöründe `association-trace.jsonl`, her kare/seçili takipçi için
bir JSON satırı içerir. `observations`, takipçiye verilen low-threshold sonrası
person tespitlerini giriş sırasıyla kaydeder; `detection_index` bu dizide sıfır
tabanlıdır, skora göre sıralanmış aday sütunu değildir. Kutu ve skor saklanır;
512-boyut embedding'ler dosyaya yazılmaz.

Her adayda eski `track_id`, aşama ve sonuç bulunur:

- `active_high`, `active_low`, `lost_high`: mevcut eşleştirme sırası.
- `class`, `iou`, `motion`, `appearance`: ilk reddeden kontrol.
- `eligible`: tüm kapıları geçti, assignment tarafından seçilmedi.
- `matched`: assignment seçti. `reward` birleşik ağırlıktır, olasılık değildir.
- `birth` / `created`: yeni track kimliği ile giriş tespiti ilişkisi.

Yalnız gerçekten hesaplanan IoU, karesel Mahalanobis mesafesi, cosine mesafesi
ve reward doludur. Erken redden sonra kalan alanlar `null`; sıfır veya başarılı
kontrol anlamına gelmez. Sonraki aşamada artık uygun olmayan aday hiç görünmez.
Kapasite nedeniyle bastırılan doğumlar ve expiry için olay yoktur; mevcut
diagnostic sayaçları geçerlidir. Eksik kaydı belirli bir ret nedeni saymayın.

Çekirdek yalnız son başarılı update'in izini tutar; başarısız update hem state
hem önceki izi korur. Üst sınır 256×512 aday + 512 doğum kaydıdır. Takipçi
kapalıyken kayıt oluşturmaz. JSONL toplamı 64 MiB ile sınırlıdır; sınır veya
yazım hatasında benchmark başarısız olur, tamamlanmış rapor yazılmaz.

Kayıt açıkken allocation/JSON/disk maliyeti vardır; süreleri normal çalışmanın
hız kıyası olarak kullanmayın. Kalite ve ham track sonuçları ayrıca karşılaştırılır.
Testler ilk-ret sırası, original-index, seçilmiş/seçilmemiş aday, active-low/lost
aşamaları, transaction korunması ve kayıt açık/kapalı sonuç eşitliğini kapsar.

Bu araç [CLEAR olay incelemesini](tracking-switch-audit.md) tamamlar: önce olayın
kare/ID'si bulunur, sonra o karedeki eski ID → yeni tespit adayı incelenir.
Tek adayın elenmesi, ilgili kapıyı kaldırmanın sekans genelinde kaliteyi
iyileştireceğini kanıtlamaz; bunun için ayrı kontrollü ablation gerekir.

## Campus geliştirme sahnesi — 2026-10-05

71 karenin tamamı aynı sabit modeller/eşiklerle yeniden çalıştırıldı. Yeni
`tracking-frames.json`, önceki `outputs/quality-campus-reid` dosyasıyla byte
düzeyinde aynı SHA256'ya sahip. Beş takipçinin CLEAR/Identity/HOTA değerleri
resmi TrackEval ile 1e-6 toleransta doğrulandı. Trace dosyası 1.424.317 byte.

CLEAR olayındaki yeni ID'nin `birth` kaydından giriş tespit indeksi bulundu;
aynı karede eski ID ile bu tespit arasındaki aday aşağıdadır. Bu yöntem kutu
yakınlığıyla indeks tahmini yapmaz. Aynı karede başka aday retleri de olabilir.

| Kare | Eski→yeni ID | IoU | Hareket mesafesi | Cosine mesafesi | İlk ret |
|---|---|---:|---:|---:|---|
| 21 | 10→12 | 0,80761 | 0,39548 | 0,21574 | appearance |
| 27 | 14→17 | 0,89704 | 0,32702 | 0,21127 | appearance |
| 29 | 17→18 | 0,87405 | 0,48512 | 0,21423 | appearance |
| 37 | 15→19 | 0,71266 | 1,96766 | 0,25282 | appearance |
| 39 | 19→20 | 0,86779 | 0,33322 | 0,24837 | appearance |
| 52 | 22→25 | 0,93941 | 0,01830 | 0,23904 | appearance |

Altı olayda da önceki kare eşleşmişti (`unmatched_frames=0`); aşama
`active_high`. IoU alt sınırı .30 ve motion üst sınırı 9.2103 geçiliyor,
fakat cosine distance üst sınırı .20 aşılıyor. Böylece bu gerçek koşudaki
eski-ID/yeni-tespit adayının doğrudan elenme nedeni appearance kapısıdır.
Diğer dört IDSW (12,18,23,49. kareler), `lost_high` aşamasında IoU .30 altına
düştüğünden eleniyor; bu adaylarda appearance hiç hesaplanmıyor.

Bu bulgu, bütün sorunları yalnız appearance eşiğini artırarak çözebileceğimiz
anlamına gelmez. Kapıyı değiştirmek sonraki state ve assignment'ları da değiştirir,
yanlış kişiyi bağlama riskini artırabilir. Sabit varsayılanlar korunmuştur.
Sonraki deney aktif eşleşmelerde appearance hard-gate etkisini, fusion etkisinden
ayrı ölçmeli; aynı geliştirme çıktısını bağımsız test diye sunmamalıdır.

## Stadtmitte ile ikinci kontrol

Aynı komut `artifacts/datasets/mot15-tud/quality-manifest.json` ve yeni
`outputs/quality-tud-trace-v1` klasörüyle 179 karenin tamamında çalıştırıldı.
Ham `tracking-frames.json` yine önceki `outputs/quality-tud-reid` ile byte
düzeyinde aynı; beş takipçinin CLEAR/Identity/HOTA referans kontrolü geçti.

| Kare | Eski→yeni ID | IoU | Hareket mesafesi | Cosine mesafesi | İlk ret |
|---|---|---:|---:|---:|---|
| 3 | 7→8 | 0,52617 | 16,10990 | hesaplanmadı | motion |
| 4 | 6→9 | 0,94469 | 0,01541 | 0,24354 | appearance |
| 20 | 10→11 | 0,96057 | 0,00106 | 0,20402 | appearance |
| 40 | 15→16 | 0,91032 | 0,09645 | 0,28103 | appearance |
| 57 | 4→19 | 0,90747 | 0,13684 | 0,21857 | appearance |
| 86 | 2→24 | 0,83419 | 0,24266 | 0,21484 | appearance |
| 89 | 24→25 | 0,94784 | 0,01603 | 0,30316 | appearance |
| 91 | 23→26 | 0,81382 | 0,16588 | 0,21910 | appearance |
| 109 | 28→29 | 0,95519 | 0,04683 | 0,23392 | appearance |

İki geliştirme sahnesindeki 15 ardışık OSNet IDSW adayının 14'ü appearance,
1'i motion kapısında reddedildi. Bu, gözlenen koşudaki doğrudan aday kararını
açıklar; kapı kaldırıldığında 14 olayın tamamının düzeleceği iddiası değildir.
ETH/PETS yeniden çalıştırılmadı, model veya eşik ayarı yapılmadı.

Yerel trace SHA256 değerleri:

- Campus: `63c4e61d5c11b30835ed620bd6b9c6070892bb21a04cc5fd2e35f343ffe32ba4`
- Stadtmitte: `6efdab14fa919b43fae5da89705bb3d40f09440c5731936cc440ca7e58c0417d`

Kontrollü takip eden deney için öncelik: aktif eşleşmedeki appearance hard-gate'i
tek başına kaldırıp mevcut fusion'ı koruyan opt-in varyant; sonra fusion için
ayrı ablation. Varsayılanı değiştirmeden FP/FN, IDF1 ve HOTA beraber incelenmeli.
Yeni ayar seçilirse başka, dokunulmamış sekansla değerlendirilmelidir.

2026-10-06: [aktif kapı ablation'ı](active-appearance-ablation.md) tamamlandı.
Ardışık kopmalar azalırken FP arttı; varsayılan değiştirilmedi.
