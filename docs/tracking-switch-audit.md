# Kare bazlı kimlik değişimi incelemesi

Tarih: 2026-10-05. Bu adım bir kalite artışı iddiası değil, sonraki kontrollü
iyileştirme için native C++ hata analizi aracıdır. Model/eşik/takip davranışı
değişmedi. Mevcut 4 sekans / 5 takipçi, toplam 1.399 kare yeniden inference
yapılmadan incelendi. Ham dosyalar ve eski metrik raporları değiştirilmedi.

## Kullanım

```powershell
build/search/Release/aegisvision_quality.exe --audit outputs/quality-campus-reid outputs/audit-campus-v1
build/search/Release/aegisvision_quality.exe --audit outputs/quality-tud-reid outputs/audit-tud-v1
build/search/Release/aegisvision_quality.exe --audit outputs/quality-transfer-eth-v1 outputs/audit-eth-v1
build/search/Release/aegisvision_quality.exe --audit outputs/quality-transfer-pets-v1 outputs/audit-pets-v1
```

Üst çıktı klasörü mevcut, hedef klasör yeni olmalıdır. Kaynak `report.json` ve
`tracking-frames.json` dosyaları 256 MiB ile sınırlıdır. Hash'leri sonuçta
kaydedilir ve okuma sonrası tekrar kontrol edilir. GT, kaynak rapordaki manifest
ile karşılaştırılır. CLEAR/Identity değerleri tekrar hesaplanır; eski rapora
uymayan sayı/skor, geçersiz kutu/ID veya eksik karede çıktı oluşturulmaz.
Metrik çekirdeğinin kare, nesne, kimlik ve işlem sınırları geçerlidir.
Hash'ler kaynak kimliğini kaydeder; dijital imza veya dosyaların bağımsız doğruluk
kanıtı değildir. Eski sonuçların TrackEval kontrolü ayrıca geçerlidir.

Her klasörde `switch-audit.json` ve `switch-events.csv` oluşur. CLI detector,
OSNet veya video yüklemez; quality executable'ın derleme bağımlılıkları korunur.

## Olayların anlamı

- Her CLEAR IDSW için tek olay: 1-based kare, GT ID, önceki/yeni prediction ID,
  önceki eşleşme karesi, IoU ve mevcut kutular.
- `unmatched_frames`: bu GT'nin son CLEAR eşleşmesinden sonraki ara kareler.
  Sıfırsa ardışık kare değişimi; pozitifse eşleşmesiz aradan sonra değişim.
- `gt_absent_frames`: bu ara karelerin kaçında GT etiketi yoktu. GT yokluğu
  fiziksel örtülmeyi kanıtlamaz; GT varken eşleşmemek detector yokluğu demek değildir.
- `new_id_first_seen`: yeni ID'nin ilk **çıktı** karesi. Takipçinin iç doğum
  durumuna erişilmez; bütün yöntemler için genel bir yeniden-doğum kanıtı değildir.
- `old_id_visible`: eski ID aynı karede başka bir prediction olarak var mı?
  Bu değer, eski track'in iç bellekte hâlâ tutulup tutulmadığını söylemez.

Ardışık ve aradan-sonra sayıları IDSW'ye toplanır. İlk-görünüm ve GT-yokluğu
sayacı bunlarla örtüşebilir; tüm sütunlar birbirine eklenmez. HOTA'nın
eşleşmeleri kullanılmaz. GT yalnız offline değerlendirmede bulunur.

## Gerçek çıktıdan bulgular

Tabloda `ardışık / aradan sonra / toplam IDSW` gösterilir.

| Sekans | IoU | İki aşamalı | Kalman tam kutu | Merkez | Merkez + OSNet |
|---|---:|---:|---:|---:|---:|
| Campus (geliştirme) | 3 / 5 / 8 | 0 / 4 / 4 | 17 / 5 / 22 | 0 / 5 / 5 | 6 / 4 / 10 |
| Stadtmitte (geliştirme) | 1 / 12 / 13 | 1 / 11 / 12 | 2 / 10 / 12 | 1 / 10 / 11 | 9 / 10 / 19 |
| ETH (gözlenmiş validation) | 3 / 6 / 9 | 1 / 2 / 3 | 25 / 7 / 32 | 12 / 5 / 17 | 24 / 10 / 34 |
| PETS (artık gözlenmiş test) | 22 / 65 / 87 | 11 / 37 / 48 | 326 / 69 / 395 | 85 / 41 / 126 | 166 / 54 / 220 |

20 takipçi/sekans raporunun bütün CLEAR/Identity alanları önceki raporlarla
eşleşti. Olay sayıları önceki resmi doğrulanmış IDSW toplamlarına eşittir.

Campus'ta OSNet değişimlerinin 10/10'u, Stadtmitte'de 18/19'u yeni ID'nin ilk
çıktı karesindedir. İki geliştirme sahnesinde toplam ardışık değişim merkezde
1, OSNet eklenince 15'tir. Dolayısıyla yalnız kayıp kimlikleri geri getirmeye
odaklanmak eksik olur: kesintisiz gözlenen kişide kimlik sürekliliği de sorun.
Bu gözlem appearance gate'i tek başına suçlamaz; aday kapıları, fused assignment
ve geçmiş state farklılaşmaları izlenmeden kesin neden çıkarılamaz.

### İncelenebilir örnek

Campus GT 7, OSNet panelinde kare 27'de ID 14→17, kare 29'da 17→18 olur;
iki olayda da `unmatched_frames=0`. Kaynak zamanları 1,04 ve 1,12 saniyedir
(`(frame-1)/25`). Kare 21'de GT 2 için ID 10→12 de aynı türdedir.

Yerel `outputs/audit-campus-v1/switches-slow.mp4`, önceki gerçek beşli videonun
16–45. karelerini 4 kat yavaş gösterir (30 kaynak kare; interpolasyon yok).
Sarı kutular GT, yeşil kutular takip çıktısıdır. Klip yalnız hata inceleme örneği;
ölçümler tüm sekanslara aittir, başarı demosu veya yeni deney değildir.

```powershell
ffmpeg -hide_banner -loglevel error -nostdin -n -i outputs/quality-campus-reid/comparison.mp4 -vf "trim=start_frame=15:end_frame=45,setpts=4*(PTS-STARTPTS)" -an -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart outputs/audit-campus-v1/switches-slow.mp4
```

## Sonraki kontrollü deney

Önce geliştirme sahnelerinde aday düzeyi IoU/motion/appearance kapı izi ekle.
Ardından appearance gate ve fusion etkisini ayrı ayrı kapatan ablation yap;
aynı detector ve embedding çıktısını paylaş, FP/FN/HOTA/IDF1'i birlikte ölç.
Öncelik yukarıdaki ardışık değişim örnekleri; henüz kanıtlanmış tracker düzeltmesi
yoktur. ETH/PETS artık gözlendi: yeni ayarlar için yeni dokunulmamış test seçilmeli.

Regresyonlar: ardışık switch, GT mevcut/yok karışık boşluk, geçmişte görülmüş
prediction, eski ID'nin hâlâ görünmesi, switch olmayan yeniden görünüm,
rapor/GT uyuşmazlığı ve mevcut çıktıyı koruma. 32 yerel CTest geçti.
