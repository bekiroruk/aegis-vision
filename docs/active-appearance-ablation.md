# Aktif takipte appearance hard-gate ablation

Protokol: 2026-10-06, ilk deneyden önce tanımlandı. Önceki sürüm `00f8829`.
[Karar izi](association-trace.md), geliştirme sahnelerindeki 15 ardışık IDSW
adayının 14'ünün appearance kapısında reddedildiğini gösterdi. Bu deney bu kapının
etkisini ayrı ölçer; encoder, fusion ağırlığı veya eşik taraması yapılmaz.

## Sabit yöntem

- Yalnız TUD-Campus (71 kare) ve TUD-Stadtmitte (179 kare), tüm kareler.
  Bunlar gözlenmiş geliştirme sahneleridir; bağımsız test değildir.
- Aynı `configs/evaluation.toml`, YOLOv8n ve OSNet x0.25 MSMT17 ONNX.
  Hash'ler [transfer protokolündeki](../configs/tracking-transfer-v1.json) model/config
  değerleriyle aynıdır; raporlarda gerçek dosya SHA256'ları kaydedilir.
- Beş eski takipçi korunur. Altıncı `kalman_reid_active`, aynı detection ve
  aynı crop embedding'lerini paylaşır; ek model çağrısı yapmaz.
- Tek değişken: aktif kimlikler için (active-high ve active-low) cosine .20
  hard-gate devre dışı; lost-high için .20 kapısı korunur.
- Sınıf, IoU .30 ve center Mahalanobis 9.2103 kontrolleri; .10/.35/.50 skorlar,
  20 kare yaşam süresi, .50 IoU/cosine fusion ve .90 EMA değişmez.
- Fusion `0.5*IoU + 0.5*cosine` olarak kalır. Negatif/zero toplam reward
  eşleşmez (solver'a 0 verilir); cosine yeniden ölçeklenmez. Yeni politika
  negatif cosine kabul edebildiğinden, genel API'de momentum .5 antipodal
  EMA tam sıfır olursa yeni gözlem yönü korunur. Deneyde momentum .90'dır.
- Low-score eşleşmeler hâlâ prototype güncellemez. Yeni varyant yalnız kalite
  CLI'ında opt-in'dir; normal video/RTSP config'i veya varsayılan değişmez.
- Yeni varyant dondurulmuş ETH/PETS transfer manifestiyle çalıştırılamaz;
  mevcut protokolün sessizce değiştirilmesi runner tarafından reddedilir.

Metrikler: HOTA/DetA/AssA, IDF1, FP/FN/IDSW ve kare bazlı switch incelemesi.
Her iki sekansta resmi TrackEval kontrolü ve eski beş takipçi ham sonuç eşitliği
gereklidir. Daha az IDSW tek başına başarı sayılmaz. Sonuca göre varyant otomatik
olarak varsayılan yapılmaz; seçilen ayar için dokunulmamış test gerekir.

## Tekrar üretme

```powershell
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-active-v1 --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17 --active-appearance-ablation
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-active-v1 --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17 --active-appearance-ablation
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-active-v1
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-active-v1
build/search/Release/aegisvision_quality.exe --audit outputs/quality-campus-active-v1 outputs/audit-campus-active-v1
build/search/Release/aegisvision_quality.exe --audit outputs/quality-tud-active-v1 outputs/audit-tud-active-v1
```

Çıktı dizinleri yeni olmalı. Altı panelli gerçek AVI, ham track JSON'u, altıncı
takipçi MOT dosyası ve rapor üretilir. Normal beşli raporlar referans/audit araçlarıyla
uyumlu kalır. Veriler/modeller/çıktılar Git'e eklenmez.

## Campus sonucu

Skorlar yüzde, sayılar CLEAR IoU .50'dir. HOTA 19 IoU eşiği ortalamasıdır.

| Yöntem | HOTA | DetA | AssA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| İki aşamalı | 51,57 | 50,51 | 52,77 | 68,90 | 4 | 83 | 55 |
| Kalman merkez | 49,94 | 53,14 | 47,08 | 67,39 | 5 | 75 | 57 |
| OSNet mevcut | 50,73 | 52,61 | 49,13 | 64,29 | 10 | 69 | 59 |
| OSNet aktif kapı kaldırılmış | 51,05 | 52,94 | 49,39 | 67,39 | 5 | 75 | 57 |

Yeni varyantta ardışık IDSW 6→0, eşleşmesiz aradan sonraki IDSW 4→5 oldu.
HOTA +0,32 yüzde puan, IDF1 +3,11 yüzde puan; FP +6, FN -2. Yani sadece
14/15 aday nedenini bulmak, tüm sekansın hatalarının aynı oranda azalacağı
anlamına gelmiyor. İki aşamalı baseline bu sahnede hâlâ daha yüksek HOTA/IDF1'e sahip.
Merkez yöntemiyle IDF1/sayıların eşitliği tüm kimlik eşleşmelerinin aynı olduğu
anlamına gelmez; HOTA farklıdır.

Altı takipçinin resmi TrackEval CLEAR/Identity/HOTA kontrolü geçti (1e-6).
Eski beş takipçinin ham kutu/ID'leri, raw detections ve GT önceki trace koşusuyla
aynı. Encoder çağrısı 676: altıncı takipçi için ek embedding inference yok.

## Stadtmitte sonucu

| Yöntem | HOTA | DetA | AssA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| İki aşamalı | 54,49 | 62,01 | 48,01 | 68,36 | 12 | 74 | 177 |
| Kalman merkez | 58,48 | 61,89 | 55,36 | 76,68 | 11 | 71 | 179 |
| OSNet mevcut | 58,83 | 62,72 | 55,28 | 75,98 | 19 | 47 | 182 |
| OSNet aktif kapı kaldırılmış | 58,48 | 61,89 | 55,36 | 76,68 | 11 | 71 | 179 |

Ardışık IDSW 9→1, aradan sonraki IDSW 10→10 oldu. IDF1 +0,70 yüzde puan,
fakat HOTA -0,35 yüzde puan; FP +24, FN -3. Bu sekansta yeni varyantın ham
kutu **ve ID** çıktıları merkez takipçisiyle tam olarak aynı; appearance
maliyeti karşılığında merkez yöntemine ek başarı gösterilmedi.

Altı takipçinin resmi referans kontrolü geçti. Eski beş takipçi ham çıktıları,
GT ve raw detections önceki trace koşusuyla aynı. Encoder çağrısı yine 1.328.
İki geliştirme videosunda toplam ardışık IDSW 15→1 oldu, fakat bu azalma
her sekansta HOTA artışına dönüşmedi. Sekans HOTA'ları pooled skor gibi ortalanmaz.

## Karar ve çıktılar

**Deneysel kalır; varsayılan değiştirilmez.** Hipotez kısmen desteklendi: sert
appearance reddi ardışık kopmaları artırıyordu, fakat bu reddi kaldırmak FP'yi
de artırdı. Sonraki ayrı deney fusion etkisi veya yalnız güçlü/tekil geometrik
adaylarda devamlılık politikası olabilir. Yeni seçimler validation üzerinde
belirlenmeli, başka dokunulmamış sekanslarda sınanmalı; bu iki sahne test değildir.
Bu turda ETH/PETS çalıştırılmadı ve yeni eşik taraması yapılmadı.

32 CTest ve 25 referans protokol testi geçti. Gerçek iki koşuda altı takipçinin
CLEAR/Identity ve 19 HOTA eşiği resmi TrackEval ile 1e-6 toleransta eşleşti.
Rapor SHA256'ları:

- Campus: `804caa0f74771e429ef9ee744aaa370ee810775e98418b6b18ca7fe833cdc79c`
- Stadtmitte: `95bcefcfdc6212622283a86fd1c5af95274b04d014767bb3ee8e05185771a9fc`

Yerel `outputs/quality-tud-active-v1/before-after.mp4`, 179 karenin tamamında
solda eski OSNet, sağda yeni varyantı gösterir; kaynak 25 FPS/7,16 saniye korunur.
`outputs/quality-campus-active-v1/before-after-slow.mp4`, 71 karenin tamamının
4 kat yavaşlatılmış karşılaştırmasıdır. İkisi de gerçek altı panelli analizden
kırpılır; yeni inference veya başarılı kare seçimi değildir. Sarı GT, yeşil
takip kutularıdır. Oynatma hızı analiz hızı değildir; bu tur kontrollü CPU hız
benchmark'ı olarak sunulmaz.
