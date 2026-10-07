# Güçlü ve tekil geometriyle sınırlı appearance esnetmesi

Protokol 2026-10-06, ilk koşudan önce belirlendi. Önceki sürüm `edac274`.
Önceki aktif-kapı ablation'ı kimlik kopmalarını azaltırken FP'yi artırdı.
Bu deney, daha korumalı bir alternatifin bu ödünleşimi değiştirip değiştirmediğini
ölçer. İyileşme varsayılmaz; eşik taraması yapılmaz.

## Sabit kural

Yedinci takipçi `kalman_reid_guarded`, eski OSNet ayarlarıyla başlar. Cosine
mesafesi .20 kapısını yalnız aşağıdaki şartların **hepsi** sağlanınca esnetir:

1. Kimlik aktif ve eşleştirme active-high aşamasında.
2. Tespit güveni yeni kimlik eşiğine (.50) en az eşit.
3. Tahmin edilen kutuyla IoU en az .70.
4. Sınıf, normal IoU .30 ve center Mahalanobis 9.2103 kapılarını geçen adaylar
   arasında bu track'in tek tespiti ve bu tespitin tek track'i var.

Tekillik, hiçbir assignment yapılmadan bütün canlı kimlikler (lost dahil) ve
bütün uygun tespitler (score >= .10, low dahil) üzerinden sayılır. Aşama sırası
başka adayı görünmez yapmaz. Low, lost, belirsiz veya zayıf geometrik eşleşmelerde
normal appearance kapısı korunur. Fusion .50, EMA .90, high-only prototype
update, yaşam süresi ve diğer eşikler aynı kalır. .70 bir başlangıç hipotezidir,
kalibre edilmiş doğruluk olasılığı değildir. Kuralın adı güvenlik garantisi vermez.

Ek geometri taraması 256×512 adayla sınırlıdır. `guarded_appearance_bypasses`
appearance sınırını aşmasına rağmen esnetilen **aday çiftleri** sayar; eşleşen
kişi veya düzeltilen IDSW sayısı değildir. Aktif-kapıyı tamamen kaldıran seçenek
ile aynı tracker config'inde birlikte açılamaz.

## Deney ve sınırlar

Yalnız gözlenmiş geliştirme sahneleri Campus (71) ve Stadtmitte (179 kare).
Mevcut model/config hash'leri ve bütün kareler korunur. Yedi bağımsız takipçi
aynı YOLO sonuçlarını, üç appearance varyantı aynı OSNet embedding'lerini paylaşır.
Eski altı takipçi ham çıktılarının değişmemesi ve resmi CLEAR/Identity/HOTA
doğrulaması kabul şartıdır. Model veya ek eşik araması yoktur.

Seçenek yalnız kalite CLI'ında opt-in; normal video/RTSP varsayılanı değişmez.
Dondurulmuş transfer manifestlerinde reddedilir. ETH/PETS bu tur çalıştırılmaz.
Bu iki geliştirme sahnesi bağımsız test sayılamaz. Sonuçtan sonra yeni bir ayar
seçilecekse ayrıca validation ve dokunulmamış test gerekir.

```powershell
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-guarded-v1 --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17 --guarded-appearance-ablation
build/search/Release/aegisvision_quality.exe configs/evaluation.toml artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-guarded-v1 --kalman --kalman-center --reid artifacts/models/osnet-x0-25-msmt17 --guarded-appearance-ablation
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-campus/quality-manifest.json outputs/quality-campus-guarded-v1
python scripts/verify_quality_reference.py mot artifacts/datasets/mot15-tud/quality-manifest.json outputs/quality-tud-guarded-v1
```

Yeni hedef dizin kullanılır. Guarded bayrağı altıncı aktif-kapı baseline'ını da
ekler; `--active-appearance-ablation` ayrıca gerekli değildir. Ham JSON, yedi
panelli gerçek video ve `kalman-reid-guarded-mot.txt` üretilir. Offline `--audit`
yedinci takipçiyi de kapsar. Kaynak/veri/model/çıktı ayrımı korunur.

## Campus sonucu

Skorlar yüzde; FP/FN/IDSW, CLEAR IoU .50 sayılarıdır.

| Yöntem | HOTA | IDF1 | IDSW | FP | FN | Ardışık IDSW |
|---|---:|---:|---:|---:|---:|---:|
| Mevcut OSNet | 50,73 | 64,29 | 10 | 69 | 59 | 6 |
| Aktif kapı tamamen kaldırılmış | 51,05 | 67,39 | 5 | 75 | 57 | 0 |
| Güçlü/tekil geometriyle esnetme | 50,66 | 65,38 | 8 | 69 | 59 | 4 |

11 aday çifte istisna uygulandı. Mevcut OSNet'e göre FP/FN değişmedi,
IDSW 2 azaldı; IDF1 +1,10 yüzde puan, HOTA -0,07 yüzde puan. Bu küçük sahnede
sonuç tek yönlü üstünlük göstermiyor. Yedi takipçinin resmi referans kontrolü
geçti; eski altı takipçi ham kutu/ID'leri ve raw detections önceki koşuyla aynı.

## Stadtmitte sonucu ve karar

| Yöntem | HOTA | IDF1 | IDSW | FP | FN | Ardışık IDSW |
|---|---:|---:|---:|---:|---:|---:|
| Mevcut OSNet | 58,83 | 75,98 | 19 | 47 | 182 | 9 |
| Aktif kapı tamamen kaldırılmış | 58,48 | 76,68 | 11 | 71 | 179 | 1 |
| Güçlü/tekil geometriyle esnetme | 59,53 | 76,85 | 13 | 45 | 184 | 3 |

15 aday çifte istisna uygulandı. Mevcut OSNet'e göre HOTA +0,70 ve IDF1 +0,88
yüzde puan; IDSW 6, FP 2 azalırken FN 2 arttı. Eski altı takipçi, ham tespitler
ve GT çıktıları birebir korundu. Resmi TrackEval CLEAR/Identity ve 19 eşikli HOTA
kontrolü yedi takipçide de 1e-6 toleransla geçti.

**Karar:** seçenek deneysel kalır; uygulama varsayılanı değişmez. Campus HOTA
düşüşü ve Stadtmitte FN artışı nedeniyle genel üstünlük iddiası yoktur. Bu iki
gözlenmiş sahnede yeni eşik taraması yapılmayacak. Bir sonraki tracking kalite
adımı, yeni validation/test verisiyle önceden sabitlenmiş kabul ölçütleridir.
GPU hızlanması veya çoklu kamera Re-ID bu deneyle tamamlanmış sayılmaz.

Yerel `outputs/quality-tud-guarded-v1/policy-comparison.mp4`, soldan sağa mevcut
OSNet / aktif kapısız / korumalı yöntemleri gösterir: 179 gerçek kare, 1920×480,
25 FPS, 7,16 saniye. Model ve videolar Git'e dahil edilmez; yukarıdaki komutlarla
ana karşılaştırma tekrar üretilebilir.

## Doğrulama ve açık CI sorunu

- 32/32 yerel CTest ve 26 referans yardımcı testi geçti.
- İki gerçek sahnede toplam 250 kare ve yedi takipçi resmi referansla doğrulandı.
- Low/lost, zayıf geometri, rakip track/tespit ve geçersiz config testleri eklendi.
- Önceki `edac274` Linux CI koşusunda canlı arşiv testi zaman aşımına uğradı;
  tracking testleri geçti. Bekleme süresi artırılmadan, hata mesajına kaynak
  konumu ve başarısız indeksleme ayrıntısı eklendi. Bu tanılama değişikliğiyle
  arşiv testi yerelde art arda üç kez geçti; uzak ortamın kök nedeni henüz
  çözülmüş sayılmaz.
