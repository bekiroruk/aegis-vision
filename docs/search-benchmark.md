# COCO kırpma araması: tekrarlanabilir başlangıç ölçümü

Bu küçük değerlendirme, COCO 2017 validation etiketlerinden sekiz benzer nesne
sınıfında (bicycle, motorcycle, bus, train, cat, dog, zebra, giraffe) sınıf başına
sekiz nesne seçer. Aynı görüntü iki kez kullanılmaz. Seçim, annotation ID'sinin
sabit SHA256 sırasını izler; kutu en az 96×96 piksel ve görüntü alanının %5'i
olmalıdır. İndekslenen şey tam sahne değil, annotation kutusunun kırpmasıdır.
Veri ve çıktı Git'e eklenmez.

Kaynak: [COCO 2017 Detection](https://cocodataset.org/dataset/detection-2017.htm)
ve [resmi indirme sayfası](https://cocodataset.org/#download). Görüntülerin telif
ve kullanım koşulları için [COCO Terms of Use](https://cocodataset.org/#termsofuse)
sayfasını kontrol edin. Depoda yalnızca indirme/seçim kodu vardır; görseller
yeniden dağıtılmaz.

## Veri hazırlama (PowerShell, proje kökünde)

```powershell
New-Item -ItemType Directory -Force artifacts/datasets/coco-search | Out-Null
curl.exe -L --fail --output artifacts/datasets/coco-search/annotations_trainval2017.zip https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
Get-FileHash artifacts/datasets/coco-search/annotations_trainval2017.zip -Algorithm MD5
tar -xf artifacts/datasets/coco-search/annotations_trainval2017.zip -C artifacts/datasets/coco-search annotations/instances_val2017.json
./work/yolo-export/Scripts/python.exe scripts/prepare_coco_search.py artifacts/datasets/coco-search/annotations/instances_val2017.json artifacts/datasets/coco-search/manifest.json --download
```

Resmi S3 yanıtındaki ETag `f4bbac642086de4f52a3fdda2de5fa2c`; indirilen
arşivin MD5 çıktısıyla karşılaştırın. Alternatif sistemlerde `python` çalıştırılabilir.
Hazırlama scripti standart Python kütüphanesini kullanır. Fotoğrafları resmi
S3 HTTPS uç noktasından tek tek çeker, manifesti yerelde oluşturur.

## İndeksleme ve ölçüm

Yerel Qdrant'ı başlatın (bkz. [arama kılavuzu](search.md)). Benchmark için
**ayrı ve boş bir koleksiyon** kullanın; komutlar hiçbir koleksiyonu silmez.

```powershell
./build/search/Release/aegisvision_search_cli.exe artifacts/models/clip-vit-b32 6333 aegis_coco_search_v1 init
./build/search/Release/aegisvision_search_cli.exe artifacts/models/clip-vit-b32 6333 aegis_coco_search_v1 batch-index artifacts/datasets/coco-search/manifest.json
./build/search/Release/aegisvision_search_cli.exe artifacts/models/clip-vit-b32 6333 aegis_coco_search_v1 evaluate artifacts/datasets/coco-search/manifest.json outputs/coco-search/report.json
```

`batch-index`, CLIP'i bir kez yükler ve 64 kırpmayı aynı süreçte indeksler;
aynı ID'lerle yeniden çalıştırılması kayıtları günceller. İşlem yarıda kalırsa
aynı komut tekrar çalıştırılabilir. `evaluate`, koleksiyon kayıt sayısının
manifestteki sayıya eşit olmasını ister, sorgu sonuçlarında manifest dışı ID
bulursa hata verir. Bu kontrol özel koleksiyon şartını destekler, ancak elle
değiştirilmiş aynı sayıda kayıt için tam doğrulama değildir.

Her sınıf için `a photo of a {class}` sorgusu yapılır. Her sorgunun 8 doğru
kırpması vardır. Recall@K = ilk K içindeki **benzersiz doğru kırpma sayısı / 8**;
rapordaki macro değer sekiz sorgunun aritmetik ortalamasıdır. Hit@K ise ilk K'de
en az bir doğru sonuç bulunan sorgu oranıdır. Bu yüzden Recall@1 en fazla 0.125
olabilir. K=1, 5 ve 10 raporlanır. Aynı görüntüyle kendini arama yapılmaz.

Bu küçük, seçilmiş altküme resmi COCO detection ölçümü değildir; CLIP'in COCO
görsellerini ön eğitimde görmediği de garanti edilemez. Metrikler yalnızca bu
manifest ve model/ön işleme sürümü için geçerlidir. Çok dilli model ve daha geniş,
bağımsız test kümesi sonraki adımdır.

## 2026-09-30 yerel başlangıç sonucu

CLIP ViT-B/32, ONNX Runtime CPU ve Qdrant ile 64 kırpma/8 sorgu:

| Metrik | Değer |
| --- | ---: |
| Macro Recall@1 | 0.109375 |
| Macro Recall@5 | 0.593750 |
| Macro Recall@10 | 0.906250 |
| Hit@1 | 0.875000 |
| Hit@5 | 1.000000 |

`bicycle` sorgusunda ilk sıraya `motorcycle` kırpması geldi; diğer yedi sorguda
ilk sonuç doğru sınıftandı. Yerel ayrıntılı rapor `outputs/coco-search/report.json`
konumundadır. Manifestte annotation JSON SHA256 ve kullanılan görsel adresi kayıtlıdır.
