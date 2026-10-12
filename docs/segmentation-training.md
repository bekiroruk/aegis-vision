# Gerçek GPU fine-tuning ve C++ karşılaştırması

2026-10-12: YOLOv8n-seg, bu bilgisayardaki RTX 3060 Laptop (6 GB) üzerinde
gerçek insan maskeleriyle 10 epoch eğitildi. Eğitim/export PyTorch ile, her iki
modelin test inference'ı **aynı native C++/OpenCV CPU FP32** uygulamasıyla yapıldı.
Bu bir eğitim iskeleti değil; ağırlık, ONNX, tahmin ve ölçüm dosyaları üretildi.
Çalışan servisin varsayılan modeli değiştirilmedi.

## Sabit veri ve deney protokolü

[Penn–Fudan'ın resmi arşivi](https://www.cis.upenn.edu/~jshi/ped_html/)
170 görüntü içerir. 96 Penn görüntüsünün isimleri sabit SHA256 sıralamasıyla
80 eğitim / 16 doğrulama olarak ayrıldı; başka konumdaki **74 Fudan görüntüsü
yalnız test** için tutuldu. Bölünme etiket/sonuç görülerek değiştirilmez.
Kontrol noktası yalnız Penn doğrulama kümesindeki Ultralytics fitness değeriyle
seçildi. Fudan test sonucu ağırlık, epoch veya eşik seçmekte kullanılmadı.

İndirilen arşivde toplam **423 instance maskesi**, test bölümünde **160 kişi**
var. Kaynak sayfadaki tarihsel 345 sayısından fark, arşivin `added-object-list.txt`
dosyasındaki 78 ek nesneyle uyumludur. Test anotasyonları orijinal piksel
maskeleridir; boyalı önizleme veya modelin otomatik etiketi değildir.

Eğitim etiketleri her instance'ın en büyük dış polygon'udur. Bu dönüşüm delikleri
doldurur ve küçük ayrık parçaları atabilir: 423 instance üzerinde polygon/orijinal
maske IoU ortalaması 0,99723, en düşük değer 0,79871. **Değerlendirme bu yaklaşık
polygon'ları değil tam orijinal maskeleri kullanır.**

- PyTorch `2.9.1+cu130`, Ultralytics `8.3.252`, ONNX `1.19.1`.
- 640 giriş, batch 4, 10 epoch, seed 20261012, workers 0, deterministic açık.
- AdamW, başlangıç LR 0,0001, son LR oranı 0,1, weight decay 0,0005.
- İlk 10 katman donduruldu; AMP, mosaic, mixup, copy-paste kapalı.
- 80 sınıflı COCO çıkış kafası korundu, yalnız person etiketleriyle fine-tuning.
- Diğer sürüme bağlı eğitim varsayılanları `training/args.yaml` içinde saklandı.
- ONNX: FP32, statik `[1,3,640,640]`, opset 12, NMS ve simplify kapalı.
- Native inference: confidence 0,35, NMS IoU 0,45, maske eşiği 0,5, en çok 100 instance.
- Resmi `pycocotools 2.0.11` COCOeval segm, IoU 0,50:0,05:0,95.

## Sonuç: iyileşme var, otomatik yayın yok

Aynı 74 Fudan görüntüsü, aynı C++ çalıştırıcı ve aynı eşikler:

| Ölçüt | Hazır model | Eğitilmiş aday |
|---|---:|---:|
| Maske AP50:95 | %61,73 | **%71,06** |
| Maske AP50 | %94,13 | %90,38 |
| Maske AP75 | %72,78 | %86,33 |
| Küçük nesne AP | %32,79 | %14,36 |
| Doğru kişi maskesi, IoU ≥0,50 | 159 | 150 |
| Fazladan kişi maskesi | 93 | 37 |
| Kaçırılan kişi | 1 | 10 |
| Eşleşen maskelerde ortalama IoU | %82,12 | %87,96 |

AP50:95 **9,34 yüzde puan** arttı; fazladan maskeler azaldı ve eşleşen maskeler
daha hassas oldu. Buna karşılık recall ve küçük nesne başarısı geriledi. Bu
yüzden aday **genel servis modelinin yerine geçirilmedi**. Diğer 79 sınıf için
kalite korunumu ölçülmedi; kişi odaklı eğitim bunları bozabilir.

Bu küçük, çoğunlukla ayakta kişiler içeren konum-transfer deneyidir; negatif
sahne testi ve genel 80 sınıflı COCO benchmark'ı değildir. Confidence 0,35'te
kesilmiş AP, düşük eşikli resmi model leaderboard skoruyla karşılaştırılmaz.
Penn train/val görüntüleri arka plan paylaşabilir. Fudan bu fine-tuning'den ayrı
tutuldu; temel modelin önceki ön-eğitim verisi için bağımsızlık garantisi yoktur.
Tek koşu sonucu istatistiksel güven aralığı veya farklı seed garantisi değildir.
Bu test artık görülmüştür; yeni iyileştirme kararları için validation, son kontrol
için yeni dokunulmamış test verisi gerekir.

## Bilgisayardaki gerçek çıktılar

- `outputs/person-finetune-v1/training/weights/best.pt`: eğitilmiş ağırlık.
- `outputs/person-finetune-v1/candidate.onnx`: C++ ile doğrulanmış aday.
- `outputs/person-finetune-v1/protocol.json`, `candidate-manifest.json`: sürüm/hash/aday durumu.
- `outputs/person-finetune-v1/training/results.csv`, `args.yaml`: eğitim geçmişi ve tüm ayarlar.
- `outputs/person-finetune-native-base-v2/metrics.json`: başlangıç ölçümü.
- `outputs/person-finetune-native-trained-v1/metrics.json`: aday ölçümü.
- Her iki native çıktı klasöründe `mask-predictions.json` ve `preview.jpg`.
- Git'te küçük, sayısal [deney kaydı](experiments/person-finetune-v1.json) bulunur;
  veri, tahmin maskeleri ve büyük model dosyaları yereldir, Git'e girmez.

## Tekrar çalıştırma

Komutlar **repo kökünde** çalıştırılır. Mevcut export ortamı, resmi temel
YOLOv8n-seg ağırlığı, COCO kategori metadata'sı ve C++ build gereklidir;
bu komutlar sıfır bilgisayara tüm bağımlılıkları kuran bir kurulum değildir.
Eğitici sürüm/GPU kontrolünde uyumsuz ortamı reddeder. Otomatik paket yükseltmez.
Tam kayan-nokta bit eşitliği farklı GPU/sürücü ortamlarında garanti edilmez.

Arşiv bir kez resmi kaynaktan alınır; mevcut dosyanın üzerine indirmeyin:

```powershell
New-Item -ItemType Directory -Force artifacts/datasets/penn-fudan
if (-not (Test-Path artifacts/datasets/penn-fudan/PennFudanPed.zip)) {
  Invoke-WebRequest 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip' -OutFile artifacts/datasets/penn-fudan/PennFudanPed.zip
}
work/yolo-export/Scripts/python.exe scripts/prepare_segmentation_training.py artifacts/datasets/penn-fudan/PennFudanPed.zip artifacts/datasets/penn-fudan/transfer-v1 artifacts/datasets/coco-search/annotations/instances_val2017.json --sha256 9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2
work/yolo-export/Scripts/python.exe scripts/train_segmentation.py artifacts/datasets/penn-fudan/transfer-v1 outputs/person-finetune-rerun
```

Hazırlanmış `transfer-v1` zaten varsa hazırlama adımını atlayın. Hazırlayıcı ve
eğitici var olan çıktı dizinini ezmez. Her tekrar için yeni çıktı adı kullanın.
Görüntü/etiket/anotasyon hash'leri, sabit split üyeliği, yinelenen görüntüler,
kaçak dosya yolları ve sınıf sıralaması kontrol edilir.

```powershell
cmake --build build/live-seg --config Release --target aegisvision_segment_quality aegisvision_segmentation_tests --parallel 4
build/live-seg/Release/aegisvision_segment_quality.exe artifacts/models/yolov8n-seg/yolov8n-seg.onnx artifacts/datasets/penn-fudan/transfer-v1/test-manifest.json outputs/person-baseline-rerun
build/live-seg/Release/aegisvision_segment_quality.exe outputs/person-finetune-rerun/candidate.onnx artifacts/datasets/penn-fudan/transfer-v1/test-manifest.json outputs/person-candidate-rerun
work/yolo-export/Scripts/python.exe scripts/evaluate_segmentation.py artifacts/datasets/penn-fudan/transfer-v1/test-manifest.json artifacts/datasets/penn-fudan/transfer-v1/annotations.json artifacts/models/yolov8n-seg/yolov8n-seg.onnx outputs/person-baseline-rerun/mask-predictions.json outputs/person-baseline-rerun/metrics.json
work/yolo-export/Scripts/python.exe scripts/evaluate_segmentation.py artifacts/datasets/penn-fudan/transfer-v1/test-manifest.json artifacts/datasets/penn-fudan/transfer-v1/annotations.json outputs/person-finetune-rerun/candidate.onnx outputs/person-candidate-rerun/mask-predictions.json outputs/person-candidate-rerun/metrics.json
work/yolo-export/Scripts/python.exe scripts/test_segmentation_training.py
work/yolo-export/Scripts/python.exe scripts/test_segmentation_eval.py
ctest --test-dir build/live-seg -C Release --output-on-failure
```

Resmi ölçüm için mevcut `artifacts/deps/quality-reference` içindeki pycocotools
gereklidir. Deneyde 588×547 boyutlu bir görüntü, C++ letterbox oranını geri
çarparken `640.0000000000001` sonucuyla hatalı reddediliyordu. Tam sayı yeniden
boyutlandırma ölçüsü dar toleransla geri kuruluyor; taşan/geçersiz ölçüler hâlâ
reddediliyor. Görüntü ve transpozu için C++ regresyon testi eklendi. İlk yarım
baseline klasörü (`person-finetune-native-base-v1`) tamamlanmış sonuç değildir;
rapor **v2** tam koşusunu kullanır.

Doğrulama: 14 veri/eğitim sözleşmesi testi, 9 değerlendirme testi, 46/46 yerel
CTest ve web durum testleri geçti. Eğitim ve her iki 74 görüntülük native koşu
tamamlandı. C++ testleri/CLI'ları yeniden derlendi. Tam build sırasında çalışan
`aegisvision_server.exe` Windows tarafından kilitli olduğundan yalnız sunucu
yeniden linklenemedi; servis kapatılmadı. Dolayısıyla açık serviste eski binary
korunur (`server_help` kontrolü de o binary üzerindedir). Normal bakımda servis
kapatılıp `aegisvision_server` hedefi yeniden derlendiğinde decoder düzeltmesi
servise de geçer; yeni aday model bununla otomatik etkinleştirilmez.

Kaynaklar: [Penn–Fudan](https://www.cis.upenn.edu/~jshi/ped_html/),
[PyTorch veri kümesi örneği](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html),
[Ultralytics segmentation](https://docs.ultralytics.com/tasks/segment/).
Veri yeniden dağıtılmaz; ticari kullanımdan önce veri/atıf koşulları ve
[model lisansı](https://www.ultralytics.com/license) ayrıca kontrol edilmelidir.
