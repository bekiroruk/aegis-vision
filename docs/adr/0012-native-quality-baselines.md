# ADR-0012: C++ doğruluk ve takip baseline'ları

Durum: Kabul edildi. Tarih: 2026-10-02.

## Bağlam

Kutulu bir görüntü veya üretilen track ID sayısı model doğruluğunu göstermez.
Detection, takip kalitesi ve işlem süresi birbirinden ayrılmalı; ileride GPU,
Re-ID veya farklı bir tracker eklenince aynı koşullarda karşılaştırılabilmelidir.
Aktif uygulama C++ kalır; geliştirme araçları bağımsız referans kontrolü yapabilir.

## Karar

- Bağımlılıksız çekirdek C++'ta sınıf duyarlı bbox AP ve tek sınıflı CLEAR/Identity
  ölçümleri hesaplar. AP, IoU .50:.05:.95 ve 101 recall noktasını kullanır; normal
  GT önce eşlenir, crowd eşleşmeleri detection alanına göre yok sayılır ve yeniden
  kullanılabilir. maxDets her görüntü/sınıfta uygulanır. GT olmayan sınıfın AP'si
  tanımsızdır ve ortalamaya dahil edilmez; boş paydalara sahte yüzde verilmez.
- Tracking CLEAR, süreklilik öncelikli bire bir eşleştirme ile TP/FP/FN ve ID switch
  hesaplar. IDF1 ayrı, tüm sekans boyunca global kimlik ataması kullanır; yalnızca
  CLEAR'ın seçtiği kare eşleşmelerinden türetilmez. Bu tam MOTChallenge/HOTA
  değerlendirmesi değildir. Veri kümesine özgü etiket filtreleri hazırlayıcıdadır.
- İsteğe bağlı `aegisvision_quality` CLI, SHA256'lı sürümlü manifestten gerçek
  görüntü/video okur ve YOLO'yu çalıştırır. Video her karede yalnızca bir kez
  infer edilir; iki takip yöntemi aynı tespitleri, bağımsız state ile kullanır.
  GT hiçbir zaman detector/tracker'a verilmez. MOT dışa aktarımında orijinal
  bir-tabanlı konumlar geri yüklenir; uygulama içinde sıfır-tabanlı xyxy kullanılır.
- Model açılışı, ısınma ve ölçülen inference ayrı tutulur. OpenCV tek thread,
  CPU/FP32, sabit model/config checksum'ları raporlanır. p50/p95 nearest-rank'tır.
  Detector süresi piksel kopyası, preprocessing, forward ve NMS içerir; video
  karşılaştırma döngüsü iki tracker ve çizim/yazım maliyetini de içerir. Bu gerçek
  zamanlı RTSP throughput garantisi veya GPU performans ölçümü değildir.
- İlk veri mevcut COCO kırpma seçiminin 64 **tam sahnesidir**: seçili sekiz sınıfa
  ait tüm etiketler, küçük/crowd nesneler dahil. Bu büyük nesne ağırlıklı, yanlı
  bir alt kümedir; 80 sınıflı resmi COCO sonucu olarak sunulmaz. MOT15'in kısa
  TUD-Stadtmitte training sekansı, resmi yeniden kodlanmış önizleme MP4'üyle
  değerlendirilir; orijinal JPEG leaderboard sonucuyla eşdeğer değildir.
- Python yalnızca veri hazırlığı ve resmi pycocotools/TrackEval karşılaştırması
  içindir. Runtime kalite ölçümü C++'tır. Model, veri ve büyük sonuç videoları
  Git'e eklenmez; ölçüm özeti, checksum ve tekrar üretim komutları belgelenir.
- Yeni/boş çıktı klasörü gerekir. Dosya kapsamı, SHA256, boyut/kare sayısı ve
  ölçüm iş yükü doğrulanır; input checksum'ları koşu başında ve sonunda kontrol
  edilir. Bu harici yazıcılara karşı atomik dosya snapshot'ı değildir; koşu
  sırasında veri dosyaları değiştirilmemelidir. Hatalı koşunun kısmi dosyaları
  korunur ama başarılı `report.json` yayımlanmaz.

## Sonuçlar ve sınırlar

İki tracker aynı tespitlerden ölçülebilir ve sonuçlar bağımsız uygulamalarla
karşılaştırılır. Bir sekans üzerinde daha iyi IDF1 genel üstünlük kanıtı değildir.
Bu koşuda eğitim, eşik taraması veya hyperparameter tuning yoktur. Bağımsız,
daha geniş evaluation/test ayrımı, HOTA, Kalman/Re-ID, çoklu kamera, segmentation
ve model fine-tuning sonraki işlerdir; mevcut baseline bunları tamamlanmış saymaz.

Protokoller: [COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py),
[TrackEval CLEAR](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py),
[TrackEval Identity](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py),
[MOT15 veri tanımı](https://motchallenge.net/data/MOT15/),
[MOT15 makalesi](https://arxiv.org/abs/1504.01942).
