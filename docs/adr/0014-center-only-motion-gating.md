# ADR-0014: Boyut titreşiminden ayrılan merkez hareket kapısı

Durum: Kabul edildi, deneysel. Tarih: 2026-10-02.

## Bağlam

ADR-0013'teki dört ölçümlü Kalman kapısı Campus'te kimlik parçalanması üretti.
Kişinin yürüyüşü/örtüşmesi kutu genişliğini değiştirirken merkez yakın kalabiliyor:
önceki koşunun kare 10 örneğinde IoU .656, merkez hatası yaklaşık (-2.24,.66) px,
genişlik hatası 28.87 px; dört boyutlu squared Mahalanobis 31.57, 13.2767'yi aşıyor.
Matris/durum hatası bulunmadı; sabit gürültü modeli boyut oynaklığına uymuyor.
Etiketli sahneyi okuyup eşiği büyütmek yerine boyut bilgisinin rolünü açıkça
ayıran bir alternatif karşılaştırılmalı. Önceki davranış korunmalı.

## Karar

- `KalmanGateMode::FullBox` varsayılan kalır; nominal chi-square(4) %99 kapısı
  `13.2767` ve önceki hesaplama değişmez. `CenterOnly` ayrı bir seçenektir.
- Center kapısı yalnız `(cx,cy)` innovation'ını ve **marjinal** kovaryans
  `S[0:2,0:2]` kullanır. Merkez ilk iki bileşen olduğundan dört boyutlu Cholesky
  faktörünün öncü 2×2 bloğu bu matrisi faktörler. İki residual forward solve ile
  beyazlatılır, karelerinin toplamı alınır. Schur complement veya dört boyutlu
  solve sonrası kısmi toplam kullanılmaz.
- Yeni nominal chi-square(2) %99 kapısı `9.2103`, ilk gerçek koşudan önce sabittir.
  Kovaryans kalibre edilmediğinden gerçek %99 doğru eşleşme garantisi verilmez.
  Konum/boyut/hız gürültüsü, tam **4B Kalman correction**, predicted-box IoU .30,
  sınıf kontrolü, aktif/high → aktif/low → kayıp/high sırası, low/high/new ve
  lifetime değerleri değiştirilmez. Görünür sonuç hâlâ yalnız detector kutusudur.
- C++ aggregate config sonuna mode eklenir; eski initializer'lar FullBox kalır.
  Center seçen çekirdek çağıran eşik değerini ayrıca `kalman_center_gate99` olarak
  ayarlar. TOML `gating_mode="center"` ve positional `--tracker kalman-center`
  nominal iki boyutlu eşiği otomatik seçer. TOML'de açık `mahalanobis_gate` override
  geçerlidir. `gating_mode` yalnız `backend="kalman"` için kabul edilir.
- Kalite CLI `--kalman-center` bağımsız panel ekler. `--kalman --kalman-center`
  dört karşılaştırmadır: IoU, iki aşamalı, FullBox Kalman, Center Kalman. Her karede
  **tek** detector sonucu ve ayrı durumlar; GT yalnız değerlendirme/çizim içindir.
  Gate mode/dimension/threshold, ham tespitler, MOT tahminleri ve süreler kaydedilir.
- İki mevcut eğitim sekansı tekrar ölçülür. Bu alternatif daha önce Campus
  hatalarından hareketle seçildiği için bu sahneler bağımsız validation/hold-out
  değildir. Eşik taraması/fine-tuning yoktur; geliştirme koşusu açıkça belirtilir.
  Genel varsayılan IoU, eski Kalman config'i ve canlı RTSP reddi korunur.

## Doğrulama ve sınırlar

Sentetik sabit merkezde genişlik/yükseklik titreşimi kimliği korumalı, aynı IoU'ya
sahip büyük merkez sıçraması reddedilmeli. Çok büyük kutu değişimi yine IoU'dan
reddedilir. Sınıf, one-to-one, düşük güven/kayıp dönüşü, aktif önceliği, kapasite
ve invalid-input rollback sözleşmeleri iki modda test edilir. Resmi TrackEval
aynı dışa aktarılan dört takipçiyi kontrol eder; önceki üç metriğin değişmemesi
ayrıca karşılaştırılır. [Sonuçlar ve tekrar üretim](../model-quality.md).

Merkez de kısmi örtüşmeyle oynayabilir; yakın/kesişen kişiler görünüş bilgisi
olmadan karışabilir. Boyut hızı predicted IoU'yu etkiler, merkez belirsizliği
hâlâ kutu boyutuyla ölçeklenir. Bu değişiklik Re-ID veya tam ByteTrack değildir.
Ayrı, daha geniş/sızıntısız veri üzerinde doğrulama ve appearance sonraki işlerdir.

Referans fikir: [Deep SORT'un konum odaklı marjinal kapısı](https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py).
Bağımsız C++ uygulamasıdır; kaynak kod kopyalanmadı. Chi-square değerleri için
[SciPy dağılım dokümanı](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html).
