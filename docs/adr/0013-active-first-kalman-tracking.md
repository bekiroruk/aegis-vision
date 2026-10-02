# ADR-0013: Aktif-önce, belirsizlik kapılı Kalman takipçisi

Durum: Kabul edildi. Tarih: 2026-10-02.

## Bağlam

Etiketli başlangıç ölçümünde mevcut iki aşamalı takipçinin IDF1'i basit IoU'dan
düşüktü. TUD-Stadtmitte kare 97'de, kare 96'da görülen aktif kimlik 3 ile son
gözlemi kare 91 olan kayıp kimlik 16 birlikte Hungarian eşleştirmesine giriyor.
Aktif kimliğin doğru kutuyla IoU'su .572, yanlış kutuyla .511; kayıp kimliğin
aynı doğru kutuyla .444, diğer kutuyla .009. Toplam ağırlık .511+.444, tek doğru
eşleşmenin .572 ağırlığından yüksek olduğu için iki yanlış kimlik eşleşmesi
seçilebiliyor. Hareket tahminini kapatmak bu örneği çözmüyor.

Yeni backend, sürekliliği koruyan eşleştirme politikası ile gürültülü hareket
gözlemlerinin kovaryanslı tahminini birlikte sunmalı. Önceki iki backend değişmeden
karşılaştırma olarak kalmalı; tek bir sekansın sonucuna göre eşik taraması yapılmamalı.

## Karar

- `KalmanTracker`, harici matematik/OpenCV bağımlılığı olmadan C++ çekirdeğinde
  eklenir. Durum `(cx,cy,w,h,vx,vy,vw,vh)`, adım bir ardışık decoded frame'dir.
  Konum/boyut ve hız belirsizliği kutu boyutuna göre ölçeklenir; bir piksel tabanı
  vardır. Init std: `2*size/20`, `10*size/160`; process std `size/20`, `size/160`;
  measurement std `size/20`. X/width bileşeni genişliğe, Y/height yüksekliğe bağlıdır.
- Eşleştirme sırası aktif/yüksek güven, kalan aktif/düşük güven, sonra kayıp/kalan
  yüksek güvendir. Böylece eski bir kayıp kimlik, hâlâ eşlenebilen aktif kimlikle
  aynı aşamada rekabet etmez. Sınıf, IoU ve squared Mahalanobis kapıları kullanılır.
  Sabit kapı `13.2767` dört ölçüm için ideal Gaussian varsayımında %99 chi-square
  quantile'dır; gerçek sahnelerde %99 doğru eşleşme garantisi değildir.
- İnovasyon Cholesky solve ile çözülür, açık inverse oluşturulmaz. Joseph formu
  ve simetrileştirme kovaryansın sayısal davranışını korur. Geçersiz prediction/
  correction bounded reset'e döner ve sayaç artar. Negatif boyut tahmininde yalnız
  o boyutun hızı durur. Filtrelenmiş/predicted kutular görünür sonuç değildir;
  yalnız gerçekten gözlenen detector kutuları döndürülür.
- Eşikler başlangıçla aynı: low .10, high .35, new .50, IoU .30, max missed 20.
  En fazla 256 saklanan track ve 512 input detection; fazla input state değişmeden
  reddedilir. Dolu kapasitede yeni doğumlar skor sırasıyla bastırılır ve sayılır;
  kullanıcı kayıtları veya mevcut kimlikler otomatik silinmez. Koordinatlar
  ±1,000,000, pozitif boyutlar en fazla 1,000,000 ile sınırlıdır.
- İsteğe bağlı `tracking.backend="kalman"` / `--tracker kalman`; varsayılan IoU
  ve eski iki aşamalı uygulama değişmez. Canlı RTSP/drop-oldest değişken zaman
  adımı gerektirdiği için bu backend stream modunda açıkça reddedilir. Henüz
  timestamp-aware dt, kamera PTS, appearance/Re-ID veya tentative tracks yoktur.
- Kalite CLI `--kalman` ile üçüncü panel/rapor ekler; bayrak olmadan iki panel
  korunur. Her karede tek detector sonucu tüm backend'lere gider. Ham person
  tespitleri audit için kaydedilir; GT hiçbir tracker'a girmez. Ek TUD-Campus
  sekansı farklı bir sahne kontrolüdür; ikisi de MOT15 training'dir, bağımsız
  hold-out test veya leaderboard değerlendirmesi değildir.

## Sonuçlar

Süreklilik hatası için deterministik regresyon testi ve sayısal/kapasite/yaşam
döngüsü testleri vardır. Kalman ile aktif-önce politikası birlikte değiştiği için
bir kalite farkı yalnız Kalman filtresine mal edilemez. Görünüş olmadan örtüşme
ve uzun kaybolmalarda doğru kimlik garantisi verilmez. Şartlara göre daha kötü
sonuç da raporlanır; varsayılan seçim ölçümle otomatik değiştirilmez.

İki gerçek sekans resmi TrackEval ile doğrulandı. Stadtmitte IDF1 %76.77,
önceki iki aşamalı %68.36, IoU %77.20; Campus'te %43.44, %68.90, %59.92.
Campus kutu genişliği değişimleri mevcut gürültü/kapı modelinde kimlik parçalanması
oluşturur; yöntem deneysel kalır. [Tam sonuç ve tekrar üretim](../model-quality.md).

Referans fikirler: [ByteTrack Kalman yaklaşımı](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/kalman_filter.py),
[DeepSORT recency cascade](https://github.com/nwojke/deep_sort/blob/master/deep_sort/linear_assignment.py).
Bu bağımsız `cx,cy,w,h` ve aktif-önce uygulaması resmi ByteTrack, DeepSORT veya
SORT değildir; ilgili projelerin kodu backend'e kopyalanmamıştır.
