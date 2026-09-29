# ADR-0002: İsteğe bağlı OpenCV görüntü adaptörü

- Durum: Kabul edildi
- Tarih: 2026-09-30

Çekirdek C++ kütüphanesi bağımlılıksız kalır. `AEGISVISION_WITH_OPENCV=ON` ayrı
`aegisvision_opencv` kütüphanesini ve `aegisvision_image` aracını etkinleştirir.
`cv::Mat` yalnızca bu sınırda kullanılır; mevcut demo `Frame` modeline gizli piksel
varsayımları eklenmez. Gerçek detector adaptörü eklenirken görüntü taşıma sözleşmesi
ayrıca tanımlanacaktır.

Eşleştirme ORB + Hamming KNN (k=2), oran testi, tekil hedef eşleşmeleri ve RANSAC
homografisinden oluşur. Dönüşüm açıkça SOURCE → TARGET yönündedir. Yetersiz özellik,
eşleşme veya inlier durumunda hata döner. Rapor inlier sayısı/oranı ve yeniden izdüşüm
RMSE'sini içerir. Bu metrikler sahneye bağımlıdır; gerçek veri değerlendirmesinin yerine geçmez.

Yerel test deterministik desen üzerinde bilinen perspektif dönüşümünü uygulayıp
tahmin edilen dönüşümün kontrol noktalarını karşılaştırır. Windows'ta Unicode dosya
yolları için ikili dosya akışı + imdecode/imencode kullanılır. DLL'ler executable
yanına kopyalanır; sistem PATH'i değiştirilmez.
