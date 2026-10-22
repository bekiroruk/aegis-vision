# AegisVision vitrin ve sunum paketi

23 Ekim 2026. Bu klasörün tamamını birlikte tutun.

## En kolay gösterim

ZIP dosyasını önce bir klasöre çıkartın. `index.html` dosyasına çift tıklayın.
Tarayıcı sunumu açar. Önceki / Sonraki düğmelerini veya sağ / sol ok tuşlarını kullanın.
Sayfanın başındaki 26 saniyelik vitrin videosu üç gerçek sahneyi tek akışta gösterir.
Tek tek videolar bağlantısı kaynak/analiz karşılaştırmalarına götürür.
Bu gösterim için internet, Docker, kamera veya çalışan analiz servisi gerekmez.

`AegisVision.pptx` düzenlenebilir PowerPoint sunumudur. Konuşmacı notlarında
anlatım ve kaynaklar bulunur. Videolar PowerPoint dosyasına gömülü değildir.
Video gösterirken bu klasördeki `showcase.mp4` dosyasını veya üç ayrı videoyu açın
veya tarayıcıdaki sunum sayfasını kullanın.

## Yaklaşık üç dakikalık anlatım

1. **Proje, 20 saniye:** AegisVision yerel videolarda nesne maskesi ve takip
   çıktısı üreten bir C++20 projesi. İşlem bilgisayarda gerçekleşiyor.
2. **Akış, 35 saniye:** Video hazırlanıyor. C++ servisi işi kuyruğa alıyor.
   YOLOv8-seg maskeleri, IoU takip mekanizması video içi kimlikleri üretiyor.
   Sonuç MP4 ve CSV olarak saklanıyor.
3. **Vitrin, 30 saniye:** showcase.mp4 dosyasını oynat. Solda kaynak,
   sağda model sonucu var. Örtüşen insanların takibinde kimlik değişimi olabilir.
4. **Üç sahne, 30 saniye:** Yaya caddesi, hareketli araç kamerası ve sabit
   güvenlik kamerasında aynı işlem hattının sonuçlarını göster.
5. **Kullanım, 35 saniye:** Hazır bilgisayarda DEMOYU-AC.cmd paneli açıyor.
   VIDEO-EKLE.cmd başka video seçtiriyor. İlk 15 saniye demo için hazırlanıyor.
   Orijinal dosya korunuyor. Panelde analiz başlatılıp MP4 indirilebiliyor.
6. **Sınırlar, 30 saniye:** Bu örneklerde maskeleme CPU üzerinde çalışıyor.
   Fiziksel kamera testi kapsam dışında. Üretim onayı veya hatasız kimlik
   takibi iddiası yok. Amaç, çalışan ve gösterilebilir bir portföy demosu.

## Verilerin anlamı

İnsan sahnesi 72 kare / 7,2 saniye; trafik ve güvenlik kamerası sahneleri
120 kare / 12 saniyedir.
Bu süreler kaynak video süresidir, inference performansı değildir.
Takip kimliği sayısı benzersiz kişi/araç sayısı değildir. Kareler boyunca tekrar
tespit edilen aynı nesne birçok maske kaydı oluşturur. Sayılar doğruluk puanı değildir.

Kaynaklar: MOT15 TUD-Stadtmitte, KITTI-17 ve PETS09-S2L1.
https://motchallenge.net/data/MOT15/
Kaynak verilerden türetilmiş videoları dışarıda yayımlamadan önce kullanım koşullarını
kontrol edin. Paket model dosyalarını veya kurulabilir uygulamayı içermez.
`manifest.json` önceki analiz çıktılarının kaynağını, iş kimliklerini ve hash'lerini korur.
Paket içindeki üç ayrı video analiz servisinin comparison çıktılarıdır.

Kod: https://github.com/bekiroruk/aegis-vision
Yeni bilgisayarda uygulamayı kurmak için repodaki docs/demo.md kılavuzunu kullanın.
