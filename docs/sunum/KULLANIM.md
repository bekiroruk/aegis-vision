# AegisVision sunum paketi

19 Ekim 2026. Bu klasörün tamamını birlikte tutun.

## En kolay gösterim

ZIP dosyasını önce bir klasöre çıkartın. `index.html` dosyasına çift tıklayın.
Tarayıcı sunumu açar. Önceki / Sonraki düğmelerini veya sağ / sol ok tuşlarını kullanın.
Demo videoları bağlantısı, iki gerçek karşılaştırma videosuna götürür.
Bu gösterim için internet, Docker, kamera veya çalışan analiz servisi gerekmez.

`AegisVision.pptx` düzenlenebilir PowerPoint sunumudur. Konuşmacı notlarında
anlatım ve kaynaklar bulunur. Videolar PowerPoint dosyasına gömülü değildir.
Video gösterirken bu klasördeki `insanlar.mp4` ve `trafik.mp4` dosyalarını açın
veya tarayıcıdaki sunum sayfasını kullanın.

## Yaklaşık üç dakikalık anlatım

1. **Proje, 20 saniye:** AegisVision yerel videolarda nesne maskesi ve takip
   çıktısı üreten bir C++20 projesi. İşlem bilgisayarda gerçekleşiyor.
2. **Akış, 35 saniye:** Video hazırlanıyor. C++ servisi işi kuyruğa alıyor.
   YOLOv8-seg maskeleri, IoU takip mekanizması video içi kimlikleri üretiyor.
   Sonuç MP4 ve CSV olarak saklanıyor.
3. **İnsan sahnesi, 30 saniye:** insanlar.mp4 dosyasını oynat. Solda kaynak,
   sağda model sonucu var. Örtüşen insanların takibinde kimlik değişimi olabilir.
4. **Trafik sahnesi, 30 saniye:** trafik.mp4 dosyasını oynat. Aynı işlem hattı
   hareketli araç kamerasından gelen farklı bir görüntüde de çalışıyor.
5. **Kullanım, 35 saniye:** Hazır bilgisayarda DEMOYU-AC.cmd paneli açıyor.
   VIDEO-EKLE.cmd başka video seçtiriyor. İlk 15 saniye demo için hazırlanıyor.
   Orijinal dosya korunuyor. Panelde analiz başlatılıp MP4 indirilebiliyor.
6. **Sınırlar, 30 saniye:** Bu örneklerde maskeleme CPU üzerinde çalışıyor.
   Fiziksel kamera testi kapsam dışında. Üretim onayı veya hatasız kimlik
   takibi iddiası yok. Amaç, çalışan ve gösterilebilir bir portföy demosu.

## Verilerin anlamı

İnsan sahnesi 72 kare / 7,2 saniye; trafik sahnesi 120 kare / 12 saniyedir.
Bu süreler kaynak video süresidir, inference performansı değildir.
Takip kimliği sayısı benzersiz kişi/araç sayısı değildir. Kareler boyunca tekrar
tespit edilen aynı nesne birçok maske kaydı oluşturur. Sayılar doğruluk puanı değildir.

Kaynaklar: MOT15 TUD-Stadtmitte ve KITTI-17.
https://motchallenge.net/data/MOT15/
Kaynak verilerden türetilmiş videoları dışarıda yayımlamadan önce kullanım koşullarını
kontrol edin. Paket model dosyalarını veya kurulabilir uygulamayı içermez.
`manifest.json` önceki analiz çıktılarının kaynağını, iş kimliklerini ve hash'lerini korur.
Paket içindeki insanlar.mp4 / trafik.mp4, bu manifestteki iki comparison dosyasının aynısıdır.

Kod: https://github.com/bekiroruk/aegis-vision
Yeni bilgisayarda uygulamayı kurmak için repodaki docs/demo.md kılavuzunu kullanın.
