# Gerçek videodan kişi takip parçaları

2026-10-14: native C++ `aegisvision_tracklets`, yerel dosyada YOLOv8n kişi
tespiti → IoU yerel takip → OSNet x0.25 özellik çıkarımı → parça prototipi üretir.
Bu adım çoklu kamera eşleştirme bileşeninin gerçek görüntü girdisini sağlar;
canlı çoklu-kamera sistemi veya etiketli Re-ID doğruluk ölçümü değildir.
Mevcut panel/servis değiştirilmedi. Runtime Python, Qdrant veya GPU gerektirmez.

## Çalıştırma

Mevcut OpenCV + search build'i ve yerel YOLO/OSNet modelleri gerekir.
Model edinme/lisans bilgisi: [OSNet](person-appearance.md).

```powershell
cmake --build build/tensorrt --config Release --target aegisvision_tracklets aegisvision_tracklets_tests --parallel 4
build/tensorrt/Release/aegisvision_tracklets.exe configs/tracklets-demo.json outputs/person-tracklets-new
```

Config yolları JSON dosyasına göre çözülür. Demo ilk 300 kareyi işler; dosyanın
tamamını işlediği veya gerçek zamanlı çalıştığı iddia edilmez. `max_frames` zorunlu
ve 1..1.000.000; FPS metadata'sı 0,001..1000 olmalı; sessiz FPS fallback yoktur.
Yeni çıktı dizini gerekir. Eski rapor üzerine yazılmaz.

- `tracked.avi`: işlenmiş tüm kareler, kişi kutuları ve kamera/yerel ID.
- `preview.jpg`: en fazla kişi gözlemi olan ilk karenin önizlemesi.
- `observations.csv`: her gözlemin karesi, göreli zamanı, ID, skor ve kutusu.
- `tracklets.json`: 512 boyutlu prototipler, kapanma nedeni, gözlem sayısı,
  uygunluk, model uzayı, video/model SHA256 ve çalıştırma ayarları.

İsteğe bağlı tarayıcıda oynatılabilir MP4 (kurulu FFmpeg, üzerine yazmaz):

```powershell
ffmpeg -n -i outputs/person-tracklets-new/tracked.avi -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart outputs/person-tracklets-new/tracked.mp4
```

Başarısız koşu kısmi AVI/CSV bırakabilir; **`tracklets.json` en son yazılır**,
tamamlanmamış çıktı association girdisi olamaz. Decode sonu ile decode hatası
OpenCV'de kesin ayrılmadığından son aktif parçalar her iki durumda da sansürlüdür.

## Takip ve kapanma sözleşmesi

YOLO CPU/FP32, 640 giriş, confidence .35, NMS .45, en fazla 300 detection;
yalnız `person` sınıfı kullanılır. Tracker bu ilk çıkarıcıda **IoU**'dur;
OSNet özellikleri yerel atamayı değiştirmez. Mevcut Kalman-ReID video CLI'ının
yerine geçmez. `update_indexed` ile her özellik tam kendi detection/ID'sine bağlanır.

Her gözlemin OSNet vektörü önce L2-normalize edilir. Parça prototipi eşit ağırlıklı
unit vektörlerin ortalamasının L2-normalizasyonudur. Sonlu olmayan, yanlış boyutlu
ve sıfır normlu özellikler hata verir; sahte feature ile devam edilmez.

- `missing_timeout`: son gözlemden sonra **max_missed_frames'den fazla** ardışık
  karede görünmeyen takip kapandı. Bitiş zamanı son gözlemindir, timeout anı değil.
- `end_of_input` / `frame_limit`: hâlâ aktif parça kesildi. **Association'a alınmaz.**
- `eligible`: timeout ile kapanmış ve en az `min_observations` gözlemi olan parça.
  Timeout fiziksel kamera çıkışını kanıtlamaz; occlusion veya ID parçalanması olabilir.
- İlk karedeki kişiler de sol taraftan kesilmiş olabilir; ilk gözlem gerçek giriş
  anı sayılmaz. Kamera geçişinin doğruluğu ayrıca etiketli veriyle ölçülmelidir.
- En fazla 512 toplam parça; kapasite aşımında sessiz silme yerine koşu hata verir.
  ID aynı epoch içinde tekrar kullanılamaz. Yeniden başlatılan kayda yeni epoch gerekir.

## Kamera eşleştirmesine aktarım

Çıkarıcı zamanları **`relative_cfr_estimate_ms`**: `round(frame_index*1000/fps)`.
PTS, duvar saati veya senkronize kamera zamanı değildir. VFR, frame drop, drift
ve farklı başlangıçlar varsa yalnız offset eklemek yeterli değildir.

İki gerçek export'un CFR zamanlarıyla ortak eksen arasındaki dönüşümün geçerli
olduğunu ayrıca doğruladıktan sonra bir saat eşleme JSON'u oluşturulur:

```json
{
  "version": 1,
  "time_basis": "synchronized_ms",
  "clock_evidence": "OPERATÖRÜN DOĞRULADIĞI SAAT EŞLEME RAPORUNUN YOLU VE ÖZETİ",
  "sources": [
    {"file": "camera-a/tracklets.json", "origin_ms": 0},
    {"file": "camera-b/tracklets.json", "origin_ms": 2500}
  ],
  "cameras": ["a", "b"],
  "transitions": [{"from":"a", "to":"b", "min_gap_ms":100, "max_gap_ms":30000}],
  "min_similarity": 0.85,
  "ambiguity_margin": 0.05
}
```

Yukarıdaki yollar/offsetler **şema örneğidir**, hazır iki-kamera verisi veya
ölçülmüş senkronizasyon değildir. Kamera adları export config'leriyle aynı olmalı.
`origin_ms` negatif olmayan ortak eksen başlangıcıdır; büyük zamanlar taşma
kontrolünden geçer. Dosya yolları saat eşleme JSON'una göredir.

```powershell
build/tensorrt/Release/aegisvision_tracklets.exe --prepare-association clock-map.json outputs/prepared-new
build/tensorrt/Release/aegisvision_multicamera.exe outputs/prepared-new/association-input.json outputs/association-new
```

Aktarım yalnız uygun kapanmış parçaları alır; karışık model uzayı, tekrar kamera,
epoch tutarsızlığı, yanlış feature, kapasite ve saat taşmasını reddeder.
`provenance.json` export SHA256'larını, origin'leri ve operatör beyanını korur.
`clock_verified_by_software=false`: yazılım beyanın doğruluğunu doğrulamaz.
`association-input.json` mevcut C++ association CLI sözleşmesine uygundur.
İki farklı kaynak dosyası sağlamak tek başına gerçek iki kamera kanıtı değildir.

## Testler ve kalan sınır

Yerel gerçek-model koşusu: `pedestrians.mp4` ilk 300 kare / 30 saniye,
768×576, 10 FPS. 1.873 kişi gözlemi, 28 yerel takip parçası; 20 timeout
kapanışının 16'sı en az üç gözlem koşulunu geçti, 4 kısa parça dışlandı.
Diğer 8 parça kare sınırında sansürlendi. Bunlar **kişi sayısı veya doğruluk
metriği değildir**. Prototiplerin tamamı 512 boyutlu; maksimum L2 norm
hatası 8,5e-9'dan küçük. Yerel çıktı `outputs/person-tracklets-v2/`;
MP4 sunum dosyası ayrıca FFmpeg ile AVI'den dönüştürülür.

Windows doğrulaması: seçili CTest 5/5, temel CTest 6/6, iki CLI aktarım testi
13/13 geçti. OSNet üç referans fixture'ında minimum cosine 1,
maksimum embedding farkı 4,992e-7. CI testleri eklendi; uzak CI sonucu
bu yerel kontrollerden çıkarılamaz.

```powershell
ctest --test-dir build/tensorrt -C Release -R "aegisvision_(tracklets|multicamera|reid)" --output-on-failure
python scripts/test_tracklets_cli.py build/tensorrt/Release/aegisvision_tracklets.exe build/tensorrt/Release/aegisvision_multicamera.exe
```

Collector testleri timeout sınırı, son gözlem zamanı, normalize ortalama, censored
kapanış, ID tekrarları, kapasite, boş kare ve kötü özellikleri kapsar.
Bridge testleri gerçek iki CLI'ı **sentetik** exportlarla birbirine bağlar;
bu testler model doğruluğunu ölçmez. Etiketli senkronize çoklu kamera veriyle
geliştirme/holdout ölçümü, eşik kalibrasyonu ve canlı/UI entegrasyonu açık iştir.
İlk [gerçek iki-view görünüm teşhisi](crossview-evaluation.md) eklendi; GT kutularda
Rank-1 %25,52 ve sıfır kabul ile mevcut model/eşik uygun bulunmadı. Tracklet
geçiş doğruluğu değil, kişi temsili için ayrı bir teşhis deneyidir.
Kişi özelliklerini içeren gerçek çıktılar yerelde `outputs/` altında, Git dışında tutulur.
