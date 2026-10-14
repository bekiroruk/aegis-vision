# Gerçek iki-kamera kişi görünümü değerlendirmesi

2026-10-14. **Ölçüm aracı tamamlandı; model bu deneyde yeterli bulunmadı.**
Bu C++ değerlendirmesi, aynı anda farklı kameralarda görülen kişilerin OSNet
özelliklerini karşılaştırır. Önceki temporal `associate_cameras` algoritmasını
değiştirmez; onun çıkış–giriş başarısını ölçmez.

## Veri ve önceden sabitlenen protokol

Kaynak: [resmî EPFL WILDTRACK sayfası](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)
ve [veri setinin CVPR 2018 yayını](https://openaccess.thecvf.com/content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html).
Kameralar senkronizedir ve görüş alanları örtüşür. Dolayısıyla mevcut
**örtüşmeyen kameralar için** kapanmış-parça geçiş testinin yerine kullanılamaz.
Saatlere yapay offset eklenmedi, aynı video iki kamera diye kopyalanmadı.

- C1 sorgu, C2 galeri. Yalnız aynı anotasyon anındaki kişiler karşılaştırılır.
- 16 anotasyon: `0,125,250,...,1875`; seçim model çalıştırılmadan sabitlendi.
- Görüntüler 1920×1080. Resmî `personID` ve `viewNum=0/1` kutuları kullanılır.
- Kutu görüntü sınırlarına kırpılır; en az 20 px genişlik ve 64 px yükseklik
  gerekir. Kırpılmış/örtülmüş zor kişiler model skoruna bakılarak elenmez.
- **Ground-truth kutular kullanılıyor:** YOLO detection, yerel takip,
  kapanmış-parça prototipi veya uçtan uca MTMC metriği değildir.
- OSNet x0.25 MSMT17-combineall CPU/FP32, mevcut hash kontrollü C++ adaptörü.
  Her kutudan ayrı 512-D normalize özellik çıkarılır; bu deneyde eğitim yoktur.
- Cosine `.85`, iki yönlü belirsizlik farkı `.05`, ilk koşudan önce sabitlendi.
  Sonuçlara bakarak eşikler düşürülmedi. Gelecek ayarlarda bu altküme artık
  **görülmüş teşhis verisidir**; kör holdout diye yeniden sunulamaz.

Kişi etiketleri yalnız skor hesaplamada kullanılır; özellik modeli ve görünüm
ataması etiketleri kullanmaz. Aynı kimlik bir view içinde iki kez bulunamaz.
Senkronizasyon kaynağın beyanıdır; bu araç kameraların saatini ayrıca ölçmez.

## Sonuç: bu aday kabul edilmedi

| Ölçüm | Sonuç |
|---|---:|
| C1 sorguları | 342 |
| C2'de filtreyi geçen karşılığı bulunan sorgular | 286 |
| Karşılığı bulunmayan sorgular | 56 |
| Doğru kişi ilk sırada | 73 / 286 = **%25,52** |
| Kabul edilen doğru eşleştirme | **0** |
| Kabul edilen yanlış eşleştirme | 0 |
| Kaçırılan mevcut karşılık | **286** |
| Karşılıksız sorguyu reddetme | 56 / 56 |
| Kabul edilenlerin precision'ı | **tanımsız (`null`)** — hiç kabul yok |
| Mevcut karşılıkları kabul etme recall'ı | **%0** |

Sıfır yanlış kabul başarı anlamına gelmez: sistem bütün eşleşmeleri reddetmiştir.
En yakın aday doğruluğu da düşük olduğundan sorun yalnız kabul eşiği değildir.
Farklı bakış açısı, örtülme, kutuya başka kişilerin girmesi ve küçük görüntüler
olası etkenlerdir; bu koşu hangisinin nedensel etkisini ayrıştırmaz.
**Üretime uygun çoklu-kamera kişi tanıma tamamlandı denilemez.**

342 sorgu, 342 bağımsız kişi değildir; zaman boyunca tekrar eden kişiler vardır.
Bu küçük, tek kamera çifti altkümesinden tüm WILDTRACK, başka ortamlar veya
gerçek saha başarısı için genelleme yapılmaz. Kamera çifti/kimlik ayrılmış
geliştirme ve dokunulmamış test verisi olmadan kalibrasyon sonucu kabul edilmez.

## Metrikler

Rank-1 yalnız galeride karşılığı bulunan sorguların paydasını kullanır.
En yüksek skor eşitliği (1e-12 tolerans) doğru sayılmaz; etiket sırasıyla
tesadüfi doğru seçim ödüllendirilmez. Kabul için skor ve karşılıklı en iyi
aday koşulu yanında her iki yönde birinci/ikinci skor farkı gerekir.

`correct_accepts` doğru kabul; `wrong_accepts` karşılığı olsun/olmasın yanlış
kimliğe kabul; `missed_counterparts` mevcut doğru karşılığı kabul edememe sayısıdır.
Yanlış kimliğe kabul edilen, karşılığı olan sorgu hem yanlış kabul hem kaçırmadır.
Precision = doğru kabul / tüm kabuller; recall = doğru kabul / karşılığı olan
sorgular. Payda sıfırsa oran `null`; %100 gibi gösterilmez.

## Tekrar üretme

Python yalnız resmî veriyi hazırlar; inference ve eşleştirme C++'tır.

```powershell
python scripts/prepare_crossview_data.py artifacts/datasets/wildtrack-crossview-new --download
cmake --build build/tensorrt --config Release --target aegisvision_crossview aegisvision_crossview_tests --parallel 4
build/tensorrt/Release/aegisvision_crossview.exe artifacts/datasets/wildtrack-crossview-new/manifest.json artifacts/models/osnet-x0-25-msmt17 outputs/crossview-new
```

Çıktı dizini yeni olmalı. `report.json` ölçüm tamamlandıktan sonra atomik
yeniden adlandırmayla yayımlanır; manifest/model uzayı ve her sorgunun kararı
rapordadır. `review.jpg` ilk andaki **ilk 8 sorguyu**, en iyi adaylarını ve
red/kabul nedenlerini gösterir; en başarılı örnekler seçilmez. CLI başarı
çıkış kodu ölçümün tamamlandığını gösterir, modelin kalite onayını değil.

Yerel sonuç: `outputs/crossview-wildtrack-v1/report.json`.
Girdi manifesti SHA256:
`183f853ad20a6b9bb5c36d08275b937feae545ca42698d937d8e95587deb4e48`.

### İndirme ve veri kaydı

Resmî ZIP 6.807.496.358 byte; script yalnız gereken 32 PNG ve 16 JSON üyesini
HTTP Range ile aldı (bu koşuda toplam 80.416.391 byte). `206`, `Content-Range`
ve sabit ETag kontrol edilir; sunucu tüm arşivi döndürürse işlem durur.
Toplam 200 MiB ve istek/üye başına 16 MiB sınırı vardır. ZIP CRC kontrolüne
ek olarak indirilen dosyaların SHA256'ları `provenance.json`'da tutulur.
Eski ZIP'in 4 GiB taşan offsetleri, yerel başlık ve tam üye adı doğrulanarak
çözülür. ZIP yolları diske doğrudan açılmaz; yalnız izin verilen hedef adları yazılır.
Başarısız hazırlık kısmi klasör bırakabilir; manifest en son yazılır ve yeni
koşu için yeni klasör kullanılır. Veri ve kişi görüntüleri Git'e eklenmez.
Verinin kullanım koşulları kaynak sağlayıcıya aittir; proje lisansı veri lisansı değildir.

## Testler ve sıradaki karar

```powershell
python scripts/test_crossview_data.py
python scripts/test_crossview_cli.py build/tensorrt/Release/aegisvision_crossview.exe
ctest --test-dir build/dev -C Release --output-on-failure
```

Testsiz skor değişikliği yerine önce hazırlama ve metrik sözleşmeleri kontrol
edilir: HTTP range/kimlik/box sınırları, etiketlerin atamayı etkilememesi,
belirsizlik, boş galeri, yanlış kimlik kabulü, kötü vektör, değiştirilmiş eşik,
hash/path doğrulaması ve eski raporu koruma.

Yerel Windows kontrolleri: temel CTest 7/7, seçili görünüm/tracklet/Re-ID
CTest 7/7, hazırlama testleri 11/11 ve CLI giriş testleri 10/10 geçti.
GitHub CI'a testler eklendi; uzak CI sonucu bu yerel ölçümden çıkarılamaz.

Sıradaki model işi: ayrı geliştirme verisinde kutu/örtülme etkisini incelemek,
daha uygun kişi temsili veya metric-learning adayını karşılaştırmak, ardından
henüz görülmemiş kamera/kimlik verisinde kalibrasyon ve test. Aynı deneyin
eşiğini düşürmek bu sonucu düzeltmiş sayılmaz. Non-overlap kamera geçişi,
canlı çoklu kamera capture ve UI entegrasyonu ayrı açık işlerdir.
