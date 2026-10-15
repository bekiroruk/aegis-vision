# Daha büyük kişi Re-ID modeli karşılaştırması

Tarih: 2026-10-15. Resmî [OSNet model deposundaki](https://huggingface.co/kaiyangzhou/osnet)
MSMT17-combineall **x1.0** ağırlıkları, mevcut **x0.25** ile aynı veride karşılaştırılır.
Bu hazır eğitilmiş model karşılaştırmasıdır; projeye özgü fine-tuning/metric learning
değildir. Çalışan servis ve tracklet konfigürasyonunun varsayılanı değiştirilmez.

## Geliştirme sonucu

Aynı C3→C4, çift kişi kimlikli 39 kare; 304 sorgu, 94 karşılığı olan sorgu.
Her iki model tek görüntü kullanır, flip ortalaması yoktur. Eşik `.85`, margin
`.05` sabittir. GT kutular kullanılır; detection/tracking/handoff ölçülmez.

| Ölçüm | x0.25 | x1.0 |
|---|---:|---:|
| Doğru ilk aday | 59 / 94 | 66 / 94 |
| Rank-1 | %62,77 | %70,21 |
| Kabul edilen doğru / yanlış eşleşme | 0 / 0 | 0 / 0 |
| Kaba ortalama özellik süresi | 21,16 ms | 67,76 ms |
| ONNX boyutu | 891.011 bayt | 8.728.952 bayt |

**+7,45 yüzde puan** geliştirme kazancı, önceden belirlenmiş +2 puan kapısını
geçti. Ancak kabul recall'ı hâlâ sıfır; precision tanımsızdır. Sıfır yanlış
kabul, hiç kabul yokken güvenilir çalışan eşleştirme anlamına gelmez.
Süreler ayrı koşularda gözlenen yaklaşık maliyettir (~3,20×), kontrollü
p50/p95 benchmark değildir. x0.25 raporu önceki deneyden aynen kullanılmıştır.

## Ayrılmış C5/C6 son kontrolü: kazanç var, üretim terfisi yok

Önceden ayrılmış 10 kare çiftinde 54 sorgunun tamamının galeride karşılığı
vardı; toplam 166 özellik çıkarıldı. İki model aynı karelerde birer kez çalıştı:

| Ölçüm | x0.25 | x1.0 |
|---|---:|---:|
| Doğru ilk aday | 22 / 54 | 27 / 54 |
| Rank-1 | %40,74 | %50,00 |
| Kabul edilen doğru / yanlış eşleşme | 0 / 0 | 0 / 0 |
| Kabul recall'ı | %0 | %0 |

İlk aday doğruluğu **+9,26 yüzde puan** arttı; yine de aday sorguların yarısında
yanlış kişiyi ilk sıraya koydu ve hiçbir eşleşmeyi kabul etmedi. Bu nedenle
**x1.0 araştırma adayı olarak tutuldu, çalışan varsayılan x0.25 korundu**.
Bu son kontrolde karşılığı olmayan sorgu bulunmadığından unknown-rejection
başarısı değerlendirilemez. Sıfır kabul nedeniyle precision da hesaplanamaz.
Geliştirme ve son kontrol yüzdeleri farklı galeriler nedeniyle birbirleriyle
doğrudan kıyaslanmaz; yalnız her kümedeki iki modelin farkı anlamlıdır.

Sonraki [32-ayarlı geliştirme eşik kalibrasyonu](reid-calibration.md) tamamlandı;
hiçbir ayar yanlış-kabul/recall hedeflerini birlikte geçmedi. Bu C5/C6 sonuçları artık
görülmüştür; yeni ayarlar için dokunulmamış bir son test gerekir. Gerekirse
ayrı eğitim kimliklerinde metric-learning yapılır. Kamera saatleri ve gerçek
tracklet geçişleri ölçülmeden canlı çoklu-kamera desteği tamamlanmış sayılmaz.

Yerel sonuçlar:

- `outputs/reid-model-holdout-baseline-v1/report.json`, SHA256 `4b3dac7f85ee8ff92537ff2d52af3e104660a9037486a9f87968ceba8dd7a9b7`.
- `outputs/reid-model-holdout-candidate-v1/report.json`, SHA256 `dbf0e365f46a2736221584d2ce1398e17b587a4a0c8b84cbb9b84f6c36343e25`.
- `artifacts/datasets/wildtrack-model-holdout-v2/manifest.json`, SHA256 `c4bb921660d1a7b0b781e4366b07329c8b331dad86a5cea5967b5679ae3981b8`.

Toplam 46.460.687 bayt indirildi. C5/C6 seçim verisine karıştırılmadı;
model/eşik değiştirilmedi veya son kontrol üstünde arama yapılmadı.

## Sabit protokol ve sınırlar

`configs/reid-model-experiment.json`, aday geliştirme inference'ından önce
yazıldı. Model/dataset SHA256, en az 30 karşılıklı sorgu, +2 puan Rank-1 ve
yanlış kabullerde artış olmaması koşulları sabitlendi. Seçici aynı model ile
augmentation deneyini ve farklı sabit modellerle `single` karşılaştırmasını
ayrı doğrular; karışmış ağırlık, veri, varyant veya holdout raporunu reddeder.

Önceden ayrılmış son kontrol: C5→C6, tek kişi kimlikleri, 60..1860 arasında
200 adımla 10 kare. Yalnız geliştirme kapısı geçildikten sonra indirilir ve
iki model birer kez ölçülür. Otomatik üretim terfisi yapılmaz. Geliştirme
verisi önceki flip deneyinde görülmüştür. Son kontrol aynı kayıttandır;
bazı kimlikler eski C1/C2 teşhisinde görülmüş olabilir. Tamamen yeni kişi,
yeni saha veya uçtan uca canlı başarı iddiası yoktur. Tekrarlanan kişiler
nedeniyle sorgular bağımsız örnekler sayılmaz.

## Paket ve güvenli yükleme

- Checkpoint: `osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`.
- Resmî depo revizyonu: `a5c5cc037c24235cda3b21085b93ad77c9616224`.
- Checkpoint SHA256: `48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727`; tam 17.273.805 bayt.
- Export ONNX SHA256: `5bf5321c31d064507acaec3aefecf3381b24673123c61403ceab491f49f9dc6a`.
- Geliştirme öncesi politika SHA256: `d7ca7134af72aaf53664ee64e60da51a0268dae53fe18d5fd30ec056c0bf2b0f`.

İndirme en fazla 32 MiB ve ayrıca tam dosya boyutu/SHA256 ile sınırlıdır.
Yalnız SHA-sabitlenmiş MIT tensor sınıfları ve seçili fabrika çalıştırılır;
upstream paket, indirme veya unrestricted pickle loader çalıştırılmaz.
`torch.load(weights_only=True)` zorunludur. Bilinen 4.101 sınıflı eğitim
başlığı dışında eksik/fazla/misshaped backbone tensoru kabul edilmez.
565 backbone tensoru yüklendi. Native taraf mimari-checkpoint-boyut-hash
eşleşmesini doğrular; x0.25 metadata'sını x1.0 gibi sunmak reddedilir.

Üç golden girdide PyTorch/native C++ karşılaştırması geçti: maksimum özellik
farkı 1,22935e-6; giriş tensoru farkı 0. Eski modelin golden testi de geçti.
Bu kontroller uygulama eşliğini gösterir, saha doğruluğunu değil. Model hash'i
embedding uzayına dahil olduğundan iki modelin vektörleri aynı galeriye karışmaz.

## Tekrar üretme

Önceden kurulmuş yerel hazırlama ortamında (ilk export Python; inference C++):

```powershell
.\work\yolo-export\Scripts\python.exe scripts/export_reid.py artifacts/models/osnet-x1-new --architecture osnet_x1_0 --reference-video artifacts/datasets/mot15-tud/TUD-Stadtmitte-raw.mp4 --reference-manifest artifacts/datasets/mot15-tud/quality-manifest.json
cmake --build build/tensorrt --config Release --target aegisvision_crossview aegisvision_reid_tests --parallel 4
build/tensorrt/Release/aegisvision_reid_tests.exe --bundle artifacts/models/osnet-x1-new
build/tensorrt/Release/aegisvision_crossview.exe artifacts/datasets/wildtrack-flip-dev-v2/manifest.json artifacts/models/osnet-x1-new outputs/reid-model-dev-new
python scripts/select_reid_flip.py configs/reid-model-experiment.json outputs/reid-flip-dev-baseline-v1/report.json outputs/reid-model-dev-new/report.json outputs/reid-model-selection-new
```

Yalnız seçim kaydı `candidate_selected: true` ise ayrılmış son kontrol:

```powershell
python scripts/prepare_crossview_data.py artifacts/datasets/reid-holdout-new --download --profile holdout
build/tensorrt/Release/aegisvision_crossview.exe artifacts/datasets/reid-holdout-new/manifest.json artifacts/models/osnet-x0-25-msmt17 outputs/reid-holdout-baseline-new
build/tensorrt/Release/aegisvision_crossview.exe artifacts/datasets/reid-holdout-new/manifest.json artifacts/models/osnet-x1-new outputs/reid-holdout-candidate-new
```

Son kontrol raporları seçim aracına verilmez; sabit model ve eşikler arasında
tek değerlendirmedir, eşik arama verisi değildir.

Seçici model hash'lerini bilerek sabitler. Başka exporter sürümü farklı ONNX
üretirse politika sessizce değiştirilmez; yeni deney ayrı politika ile başlar.
Önceki veri yoksa [hazırlama talimatı](reid-flip-experiment.md#tekrar-üretme)
izlenir; baseline de aynı C++ aracında yeniden üretilir. Her çıktı dizini yeni
olmalıdır. Fotoğraflar, kişi vektörleri ve model ağırlıkları Git'e gönderilmez.

Yerel geliştirme kanıtları: `outputs/reid-model-dev-candidate-v1/report.json`,
`outputs/reid-model-selection-v1/selection.json`. Baseline:
`outputs/reid-flip-dev-baseline-v1/report.json`.
Aday rapor SHA256: `570596372375b0c7cca1552dad671428954bfecc12859009e011815d11ca6e5a`.
Yeni native adaptörle eski model yeniden çalıştırıldı: 304 sorgunun bütün
skorları/kararları ve embedding uzayı aynen korundu. Kontrol raporu:
`outputs/reid-model-baseline-regression-v1/report.json`.

### Eski ZIP arşivinde sınır düzeltmesi

İlk C5/C6 indirmesi `Overlapped entries` ile güvenli biçimde durdu;
`wildtrack-model-holdout-v1` kısmi kaldı ve geçerli manifest yayımlanmadı.
Arşivdeki ZIP32 adres taşması, Python'un sıralı ham adreslerden hesapladığı
sonraki-dosya sınırına sahte girişler yerleştiriyordu. C6/60 gerçek başlığı
5.001.379.603, sıkıştırılmış veri boyutu 2.493.115 bayt iken otomatik sınır
5.002.142.689 idi; doğrulanmış sonraki C6/65 başlığı 5.003.872.827'dir.

Hazırlayıcı artık sonraki **tam adı doğrulanmış fiziksel ZIP başlığını** en
fazla 32 adayla arar ve overlap kontrolüne bu sınırı verir. Overlap/CRC,
HTTPS Range/ETag ve indirme bütçesi kontrolleri korunur. Sahte sınır,
gerçek çakışma, bozuk CRC, geçersiz merkez dizin ve sınırlı arama testleri
eklendi (hazırlayıcı 15/15). Yeniden indirme ayrı `v2` dizinine yapılır;
eski kısmi çıktı üzerine yazılmaz. Bu düzeltme kişi/veri seçim kuralını değiştirmez.

Kontroller: export hazırlayıcısı 15/15, seçim 23/23, crossview CLI 12/12,
Release temel CTest 7/7 ve seçili CTest 7/7 geçti. İlk Debug CTest denemesinde
beş executable bulunamadı; bu testler mevcut Release yapısında başarıyla
çalıştırıldı. Uzak CI sonucu yerel testlerden çıkarılamaz.
