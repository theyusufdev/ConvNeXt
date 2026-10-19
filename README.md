# 🌊 ConvNeXt ile Deniz Canlıları Sınıflandırma
### *Deep Learning Marine Creature Image Classification with ConvNeXt-Large*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15%2B-orange)](https://pytorch.org/vision/stable/index.html)
[![Model](https://img.shields.io/badge/Backbone-ConvNeXt--Large-success)](#-model-mimarisi)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-90.32%25-brightgreen)](#-eğitim-ve-performans-sonuçları)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Bu proje, modern konvolüsyonel sinir ağı mimarisi olan **ConvNeXt-Large** kullanarak 23 farklı deniz canlısı sınıfını yüksek doğrulukla sınıflandıran uçtan uca bir **Transfer Learning (Aktarımlı Öğrenme)** derin öğrenme çalışmasıdır.

---

## 📌 İçindekiler
- [Proje Genel Bakış](#-proje-genel-bakış)
- [Öne Çıkan Özellikler](#-öne-çıkan-özellikler)
- [Veri Kümesi](#-veri-kümesi)
- [Model Mimarisi](#-model-mimarisi)
- [Eğitim ve Performans Sonuçları](#-eğitim-ve-performans-sonuçları)
- [Kurulum ve Gereksinimler](#-kurulum-ve-gereksinimler)
- [Kullanım Rehberi](#-kullanım-rehberi)
- [Proje Dizin Yapısı](#-proje-dizin-yapısı)
- [Gelecek Geliştirmeler](#-gelecek-geliştirmeler)
- [Lisans](#-lisans)

---

## 🔍 Proje Genel Bakış

Geleneksel CNN'lerin ve modern Vision Transformer (ViT) mimarilerinin tasarım felsefelerinden esinlenerek geliştirilen **ConvNeXt**, saf konvolüsyonel bloklarla transformatör düzeyinde doğruluk ve ölçeklenebilirlik sunar.

Bu projede ImageNet üzerinde önceden eğitilmiş (pre-trained) `ConvNeXt_Large` modeli baz alınmış, omurga (feature extractor) dondurularak sınıflandırma başlığı **23 deniz canlısı sınıfına** uyarlanmıştır. Yalnızca 10 epoch'luk eğitim ile **%90.32 test doğruluğuna** ulaşılmıştır.

---

## ✨ Öne Çıkan Özellikler

- 🚀 **State-of-the-Art Backbone**: Torchvision `ConvNeXt_Large_Weights.DEFAULT` önceden eğitilmiş ağırlıkları.
- ⚡ **Hızlı ve Verimli Eğitim**: Dondurulmuş katmanlar sayesinde 197M parametrenin yalnızca son katmanı (~38.4K parametre) eğitilerek GPU/CPU kaynakları optimize edilmiştir.
- 🎯 **Yüksek Başarım**: Test veri setinde 10 epoch sonucunda **%90.32 test doğruluğu**.
- 🛠 **Otomatik Önişleme**: Mimariye özgü `auto_transforms` entegrasyonu ile doğru normalize ve boyutlandırma (224x224).
- 🖼 **Inference Desteği**: Tekil test fotoğrafları ile anında tahminleme yapabilme altyapısı.

---

## 🗂 Veri Kümesi

Veri seti `train` ve `test` olmak üzere iki ana bölümden oluşmakta ve toplamda **23 sınıf** içermektedir:

| # | Sınıf Adı | Açıklama | # | Sınıf Adı | Açıklama |
|---|-----------|----------|---|-----------|----------|
| 1 | **Clams** | İstiridye / Midye | 13 | **Puffers** | Balon Balığı |
| 2 | **Corals** | Mercanlar | 14 | **Sea Rays** | Vatoz |
| 3 | **Crabs** | Yengeç | 15 | **Sea Urchins** | Deniz Kestanesi |
| 4 | **Dolphin** | Yunus | 16 | **Seahorse** | Deniz Atı |
| 5 | **Eel** | Müren / Yılan Balığı | 17 | **Seal** | Fok |
| 6 | **Fish** | Çeşitli Balıklar | 18 | **Sharks** | Köpekbalığı |
| 7 | **Jelly Fish**| Denizanası | 19 | **Shrimp** | Karides |
| 8 | **Lobster** | Istakoz | 20 | **Squid** | Kalamar |
| 9 | **Nudibranchs** | Deniz Tavşanı | 21 | **Starfish** | Denizyıldızı |
| 10 | **Octopus** | Ahtapot | 22 | **Turtle_Tortoise** | Deniz Kaplumbağası |
| 11 | **Otter** | Su Samuru | 23 | **Whale** | Balina |
| 12 | **Penguin** | Penguen | | | |

---

## 🏗 Model Mimarisi

Transfer Learning yaklaşımında `ConvNeXt-Large` omurgası korunmuş, sınıflandırma katmanı `23` çıkış için yeniden tanımlanmıştır:

```text
Giriş Görüntüsü (3x224x224)
         │
         ▼
┌───────────────────────────────────────────┐
│     ConvNeXt-Large Feature Extractor      │  ◄── Donduruldu (requires_grad = False)
│  (CNBlocks, LayerNorm2d, Downsampling...) │      ~196.2 M Parametre
└───────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│           AdaptiveAvgPool2d (1x1)         │
└───────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│       Özelleştirilmiş Sınıflandırıcı       │  ◄── Eğitilebilir (Trainable)
│  Linear(in_features=1536, out_features=23)│      38,423 Parametre
└───────────────────────────────────────────┘
         │
         ▼
23 Sınıf İçin Olasılık Dağılımı (Softmax / CrossEntropy)
```

### Parametre İstatistiği

| Parametre Türü | Adet |
|----------------|------|
| **Toplam Parametre Sayısı** | `196,265,687` (~196.2M) |
| **Eğitilen (Trainable) Parametre** | `38,423` (%0.02) |
| **Dondurulan (Non-trainable) Parametre** | `196,227,264` |

---

## 📊 Eğitim ve Performans Sonuçları

- **Kayıp Fonksiyonu:** Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Optimizasyon:** Adam Optimizer (`lr = 0.001`)
- **Batch Boyutu:** 32
- **Epoch:** 10

### Epoch Bazlı İlerleme Tablosu

| Epoch | Train Loss | Train Acc (%) | Test Loss | Test Acc (%) |
|:-----:|:----------:|:-------------:|:---------:|:------------:|
| **1** | 0.7737 | %81.52 | 0.4309 | %87.29 |
| **2** | 0.4150 | %88.01 | 0.3891 | %88.89 |
| **3** | 0.3514 | %89.84 | 0.3705 | %89.27 |
| **4** | 0.3175 | %90.74 | 0.3571 | %89.95 |
| **5** | 0.2952 | %91.45 | 0.3600 | %89.52 |
| **6** | 0.2698 | %92.37 | 0.3637 | %89.82 |
| **7** | 0.2562 | %92.43 | 0.3580 | %89.98 |
| **8** | 0.2418 | %93.03 | 0.3575 | %89.84 |
| **9** | 0.2298 | %93.05 | 0.3552 | %90.31 |
| **10** | **0.2163** | **%93.42** | **0.3559** | **%90.32** |

> 🎯 **Sonuç:** Model, ezberleme (overfitting) yaşamadan test setinde dengeli bir biçimde **%90.32** doğruluğa ulaşmıştır.

---

## 💻 Kurulum ve Gereksinimler

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/kullanici_adiniz/ConvNeXt.git
cd ConvNeXt
```

### 2. Sanal Ortam Oluşturun ve Aktive Edin
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Gerekli Paketleri Yükleyin
```bash
pip install torch torchvision torchinfo pillow matplotlib
```

*(CUDA destekli PyTorch için sisteminize uygun komutu [PyTorch Resmi Sayfası](https://pytorch.org/get-started/locally/)'ndan seçebilirsiniz.)*

---

## 🚀 Kullanım Rehberi

### Jupyter Notebook ile Çalıştırma
Tüm veri yükleme, model kurulumu, eğitim döngüsü ve tahmin adımları `model.ipynb` dosyasında yer almaktadır:
```bash
jupyter notebook model.ipynb
```

### Örnek Görsel ile Tahmin (Inference)
Proje kök dizinindeki veya harici bir görüntü dosyasını sınıflandırmak için:

```python
import torch
import torchvision
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Ağırlıkları ve dönüşümleri yükleyin
weights = torchvision.models.ConvNeXt_Large_Weights.DEFAULT
auto_transforms = weights.transforms()

# 2. Test görselini dönüştürün
image_path = "dolphin.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = auto_transforms(image).unsqueeze(0).to(device)

# 3. Tahmin üretin
model.eval()
with torch.inference_mode():
    outputs = model(input_tensor)
    predicted_idx = outputs.argmax(dim=1).item()

print(f"Tahmin Edilen Sınıf: {class_names[predicted_idx]}")
```

---

## 📁 Proje Dizin Yapısı

```plaintext
ConvNeXt/
│
├── deniz_canlilari/             # Veri kümesi dizini
│   ├── train/                   # 23 sınıflı eğitim verisi
│   └── test/                    # 23 sınıflı test/doğrulama verisi
│
├── model.ipynb                  # Model eğitimi, metrikler ve görselleştirme notebook'u
├── main.py                      # Giriş / yardımcı betik
├── clams.jpg                    # Örnek test görseli
├── crabs.jpg                    # Örnek test görseli
├── dolphin.jpg                  # Örnek test görseli
└── README.md                    # Proje dokümantasyonu
```

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak sunulmaktadır.
Dilediğiniz gibi kullanabilir, geliştirebilir ve paylaşabilirsiniz.
