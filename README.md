# An Artificial Intelligence Based Agronomist

> **Bachelor's Final Year Project** — Electrical Engineering  
> Sukkur IBA University, Pakistan | 2022  
> Supervised by: **Dr. Gulsher Ali Lund Baloch**  
> Team: Ghulam Sarwar · Fatima Zehra Alvi · Saeed Akbar Khan

---

## 🏆 Awards & Recognition

| Competition | Organiser | Result |
|---|---|---|
| DICE AFS 2021 | MNS University, Multan | 🥇 **Winner — PKR 1,00,000** |
| Hult Prize 2022 | Sukkur IBA University | 🥉 2nd Runner-up |
| Lab2Market | PAF IAST Hari Pur | 🏅 Top 5 Teams |
| Ideagist Pakistan 2021 | National | City Round Qualifier |

---

## Overview

**Agronomist** is an Android mobile application that identifies plant species from a photo and instantly provides detailed information — including the plant's name, growing season, region, and care instructions. Built to help everyday people, farmers, and home gardeners learn about plants without needing an expert.

> Motivation: During the pandemic, people were focused on home gardening but had no easy way to identify or learn about plants. Existing apps provided region-limited solutions. Agronomist was built to serve Pakistani users with locally relevant plant data.

---

## App Features

- 📸 Take or select a photo of any plant
- 🌿 Instant plant identification using a CNN model
- 📋 Detailed plant info — name, season, region, description, growing tips
- 🌍 Pakistan-focused regional plant data
- 🔒 Secure — HTTPS with CIA Triad (Confidentiality, Integrity, Availability via Azure)
- 🇵🇰 Bilingual support planned (English + Urdu)

**Example output from the app:**
```
Plant:       Petunia Titan Giant White
Season:      Spring
Region:      Pakistan
Description: Petunias like fertile soil which drains well and is neutral
             to slightly acid (pH 6.0 to 7.0). Light, sandy soil is ideal...
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile Frontend | Flutter (Dart) |
| AI Backend | Python Flask |
| CNN Model | TensorFlow / Keras |
| Server Hosting | Microsoft Azure |
| Security | HTTPS / TLS 1.2 (CIA Triad) |

---

## Dataset

Images collected manually using mobile phones at different times and angles.

| Property | Details |
|---|---|
| Total flower classes | 20 species |
| Images per class | ~230 images |
| Total images | ~4,600 images |
| Pre-processing | Manual cropping to remove background |
| Split per class | Train: 160 · Validation: 30 · Test: 10 |

---

## Model Training — Three Experiments

### Experiment 1 — VGG-16, uncropped data, 10 classes
- Pre-trained VGG-16 (13 conv + 13 MaxPool + 3 Dense)
- Adam lr=0.0001, 20 epochs
- **Result: 47% accuracy** — poor due to noisy uncropped images

### Experiment 2 — Custom CNN, cropped data, 10 classes ⭐ Best Result
- 3× Conv2D (128, 512, 1024 filters, 3×3, ReLU) + BatchNorm + MaxPool (3×3) + Dropout (0.4)
- Flatten → Dense 512 (ReLU) → Dense 10 (Softmax)
- Adam lr=0.00001, 30 epochs
- **Result: 97.6% training accuracy, 98% validation accuracy**

### Experiment 3 — Custom CNN, cropped data, 20 classes
- 2× Conv2D (32, 64 filters, 3×3, ReLU) + MaxPool (3×3) + Dropout (0.6)
- Flatten → Dense 512 (ReLU) → Dense 20 (Softmax)
- Adam lr=0.00001, 20 epochs
- **Result: 92% training, 91% validation accuracy**

> **Key learning:** Manually cropping images to remove background noise was the single biggest factor — it boosted accuracy from 47% to 97.6%.

![Model Architecture](images/model_architecture.png)

---

## System Pipeline

```
Mobile photo (Flutter app)
        │
        ▼  HTTPS request
Python Flask API (Azure server)
        │
        ▼
Image pre-processing (resize 150×150, normalise /255)
        │
        ▼
Custom CNN inference
        │
        ▼  JSON response
Flutter app displays:
  Plant name · Season · Region · Description
```

---

## Project Structure

```
AI-Agronomist-FYP/
├── src/
│   ├── train_vgg16.py           ← Experiment 1: VGG-16 transfer learning
│   ├── train_custom_cnn.py      ← Experiment 2 & 3: Custom CNN (best model)
│   └── predict.py               ← Inference script
├── models/
│   └── class_names.json         ← Flower class label mapping
├── images/
│   └── model_architecture.png   ← CNN architecture diagram
├── docs/
│   ├── FYP_presentation.pdf     ← Final FYP presentation (21 slides)
│   └── Poster.pdf               ← DICE AFS 2021 winning poster
├── app/
│   └── README.md                ← Flutter app + Flask backend details
└── requirements.txt
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare dataset folders
# dataset/train/<classname>/ and dataset/val/<classname>/

# Train best model (Custom CNN, 10 classes)
python src/train_custom_cnn.py

# Predict on a new image
python src/predict.py --image rose.jpg --model custom_cnn
```

---

## Future Work

- Plant disease detection and crop yield prediction
- Full Urdu language support
- Expansion to national-level plant database
- Collaboration with NGOs and Pakistan agriculture departments
- Applied for ISF World Bank grant through HEC Pakistan

---

## Team

| Name | University |
|---|---|
| **Ghulam Sarwar** | Sukkur IBA University |
| Fatima Zehra Alvi | Sukkur IBA University |
| Saeed Akbar Khan | Sukkur IBA University |

Supervisor: **Dr. Gulsher Ali Lund Baloch**

---

## License

MIT © Ghulam Sarwar, Fatima Zehra Alvi, Saeed Akbar Khan — 2022
