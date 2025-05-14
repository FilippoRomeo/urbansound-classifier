# 🎧 Urban Sound Classifier – CNN with Mel Spectrograms

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify urban sounds using the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset. It follows the official **10-fold cross-validation protocol** using **Mel spectrograms** as input features.

---

## 🗂 Dataset Overview

**UrbanSound8K** contains 8732 short audio clips (≤ 4s), categorized into 10 common urban sound classes:

- air_conditioner
- car_horn
- children_playing
- dog_bark
- drilling
- engine_idling
- gun_shot
- jackhammer
- siren
- street_music

Each clip is assigned to one of **10 predefined folds**, which must be used for valid cross-validation.

---

## 📁 Project Structure

```
urban_sound_classifier/
├── config.py                # Global settings and paths
├── download_dataset.py      # Downloads UrbanSound8K using soundata
├── preprocess_audio.py      # Converts .wav → Mel spectrograms (.npy)
├── dataset.py               # PyTorch Dataset with fold support
├── model.py                 # CNN architecture
├── train.py                 # 10-fold cross-validation training
├── evaluate.py              # Evaluation + metrics for individual folds
├── checkpoints/             # Saved models (1 per fold)
├── features/                # Precomputed spectrograms
└── UrbanSound8K/            # Raw dataset audio + metadata
```

---

## 🧠 Pipeline Overview

```
.wav → Mel Spectrogram (.npy)
     ↓
CNN Classifier (fold-wise)
     ↓
Model per Fold → Evaluation → Confusion Matrix + Report
```

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install torch librosa soundata scikit-learn matplotlib seaborn
```

---

### 2. Download Dataset

```bash
python download_dataset.py
```

This uses `soundata` to download and validate the UrbanSound8K dataset into the `UrbanSound8K/` folder.

---

### 3. Preprocess Audio

```bash
python preprocess_audio.py
```

This converts each `.wav` file into a **log-scaled Mel spectrogram** and saves it as a `.npy` array under `features/{class}/{clip_id}.npy`.

---

### 4. Train the Classifier

```bash
python train.py
```

This performs **10-fold cross-validation**, training a separate model for each fold and saving them under `checkpoints/`.

---

### 5. Evaluate a Fold

```bash
python evaluate.py --fold 1
```

Generates:
- Accuracy
- Classification report
- Confusion matrix heatmap

---

## 📊 Example Results (Fold 1)

```
✅ Accuracy for Fold 1: 0.6667

🧾 Classification Report:
air_conditioner     f1-score: 0.35
car_horn            f1-score: 0.97
children_playing    f1-score: 0.82
...
macro avg           f1-score: 0.69
```

---

## ✅ Key Features

- ✅ Uses **official 10-fold split** — no data leakage
- ✅ **Mel spectrogram preprocessing** via librosa
- ✅ Clean, modular PyTorch codebase
- ✅ Per-fold **model saving** and evaluation
- ✅ **Confusion matrix visualization**
- ⚙️ Ready for model upgrades (ResNet, AST, transformers)

---

## 🧠 Ideas for Expansion

- Add **SpecAugment** or `torchaudio` transforms
- Use a **deeper CNN or pretrained model**
- Deploy as a **real-time audio classifier** with mic input
- Export results and plots to PDF/CSV for reporting

---

## 📄 Citation

If using the dataset or referencing this structure, cite:

> J. Salamon, C. Jacoby, and J.P. Bello,  
> “A Dataset and Taxonomy for Urban Sound Research,”  
> ACM Multimedia 2014.

---

## 👤 Author

Created by [https://github.com/FilippoRomeo]   
📫 Reach out for collaboration, ideas, or improvements!

---

## 🧪 License

This repo is open-source under the MIT License. UrbanSound8K is shared for non-commercial research purposes. See their terms of use [here](https://urbansounddataset.weebly.com/urbansound8k.html).
