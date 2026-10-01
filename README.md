# Urban Sound Classifier

A PyTorch audio-classification experiment built on **UrbanSound8K**, using Mel spectrograms as input to a convolutional neural network and respecting the dataset's official 10-fold evaluation protocol.

## Pipeline

```text
UrbanSound8K audio
       ↓
log-Mel spectrogram preprocessing
       ↓
fold-aware PyTorch dataset
       ↓
CNN classifier
       ↓
per-fold checkpoints
       ↓
accuracy, classification report, confusion matrix
```

## Why the fold protocol matters

UrbanSound8K contains 8,732 labelled clips across 10 sound classes. The dataset ships with predefined folds, so evaluation should preserve those folds instead of randomly splitting clips. This repository keeps that structure throughout training and evaluation.

Classes include:

`air_conditioner` · `car_horn` · `children_playing` · `dog_bark` · `drilling` · `engine_idling` · `gun_shot` · `jackhammer` · `siren` · `street_music`

## Project structure

```text
urban_sound_classifier/
├── config.py
├── download_dataset.py
├── preprocess_audio.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── checkpoints/
├── features/
└── UrbanSound8K/
```

## Quick start

Install the main dependencies:

```bash
pip install torch librosa soundata scikit-learn matplotlib seaborn
```

Download and validate UrbanSound8K:

```bash
python download_dataset.py
```

Convert the audio into log-Mel spectrogram arrays:

```bash
python preprocess_audio.py
```

Train across the predefined folds:

```bash
python train.py
```

Evaluate an individual fold:

```bash
python evaluate.py --fold 1
```

Evaluation produces accuracy, a classification report, and a confusion matrix for the selected fold.

## Example result

An example recorded for fold 1 reached approximately **0.67 accuracy**. That value is included as an experiment result, not as a benchmark claim for other environments or model revisions.

## Stack

`Python` `PyTorch` `librosa` `soundata` `scikit-learn` `Mel spectrograms` `CNN`

## What this project demonstrates

- fold-aware dataset handling rather than random train/test splitting
- audio preprocessing into model-ready time-frequency representations
- a complete PyTorch training and checkpoint workflow
- per-fold evaluation and visual diagnostics

## Possible next experiments

The current CNN is intentionally straightforward. Natural extensions include SpecAugment, stronger convolutional backbones, pretrained audio encoders, or a realtime inference interface.

## Dataset citation

J. Salamon, C. Jacoby, and J. P. Bello, *A Dataset and Taxonomy for Urban Sound Research*, ACM Multimedia, 2014.

UrbanSound8K has its own usage terms; refer to the dataset documentation when redistributing or using the audio.
