# preprocess_audio.py

import os
import librosa
import numpy as np
import soundata
import tqdm
from config import DATASET_PATH, FEATURES_PATH, SAMPLE_RATE, N_MELS, HOP_LENGTH, DURATION

# 4 seconds * 22050 samples/sec
SAMPLES_PER_CLIP = SAMPLE_RATE * DURATION

def extract_mel_spectrogram(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # If the clip is shorter than required, pad it
    if len(y) < SAMPLES_PER_CLIP:
        y = np.pad(y, (0, SAMPLES_PER_CLIP - len(y)))
    else:
        y = y[:SAMPLES_PER_CLIP]

    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def preprocess_dataset():
    os.makedirs(FEATURES_PATH, exist_ok=True)

    dataset = soundata.initialize('urbansound8k', data_home=DATASET_PATH)
    dataset.validate()

    print("Extracting Mel spectrograms...")

    for clip_id in tqdm.tqdm(dataset.clip_ids):
        clip = dataset.clip(clip_id)
        audio_path = clip.audio_path
        label = clip.tags.labels[0] if clip.tags.labels else "unknown"


        if label == "unknown":
            continue

        class_dir = os.path.join(FEATURES_PATH, label)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, f"{clip_id}.npy")

        try:
            mel_spec = extract_mel_spectrogram(audio_path)
            np.save(save_path, mel_spec)
        except Exception as e:
            print(f"❌ Error processing {clip_id}: {e}")

    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    preprocess_dataset()
