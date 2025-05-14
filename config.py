# config.py

import os

# Base path where everything will live
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Path where soundata will download UrbanSound8K
DATASET_PATH = os.path.join(BASE_DIR, "UrbanSound8K")

# Directory where we’ll save Mel spectrograms
FEATURES_PATH = os.path.join(BASE_DIR, "features")

# Model saving
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints")
LOGS_PATH = os.path.join(BASE_DIR, "logs")

# Training settings
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 4  # seconds
N_CLASSES = 10

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
