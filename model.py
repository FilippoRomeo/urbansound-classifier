# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_CLASSES

class UrbanSoundCNN(nn.Module):
    def __init__(self):
        super(UrbanSoundCNN, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsample by 2
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # You may need to adjust this based on input spectrogram size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 21, 256),  # adjust if your spectrogram shape is different
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, N_CLASSES)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.fc(x)
        return x
