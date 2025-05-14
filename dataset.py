# dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class UrbanSoundDataset(Dataset):
    def __init__(self, features_path, fold_list, metadata_csv="UrbanSound8K/metadata/UrbanSound8K.csv", transform=None):
        self.features_path = features_path
        self.transform = transform
        self.data = []
        self.labels = []
        self.label_to_index = {}
        self.index_to_label = {}
        self.metadata = pd.read_csv(metadata_csv)

        class_names = sorted(os.listdir(features_path))
        for idx, label in enumerate(class_names):
            self.label_to_index[label] = idx
            self.index_to_label[idx] = label

        # Filter by folds
        subset = self.metadata[self.metadata["fold"].isin(fold_list)]

        for _, row in subset.iterrows():
            label = row["class"]
            clip_name = row["slice_file_name"].split('.')[0]
            file_path = os.path.join(features_path, label, f"{clip_name}.npy")
            if os.path.exists(file_path):
                self.data.append(file_path)
                self.labels.append(self.label_to_index[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.load(self.data[idx])
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, freq, time)
        y = self.labels[idx]
        return x, y
