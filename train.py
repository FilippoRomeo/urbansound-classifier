# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import UrbanSoundDataset
from model import UrbanSoundCNN
from config import FEATURES_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_PATH
import os 
import numpy as np

def train_fold(train_loader, val_loader, fold_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UrbanSoundCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save model for this fold
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f"model_fold_{fold_idx}.pt"))

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Fold {fold_idx}: Accuracy = {acc:.4f}")
    return acc
def cross_validate():
    fold_accuracies = []
    for test_fold in range(1, 11):
        train_folds = [f for f in range(1, 11) if f != test_fold]
        print(f"🔁 Running Fold {test_fold}...")

        train_dataset = UrbanSoundDataset(FEATURES_PATH, fold_list=train_folds)
        val_dataset = UrbanSoundDataset(FEATURES_PATH, fold_list=[test_fold])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        acc = train_fold(train_loader, val_loader, test_fold)
        fold_accuracies.append(acc)

    avg_acc = np.mean(fold_accuracies)
    print(f"\n📊 Average 10-Fold Accuracy: {avg_acc:.4f}")

if __name__ == "__main__":
    cross_validate()
