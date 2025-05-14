# evaluate.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from dataset import UrbanSoundDataset
from model import UrbanSoundCNN
from config import FEATURES_PATH, BATCH_SIZE, CHECKPOINT_PATH

def evaluate_fold(fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Evaluating Fold {fold} on device: {device}")

    # Load test dataset for this fold
    test_dataset = UrbanSoundDataset(FEATURES_PATH, fold_list=[fold])
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Load model checkpoint for this fold
    model_path = f"{CHECKPOINT_PATH}/model_fold_{fold}.pt"
    model = UrbanSoundCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Accuracy for Fold {fold}: {accuracy:.4f}")

    # Per-class metrics
    label_map = test_dataset.index_to_label
    class_names = [label_map[i] for i in range(len(label_map))]
    print("\n🧾 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, help="Which fold to evaluate (1–10)")
    args = parser.parse_args()

    evaluate_fold(args.fold)
