# src/evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from dataset import load_data, SkinDataset, get_transforms, LABEL_NAMES
from model import build_model

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = 'data/archive'
MODEL_PATH = 'models/best_model.pth'
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Load model ───────────────────────────────────────────────────────────────
def load_trained_model():
    model = build_model(num_classes=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


# ── Get predictions ──────────────────────────────────────────────────────────
def get_predictions(model, loader):
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


# ── Plot confusion matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved to models/confusion_matrix.png")


# ── Plot ROC curves ──────────────────────────────────────────────────────────
def plot_roc_curves(labels, probs):
    plt.figure(figsize=(10, 7))

    for i, name in enumerate(LABEL_NAMES):
        # One-vs-rest ROC for each class
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _   = roc_curve(binary_labels, probs[:, i])
        auc           = roc_auc_score(binary_labels, probs[:, i])
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})')

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — One vs Rest')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('models/roc_curves.png', dpi=150)
    plt.show()
    print("ROC curves saved to models/roc_curves.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def evaluate():
    _, _, test_df = load_data(DATA_DIR)

    test_ds     = SkinDataset(test_df, DATA_DIR, transform=get_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    model = load_trained_model()
    preds, labels, probs = get_predictions(model, test_loader)

    # ── Classification report ─────────────────────────────────────────────
    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(labels, preds, target_names=LABEL_NAMES))

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds)
    plot_roc_curves(labels, probs)


if __name__ == '__main__':
    evaluate()