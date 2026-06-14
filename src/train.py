# src/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import load_data, SkinDataset, get_transforms
from model import build_model

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR   = 'data/archive'
SAVE_PATH  = 'models/best_model.pth'
EPOCHS     = 15
BATCH_SIZE = 32
LR         = 1e-3
NUM_CLASSES = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ── Data ────────────────────────────────────────────────────────────────────
def get_loaders():
    train_df, val_df, _ = load_data(DATA_DIR)

    train_ds = SkinDataset(train_df, DATA_DIR, transform=get_transforms('train'))
    val_ds   = SkinDataset(val_df,   DATA_DIR, transform=get_transforms('val'))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ── Train one epoch ─────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds      = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Validate ────────────────────────────────────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds      = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Main training loop ───────────────────────────────────────────────────────
def train():
    os.makedirs('models', exist_ok=True)

    train_loader, val_loader = get_loaders()
    model     = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✅ Best model saved (val_loss: {val_loss:.4f})")

    print("\nTraining complete!")
    print(f"Best model saved to: {SAVE_PATH}")


if __name__ == '__main__':
    train()
    