# src/dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ── Label mapping ──────────────────────────────────────────────────────────
LABEL_MAP = {
    'nv'   : 0,  # Melanocytic Nevi (benign mole)
    'mel'  : 1,  # Melanoma
    'bkl'  : 2,  # Benign Keratosis
    'bcc'  : 3,  # Basal Cell Carcinoma
    'akiec': 4,  # Actinic Keratosis
    'vasc' : 5,  # Vascular Lesion
    'df'   : 6,  # Dermatofibroma
}

LABEL_NAMES = [
    'Melanocytic Nevi', 'Melanoma', 'Benign Keratosis',
    'Basal Cell Carcinoma', 'Actinic Keratosis',
    'Vascular Lesion', 'Dermatofibroma'
]

# ── Helper: find image path across both part folders ───────────────────────
def find_image_path(image_id, data_dir):
    """
    HAM10000 images are split across two folders.
    This checks both and returns the correct path.
    """
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        path = os.path.join(data_dir, part, image_id + '.jpg')
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image {image_id} not found in either part folder.")


# ── Dataset class ──────────────────────────────────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = find_image_path(row['image_id'], self.data_dir)
        image    = Image.open(img_path).convert('RGB')
        label    = LABEL_MAP[row['dx']]

        if self.transform:
            image = self.transform(image)

        return image, label


# ── Transforms ─────────────────────────────────────────────────────────────
def get_transforms(phase):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


# ── Load and split data ────────────────────────────────────────────────────
def load_data(data_dir):
    """
    Reads the metadata CSV and splits into train / val / test sets.
    Returns three DataFrames.
    """
    csv_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    df = pd.read_csv(csv_path)

    # Keep only columns we need
    df = df[['image_id', 'dx']]

    # Split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.30,
                                         stratify=df['dx'], random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50,
                                         stratify=temp_df['dx'], random_state=42)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_DIR = 'data/archive'

    train_df, val_df, test_df = load_data(DATA_DIR)

    # Create datasets
    train_ds = SkinDataset(train_df, DATA_DIR, transform=get_transforms('train'))
    val_ds   = SkinDataset(val_df,   DATA_DIR, transform=get_transforms('val'))

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")   # should be [32, 3, 224, 224]
    print(f"Labels:      {labels}")
    print("dataset.py works correctly!")