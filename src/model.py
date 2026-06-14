# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=7, freeze_base=True):
    """
    Loads EfficientNet-B0 pretrained on ImageNet.
    Replaces the final layer to output num_classes (7 skin conditions).
    """
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all base layers — we only train the final head
    # This is Transfer Learning: reuse what ImageNet already taught it
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head for our 7 classes
    # EfficientNet's final layer outputs 1280 features
    classifier = model.classifier[1]
    assert isinstance(classifier, nn.Linear)
    in_features = classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == '__main__':
    model = build_model(num_classes=7)

    # Count trainable parameters
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")
    print("model.py works correctly!")