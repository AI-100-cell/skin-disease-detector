# src/predict.py
import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.append('src')
from model import build_model

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = 'models/best_model.pth'
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_NAMES = [
    'Melanocytic Nevi (Benign Mole)',
    'Melanoma (Skin Cancer)',
    'Benign Keratosis',
    'Basal Cell Carcinoma',
    'Actinic Keratosis',
    'Vascular Lesion',
    'Dermatofibroma'
]

# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Load model ───────────────────────────────────────────────────────────────
def load_model():
    model = build_model(num_classes=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# ── Predict ──────────────────────────────────────────────────────────────────
def predict(image_path):
    image  = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    model  = load_model()
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    # Top 3 predictions
    top3 = torch.topk(probs, 3)
    print(f"\n🔬 Skin Analysis Results for: {image_path}")
    print("─" * 45)
    for i in range(3):
        idx        = top3.indices[i].item()
        confidence = top3.values[i].item() * 100
        print(f"#{i+1} {LABEL_NAMES[idx]:<35} {confidence:.1f}%")
    print("─" * 45)
    print("⚠️  For medical diagnosis consult a dermatologist.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
    else:
        predict(sys.argv[1])