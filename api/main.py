# api/main.py
import sys
import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision import transforms

sys.path.append('src')
from model import build_model

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = 'models/best_model.pth'
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_NAMES = [
    'Melanocytic Nevi (Benign Mole)',
    'Melanoma (Skin Cancer)',
    'Benign Keratosis',
    'Basal Cell Carcinoma',
    'Actinic Keratosis',
    'Vascular Lesion',
    'Dermatofibroma'
]

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title='SkinScan AI API')

# Allow frontend to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Load model once at startup ───────────────────────────────────────────────
model = build_model(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get('/')
def root():
    return {'message': 'SkinScan AI API is running'}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor   = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    # Top 3 results
    top3 = torch.topk(probs, 3)
    results = [
        {
            'condition' : LABEL_NAMES[top3.indices[i].item()],
            'confidence': round(top3.values[i].item() * 100, 1)
        }
        for i in range(3)
    ]

    return JSONResponse({'predictions': results})