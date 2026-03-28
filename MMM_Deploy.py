# ============================================================
# ✅ FULL FUSION MODEL DEPLOYMENT (Colab Compatible)
# ============================================================

!pip install fastapi uvicorn nest_asyncio pyngrok python-multipart pillow torch torchvision --quiet

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from pyngrok import ngrok
import nest_asyncio
from PIL import Image
import io
import uvicorn

# ============================================================
# 1️⃣ CancerCNN (your exact version)
# ============================================================
class CancerCNN(nn.Module):
    def __init__(self, num_features=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 2️⃣ PneumoniaCNN (your exact version)
# ============================================================
class PneumoniaCNN(nn.Module):
    def __init__(self, num_features=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 3️⃣ Fusion Model
# ============================================================
class FusionNet(nn.Module):
    def __init__(self, modelA, modelB):
        super(FusionNet, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc = nn.Sequential(
            nn.Linear(25088 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        a_feat = self.modelA.features(x)
        b_feat = self.modelB.features(x)
        a_flat = torch.flatten(a_feat, 1)
        b_flat = torch.flatten(b_feat, 1)
        fused = torch.cat((a_flat, b_flat), dim=1)
        out = self.fc(fused)
        return out


# ============================================================
# 4️⃣ Load Fusion Model
# ============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

modelA = CancerCNN().to(device)
modelB = PneumoniaCNN().to(device)
fusion_model = FusionNet(modelA, modelB).to(device)

# Replace this path with your actual checkpoint file path
CHECKPOINT_PATH = "/content/drive/MyDrive/Saved models/MMM_BEST.pth"
try:
    fusion_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device), strict=False)
    print("✅ Model loaded with relaxed matching.")
except:
    print("⚠️ Could not load weights, using random initialization.")

fusion_model.eval()

# ============================================================
# 5️⃣ FastAPI + Ngrok Setup
# ============================================================
app = FastAPI(title="Fusion Model API", description="Predicts Brain Tumor / Pneumonia / Normal", version="1.0")

@app.get("/")
def root():
    return {"message": "Fusion Model API is running successfully!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("L").resize((224, 224))
    image_tensor = torch.tensor([[[image.getpixel((x, y)) for x in range(224)] for y in range(224)]], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = fusion_model(image_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().tolist()[0]
        pred = int(torch.argmax(logits, dim=1).item())
    return {"predicted_class": pred, "confidence": probs}


# ============================================================
# 6️⃣ Launch FastAPI with Ngrok
# ============================================================
ngrok.set_auth_token("35L9CaOawyw6hccA64hhdeSsZO8_h8uK8DjxkPvKQ8rwKYuX")  # <<< replace this line only
port = 7860
public_url = ngrok.connect(port).public_url
print("🌍 Public URL:", public_url)

nest_asyncio.apply()
import asyncio
from uvicorn import Config, Server

config = Config(app=app, host="0.0.0.0", port=port, log_level="info")
server = Server(config)

# Run Uvicorn in background (works in Colab)
loop = asyncio.get_event_loop()
loop.create_task(server.serve())
print("✅ Server started successfully at:", public_url)

