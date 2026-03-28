# full_fusion_4class_pipeline.py
# Run in Colab. Adjust the model path variables below if your files live elsewhere.

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -------------------------
# CONFIG: edit these paths if required
# -------------------------
CANCER_ZIP = "/content/drive/MyDrive/Datasets/Cancer grey scale.zip"
PNEUMONIA_ZIP = "/content/drive/MyDrive/Datasets/chest_xray_pnemonia.zip"

# Your pretrained individual-model weights (set to actual paths)
cancer_model_path = "/content/drive/MyDrive/Saved models/Brain_Tumor_Binary_classification.pth"
pneumonia_model_path = "/content/drive/MyDrive/Saved models/PNEUMONIA_COLAB_SAVED"  # change to .pth if needed

# Where to save the trained fusion model
fusion_save_path = "/content/drive/MyDrive/Saved models/fusion_model_4class_state_dict.pth"

# -------------------------
# Mount Drive (Colab)
# -------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass  # not in colab — assume files are already accessible

# -------------------------
# Extract zips (if not already extracted)
# -------------------------
os.makedirs("/content/data_extract", exist_ok=True)
if os.path.exists(CANCER_ZIP):
    with zipfile.ZipFile(CANCER_ZIP, 'r') as z:
        z.extractall("/content/data_extract/cancer")
if os.path.exists(PNEUMONIA_ZIP):
    with zipfile.ZipFile(PNEUMONIA_ZIP, 'r') as z:
        z.extractall("/content/data_extract/pneumonia")

print("Extraction done (if zips existed). Check /content/data_extract")

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print("Using device:", device)

# -------------------------
# Define the exact architectures you used (copied from your code)
# -------------------------
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

            nn.Linear(64, 1)  # logits for binary (original training)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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

# -------------------------
# Combined Dataset builder
# -------------------------
# We'll build the combined list by:
# - Loading cancer ImageFolder and using your binary mapping mapping={0:0,1:0,2:0,3:1} where label 3 means 'Cancer'
# - Loading pneumonia images from extracted pneumonia folders (NORMAL vs PNEUMONIA)
#
# Final label mapping:
# 0 -> Cancer
# 1 -> No_Cancer
# 2 -> Pneumonia
# 3 -> No_Pneumonia

def gather_combined_dataset(cancer_root, pneumonia_root, transform=None):
    samples = []

    # 1) cancer dataset: user earlier used ImageFolder('/content/cancer_dataset/Data')
    cancer_folder = os.path.join(cancer_root, "Data") if os.path.isdir(os.path.join(cancer_root, "Data")) else cancer_root
    if os.path.isdir(cancer_folder):
        # Use ImageFolder to get mapping of classes
        folder_ds = datasets.ImageFolder(cancer_folder, transform=None)
        # folder_ds.classes and folder_ds.samples available
        # binary mapping used in your training:
        binary_mapping = {0:0, 1:0, 2:0, 3:1}
        for path, orig_label in folder_ds.samples:
            mapped = binary_mapping.get(orig_label, 0)
            if mapped == 1:
                label = 0  # Cancer
            else:
                label = 1  # No_Cancer
            samples.append((path, label))
    else:
        print("Warning: cancer folder not found at", cancer_folder)

    # 2) pneumonia dataset: find images under train/val/test directories
    # typical structure after unzip: /content/data_extract/pneumonia/chest_xray/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
    # try to find chest_xray folder automatically
    found = False
    for root, dirs, files in os.walk(pneumonia_root):
        if 'chest_xray' in root and ('train' in root or 'test' in root or 'val' in root):
            found = True
            # we will collect from all NORMAL and PNEUMONIA directories under pneumonia_root
    # attempt typical path:
    chest_xray_base = None
    possible = [
        os.path.join(pneumonia_root, "chest_xray", "chest_xray"),
        os.path.join(pneumonia_root, "chest_xray"),
        pneumonia_root
    ]
    for p in possible:
        if os.path.isdir(p):
            chest_xray_base = p
            break
    if chest_xray_base is None:
        # fallback to search
        for root, dirs, files in os.walk(pneumonia_root):
            if 'train' in root and os.path.isdir(root):
                chest_xray_base = root.rsplit('/', 1)[0]
                break
    if chest_xray_base is None:
        print("Warning: could not auto-locate chest_xray base. You may need to adjust pneumonia_root.")
    else:
        # gather from train/val/test subfolders if present
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(chest_xray_base, split)
            if not os.path.isdir(split_path):
                continue
            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                if cls.upper().startswith('PNEUMONIA'):
                    label = 2
                else:
                    label = 3  # No_Pneumonia (NORMAL)
                for fname in os.listdir(cls_path):
                    fpath = os.path.join(cls_path, fname)
                    if os.path.isfile(fpath):
                        samples.append((fpath, label))

    # Done
    return samples

# Determine extracted folder roots
cancer_root = "/content/data_extract/cancer" if os.path.isdir("/content/data_extract/cancer") else "/content/cancer_dataset"
pneumonia_root = "/content/data_extract/pneumonia" if os.path.isdir("/content/data_extract/pneumonia") else "/content/Pneumonia_dataset"

print("Using cancer_root:", cancer_root)
print("Using pneumonia_root:", pneumonia_root)

# Prepare transform consistent for combined training
# Use grayscale + center crop like pneumonia transforms you previously used and normalize using computed stats (if known).
XRAY_MEAN = [0.5831533670425415]  # if you calculated earlier
XRAY_STD = [0.16427507996559143]

common_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=XRAY_MEAN, std=XRAY_STD)
])

samples = gather_combined_dataset(cancer_root, pneumonia_root, transform=None)
print(f"Collected {len(samples)} samples (combined)")

# Create a simple Dataset from sampled file paths
class FileListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, int(label)

dataset = FileListDataset(samples, transform=common_transform)

# train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print("Train samples:", len(train_dataset), "Val samples:", len(val_dataset))

# -------------------------
# Load pretrained individual models (your saved architectures)
# -------------------------
# Use the architectures you trained earlier (CancerCNN and PneumoniaCNN)
modelA = CancerCNN(num_features=1)
modelB = PneumoniaCNN(num_features=1)

# If the pneumonia saved file is a directory or named differently, adjust accordingly.
# Try to load; if fail, throw a helpful error.
try:
    sdA = torch.load(cancer_model_path, map_location=device)
    # sdA might be either a state_dict or full model — handle both
    if isinstance(sdA, dict) and any(k.startswith('features') or k.startswith('classifier') for k in sdA.keys()):
        modelA.load_state_dict(sdA)
    else:
        # It might be a saved full model object:
        try:
            modelA = sdA
            print("Loaded full cancer model object.")
        except Exception:
            modelA.load_state_dict(sdA)
    print("Loaded cancer pretrained weights.")
except Exception as e:
    print("Error loading cancer model:", e)
    raise

try:
    sdB = torch.load(pneumonia_model_path, map_location=device)
    if isinstance(sdB, dict) and any(k.startswith('features') or k.startswith('classifier') for k in sdB.keys()):
        modelB.load_state_dict(sdB)
    else:
        modelB = sdB
        print("Loaded full pneumonia model object.")
    print("Loaded pneumonia pretrained weights.")
except Exception as e:
    print("Error loading pneumonia model:", e)
    raise

modelA.to(device).eval()
modelB.to(device).eval()

# Remove classifier parts and keep feature extractors for fusion
# (If classifier is nn.Sequential, replace it with Identity to avoid accidental forward through classifiers)
if hasattr(modelA, 'classifier'):
    modelA.classifier = nn.Identity()
if hasattr(modelB, 'classifier'):
    modelB.classifier = nn.Identity()

# -------------------------
# Build Fusion model dynamically (compute feature sizes)
# -------------------------
class FusionNet4Class(nn.Module):
    def __init__(self, modelA_feat, modelB_feat, num_classes=4):
        super().__init__()
        self.modelA_feat = modelA_feat
        self.modelB_feat = modelB_feat

        # compute flattened sizes
        with torch.no_grad():
            dummy = torch.zeros(1,1,224,224).to(device)
            a = self.modelA_feat(dummy)
            b = self.modelB_feat(dummy)
            fa = a.view(1, -1).shape[1]
            fb = b.view(1, -1).shape[1]
            fused = fa + fb
        print(f"feature sizes -> A: {fa}, B: {fb}, fused: {fused}")

        self.head = nn.Sequential(
            nn.Linear(fused, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        a = self.modelA_feat(x)
        b = self.modelB_feat(x)
        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)
        fused = torch.cat([a, b], dim=1)
        out = self.head(fused)
        return out

fusion_model = FusionNet4Class(modelA.features, modelB.features, num_classes=4).to(device)
print("Fusion model created.")


from PIL import Image, UnidentifiedImageError

def is_image_file(path):
    try:
        Image.open(path).verify()  # just verify, do not load
        return True
    except (UnidentifiedImageError, IOError, OSError):
        return False


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=1e-4)


dataset = FileListDataset(samples, transform=common_transform)

# Make sure dataset is clean
print(f"Total valid images: {len(dataset)}")

# Split
total_len = len(dataset)
train_size = int(0.8 * total_len)
val_size = total_len - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print("Train samples:", len(train_dataset), "Val samples:", len(val_dataset))


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return val_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)


import time

num_epochs = 20
start_time = time.time()  # overall timer

for epoch in range(num_epochs):
    epoch_start = time.time()  # timer for this epoch

    train_loss, train_acc = train_one_epoch(fusion_model, train_loader, optimizer, criterion)
    val_loss, val_acc, preds, labels = validate(fusion_model, val_loader, criterion)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    total_time_elapsed = epoch_end - start_time

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.2f}s | Total Elapsed: {total_time_elapsed:.2f}s")


# Save the trained fusion model safely
torch.save(fusion_model.state_dict(), "/content/drive/MyDrive/Saved models/MMM_BEST.pth")
print("Fusion model state_dict saved at:", "/content/drive/MyDrive/Saved models/MMM_BEST.pth")


from torch.utils.data import DataLoader, random_split

# Use your existing dataset
full_dataset = dataset  # FileListDataset with all samples and common_transform

# Let's split out a test set (e.g., 10% of total)
test_size = int(0.1 * len(full_dataset))
remaining_size = len(full_dataset) - test_size
remaining_dataset, test_dataset = random_split(full_dataset, [remaining_size, test_size])

# Create test loader
batch_size = 16
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Test samples: {len(test_dataset)}")


import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# -------------------------
# Load the saved fusion model
# -------------------------
fusion_model_loaded = FusionNet4Class(modelA.features, modelB.features, num_classes=4).to(device)
fusion_model_loaded.load_state_dict(torch.load("/content/drive/MyDrive/Saved models/MMM_BEST.pth", map_location=device))
fusion_model_loaded.eval()
print("Fusion model loaded successfully!")

# -------------------------
# Evaluate function
# -------------------------
def evaluate_model(model, loader, class_names=['Cancer', 'No_Cancer', 'Pneumonia', 'No_Pneumonia']):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return all_labels, all_preds, cm

# -------------------------
# Example usage
# -------------------------
# Make sure you have a test_loader
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
labels, preds, cm = evaluate_model(fusion_model_loaded, test_loader)
