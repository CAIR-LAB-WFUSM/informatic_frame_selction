#%%
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

from torch import nn, optim
import pickle



from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Veriler için dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Veri setini yükleme
dataset_dir = "./maymun_2/"
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Veri ayırma (%60 - %10 - %30)
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import pickle

# Mevcut veri bölmelerindeki indeksleri kaydetme
train_indices = train_dataset.indices
val_indices = val_dataset.indices
test_indices = test_dataset.indices

# İndeksleri bir dosyaya kaydedin
split_indices = {
    "train": train_indices,
    "val": val_indices,
    "test": test_indices
}

with open("maymun_ciceği_4_sınıf.pkl", "wb") as f:
    pickle.dump(split_indices, f)
    
    

#%%




from transformers import AutoImageProcessor, ViTModel
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

# Model ve işlemciyi yükleme
def load_dino(ckpt_id="facebook/dino-vitb16"):
    processor = AutoImageProcessor.from_pretrained(ckpt_id)
    model = ViTModel.from_pretrained(ckpt_id)
    return model, processor
"""
# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalleştirme
])

"""

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Veri setini yükleme
dataset_dir = "./maymun_2/"
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Veri ayırma (%70 - %10 - %20)
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model ve işlemciyi başlatma
ckpt_id = "facebook/dino-vitb16"
model, processor = load_dino(ckpt_id)

# Baş kısmını sınıflarınıza uyarlama
model.classifier = nn.Linear(model.config.hidden_size, 4)  # 4 class
model = model.cuda()

# Tüm katmanların eğitilebilir olmasını sağlama
for param in model.parameters():
    param.requires_grad = True

# Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Eğitim döngüsü
best_val_loss = float('inf')
best_model_path = "best_model_dino.pth"

for epoch in range(10):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()

        # Görüntüleri işlemciye uygun şekilde işleme
        inputs = processor(images, return_tensors="pt").to("cuda")  # Düzgün tensör formatı
        outputs = model(**inputs,)
        logits = model.classifier(outputs.last_hidden_state[:,0])  # Havuzlama
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(loss)

        train_loss += loss.item()

    # Doğrulama döngüsü
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            inputs = processor(images, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            logits = model.classifier(outputs.last_hidden_state[:,0]) 
            loss = criterion(logits, labels)
            val_loss += loss.item()

    # En iyi modeli kaydetme
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

print("Eğitim tamamlandı.")





#%%
# Test
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.cuda()
        inputs = processor(images, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        logits = model.classifier(outputs.last_hidden_state[:,0]) 
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Performans metrikleri
print("Test Sonuçları:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))


#%%



















