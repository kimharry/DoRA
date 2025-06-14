# train_resnet_only.py

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import torchvision.models.resnet as resnet
from torchvision.models.resnet import ResNet18_Weights
from tqdm import tqdm

# resnet_ckpt = "resnet18_pretrained.pth"  # ResNet18 pretrained path 
class PureResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(PureResNet18, self).__init__()
        self.model = resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # self.model.load_state_dict(torch.load(resnet_ckpt, map_location="cpu"), strict=True)
        if num_classes != 100:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────── 하이퍼파라미터 설정 ──────
epochs = 30
batch_size = 64
lr = 0.001
num_classes = 100

# ────── 데이터셋 ──────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
seed = 26
generator = torch.Generator().manual_seed(seed)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24)

# ────── 모델 ──────
model = PureResNet18(num_classes=num_classes).to(device)

# ────── 학습 설정 ──────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ────── 학습 루프 ──────
for epoch in range(epochs):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        train_loader_tqdm.set_postfix(
            loss=running_loss / total if total > 0 else 0,
            acc=100. * correct / total if total > 0 else 0
        )

    avg_loss = running_loss / total
    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

    model.eval()
    val_loss, val_total, val_correct = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    avg_val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs} Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
# ────── 저장 ──────
torch.save(model.model.state_dict(), "resnet18_pretrained.pth")
print("✅ ResNet 저장 완료: resnet18_pretrained.pth")
