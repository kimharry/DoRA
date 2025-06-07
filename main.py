import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model.mod_resnet18 import ResNet18
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18(num_classes=100)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # or (84, 84) for Conv4
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(10):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Acc: {100 * correct / total:.2f}%, Epoch {epoch+1}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
