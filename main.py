import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model.mod_resnet18 import ResNet18
from torch.utils.data import DataLoader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'mps'

# pretrain on cifar10
model = ResNet18(num_classes=10, pretrained=False)
model.to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10 = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

train_loader = DataLoader(cifar10, batch_size=64, shuffle=True)

for epoch in range(10):
    for images, labels in train_loader:
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item()}")

model.eval()
test_loader = DataLoader(cifar10, batch_size=64, shuffle=False)

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    print(f"Test loss: {loss.item()}")