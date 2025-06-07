import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model.mod_resnet18 import ResNet18
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--fine_tune_epoch', '-fe', type=int, default=10)
parser.add_argument('--train_epoch', '-te', type=int, default=10)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18(num_classes=100)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fine_tune_epoch)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(args.fine_tune_epoch):
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Acc: {100 * correct / total:.2f}%, Epoch {epoch+1}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
