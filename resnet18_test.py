import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model.resnet18 import ResNet18
from model.aug_pred_module import AugPredModule, Augmentation

from tqdm import tqdm
import os
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default="CIFAR100")
parser.add_argument('--epoch', '-e', type=int, default=50)
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--lr', '-lr', type=float, default=0.001)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

if args.dataset == "CIFAR100":
    model = ResNet18(num_classes=100).to(device)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
elif args.dataset == "CIFAR10":
    model = ResNet18(num_classes=10).to(device)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
elif args.dataset == "MiniImageNet":
    model = ResNet18(num_classes=100).to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

log_dir = os.path.join('logs', args.dataset, datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

print(f"TensorBoard logs will be saved to: {os.path.abspath(log_dir)}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

optim = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epoch)
criterion = nn.CrossEntropyLoss()


best_val_loss = float('inf')

for epoch in range(args.epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Train]')):
        images = images.to(device)
        labels = labels.to(device)

        optim.zero_grad()
    
        output = model(images)
        
        loss = criterion(output, labels)
        loss.backward()
        optim.step()
        
        running_loss += loss.item()
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Test]')):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            val_loss += criterion(output, labels).item()
            correct += (output.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    lr_scheduler.step()
    
    writer.add_scalar('train/lr', optim.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/accuracy', 100 * correct / total, epoch)
    writer.add_scalar('train/loss', running_loss / len(train_loader), epoch)

    writer.add_scalar('val/accuracy', 100 * correct / total, epoch)
    writer.add_scalar('val/loss', val_loss / len(test_loader), epoch)

    print(f"Epoch {epoch+1}/{args.epoch}", end=" - ")
    print(f"Train Loss: {running_loss / len(train_loader)}", end=", ")
    print(f"Train Acc: {100 * correct / total:.2f}%", end=", ")
    print(f"Val Loss: {val_loss / len(test_loader)}", end=", ")
    print(f"Val Acc: {100 * correct / total:.2f}%")

    if val_loss / len(test_loader) < best_val_loss:
        best_val_loss = val_loss / len(test_loader)
        torch.save(model.state_dict(), "model/resnet18.pth")
        print(f"Best validation loss updated: {best_val_loss}")