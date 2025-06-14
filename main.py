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
parser.add_argument('--dataset', '-d', type=str, default="MiniImageNet")
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--resnet_epoch', '-re', type=int, default=10)
parser.add_argument('--aug_epoch', '-ae', type=int, default=50)
parser.add_argument('--resnet_lr', '-rlr', type=float, default=0.001)
parser.add_argument('--aug_lr', '-alr', type=float, default=0.001)
parser.add_argument('--lambda_', '-l', type=float, default=0.1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "resnet": None,
    "aug_pred": AugPredModule().to(device),
    }

if args.dataset == "CIFAR100":
    models["resnet"] = ResNet18(num_classes=100).to(device)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
elif args.dataset == "CIFAR10":
    models["resnet"] = ResNet18(num_classes=10).to(device)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
elif args.dataset == "MiniImageNet":
    models["resnet"] = ResNet18(num_classes=100).to(device)
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
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

log_dir = os.path.join('logs', args.dataset, str(args.aug_epoch), datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

print(f"TensorBoard logs will be saved to: {os.path.abspath(log_dir)}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

# Phase 1: Train ResNet
optim_resnet = torch.optim.Adam(models["resnet"].parameters(), lr=args.resnet_lr)
lr_scheduler_resnet = torch.optim.lr_scheduler.CosineAnnealingLR(optim_resnet, T_max=args.resnet_epoch)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')

for epoch in range(args.resnet_epoch):
    models["resnet"].train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.resnet_epoch} [Train]')):
        images = images.to(device)
        labels = labels.to(device)

        optim_resnet.zero_grad()
    
        output = models["resnet"](images)
        
        loss = criterion(output, labels)
        loss.backward()
        optim_resnet.step()
        
        running_loss += loss.item()
        running_correct += (output.argmax(dim=1) == labels).sum().item()
        running_total += labels.size(0)

    models["resnet"].eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.resnet_epoch} [Test]')):
            images = images.to(device)
            labels = labels.to(device)

            output = models["resnet"](images)

            val_loss += criterion(output, labels).item()
            val_correct += (output.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    lr_scheduler_resnet.step()
    
    writer.add_scalar('resnet/train/lr', optim_resnet.param_groups[0]['lr'], epoch)
    writer.add_scalar('resnet/train/accuracy', 100 * running_correct / running_total, epoch)
    writer.add_scalar('resnet/train/loss', running_loss / len(train_loader), epoch)

    writer.add_scalar('resnet/val/accuracy', 100 * val_correct / val_total, epoch)
    writer.add_scalar('resnet/val/loss', val_loss / len(test_loader), epoch)

    # mean of parameters
    for name, param in models["resnet"].named_parameters():
        writer.add_scalar(f'resnet/{name}_mean', param.mean(), epoch)
        if param.grad is not None:
            writer.add_scalar(f'resnet/{name}_grad_mean', param.grad.mean(), epoch)

    print(f"Epoch {epoch+1}/{args.resnet_epoch}", end=" - ")
    print(f"Train Loss: {running_loss / len(train_loader)}", end=", ")
    print(f"Train Acc: {100 * running_correct / running_total:.2f}%", end=", ")
    print(f"Val Loss: {val_loss / len(test_loader)}", end=", ")
    print(f"Val Acc: {100 * val_correct / val_total:.2f}%")

    if val_loss / len(test_loader) < best_val_loss:
        best_val_loss = val_loss / len(test_loader)
        torch.save(models["resnet"].state_dict(), os.path.join(log_dir, "resnet.pth"))
        print(f"Best validation loss updated: {best_val_loss}")


#  Phase 2: Train AugPredModule
optim_aug_pred = torch.optim.Adam(models["aug_pred"].parameters(), lr=args.aug_lr)
lr_scheduler_aug_pred = torch.optim.lr_scheduler.CosineAnnealingLR(optim_aug_pred, T_max=args.aug_epoch)


best_val_loss = float('inf')

for epoch in range(args.aug_epoch):
    models["resnet"].train()
    models["aug_pred"].train()
    running_loss = 0.0
    running_aug_pred_loss = 0.0
    running_resnet_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.aug_epoch} [Train]')):
        images = images.to(device)
        labels = labels.to(device)

        optim_resnet.zero_grad()
        optim_aug_pred.zero_grad()
        
        aug_pred = models["aug_pred"](images)
        images = Augmentation(images, aug_pred)

        output = models["resnet"](images)
        
        temp_acc = (output.argmax(dim=1) == labels).sum().item() / labels.size(0)

        temp_resnet_loss = criterion(output, labels)
        temp_aug_pred_loss = models["aug_pred"].loss_fn(temp_acc)
        
        loss = temp_resnet_loss + args.lambda_ * temp_aug_pred_loss
        
        loss.backward()
        optim_resnet.step()
        optim_aug_pred.step()
        
        running_loss += loss.item()
        running_aug_pred_loss += temp_aug_pred_loss
        running_resnet_loss += temp_resnet_loss.item()
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    models["resnet"].eval()
    models["aug_pred"].eval()
    val_loss = 0.0
    val_aug_pred_loss = 0.0
    val_resnet_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.aug_epoch} [Test]')):
            images = images.to(device)
            labels = labels.to(device)

            aug_pred = models["aug_pred"](images)
            images = Augmentation(images, aug_pred)

            output = models["resnet"](images)

            temp_resnet_loss = criterion(output, labels)
            temp_aug_pred_loss = models["aug_pred"].loss_fn(temp_acc)

            val_loss += temp_resnet_loss.item() + args.lambda_ * temp_aug_pred_loss
            val_aug_pred_loss += temp_aug_pred_loss
            val_resnet_loss += temp_resnet_loss.item()
            correct += (output.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    lr_scheduler_aug_pred.step()
    lr_scheduler_resnet.step()
    
    writer.add_scalar('aug_pred/train/lr', optim_aug_pred.param_groups[0]['lr'], epoch)
    writer.add_scalar('aug_pred/train/accuracy', 100 * correct / total, epoch)
    writer.add_scalar('aug_pred/train/loss', running_loss / len(train_loader), epoch)
    writer.add_scalar('aug_pred/train/aug_pred_loss', running_aug_pred_loss / len(train_loader), epoch)
    writer.add_scalar('aug_pred/train/resnet_loss', running_resnet_loss / len(train_loader), epoch)

    writer.add_scalar('aug_pred/val/accuracy', 100 * correct / total, epoch)
    writer.add_scalar('aug_pred/val/loss', val_loss / len(test_loader), epoch)
    writer.add_scalar('aug_pred/val/aug_pred_loss', val_aug_pred_loss / len(test_loader), epoch)
    writer.add_scalar('aug_pred/val/resnet_loss', val_resnet_loss / len(test_loader), epoch)

    # mean of parameters
    for name, param in models["aug_pred"].named_parameters():
        writer.add_scalar(f'aug_pred/{name}_mean', param.mean(), epoch)
        if param.grad is not None:
            writer.add_scalar(f'aug_pred/{name}_grad_mean', param.grad.mean(), epoch)

    print(f"Epoch {epoch+1}/{args.aug_epoch}", end=" - ")
    print(f"Train Loss: {running_loss / len(train_loader)}", end=", ")
    print(f"Train Acc: {100 * correct / total:.2f}%", end=", ")
    print(f"Val Loss: {val_loss / len(test_loader)}", end=", ")
    print(f"Val Acc: {100 * correct / total:.2f}%")

    if val_loss / len(test_loader) < best_val_loss:
        best_val_loss = val_loss / len(test_loader)
        torch.save(models["resnet"].state_dict(), os.path.join(log_dir, "resnet.pth"))
        torch.save(models["aug_pred"].state_dict(), os.path.join(log_dir, "aug_pred.pth"))
        print(f"Best validation loss updated: {best_val_loss}")