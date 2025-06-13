import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model.resnet18 import ResNet18
from model.aug_pred_module import AugPredModule

from tqdm import tqdm
import os
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', type=int, default=150)
parser.add_argument('--aug_epoch', '-ae', type=int, default=75)
parser.add_argument('--cycle', '-c', type=int, default=2)
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--lr', '-lr', type=float, default=0.0001)
parser.add_argument('--lambda_', '-l', type=float, default=1)
args = parser.parse_args()

def aug_pred_loss(acc):
    return -torch.log(acc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "resnet": ResNet18(num_classes=100).to(device),
    "aug_pred": AugPredModule().to(device),
    }

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)
dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

print(f"TensorBoard logs will be saved to: {os.path.abspath(log_dir)}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

optim_resnet = torch.optim.Adam(models["resnet"].parameters(), lr=args.lr)
lr_scheduler_resnet = torch.optim.lr_scheduler.CosineAnnealingLR(optim_resnet, T_max=args.epoch)
criterion = nn.CrossEntropyLoss()

optim_aug_pred = torch.optim.Adam(models["aug_pred"].parameters(), lr=args.lr)
lr_scheduler_aug_pred = torch.optim.lr_scheduler.CosineAnnealingLR(optim_aug_pred, T_max=args.aug_epoch)


best_val_loss = float('inf')

for epoch in range(args.epoch):
    models["resnet"].train()
    if epoch < args.aug_epoch:
        models["aug_pred"].train()
    else:
        models["aug_pred"].eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Train]')):
        images = images.to(device)
        labels = labels.to(device)

        optim_resnet.zero_grad()
        optim_aug_pred.zero_grad()
        
        aug_pred = models["aug_pred"](images)
        images = transforms.ColorJitter(brightness=aug_pred[:, 0], contrast=aug_pred[:, 1], saturation=aug_pred[:, 2], hue=aug_pred[:, 3])(images)

        output = models["resnet"](images)
        
        temp_acc = (output.argmax(dim=1) == labels).sum().item() / labels.size(0)

        temp_resnet_loss = criterion(output, labels)
        temp_aug_pred_loss = aug_pred_loss(temp_acc)
        
        if epoch < args.aug_epoch:
            loss = temp_resnet_loss + args.lambda_ * temp_aug_pred_loss
        else:
            loss = temp_resnet_loss
        
        loss.backward()
        optim_resnet.step()
        optim_aug_pred.step()
        
        running_loss += loss.item()
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    lr_scheduler_resnet.step()
    lr_scheduler_aug_pred.step()
    
    writer.add_scalar('train/loss', running_loss / (batch_idx + 1), epoch)
    writer.add_scalar('train/accuracy', correct / total, epoch)
    writer.add_scalar('train/lr', optim_resnet.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/aug_pred_loss', temp_aug_pred_loss.item(), epoch)
    writer.add_scalar('train/resnet_loss', temp_resnet_loss.item(), epoch)

    writer.add_scalar('aug_pred/weight', models["aug_pred"].weight.data.mean().item(), epoch)
    writer.add_scalar('aug_pred/bias', models["aug_pred"].bias.data.mean().item(), epoch)