import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model.resnet18 import ResNet18
from utils import train_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--main_epoch', type=int, default=3)
parser.add_argument('--main_batch_size', type=int, default=64)
parser.add_argument('--main_lr', type=float, default=0.001)

parser.add_argument('--fine_tune_epoch', type=int, default=30)
parser.add_argument('--fine_tune_batch_size', type=int, default=64)
parser.add_argument('--fine_tune_lr', type=float, default=0.001)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--fcn_ckpt', type=str, default="fcn_phase1.pth")  # phase1 결과 경로 (optional)
parser.add_argument('--resnet_ckpt', type=str, default="resnet18_pretrained.pth")  # ResNet18 pretrained 모델 경로 (optional)

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = ResNet18(num_classes=100, fcn_ckpt=args.fcn_ckpt, resnet_ckpt=args.resnet_ckpt)
model.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
seed = 26
generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)


train_model(model, train_dataset, test_dataset, args, fine_tune=False)

train_model(model, train_dataset, test_dataset, args, fine_tune=True)
