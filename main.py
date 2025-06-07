import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model.mod_resnet18 import ResNet18
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--main_epoch', '-me', type=int, default=10)
parser.add_argument('--main_batch_size', '-mbs', type=int, default=64)
parser.add_argument('--main_lr', '-mlr', type=float, default=0.001)
parser.add_argument('--fine_tune_epoch', '-fe', type=int, default=10)
parser.add_argument('--fine_tune_batch_size', '-fbs', type=int, default=64)
parser.add_argument('--fine_tune_lr', '-flr', type=float, default=0.001)
parser.add_argument('--num_workers', '-nw', type=int, default=24)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18(num_classes=100)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4728, 0.4489, 0.4047], std=[0.2841, 0.2758, 0.2909]),
])

dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


train_model(model, train_dataset, test_dataset, args)

train_model(model, train_dataset, test_dataset, args, fine_tune=True)
