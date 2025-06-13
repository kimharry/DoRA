'''Augmentation Parameter Prediction Module in PyTorch.

Reference:
Loss Prediction Module from
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms
import math

class AugPredModule(nn.Module):
    def __init__(self):
        super(AugPredModule, self).__init__()
        # Based on AlexNet architecture due to its simplicity and performance
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3), nn.ReLU(), #Conv2d(입력채널수, 출력채널수, 필터 크기)
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 1), nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*3*3, 1024), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,512), nn.ReLU(),
            nn.Linear(512,6)
            )

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 256*3*3)
        out = self.classifier(out)
        out = out.view(-1, 2, 3) # weight, bias
        return out

    def loss_fn(self, acc):
        return -math.log(acc+1e-10)

def Augmentation(images, aug_pred):
    # images: (batch_size, 3, w, h)
    # aug_pred: (batch_size, 2, 3)
    # Multiplication is channel-wise
    return images * aug_pred[:, 0].view(-1, 3, 1, 1) + aug_pred[:, 1].view(-1, 3, 1, 1)