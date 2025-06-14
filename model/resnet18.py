import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.resnet import ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.model = resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # self.model = resnet.resnet18()
        # weights = torch.load("model/resnet18.pth")
        # # remove model prefix
        # weights = {k.replace("model.", ""): v for k, v in weights.items()}
        # self.model.load_state_dict(weights)
        
        if num_classes != 100:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)