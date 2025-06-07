import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.resnet import ResNet18_Weights


class ResNet18(nn.Module):
    """
    A wrapper class for ResNet18 from torchvision.
    
    Args:
        num_classes (int): Number of output classes. Default is 1000 (ImageNet classes).
        pretrained (bool): If True, loads weights pre-trained on ImageNet. Default is True.
    """
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.model = resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        if num_classes != 1000:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    
    x = torch.randn(1, 3, 224, 224)
    
    output = model(x)
    print(f"Output shape: {output.shape}")
