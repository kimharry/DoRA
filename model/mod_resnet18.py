import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.resnet import ResNet18_Weights

class ChannelWiseMultiplier(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # One learnable weight per input channel
        self.weights = nn.Parameter(torch.ones(1, num_channels, 1, 1))  # shape: (1, C, 1, 1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        return x * self.weights

class ResNet18(nn.Module):
    """
    A wrapper class for ResNet18 from torchvision.
    
    Args:
        num_classes (int): Number of output classes. Default is 1000 (ImageNet classes).
        pretrained (bool): If True, loads weights pre-trained on ImageNet. Default is True.
    """
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.model = resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # add a channel-wise fc layer at the front of the model
        self.channel_wise_multiplier = ChannelWiseMultiplier(3)
        
        if num_classes != 100:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.channel_wise_multiplier(x)
        return self.model(x)


if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    
    x = torch.randn(1, 3, 224, 224)
    
    output = model(x)
    print(f"Output shape: {output.shape}")
