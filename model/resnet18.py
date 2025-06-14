import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.resnet import ResNet18_Weights


# parameter prediction network
class AugParamNet(nn.Module):
    def __init__(self, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x):
        x = self.net(x).squeeze(-1).squeeze(-1)  # (B,16)
        return self.fc(x)  # (B,4)

# apply color jitter augmentation
def apply_color_jitter(x, params):
    r, g, b, bright = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    x_r = x[:, 0] * r[:, None, None]
    x_g = x[:, 1] * g[:, None, None]
    x_b = x[:, 2] * b[:, None, None]
    x_aug = torch.stack([x_r, x_g, x_b], dim=1) + bright[:, None, None, None]
    return x_aug.clamp(0, 1)



class ResNet18(nn.Module):
    def __init__(self, num_classes=100, fcn_ckpt=None, resnet_ckpt=None):
        super().__init__()
        self.model = resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if resnet_ckpt is not None:
            self.model.load_state_dict(torch.load(resnet_ckpt, map_location="cpu"), strict=True)

        # augmentation network added
        self.fcn = AugParamNet()
        if fcn_ckpt is not None:
            self.fcn.load_state_dict(torch.load(fcn_ckpt, map_location="cpu"), strict=True)

        if num_classes != 100:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        p = torch.tanh(self.fcn(x))
        p_scaled = torch.stack([
            1.0 + 0.2 * p[:, 0],  # R gain 
            1.0 + 0.2 * p[:, 1],  # G gain
            1.0 + 0.2 * p[:, 2],  # B gain
            0.2 * p[:, 3]         # brightness 
        ], dim=1)

        x_aug = apply_color_jitter(x, p_scaled)
        out = self.model(x_aug)
        return out, p_scaled
        # return out
        # return self.model(x)  # for fine-tuning only
