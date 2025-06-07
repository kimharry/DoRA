import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()
dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)

all_pixels = torch.cat([img.view(-1) for img, _ in dataset])
global_mean = all_pixels.mean().item()

channels_sum = torch.zeros(3)
for img, _ in dataset:
    channels_sum += img.sum(dim=(1,2))
channel_means = channels_sum / (len(dataset)*224*224)

channels_sq_sum = torch.zeros(3)
for img, _ in dataset:
    channels_sq_sum += (img**2).sum(dim=(1,2))
channel_stds = (channels_sq_sum/(len(dataset)*224*224) - channel_means**2).sqrt()

print(f"mean img: {global_mean:.4f}")
print(f"mean per channel img: {channel_means}")
print(f"std per channel img: {channel_stds}")
