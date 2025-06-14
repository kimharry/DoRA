import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
from tqdm import tqdm  # tqdm 추가

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root="data/miniimagenet", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
seed = 26
generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=24)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=24)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AugParamNet(nn.Module):
    def __init__(self, out_dim=4):  # [R, G, B gain, brightness]
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x):
        x = self.net(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = AugParamNet().to(device)

def sample_random_params(batch_size, device):
    r = torch.empty(batch_size).uniform_(0.8, 1.2)
    g = torch.empty(batch_size).uniform_(0.8, 1.2)
    b = torch.empty(batch_size).uniform_(0.8, 1.2)
    brightness = torch.empty(batch_size).uniform_(-0.2, 0.2)
    return torch.stack([r, g, b, brightness], dim=1).to(device)


def apply_augmentation(x, params):
    # x: (B, 3, H, W), params: (B, 4)
    r, g, b, bright = params[:,0], params[:,1], params[:,2], params[:,3]
    x_r = x[:,0] * r[:,None,None]
    x_g = x[:,1] * g[:,None,None]
    x_b = x[:,2] * b[:,None,None]
    x_aug = torch.stack([x_r, x_g, x_b], dim=1) + bright[:,None,None,None]
    return x_aug.clamp(0, 1)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()
    total_loss = 0

    for x_raw, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):  # tqdm 적용
        x_raw = x_raw.to(device)
        p_gt = sample_random_params(x_raw.size(0), device)
        x_aug = apply_augmentation(x_raw, p_gt)
        
        p_pred = model(x_aug)
        loss = loss_fn(p_pred, p_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")


torch.save(model.state_dict(), "fcn_phase1.pth")