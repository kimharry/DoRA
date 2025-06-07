# DoRA: Domain-adaptive ResNet with Augmentation

DoRA (Domain-adaptive ResNet with Augmentation) is a deep learning framework designed to enhance image classification performance through learned color transformations. This project implements a modified ResNet18 architecture that automatically learns optimal color jittering parameters during training, enabling the model to adapt to different visual domains more effectively.

The key innovation lies in the channel-wise multiplier layer that learns to adjust color channels dynamically, allowing the model to discover the most beneficial color transformations for the specific dataset. The training process follows a two-phase approach: first optimizing the main network weights while keeping augmentation parameters fixed, followed by fine-tuning the augmentation weights while freezing the main network.

This approach is particularly useful for domain adaptation tasks where the target domain's color distribution may differ from the source domain, or when the optimal color augmentation strategy is not known in advance.


## Project Structure

```
.
├── main.py                 # Main training script
├── model/
│   └── mod_resnet18.py    # Modified ResNet18 implementation
└── utils.py                # Training utilities and helper functions
```

## Features

- **Modified ResNet18**: Custom implementation of ResNet18 with an added channel-wise multiplier layer for color jittering
- **Two-Phase Training**:
  - Phase 1: Train the main ResNet18 model while keeping color jittering weights fixed
  - Phase 2: Fine-tune only the color jittering weights while keeping the main model fixed
- **Flexible Configuration**: Adjustable hyperparameters for both training phases
- **TensorBoard Integration**: Training progress and metrics can be visualized using TensorBoard


## Model Architecture

The model consists of:
1. A channel-wise multiplier layer that learns color jittering weights
2. A standard ResNet18 backbone
3. A fully connected classification head

## Training Process

1. **Main Training Phase**:
   - The ResNet18 backbone is trained
   - Color jittering weights are frozen

2. **Fine-tuning Phase**:
   - The ResNet18 backbone is frozen
   - Only the color jittering weights are updated

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- tensorboard