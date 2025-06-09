# DoRA: Domain-adaptive ResNet with Augmentation

DoRA (Domain-adaptive ResNet with Augmentation) is a deep learning framework designed to enhance image classification performance through learned data augmentation. This project implements a modified ResNet18 architecture that automatically learns optimal augmentation parameters during training, including both spatial (crop/resize) and color transformations.

The key innovations are:
1. **Learnable Crop and Resize**: A spatial transformation layer that learns the optimal crop region for each input image.
2. **Channel-wise Color Jittering**: A learnable layer that adjusts color channels dynamically.
3. **End-to-End Training**: The augmentation parameters are learned jointly with the main model weights.

This approach is particularly useful for domain adaptation tasks where the target domain's visual characteristics may differ from the source domain, or when the optimal augmentation strategy is not known in advance.


## Project Structure

```
.
├── main.py                 # Main training script
├── model/
│   └── mod_resnet18.py    # Modified ResNet18 implementation
└── utils.py                # Training utilities and helper functions
```

## Features

- **Modified ResNet18**: Custom implementation of ResNet18 with learnable augmentation layers
- **Learnable Augmentations**:
  - **Spatial Transformations**: Automatic learning of optimal crop regions
  - **Color Jittering**: Channel-wise adaptive color transformations
- **Flexible Configuration**: Adjustable hyperparameters for both the model and augmentation layers
- **End-to-End Training**: Augmentation parameters are learned jointly with the model
- **TensorBoard Integration**: Training progress and metrics can be visualized using TensorBoard


## Model Architecture

The model consists of:
1. **Augmentation Module**:
   - Learnable crop and resize layer with adjustable region
   - Channel-wise color jittering with learnable weights
2. **Backbone**: Standard ResNet18
3. **Classification Head**: Fully connected layer for final predictions

## Training Process

The model is trained end-to-end with the following components:

1. **Forward Pass**:
   - Input images first pass through the augmentation module
   - Augmented images are then processed by the ResNet18 backbone
   - Final predictions are made by the classification head

2. **Backward Pass**:
   - Gradients flow through both the main network and augmentation layers
   - All parameters (model weights and augmentation parameters) are updated simultaneously
   - The model learns optimal augmentation strategies specific to the task

3. **Inference**:
   - The same augmentation pipeline is applied during inference
   - Ensures consistency between training and evaluation