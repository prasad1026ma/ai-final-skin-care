import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from Classification.modeling.res_net import ResNet
from Classification.utilities.skin_dataset import SkinDataset
import os


def normalization(input):
    """Normalize input to range [0, 1]"""
    input_min = input.min()
    input_max = input.max()
    if input_max - input_min > 0:
        normalized_input = (input - input_min) / (input_max - input_min)
    else:
        normalized_input = input
    return normalized_input


# load the resnet model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

model = ResNet(num_classes=5)
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# load and preprocess the samp image
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image_path = f"{BASE_DIR}/data/images/-4596787619832414799.png"
img = Image.open(image_path).convert('RGB')
transform = SkinDataset.get_transforms(train=False, input_size=224)
img_tensor = transform(img).unsqueeze(0)

# Get the weights of the first convolutional layer
first_conv_layer = model.backbone.conv1
first_conv_weights = first_conv_layer.weight.data.cpu().numpy()
num_kernels = first_conv_weights.shape[0]
in_channels = first_conv_weights.shape[1]

print(f"First conv layer has {num_kernels} kernels with {in_channels} input channels")

# Create a grid of kernel images
rows = int(np.sqrt(num_kernels / 2))
cols = int(np.ceil(num_kernels / rows))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
axes = axes.flatten() if num_kernels > 1 else [axes]

for i in range(num_kernels):
    kernel = first_conv_weights[i]

    # For RGB input (3 channels), show as color or averaged
    if in_channels == 3:
        kernel_image = kernel.transpose(1, 2, 0)
        kernel_image = normalization(kernel_image)
    else:
        kernel_image = kernel.mean(axis=0)
        kernel_image = normalization(kernel_image)

    if in_channels == 3:
        axes[i].imshow(kernel_image)
    else:
        axes[i].imshow(kernel_image, interpolation='nearest', cmap='gray')

    axes[i].axis('off')
    axes[i].set_title(f'K{i}', fontsize=8)

plt.suptitle('First Convolutional Layer Kernels')
plt.tight_layout()
plt.savefig('kernel_grid.png', dpi=150)
print("Saved kernel_grid.png")

# Apply the first convolutional layer to the sample image
with torch.no_grad():
    output = first_conv_layer(img_tensor)

# Convert output from (1, num_channels, H, W) to (num_channels, H, W)
output = output.squeeze(0)

# Create a grid showing the result of each kernel
num_channels = output.shape[0]
rows = int(np.ceil(np.sqrt(num_channels)))
cols = int(np.ceil(num_channels / rows))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
axes = axes.flatten() if num_channels > 1 else [axes]

for i in range(num_channels):
    feature_map = output[i].detach().numpy()
    feature_normalized = normalization(feature_map)

    axes[i].imshow(feature_normalized, cmap='gray', interpolation='nearest')
    axes[i].axis('off')
    axes[i].set_title(f'Filter {i}', fontsize=8)

# Hide extra subplots if any
for i in range(num_channels, len(axes)):
    axes[i].axis('off')

plt.suptitle('Feature Maps After First Convolution')
plt.tight_layout()
plt.savefig('image_transform_grid.png', dpi=150)
print("Saved image_transform_grid.png")

# Track progression through early ResNet layers
with torch.no_grad():
    x0 = img_tensor  # Original image
    x1 = model.backbone.conv1(x0)  # First convolution
    x2 = model.backbone.bn1(x1)  # Batch norm
    x3 = model.backbone.relu(x2)  # ReLU activation
    x4 = model.backbone.maxpool(x3)  # Max pooling

stages = [
    ('Original', x0),
    ('Conv1', x1),
    ('BatchNorm', x2),
    ('ReLU', x3),
    ('MaxPool', x4)
]

fig, axes = plt.subplots(1, len(stages), figsize=(len(stages) * 3, 3))

for idx, (name, tensor) in enumerate(stages):
    # Average across all channels to get single image
    vis = tensor[0].mean(dim=0).detach().numpy()
    vis_normalized = normalization(vis)

    axes[idx].imshow(vis_normalized, cmap='viridis')
    axes[idx].set_title(name, fontsize=12)
    axes[idx].axis('off')

plt.suptitle('Feature Map Progression Through ResNet Layers')
plt.tight_layout()
plt.savefig('feature_progression.png', dpi=150)
print("Saved feature_progression.png")