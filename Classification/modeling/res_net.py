import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Pretrained Res-net
        self.backbone = models.resnet18(pretrained=True)

        # Get number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)