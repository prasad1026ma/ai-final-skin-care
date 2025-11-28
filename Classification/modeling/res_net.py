import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet(num_classes, pretrained=True):
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
