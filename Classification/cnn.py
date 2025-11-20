import torch
import torch.nn as nn
import torch.nn.functional as F


class Cnn(nn.Module):
    def __init__(self, num_classes=5, input_size=224):
        """
        Deeper version with more convolutional layers for better feature extraction
        """
        super().__init__()

        # 4 different convolution layers
        self.convolution1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.convolution3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.convolution4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # add in a pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # calc feat size
        feature_size = (input_size // 16) * (input_size // 16) * 256

        # three fully connected connected layers
        self.fc1 = nn.Linear(feature_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.convolution1(x)))
        x = self.pool(F.relu(self.convolution2(x)))
        x = self.pool(F.relu(self.convolution3(x)))
        x = self.pool(F.relu(self.convolution4(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x