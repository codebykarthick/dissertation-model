import torch
import torch.nn as nn
import torch.nn.functional as F


class LFD_CNN(nn.Module):
    """
    Bespoke lightweight architecture for generating the probability of positives.
    """

    def __init__(self):
        super(LFD_CNN, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Normalization and Pooling
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial size

        # Fully Connected Layers (Classifier)
        # Assuming input image is 64x64 after pooling
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)  # Single neuron for binary classification
        self.dropout = nn.Dropout(0.3)  # Prevent overfitting

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))

        # Flatten before FC layer
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for probability output
        return x
