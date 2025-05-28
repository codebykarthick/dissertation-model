import torch
import torch.nn as nn
import torch.nn.functional as F


class LFD_CNN(nn.Module):
    """
    Lightweight CNN with Global Average Pooling.
    """

    def __init__(self, num_channels=16):
        """Creates the model instance for training / evaluation.

        Args:
            num_channels (int, optional): The number of starting filter channels. Defaults to 16.
        """
        super(LFD_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            num_channels, 2 * num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * num_channels)
        self.conv3 = nn.Conv2d(2 * num_channels, 4 *
                               num_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * num_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(4 * num_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward propagation step of the network.

        Args:
            x (torch.Tensor): The input image to be fed after the transformations.

        Returns:
            torch.Tensor: The resulting logits after one full forward prop step for a batch.
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
