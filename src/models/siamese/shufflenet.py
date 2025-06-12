from typing import cast

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ShuffleNet_V2_X1_0_Weights


class SiameseShuffleNet(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_units=256):
        super().__init__()
        model = models.shufflenet_v2_x1_0(
            weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)

        in_features = cast(nn.Linear, model.fc).in_features

        # Remove the final pooling and classifier layers, keep feature extractor backbone
        self.feature_extractor = nn.Sequential(*list(model.children())[:-2])
        # Freeze feature extractor weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.BatchNorm1d(hidden_units),  # Optional: Batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units // 2),  # Another hidden layer
            nn.BatchNorm1d(hidden_units // 2),  # Optional: Batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            # Output layer for binary classification
            nn.Linear(hidden_units // 2, 1)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input1, input2):
        # Get the representational embedding for anchor
        output1 = self.forward_once(input1)
        # Get the representational embedding for actual image
        output2 = self.forward_once(input2)
        # Get the absolute distance vector to be used for the actual classification.
        diff = torch.abs(output1 - output2)
        logit = self.classifier(diff)

        return logit  # Apply sigmoid during inference for probability
