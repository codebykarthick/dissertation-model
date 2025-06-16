

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights


class SiameseEfficientNet(nn.Module):
    def __init__(self, hidden_units=512, dropout_rate=0.5):
        super(SiameseEfficientNet, self).__init__()
        # Load pretrained EfficientNet B0 backbone
        self.model = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT)
        # Determine feature dimension from the original classifier
        in_features = self.model.classifier[1].in_features
        # Remove the pretrained classifier head
        self.model.classifier = nn.Identity()
        # Adaptive pooling to get fixed-size feature vectors
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Freeze feature extractor layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        # Embedding head for metric learning
        self.embedding_net = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.BatchNorm1d(hidden_units // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units // 2, hidden_units // 2),
        )

    def forward_once(self, x):
        # Extract features through backbone
        x = self.model.features(x)
        # Pool and flatten
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """Return the normalized embedding for a single input."""
        features = self.forward_once(x)
        embedding = self.embedding_net(features)
        # L2-normalize embeddings for metric learning
        return F.normalize(embedding, p=2, dim=1)
