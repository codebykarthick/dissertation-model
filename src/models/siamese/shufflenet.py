import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0


class SiameseShuffleNet(nn.Module):
    def __init__(self, in_features=1024, hidden_units=512, dropout_rate=0.5):
        super(SiameseShuffleNet, self).__init__()
        self.model = shufflenet_v2_x1_0(pretrained=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

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
            nn.Linear(hidden_units // 2, hidden_units // 2)
        )

    def forward_once(self, x):
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """Return the embedding for a single input."""
        features = self.forward_once(x)
        embedding = self.embedding_net(features)
        return embedding
