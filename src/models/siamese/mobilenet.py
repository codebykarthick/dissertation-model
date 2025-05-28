import torch
import torch.nn as nn
import torchvision.models as models


class SiameseMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        # Remove classifier, keep feature extractor
        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(576, 128),  # Adjust input size if necessary
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        diff = torch.abs(output1 - output2)
        logit = self.classifier(diff)
        return logit  # Apply sigmoid during inference for probability
