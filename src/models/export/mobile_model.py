from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileInferenceModel(torch.nn.Module):
    def __init__(self, yolo_model: nn.Module, classifier_model: nn.Module, num_passes: int = 10):
        super().__init__()
        self.yolo_model = yolo_model
        self.classifier_model = classifier_model
        self.num_passes = num_passes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize input
        x = x.clone()
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # x: (1, C, H, W)
        # Assume output is (N, 6) -> [x1, y1, x2, y2, conf, class]
        yolo_out = self.yolo_model(x)
        box = yolo_out[0, 0:4].int()  # Take first box only

        x1 = torch.clamp(box[0], min=0)
        y1 = torch.clamp(box[1], min=0)
        x2 = torch.clamp(box[2], max=x.shape[3])
        y2 = torch.clamp(box[3], max=x.shape[2])

        cropped = x[0:1, :, y1:y2, x1:x2]  # Keep batch dim
        _, _, h, w = cropped.shape
        scale = min(224 / h, 224 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = F.interpolate(cropped, size=(
            new_h, new_w), mode='bilinear', align_corners=False)
        padded = torch.zeros((1, 3, 224, 224), device=x.device)
        top = (224 - new_h) // 2
        left = (224 - new_w) // 2
        padded[:, :, top:top+new_h, left:left+new_w] = resized
        resized = padded

        self.classifier_model.train()  # Enable dropout

        probs = torch.zeros((self.num_passes, 1), dtype=torch.float32)
        for i in range(self.num_passes):
            logits = self.classifier_model(resized)
            probs[i, 0] = torch.sigmoid(logits.view(-1))[0]

        mean_prob = probs.mean(dim=0)
        std_prob = probs.std(dim=0)

        return mean_prob, std_prob
