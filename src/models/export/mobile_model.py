from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileRoICropModel(torch.nn.Module):
    def __init__(self, yolo_model: nn.Module, conf_threshold: float = 0.3):
        super().__init__()
        self.yolo_model = yolo_model
        self.threshold = conf_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep original input
        orig = x.clone()
        # Original dimensions
        _, _, h0, w0 = x.shape
        target_size = 640
        # Compute scale and resize letterbox
        scale = min(target_size / float(h0), target_size / float(w0))
        new_h, new_w = int(h0 * scale), int(w0 * scale)
        resized = F.interpolate(x, size=(new_h, new_w),
                                mode='bilinear', align_corners=False)
        # Compute padding
        pad_h, pad_w = target_size - new_h, target_size - new_w
        top, left = pad_h // 2, pad_w // 2
        # Create padded image with gray background
        gray = torch.tensor([0.447], device=x.device)
        padded = gray * \
            torch.ones((1, 3, target_size, target_size), device=x.device)
        padded[:, :, top:top + new_h, left:left + new_w] = resized
        x_letter = padded
        # Run YOLO model
        yolo_out = self.yolo_model(x_letter)
        detections = yolo_out[0]  # shape [N, 6]
        confidences = detections[:, 4]
        best_conf, best_idx = confidences.max(0)
        # If no detection above threshold, return the full original image
        if best_conf < self.threshold:
            # No detection above threshold: return the full original image
            return orig
        # Map best box back to original coordinates
        box = detections[best_idx, :4]

        # Has to be mapped individually to be torchscript compatible.
        x1_p = box[0]
        y1_p = box[1]
        x2_p = box[2]
        y2_p = box[3]

        x1_o = (x1_p - left) / scale
        y1_o = (y1_p - top) / scale
        x2_o = (x2_p - left) / scale
        y2_o = (y2_p - top) / scale
        x1_o = x1_o.clamp(0.0, float(w0))
        y1_o = y1_o.clamp(0.0, float(h0))
        x2_o = x2_o.clamp(0.0, float(w0))
        y2_o = y2_o.clamp(0.0, float(h0))
        # Convert coordinates to integer indices
        x1_i = int(x1_o.item())
        y1_i = int(y1_o.item())
        x2_i = int(x2_o.item())
        y2_i = int(y2_o.item())
        # Crop the original full-resolution tensor
        cropped = orig[:, :, y1_i:y2_i, x1_i:x2_i]
        return cropped


class MobileDecisionModel(torch.nn.Module):
    def __init__(self, decision_model: nn.Module, num_passes: int = 10):
        super().__init__()
        self.model = decision_model
        self.passes = num_passes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize input
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        probs = torch.zeros(
            (self.passes, 1), dtype=torch.float32, device=x.device)

        for i in range(self.passes):
            logits = self.model(x)
            probs[i, 0] = torch.sigmoid(logits.view(-1))[0]

        mean_prob = probs.mean(dim=0)
        std_prob = probs.std(dim=0)

        return mean_prob, std_prob
