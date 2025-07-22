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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # keep original input for full-resolution crop
        orig = x.clone()
        # Normalize input
        # Letterbox to 640×640: preserve aspect ratio, pad with gray
        _, _, h0, w0 = x.shape
        target_size = 640
        # compute scale and new size
        scale = min(target_size / float(h0), target_size / float(w0))
        new_h, new_w = int(h0 * scale), int(w0 * scale)
        # resize
        resized = F.interpolate(x, size=(new_h, new_w),
                                mode='bilinear', align_corners=False)
        # compute padding
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top = pad_h // 2
        left = pad_w // 2
        # create padded tensor filled with gray (0.447)
        gray = torch.tensor([0.447], device=x.device)
        padded = gray * \
            torch.ones((1, 3, target_size, target_size), device=x.device)
        padded[:, :, top:top + new_h, left:left + new_w] = resized
        # override x to the letterboxed version
        x = padded
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).view(1, 3, 1, 1)

        # x: (1, C, H, W)
        # Assume output is (N, 6) -> [x1, y1, x2, y2, conf, class]
        yolo_out = self.yolo_model(x)
        # Select detection with highest confidence from the first image
        # yolo_out: shape [1, N, 6] -> detections of shape [N, 6]
        detections = yolo_out[0]
        # confidences is the 5th column
        confidences = detections[:, 4]
        # find index of max confidence
        best_idx = confidences.argmax()
        # use the highest-confidence box coordinates
        box = detections[best_idx, :4]

        # Map padded box back to original image coordinates
        x1_p = box[0]
        y1_p = box[1]
        x2_p = box[2]
        y2_p = box[3]
        # remove padding then divide by scale to get original coords
        x1_o = (x1_p - left) / scale
        y1_o = (y1_p - top) / scale
        x2_o = (x2_p - left) / scale
        y2_o = (y2_p - top) / scale
        # clamp to original image size
        x1_o = x1_o.clamp(0.0, float(w0))
        y1_o = y1_o.clamp(0.0, float(h0))
        x2_o = x2_o.clamp(0.0, float(w0))
        y2_o = y2_o.clamp(0.0, float(h0))
        # cast mapped coords to int for cropping
        x1_i = int(x1_o)
        y1_i = int(y1_o)
        x2_i = int(x2_o)
        y2_i = int(y2_o)
        # crop the original full-resolution tensor
        orig_cropped = orig[:, :, y1_i:y2_i, x1_i:x2_i]

        # Use original-resolution crop (not the 640x640 letterboxed version) for classification
        cropped = orig_cropped
        _, _, h, w = cropped.shape
        if h == 0 or w == 0:
            # Fallback: use the full original-resolution image
            cropped = orig
            _, _, h, w = cropped.shape

        # TODO: We will do the resize outside in React Native side and move this inference to a
        # separate torchscript wrapper
        # Aspect-ratio preserving resize to fit within 224x224, then pad with black
        scale_cls = min(224.0 / float(h), 224.0 / float(w))
        resized_small = F.interpolate(
            cropped, scale_factor=(scale_cls, scale_cls), mode='bilinear', align_corners=False
        )
        new_h, new_w = resized_small.shape[-2], resized_small.shape[-1]
        padded_cls = torch.zeros((1, 3, 224, 224), device=x.device)
        top = (224 - new_h) // 2
        left = (224 - new_w) // 2
        padded_cls[:, :, top:top + new_h, left:left + new_w] = resized_small
        resized = padded_cls

        resized = (resized - mean) / std

        probs = torch.zeros((self.num_passes, 1), dtype=torch.float32)
        for i in range(self.num_passes):
            logits = self.classifier_model(resized)
            probs[i, 0] = torch.sigmoid(logits.view(-1))[0]

        mean_prob = probs.mean(dim=0)
        std_prob = probs.std(dim=0)

        return mean_prob, std_prob, orig_cropped
