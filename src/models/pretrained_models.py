import torch.nn as nn
import torchvision.models as models


def get_mobilenetv3():
    """
    Get a mobilenet v3 model with all layers frozen except for final and
    output layer for finetuning.
    """
    model = models.mobilenet_v3_large(pretrained=True)

    # Optionally, freeze the feature extraction layers to only fine-tune the classifier
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier with a new linear layer for binary classification
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid()
    )

    return model


def get_efficientnet():
    """
    Get an EfficientNet-B7 model with all layers frozen except 
    for the final classifier layer for finetuning.
    This is used as a substitute for EfficientNet-Lite, 
    which is not available in torchvision.models.
    """
    # Load a pretrained EfficientNet-B7 model
    model = models.efficientnet_b7(pretrained=True)

    # Freeze the feature extraction layers to only fine-tune the classifier
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier with a new linear layer for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
        nn.Sigmoid()
    )

    return model
