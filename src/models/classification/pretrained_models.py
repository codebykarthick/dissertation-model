from typing import cast

import timm
import torch.nn as nn
import torchvision.models as models
from torchvision import models
from torchvision.models import (
    EfficientNet,
    EfficientNet_B0_Weights,
    MobileNetV3,
    ShuffleNet_V2_X1_0_Weights,
    ShuffleNetV2,
)
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights


def get_mobilenetv3_tuned(dropout_rate=0.2, hidden_units=128) -> MobileNetV3:
    """Generate a fine-tunable crafted instance of MobileNetV3 for the task of prediction.

    Args:
        dropout_rate (float, optional): The amount of dropout to be used in the fine-tune layer. Defaults to 0.2.
        hidden_units (int, optional): The number of hidden units in the dense layer. Defaults to 128.

    Returns:
        MobileNetV3: The modified instance of the MobileNetV3 architecture to be fine-tuned.
    """
    model = models.mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.DEFAULT)

    # Freeze the feature extraction layers that are already pre-trained.
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the number of input features for the classifier from the previously defined layer
    in_features = cast(nn.Linear, model.classifier[0]).in_features

    # Replace the classifier with a new, more complex sequential module
    model.classifier = nn.Sequential(
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

    return model


def get_efficientnet_tuned(dropout_rate=0.3, hidden_units=256) -> EfficientNet:
    """Generate a custom fine-tunable instance of EfficientNet architecture for the task.

    Args:
        dropout_rate (float, optional): The amount of dropout to be used in the fine-tune layer. Defaults to 0.3.
        hidden_units (int, optional): The number of hidden units in the dense layer. Defaults to 256.

    Returns:
        EfficientNet: The modified instance of the EfficientNet architecture to be fine-tuned.
    """

    # Load a pretrained EfficientNet-B7 model
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # Freeze the feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the number of input features for the classifier
    # The first layer is a dropout, second is the actual linear layer from which we can extract the dimensions.
    in_features = cast(nn.Linear, model.classifier[1]).in_features

    # Replace the classifier with a new, more complex sequential module
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, hidden_units),
        nn.BatchNorm1d(hidden_units),  # Optional: Batch normalization
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_units, hidden_units // 2),  # Another hidden layer
        nn.BatchNorm1d(hidden_units // 2),  # Optional: Batch normalization
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        # Output layer for binary classification
        nn.Linear(hidden_units // 2, 1)
    )

    return model


def get_shufflenet_tuned(dropout_rate=0.3, hidden_units=256) -> ShuffleNetV2:
    model = models.shufflenet_v2_x1_0(
        weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)

    # Freeze the feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    in_features = cast(nn.Linear, model.fc).in_features

    model.fc = nn.Sequential(
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

    return model


def get_tinyvit_tuned(model_name: str = 'tiny_vit_11m_224.in1k', dropout_rate: float = 0.2, hidden_units: int = 128) -> nn.Module:
    """Generate a custom fine-tunable instance of TinyViT architecture for the task.

    Args:
        model_name (str, optional): Name of the TinyViT model in timm. Defaults to 'tiny_vit_11m_224.in1k'.
        dropout_rate (float, optional): Dropout rate for the classifier. Defaults to 0.2.
        hidden_units (int, optional): Number of units in the first hidden layer. Defaults to 128.

    Returns:
        nn.Module: The modified TinyViT model for fine-tuning.
    """
    # Create a pretrained TinyViT model
    model = timm.create_model(model_name, pretrained=True)

    # Freeze all parameters (feature extractor)
    for param in model.parameters():
        param.requires_grad = False

    # Number of features from the model (embedding dimension)
    in_features = model.num_features

    # Replace the original classification head with a new sequential head
    model.head = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.BatchNorm1d(hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_units, hidden_units // 2),
        nn.BatchNorm1d(hidden_units // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_units // 2, 1)
    )

    return model
