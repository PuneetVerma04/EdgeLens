import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights


def get_resnet50(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """ResNet50 with pretrained ImageNet weights, final FC replaced for num_classes."""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_mobilenetv2(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """MobileNetV2 with pretrained ImageNet weights, classifier head replaced."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model
