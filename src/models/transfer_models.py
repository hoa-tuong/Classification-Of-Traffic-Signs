import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, resnet18, ResNet18_Weights


def get_transfer_model(model_name: str = "mobilenet_v3", num_classes: int = 250, pretrained: bool = True):
    """
    Khoi tao mo hinh Transfer Learning (MobileNetV3 hoac ResNet18) cho 250 lop bien bao.
    Thich hop cho so sanh thuc nghiem trong do an.
    """
    if model_name.lower() == "mobilenet_v3":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    elif model_name.lower() == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Khong ho tro model_name='{model_name}'. Chon 'mobilenet_v3' hoac 'resnet18'.")
