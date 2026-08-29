import torch.nn as nn
from torchvision.models import resnet18


def build_model(num_classes: int):
    model = resnet18(weights="DEFAULT")

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model