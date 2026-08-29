import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_model(
    architecture: str,
    num_classes: int,
) -> nn.Module:
    """
    Create and return the requested model architecture.

    Args:
        architecture: Model name, e.g. "resnet18".
        num_classes: Number of output classes.

    Returns:
        PyTorch model.
    """

    architecture = architecture.lower()

    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT

        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(
            in_features,
            num_classes
        )

        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}"
    )
