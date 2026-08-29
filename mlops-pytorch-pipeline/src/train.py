import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from dataset import get_dataloaders
from model import build_model


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (
        total_loss / len(loader),
        correct / total
    )


def main():
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = get_dataloaders(
        data_dir=config["dataset"]["root"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"]
    )

    model = build_model(
        config["model"]["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]

    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_path = Path(
        config["output"]["checkpoint_path"]
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(epochs):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        metrics = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4)
        }

        print(json.dumps(metrics))

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                json.dumps(
                    {"event": "early_stopping"}
                )
            )
            break


if __name__ == "__main__":
    main()