import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    """Load YAML training configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_json(data, path: Path) -> None:
    """Save Python data as formatted JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Clear gradients from previous batch.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(inputs)

        # Calculate loss.
        loss = criterion(outputs, targets)

        # Backpropagation.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Accumulate loss.
        total_loss += loss.item() * inputs.size(0)

        # Calculate predictions.
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model without updating weights."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def save_periodic_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    best_val_loss: float,
    patience_counter: int,
    config: dict,
    checkpoint_dir: Path,
    keep_last: int,
) -> Path:
    """
    Save a training checkpoint that can be used to resume training.

    Only the most recent `keep_last` periodic checkpoints are retained.
    """

    checkpoint_path = (
        checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "config": config,
        },
        checkpoint_path,
    )

    # Find all periodic checkpoints.
    checkpoint_files = list(
        checkpoint_dir.glob("checkpoint_epoch_*.pt")
    )

    # Sort by epoch number.
    checkpoint_files.sort(
        key=lambda path: int(
            path.stem.split("_")[-1]
        )
    )

    # Remove oldest checkpoints.
    while len(checkpoint_files) > keep_last:
        oldest_checkpoint = checkpoint_files.pop(0)
        oldest_checkpoint.unlink()

    return checkpoint_path

def find_latest_checkpoint(
    checkpoint_dir: Path,
) -> Path | None:

    checkpoints = list(
        checkpoint_dir.glob("checkpoint_epoch_*.pt")
    )

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda path: int(
            path.stem.split("_")[-1]
        )
    )

    return checkpoints[-1]

def save_best_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    checkpoint_dir: Path,
) -> None:
    """Save the best performing model and its metrics."""

    best_model_path = checkpoint_dir / "best_model.pt"
    best_metrics_path = checkpoint_dir / "best_metrics.json"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        best_model_path,
    )

    save_json(
        metrics,
        best_metrics_path,
    )


def save_final_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    checkpoint_dir: Path,
) -> None:
    """Save model state and metrics from the final completed epoch."""

    final_model_path = checkpoint_dir / "final_model.pt"
    final_metrics_path = checkpoint_dir / "final_metrics.json"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        final_model_path,
    )

    save_json(
        metrics,
        final_metrics_path,
    )


def plot_loss(
    metrics_history: list[dict],
    output_dir: Path,
) -> None:
    """Plot training and validation loss."""

    epochs = [
        metric["epoch"]
        for metric in metrics_history
    ]

    train_losses = [
        metric["train_loss"]
        for metric in metrics_history
    ]

    val_losses = [
        metric["val_loss"]
        for metric in metrics_history
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = output_dir / "loss_curve.png"

    plt.savefig(plot_path)
    plt.close()


def plot_accuracy(
    metrics_history: list[dict],
    output_dir: Path,
) -> None:
    """Plot training and validation accuracy."""

    epochs = [
        metric["epoch"]
        for metric in metrics_history
    ]

    train_accuracy = [
        metric["train_accuracy"]
        for metric in metrics_history
    ]

    val_accuracy = [
        metric["val_accuracy"]
        for metric in metrics_history
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = output_dir / "accuracy_curve.png"

    plt.savefig(plot_path)
    plt.close()


def main():
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    # Docker path.
    config_path = Path(
        "/app/configs/training_config.yaml"
    )

    # Local development path.
    if not config_path.exists():
        project_root = Path(__file__).resolve().parent.parent
        config_path = (
            project_root
            / "configs"
            / "training_config.yaml"
        )

    config = load_config(
        str(config_path)
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    device_info = {
        "event": "training_started",
        "device": str(device),
    }

    if torch.cuda.is_available():
        device_info["gpu"] = (
            torch.cuda.get_device_name(0)
        )

    print(
        json.dumps(device_info),
        flush=True,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get(
            "num_workers",
            0,
        ),
    )

    # ---------------------------------------------------------
    # Optimizer and loss
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    checkpoint_dir = Path(
        config["output"]["checkpoint_dir"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_history_path = (
        checkpoint_dir / "metrics_history.json"
    )

    # ---------------------------------------------------------
    # Training settings
    # ---------------------------------------------------------

    epochs = config["training"]["epochs"]

    patience = config["training"][
        "early_stopping_patience"
    ]

    checkpoint_every = config["output"].get(
        "checkpoint_every",
        5,
    )

    keep_last_checkpoints = config["output"].get(
        "keep_last_checkpoints",
        2,



    # ---------------------------------------------------------
    # State
    # ---------------------------------------------------------
    metrics_history = []
    final_metrics = None
    final_epoch = 0

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    resume_training = config["training"].get(
        "resume",
        False,
    )

    if resume_training:
        latest_checkpoint = find_latest_checkpoint(
            checkpoint_dir
        )

        if latest_checkpoint is not None:
            print(
                json.dumps({
                    "event": "checkpoint_found",
                    "path": str(latest_checkpoint),
                }),
                flush=True,
            )

            checkpoint = torch.load(
                latest_checkpoint,
                map_location=device,
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            start_epoch = checkpoint["epoch"]

            best_val_loss = checkpoint.get(
                "best_val_loss",
                float("inf"),
            )

            patience_counter = checkpoint.get(
                "patience_counter",
                0,
            )

            print(
                json.dumps({
                    "event": "training_resumed",
                    "checkpoint": str(latest_checkpoint),
                    "last_completed_epoch": start_epoch,
                    "next_epoch": start_epoch + 1,
                }),
                flush=True,
            )
    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Metrics for current epoch.
        current_metrics = {
            "epoch": current_epoch,
            "train_loss": round(
                train_loss,
                4,
            ),
            "train_accuracy": round(
                train_acc,
                4,
            ),
            "val_loss": round(
                val_loss,
                4,
            ),
            "val_accuracy": round(
                val_acc,
                4,
            ),
        }

        # Structured stdout logging.
        print(
            json.dumps(current_metrics),
            flush=True,
        )

        # -----------------------------------------------------
        # Metrics history
        # -----------------------------------------------------

        metrics_history.append(
            current_metrics
        )

        # Save metrics after EVERY epoch.
        save_json(
            metrics_history,
            metrics_history_path,
        )

        final_metrics = current_metrics
        final_epoch = current_epoch

        # -----------------------------------------------------
        # Best model
        # -----------------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_metrics = {
                **current_metrics,
                "best_val_loss": round(
                    best_val_loss,
                    4,
                ),
            }

            save_best_model(
                model=model,
                optimizer=optimizer,
                epoch=current_epoch,
                metrics=best_metrics,
                config=config,
                checkpoint_dir=checkpoint_dir,
            )

            print(
                json.dumps(
                    {
                        "event": "best_model_saved",
                        "epoch": current_epoch,
                        "val_loss": round(
                            val_loss,
                            4,
                        ),
                        "path": str(
                            checkpoint_dir
                            / "best_model.pt"
                        ),
                    }
                ),
                flush=True,
            )

        else:
            patience_counter += 1

        # -----------------------------------------------------
        # Periodic checkpoint
        # -----------------------------------------------------

        if (
            current_epoch
            % checkpoint_every
            == 0
        ):
            checkpoint_path = (
                save_periodic_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=current_epoch,
                    metrics=current_metrics,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    keep_last=keep_last_checkpoints,
                )
            )

            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "epoch": current_epoch,
                        "path": str(
                            checkpoint_path
                        ),
                    }
                ),
                flush=True,
            )

        # -----------------------------------------------------
        # Early stopping
        # -----------------------------------------------------

        if patience_counter >= patience:
            print(
                json.dumps(
                    {
                        "event": "early_stopping",
                        "epoch": current_epoch,
                        "best_val_loss": round(
                            best_val_loss,
                            4,
                        ),
                    }
                ),
                flush=True,
            )

            break

    # ---------------------------------------------------------
    # Save final model
    # ---------------------------------------------------------

    if final_metrics is not None:
        save_final_model(
            model=model,
            optimizer=optimizer,
            epoch=final_epoch,
            metrics=final_metrics,
            config=config,
            checkpoint_dir=checkpoint_dir,
        )

        print(
            json.dumps(
                {
                    "event": "final_model_saved",
                    "epoch": final_epoch,
                    "path": str(
                        checkpoint_dir
                        / "final_model.pt"
                    ),
                }
            ),
            flush=True,
        )

    # ---------------------------------------------------------
    # Create plots
    # ---------------------------------------------------------

    if metrics_history:
        plot_loss(
            metrics_history,
            checkpoint_dir,
        )

        plot_accuracy(
            metrics_history,
            checkpoint_dir,
        )

        print(
            json.dumps(
                {
                    "event": "plots_saved",
                    "loss_plot": str(
                        checkpoint_dir
                        / "loss_curve.png"
                    ),
                    "accuracy_plot": str(
                        checkpoint_dir
                        / "accuracy_curve.png"
                    ),
                }
            ),
            flush=True,
        )

    # ---------------------------------------------------------
    # Training complete
    # ---------------------------------------------------------

    print(
        json.dumps(
            {
                "event": "training_complete",
                "epochs_completed": final_epoch,
                "best_val_loss": round(
                    best_val_loss,
                    4,
                ),
                "best_model": str(
                    checkpoint_dir
                    / "best_model.pt"
                ),
                "final_model": str(
                    checkpoint_dir
                    / "final_model.pt"
                ),
                "metrics_history": str(
                    metrics_history_path
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

