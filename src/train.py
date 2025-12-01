"""Training script for the currency classifier CNN."""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model import CurrencyClassifier
from src.dataset import BanknoteDataset, get_train_transforms, get_val_transforms, create_splits


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total if total > 0 else 0.0


def train(model, train_loader, val_loader, epochs=30, lr=0.001, device="cpu",
          save_dir="models", patience=5):
    """Train the model with early stopping and learning rate scheduling.

    Args:
        model: The CNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs
        lr: Initial learning rate
        device: Device to train on (cpu or cuda)
        save_dir: Directory to save model checkpoints
        patience: Early stopping patience (epochs without improvement)

    Returns:
        dict with training history (losses, accuracies per epoch)
    """
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    print(f"Training on {device}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"Classes: {model.backbone.classifier[-1].out_features}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        val_acc = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        # Record history
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "train_acc": train_acc,
            }, checkpoint_path)
            print(f"  -> New best model saved! (Val Acc: {val_acc:.1f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.1f}%")
                break

    # Save training curves
    plot_training_curves(history, save_dir)

    return history


def plot_training_curves(history, save_dir):
    """Save training curves as a PNG image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Train the currency classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--save-dir", type=str, default="models", help="Where to save model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print("Loading dataset...")
    train_ds, val_ds = create_splits(args.data_dir, val_ratio=0.2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    num_classes = len(set(train_ds.dataset.classes))
    print(f"Found {num_classes} currency classes")

    model = CurrencyClassifier(num_classes=num_classes)
    print(f"Trainable parameters: {model.get_trainable_params():,}")

    # Train
    history = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        save_dir=args.save_dir, patience=args.patience
    )

    print("\nTraining complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.1f}%")


if __name__ == "__main__":
    main()
