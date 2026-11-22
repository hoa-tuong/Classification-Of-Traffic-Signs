from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history: dict, save_path: str | Path | None = None):
    """Ve bieu do Loss va Accuracy qua cac epoch va luu thanh file anh."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    axes[0].set_title("Loss qua tung Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curve
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], "r--s", label="Val Accuracy")
    axes[1].set_title("Accuracy qua tung Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, num_classes: int | None = None, save_path: str | Path | None = None):
    """Ve va luu bieu do ma tran nham lan (Confusion Matrix)."""
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = max(max(y_true), max(y_pred)) + 1

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.title(f"Confusion Matrix ({num_classes} Classes)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class ID", fontsize=11)
    plt.ylabel("True Class ID", fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

