from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.metrics import calculate_classification_metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float, list[int], list[int], list[list[float]]]:
    """
    Danh gia mo hinh tren tap du lieu (Test/Validation).

    Returns:
        tuple (avg_loss, accuracy, all_preds, all_targets, all_probs)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy, all_preds, all_targets, all_probs


def evaluate_and_report(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_names: list[str] | None = None,
    save_report_path: str | Path | None = None
) -> dict:
    """
    Danh gia mo hinh va tao bao cao chi tiet (Accuracy, Precision, Recall, F1).
    Tu dong luu bao cao vao file neu save_report_path duoc chi dinh.
    """
    avg_loss, accuracy, all_preds, all_targets, _ = evaluate_model(model, dataloader, criterion, device)
    metrics = calculate_classification_metrics(all_targets, all_preds, target_names=target_names)
    metrics["test_loss"] = avg_loss

    if save_report_path:
        out_path = Path(save_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Độ chính xác (Accuracy): {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Macro F1-Score: {metrics['macro_f1']*100:.2f}%\n")
            f.write(f"Macro Precision: {metrics['macro_precision']*100:.2f}%\n")
            f.write(f"Macro Recall: {metrics['macro_recall']*100:.2f}%\n")
            f.write(f"Test Loss: {metrics['test_loss']:.4f}\n\n")
            f.write("Chi tiết từng lớp:\n")
            f.write(metrics["report_text"])

    return metrics
