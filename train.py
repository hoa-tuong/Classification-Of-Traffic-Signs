import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import TrafficSignDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models import TrafficSignCNN, get_transfer_model
from src.training.train import train_one_epoch, validate
from src.utils.metrics import calculate_classification_metrics
from src.utils.visualizer import plot_training_history, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Huan luyen mo hinh phan loai bien bao giao thong")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Duong dan toi file config")
    parser.add_argument("--model", type=str, default=None, help="Chon loai model: custom_cnn, mobilenet_v3, resnet18")
    parser.add_argument("--epochs", type=int, default=None, help="So epoch muon huan luyen")
    args = parser.parse_args()

    # Doc config
    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["project"]["device"] == "cuda" else "cpu")
    print(f"[+] Thiet bi tinh toan dang su dung: {device}")

    # Kiem tra du lieu
    summary_path = Path(cfg["data"]["summary_csv"])
    if not summary_path.exists():
        print(f"[!] Chua tim thay file {summary_path}!")
        print("[i] Vui long chay script tien xu ly truoc: python scripts/prepare_dataset.py")
        return

    # Transforms
    img_size = tuple(cfg["data"]["img_size"])
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)

    # Khoi tao Datasets voi base_dir chuan
    data_base_dir = Path(cfg["project_root"]) / "data"
    train_dataset = TrafficSignDataset(summary_path, base_dir=data_base_dir, split="train", transform=train_transform)
    val_dataset = TrafficSignDataset(summary_path, base_dir=data_base_dir, split="val", transform=val_transform)
    test_dataset = TrafficSignDataset(summary_path, base_dir=data_base_dir, split="test", transform=val_transform)

    print(f"[+] So luong mau: Train = {len(train_dataset)} | Val = {len(val_dataset)} | Test = {len(test_dataset)}")

    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Khoi tao Mo hinh
    num_classes = cfg["model"]["num_classes"]
    model_name = cfg["model"]["name"]
    print(f"[+] Khoi tao kien truc: {model_name} voi {num_classes} lop...")

    if model_name == "custom_cnn":
        model = TrafficSignCNN(num_classes=num_classes, dropout_rate=cfg["model"]["dropout_rate"])
    else:
        model = get_transfer_model(model_name=model_name, num_classes=num_classes, pretrained=True)

    model = model.to(device)

    # Loss va Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["step_size"],
        gamma=cfg["training"]["gamma"]
    )

    # Thu muc checkpoint va output
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path(cfg["paths"]["best_model_path"])

    fig_dir = Path(cfg["paths"]["outputs_figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience = cfg["training"].get("early_stopping_patience", 4)
    patience_counter = 0

    print(f"\n[+] Bat dau huan luyen ({cfg['training']['epochs']} epochs)...")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{cfg['training']['epochs']:02d}] - LR: {lr_curr:.5f} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_val_acc,
                "num_classes": num_classes,
                "img_size": img_size
            }, best_model_path)
            print(f"    --> Da luu trong so tot nhat vao {best_model_path} (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Kich hoat Early Stopping sau {epoch} epochs vi Val Acc khong cai thien.")
                break

    # Ve bieu do huan luyen
    loss_curve_path = fig_dir / "loss_acc_curve.png"
    plot_training_history(history, save_path=loss_curve_path)
    print(f"[+] Bieu do qua trinh huan luyen da luu tai: {loss_curve_path}")

    # Danh gia tren tap Test voi checkpoint tot nhat
    print("\n[+] Bat dau danh gia toan dien tren tap Test...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Doc ten lop
    labels_csv = Path(cfg["data"]["labels_path"])
    class_names = [f"Class {i}" for i in range(num_classes)]
    if labels_csv.exists():
        ldf = pd.read_csv(labels_csv)
        class_names = [f"{r['sign_code']} ({r['class_id']})" for _, r in ldf.iterrows()]

    metrics = calculate_classification_metrics(all_targets, all_preds)
    print(f"[+] Ket qua kiem thu tap Test:")
    print(f"    - Accuracy:        {metrics['accuracy']*100:.2f}%")
    print(f"    - Macro Precision: {metrics['macro_precision']*100:.2f}%")
    print(f"    - Macro Recall:    {metrics['macro_recall']*100:.2f}%")
    print(f"    - Macro F1-score:  {metrics['macro_f1']*100:.2f}%")

    # Luu bao cao phan loai chi tiet
    report_file = Path(cfg["paths"]["classification_report_path"])
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(metrics["report_text"], encoding="utf-8")
    print(f"[+] Bao cao chi tiet da duoc luu tai: {report_file}")

    # Ve Confusion Matrix
    cm_path = fig_dir / "confusion_matrix.png"
    plot_confusion_matrix(all_targets, all_preds, num_classes=num_classes, save_path=cm_path)
    print(f"[+] Ma tran nham lan (Confusion Matrix) da duoc luu tai: {cm_path}")
    print("[+] Chuc mung! Do an da hoan thanh huan luyen va danh gia hoan hao!")


if __name__ == "__main__":
    main()
