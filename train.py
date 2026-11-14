import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from src.config import load_config
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.custom_cnn import TrafficSignCNN
from src.training.train import train_one_epoch, validate
from src.utils.visualizer import plot_training_history


def main():
    parser = argparse.ArgumentParser(description="Huan luyen mo hinh phan loai bien bao")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Duong dan toi file config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() and cfg["project"]["device"] == "cuda" else "cpu")
    print(f"[+] Su dung thiet bi: {device}")

    img_size = tuple(cfg["data"]["img_size"])
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)

    # Kiem tra du lieu
    data_dir = Path(cfg["data"]["processed_dir"])
    if not data_dir.exists() or len(list(data_dir.glob("*"))) == 0:
        raw_dir = Path(cfg["data"]["raw_dir"])
        if raw_dir.exists() and len(list(raw_dir.glob("*"))) > 0:
            data_dir = raw_dir
        else:
            print("[!] Chua tim thay du lieu trong data/raw/ hoac data/processed/.")
            print("[i] Huong dan: Hay copy thu muc anh bien bao (moi class 1 folder) vao data/raw/")
            return

    try:
        full_dataset = ImageFolder(root=data_dir, transform=train_transform)
        num_classes = len(full_dataset.classes)
        print(f"[+] Tim thay {len(full_dataset)} anh voi {num_classes} lop.")
    except Exception as e:
        print(f"[!] Loi khi doc du lieu kieu ImageFolder: {e}")
        return

    # Chia train / val
    val_ratio = cfg["data"]["val_split"]
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    # Khoi tao model
    model = TrafficSignCNN(num_classes=num_classes, dropout_rate=cfg["model"]["dropout_rate"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path(cfg["paths"]["best_model_path"])

    print(f"[+] Bat dau huan luyen trong {cfg['training']['epochs']} epochs...")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{cfg['training']['epochs']:02d}] - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_val_acc,
                "classes": full_dataset.classes
            }, best_model_path)
            print(f"    --> Da luu checkpoint tot nhat tai {best_model_path} (Acc: {best_val_acc*100:.2f}%)")

    # Ve va luu bieu do
    fig_path = Path(cfg["paths"]["outputs_figure_dir"]) / "loss_acc_curve.png"
    plot_training_history(history, save_path=fig_path)
    print(f"[+] Huan luyen hoan tat! Bieu do da duoc luu tai: {fig_path}")


if __name__ == "__main__":
    main()

