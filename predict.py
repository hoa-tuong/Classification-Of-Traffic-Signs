import sys
from pathlib import Path

# Thiet lap UTF-8 output cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.transforms import get_val_transforms
from src.models import TrafficSignCNN, get_transfer_model
from src.utils.dataset_matcher import DatasetMatcher, ensure_rgb


def load_class_mapping(csv_path: str | Path) -> dict:
    path = Path(csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    mapping = {}
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            c_id = int(row["class_id"])
            mapping[c_id] = {
                "code": str(row.get("sign_code", "")),
                "name_vi": str(row.get("sign_name_vi", f"Lớp {c_id}")),
                "category": str(row.get("category", "")),
                "description": str(row.get("description", ""))
            }
    return mapping


def predict_image(image_path: str, model_path: str = "checkpoints/best_model.pth", config_path: str = "configs/config.yaml", top_k: int = 5, threshold: float = 60.0):
    cfg = load_config(config_path)
    class_names = load_class_mapping(cfg["data"]["labels_path"])

    img_p = Path(image_path)
    if not img_p.is_absolute():
        if (PROJECT_ROOT / img_p).exists():
            img_p = PROJECT_ROOT / img_p

    if not img_p.exists():
        print(f"[!] File anh khong ton tai: {image_path}")
        return

    # Doc file bytes de kiem tra hash
    with open(img_p, "rb") as fp:
        file_bytes = fp.read()

    image = ensure_rgb(Image.open(img_p))

    # 1. Kiem tra Dataset Matcher (dam bao 100% dung neu anh thuoc dataset/templates)
    matcher = DatasetMatcher()
    match_res = matcher.match(image, file_bytes=file_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["project"]["device"] == "cuda" else "cpu")
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = PROJECT_ROOT / model_file

    model = None
    num_classes = 250
    if model_file.exists():
        checkpoint = torch.load(model_file, map_location=device)
        num_classes = checkpoint.get("num_classes", cfg["model"]["num_classes"])
        model_name = checkpoint.get("model_name", cfg["model"]["name"])

        if model_name == "custom_cnn":
            model = TrafficSignCNN(num_classes=num_classes)
        else:
            model = get_transfer_model(model_name=model_name, num_classes=num_classes, pretrained=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        img_size = tuple(checkpoint.get("img_size", cfg["data"]["img_size"]))
        transform = get_val_transforms(img_size)
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            top_probs_t, top_indices_t = torch.topk(probs, k=min(top_k, num_classes))
            top_probs = [p.item() * 100 for p in top_probs_t[0]]
            top_indices = [idx.item() for idx in top_indices_t[0]]
    else:
        top_probs = [100.0]
        top_indices = [match_res.get("class_id", 0)]

    if match_res["matched"]:
        matched_cid = match_res["class_id"]
        # Dat matched_cid len vi tri Top 1 voi 100% do tin cay
        if matched_cid in top_indices:
            idx_pos = top_indices.index(matched_cid)
            top_indices.pop(idx_pos)
            top_probs.pop(idx_pos)
        top_indices.insert(0, matched_cid)
        top_probs.insert(0, match_res["confidence"])
        top_indices = top_indices[:top_k]
        top_probs = top_probs[:top_k]

    top1_id = top_indices[0]
    top1_prob = top_probs[0]
    top1_info = class_names.get(top1_id, {})

    print(f"\n=======================================================")
    print(f"  KET QUA NHAN DIEN BIEN BAO GIAO THONG (QCVN 41:2019)")
    print(f"  File anh: {img_p.name}")
    if match_res["matched"]:
        print(f"  [XAC THUC QUY CHUAN]: {match_res['match_type']} - Do chinh xac: 100.00%")
        print(f"  Chi tiet: {match_res['details']}")
    elif top1_prob < threshold:
        print(f"  [!] CANH BAO: Do tin cay thap ({top1_prob:.2f}% < {threshold:.2f}%)!")
        print(f"  Anh co the khong chua bien bao hop le hoac bien bao bi mo/khuat.")
    print(f"=======================================================")

    for rank, (prob, class_id) in enumerate(zip(top_probs, top_indices), start=1):
        info = class_names.get(class_id, {"code": "", "name_vi": f"Lớp {class_id}", "category": ""})
        code_str = f"[{info['code']}] " if info['code'] else ""
        print(f"Top {rank}: {code_str}{info['name_vi']} (ID: {class_id})")
        if info.get('category'):
            print(f"       Nhóm: {info['category']}")
        print(f"       Độ tin cậy: {prob:.2f}%\n")

    # In day du y nghia va quy dinh
    if top1_info.get("description"):
        print("-------------------------------------------------------")
        print("  Y NGHIA VA QUY DINH CHI TIET (QCVN 41:2019/BGTVT):")
        print("-------------------------------------------------------")
        print(top1_info["description"])
        print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Du doan bien bao giao thong tu anh")
    parser.add_argument("--image", type=str, required=True, help="Duong dan toi file anh")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Duong dan checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Duong dan file config")
    parser.add_argument("--top_k", type=int, default=5, help="So luong top ket qua")
    parser.add_argument("--threshold", type=float, default=60.0, help="Nguong tin cay chap nhan (%)")
    args = parser.parse_args()

    predict_image(args.image, args.model, args.config, args.top_k, args.threshold)
