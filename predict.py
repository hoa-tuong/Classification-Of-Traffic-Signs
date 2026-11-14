import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.transforms import get_val_transforms
from src.models.custom_cnn import TrafficSignCNN


def load_class_mapping(csv_path: str | Path) -> dict:
    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        return dict(zip(df["class_id"], df["sign_name"]))
    return {}


def predict_image(image_path: str, model_path: str, config_path: str = "configs/config.yaml", top_k: int = 3):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["project"]["device"] == "cuda" else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    num_classes = len(checkpoint.get("classes", [])) or cfg["model"]["num_classes"]

    model = TrafficSignCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    img_size = tuple(cfg["data"]["img_size"])
    transform = get_val_transforms(img_size)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, num_classes))

    class_names = load_class_mapping(cfg["data"]["labels_path"])

    print(f"\n[+] Ket qua du doan cho anh: {image_path}")
    for rank, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), start=1):
        class_id = idx.item()
        name = class_names.get(class_id, f"Lop {class_id}")
        print(f"Top {rank}: {name} (Class ID: {class_id}) - Do tin cay: {prob.item()*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Du doan bien bao tu anh bat ky")
    parser.add_argument("--image", type=str, required=True, help="Duong dan toi file anh can du doan")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Duong dan toi file model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Duong dan toi file config")
    parser.add_argument("--top_k", type=int, default=3, help="So luong top du doan can hien thi")
    args = parser.parse_args()

    predict_image(args.image, args.model, args.config, args.top_k)

