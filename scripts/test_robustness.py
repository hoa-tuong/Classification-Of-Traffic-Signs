import sys
from pathlib import Path
import torch
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transfer_models import get_transfer_model
from src.data.transforms import get_val_transforms

def main():
    print("=======================================================")
    print("  TESTING PURE NEURAL NETWORK (NO DATASET MATCHER)")
    print("=======================================================")

    device = torch.device("cpu")
    model = get_transfer_model("mobilenet_v3", 250, pretrained=False)
    ckpt = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = get_val_transforms((64, 64))
    df_labels = pd.read_csv("data/class_labels.csv")
    name_map = {int(row["class_id"]): row["sign_name_vi"] for _, row in df_labels.iterrows()}

    # Test samples
    test_paths = [
        ("Real Vietnamese street crop (Class 1: Cấm ngược chiều)", "data/processed/test/1/real_1_test_0.jpg", 1),
        ("Realistic synth with natural background (Class 0: Đường cấm)", "data/processed/test/0/synth_0_test_0.jpg", 0),
        ("Realistic synth with 3D angle (Class 124: Đi thẳng)", "data/processed/test/124/synth_124_test_0.jpg", 124),
        ("Real Vietnamese street crop (Class 10: Cấm xe tải/khách)", "data/processed/test/10/real_10_test_0.jpg", 10),
    ]

    for desc, path_str, true_cid in test_paths:
        p = PROJECT_ROOT / path_str
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        tensor = transform(im).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, 5)

        top1_id = top_indices[0].item()
        top1_prob = top_probs[0].item() * 100
        is_correct = (top1_id == true_cid)
        status = "PASSED (CORRECT)" if is_correct else "FAILED"

        print(f"\n[Test Case] {desc}")
        print(f"  File: {p.name}")
        print(f"  True Class ID: {true_cid} ({name_map.get(true_cid, '')})")
        print(f"  AI Predicted:  {top1_id} ({name_map.get(top1_id, '')})")
        print(f"  Top 1 Prob:    {top1_prob:.2f}% | Status: {status}")
        print("  Top 3 Candidates:")
        for rank, (cid, pr) in enumerate(zip(top_indices[:3], top_probs[:3]), start=1):
            print(f"    {rank}. [{cid.item()}] {name_map.get(cid.item(), '')}: {pr.item()*100:.2f}%")

if __name__ == "__main__":
    main()
