from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.transforms import get_val_transforms
from src.models.custom_cnn import TrafficSignCNN

st.set_page_config(page_title="Nhan dien bien bao giao thong", layout="centered")

st.title("Phan Loai Bien Bao Giao Thong (AI Demo)")
st.markdown("Ung dung demo do an mon **Machine Learning / Deep Learning**")


@st.cache_resource
def load_cached_model(model_path: str, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficSignCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


@st.cache_data
def load_labels(csv_path: str) -> dict:
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        return dict(zip(df["class_id"], df["sign_name"]))
    return {}


# Load config
cfg = load_config()
model_path = Path(cfg["paths"]["best_model_path"])
labels_map = load_labels(cfg["data"]["labels_path"])

if not model_path.exists():
    st.warning(f"Chua tim thay file trong so tai `{model_path}`. Hay chay `python train.py` de huan luyen mo hinh truoc!")
else:
    checkpoint = torch.load(model_path, map_location="cpu")
    num_classes = len(checkpoint.get("classes", [])) or cfg["model"]["num_classes"]
    model, device = load_cached_model(str(model_path), num_classes)
    transform = get_val_transforms(tuple(cfg["data"]["img_size"]))

    uploaded_file = st.file_uploader("Tai len anh bien bao (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Anh da tai len", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, k=5)

        with col2:
            st.subheader("Ket qua du doan:")
            results_data = []
            for prob, idx in zip(top_probs, top_indices):
                c_id = idx.item()
                c_name = labels_map.get(c_id, f"Lop {c_id}")
                results_data.append({"Bien bao": c_name, "Do tin cay (%)": round(prob.item() * 100, 2)})

            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, hide_index=True)

            top1_name = results_data[0]["Bien bao"]
            top1_score = results_data[0]["Do tin cay (%)"]
            st.success(f"**Du doan chinh xac nhat:** {top1_name} ({top1_score}%)")

