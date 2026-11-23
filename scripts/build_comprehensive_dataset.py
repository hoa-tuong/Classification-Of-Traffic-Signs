import sys
import shutil
import random
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from torchvision import transforms

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_CSV = DATA_DIR / "class_labels.csv"
DETECTION_ROOT = DATA_DIR / "raw" / "traffic-sign-detection-vietnam" / "dataset"

# Map 82 detection classes to official 250 QCVN classes
MAP_82_TO_250 = {
    0: 1,    # P.102: Cấm đi ngược chiều
    1: 128,  # R.301e: Hướng đi phải theo
    2: 44,   # P.127: Tốc độ 40 km/h -> P.127
    3: 10,   # P.107: Cấm ô tô tải và ô tô khách
    4: 7,    # P.106a: Cấm xe ô tô tải
    5: 27,   # P.117: Hạn chế chiều cao
    6: 2,    # P.103a: Cấm xe ô tô
    7: 216,  # W.233: Nguy hiểm khác
    8: 199,  # W.221b: Đoạn đường có gờ giảm tốc
    9: 158,  # W.202a: Nhiều chỗ ngoặt nguy hiểm
    10: 118, # IE.468a: Chướng ngại vật phía trước
    12: 44,  # P.127: Tốc độ 60 km/h -> P.127
    13: 5,   # P.104: Cấm mô tô
    15: 27,  # P.117: Hạn chế chiều cao
    16: 50,  # P.128: Cấm bóp còi
    17: 32,  # P.123a: Cấm rẽ trái
    18: 33,  # P.123b: Cấm rẽ phải
    19: 34,  # P.124a1: Cấm quay đầu xe
    20: 38,  # P.124c: Cấm quay đầu và rẽ trái
    21: 39,  # P.124d: Cấm quay đầu và rẽ phải
    22: 182, # W.209: Giao nhau có tín hiệu đèn
    23: 52,  # P.130: Cấm dừng xe và đỗ xe
    24: 53,  # P.131a: Cấm đỗ xe
    25: 63,  # P.139: Cấm đi thẳng và rẽ phải
    26: 61,  # P.137: Cấm rẽ trái và phải
    27: 155, # W.201b: Chỗ ngoặt nguy hiểm bên trái
    28: 154, # W.201a: Chỗ ngoặt nguy hiểm bên phải
    29: 170, # W.207a: Giao nhau với đường không ưu tiên
    30: 164, # W.205a: Giao nhau cùng cấp
    31: 5,   # P.104: Cấm xe máy
    32: 181, # W.208: Giao nhau với đường ưu tiên
    33: 137, # R.305: Đường dành cho người đi bộ
    34: 161, # W.203b: Đường hẹp bên trái
    35: 162, # W.203c: Đường hẹp bên phải
    36: 160, # W.203a: Đường hẹp hai bên
    37: 6,   # P.105: Cấm xe ô tô và xe máy
    38: 199, # W.221b: Gờ giảm tốc
    39: 44,  # P.127: Tốc độ 50 km/h -> P.127
    40: 44,  # P.127: Tốc độ 70 km/h -> P.127
    41: 44,  # P.127: Tốc độ 80 km/h -> P.127
    42: 183, # W.210: Giao nhau với đường sắt có rào chắn
    43: 36,  # P.124b1: Cấm ô tô quay đầu
    44: 11,  # P.107a: Cấm ô tô khách
    45: 42,  # P.125: Cấm vượt
    46: 205, # W.225: Trẻ em
    47: 204, # W.224: Người đi bộ cắt ngang
    48: 59,  # P.135: Hết tất cả các lệnh cấm
    49: 127, # R.301d: Đi về bên trái
    50: 207, # W.227: Công trường
    51: 71,  # I.407a: Đường một chiều
    52: 126, # R.301c: Hướng đi phải theo (rẽ trái)
    53: 125, # R.301b: Hướng đi phải theo (rẽ phải)
    56: 135, # R.303: Vòng xuyến
    57: 44,  # P.127: Tốc độ 10 km/h -> P.127
    58: 44,  # P.127: Tốc độ 100 km/h -> P.127
    59: 44,  # P.127: Tốc độ 110 km/h -> P.127
    60: 44,  # P.127: Tốc độ 120 km/h -> P.127
    61: 44,  # P.127: Tốc độ 20 km/h -> P.127
    62: 44,  # P.127: Tốc độ 30 km/h -> P.127
    63: 44,  # P.127: Tốc độ 90 km/h -> P.127
    64: 123, # R.122: Dừng lại (Stop)
    65: 76,  # I.409: Chỗ quay xe
    66: 54,  # P.131b: Cấm đỗ ngày lẻ
    67: 55,  # P.131c: Cấm đỗ ngày chẵn
    68: 74,  # I.408: Nơi đỗ xe
    71: 40,  # P.124e: Cấm ô tô rẽ trái và quay đầu
    72: 230, # W.244: Đoạn đường hay xảy ra tai nạn
    73: 218, # W.235: Đường đôi
    74: 4,   # P.103c: CẤM XE Ô TÔ RẼ TRÁI
    75: 197, # W.220: Dốc lên nguy hiểm
    76: 186, # W.212: Cầu hẹp
    77: 198, # W.221a: Đường không bằng phẳng
    79: 150, # R.420: Bắt đầu khu đông dân cư
    80: 151, # R.421: Hết khu đông dân cư
    81: 200, # W.222a: Đường trơn
}

def generate_natural_background(w: int, h: int) -> np.ndarray:
    bg_type = random.choice(["sky", "asphalt", "greenery", "building", "gradient", "noisy"])
    if bg_type == "sky":
        c1 = np.array([random.randint(180, 255), random.randint(190, 240), random.randint(110, 180)], dtype=np.float32)
        c2 = np.array([random.randint(220, 255), random.randint(220, 255), random.randint(200, 255)], dtype=np.float32)
        grad = np.linspace(0, 1, h)[:, None, None]
        bg = (c1 * (1 - grad) + c2 * grad).astype(np.uint8)
        bg = np.tile(bg, (1, w, 1))
    elif bg_type == "asphalt":
        gray_val = random.randint(90, 160)
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        bg = np.clip(gray_val + noise, 0, 255).astype(np.uint8)
    elif bg_type == "greenery":
        base_g = np.array([random.randint(20, 60), random.randint(80, 150), random.randint(20, 60)], dtype=np.int16)
        noise = np.random.randint(-25, 25, (h, w, 3), dtype=np.int16)
        bg = np.clip(base_g + noise, 0, 255).astype(np.uint8)
    elif bg_type == "building":
        base_b = np.array([random.randint(140, 200), random.randint(150, 210), random.randint(160, 220)], dtype=np.int16)
        noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
        bg = np.clip(base_b + noise, 0, 255).astype(np.uint8)
    elif bg_type == "gradient":
        c1 = np.random.randint(80, 220, 3).astype(np.float32)
        c2 = np.random.randint(80, 220, 3).astype(np.float32)
        grad = np.linspace(0, 1, w)[None, :, None]
        bg = (c1 * (1 - grad) + c2 * grad).astype(np.uint8)
        bg = np.tile(bg, (h, 1, 1))
    else:
        bg = np.random.randint(80, 200, (h, w, 3), dtype=np.uint8)

    return cv2.GaussianBlur(bg, (5, 5), 0)

def get_template_alpha_mask(bgr_img: np.ndarray) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for cx, cy in corners:
        if np.max(bgr_img[cy, cx]) < 30:
            cv2.floodFill(bgr_img.copy(), flood_mask, (cx, cy), 0,
                          loDiff=(20, 20, 20), upDiff=(20, 20, 20))
    outer_bg = flood_mask[1:-1, 1:-1]
    return np.where(outer_bg > 0, 0, 255).astype(np.uint8)

def augment_template(template_path: Path, target_size=(64, 64)) -> Image.Image:
    bgr = cv2.imread(str(template_path))
    h, w = bgr.shape[:2]
    sign_mask = get_template_alpha_mask(bgr)
    natural_bg = generate_natural_background(w, h)
    
    alpha = (sign_mask[:, :, None] / 255.0).astype(np.float32)
    composite = (bgr.astype(np.float32) * alpha + natural_bg.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    
    # Random diagonal shadow
    if random.random() < 0.35:
        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h
        s_mask = np.ones((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                if (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) > 0:
                    s_mask[y, x] = random.uniform(0.6, 0.75)
        composite = (composite.astype(np.float32) * cv2.GaussianBlur(s_mask, (15, 15), 0)[:, :, None]).clip(0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)

    # Rotation (-22 to +22)
    angle = random.uniform(-22, 22)
    img_pil = img_pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))

    # 3D Perspective
    pt = transforms.RandomPerspective(distortion_scale=random.uniform(0.15, 0.35), p=0.85, fill=(128, 128, 128))
    img_pil = pt(img_pil)

    # Blur
    if random.random() < 0.3:
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))

    # Color jitter
    img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.75, 1.25))
    img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.8, 1.25))

    return img_pil.resize(target_size, Image.Resampling.LANCZOS)

def main():
    random.seed(42)
    np.random.seed(42)
    print("=================================================================")
    print("  BUILDING COMPREHENSIVE DATASET (REAL STREET CROPS + REALISTIC SYNTH)")
    print("=================================================================")

    df_labels = pd.read_csv(LABELS_CSV)
    assert len(df_labels) == 250
    print(f"[+] Loaded 250 official classes (0 to 249).")

    # Clean processed dir
    for split in ["train", "val", "test"]:
        s_dir = PROCESSED_DIR / split
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate realistic synthetic samples for all 250 classes
    tpl_map = {int(f.stem.split("_")[0]): f for f in TEMPLATES_DIR.glob("*.png")}
    synthetic_records = []
    
    synth_splits = {"train": 10, "val": 2, "test": 2}
    print("\n[+] Generating realistic synthetic samples for 250 classes (with natural backgrounds & 3D perspective)...")
    for split, count in synth_splits.items():
        for cid in tqdm(range(250), desc=f"Synth {split}"):
            tpl_path = tpl_map[cid]
            c_dir = PROCESSED_DIR / split / str(cid)
            c_dir.mkdir(parents=True, exist_ok=True)
            row = df_labels[df_labels["class_id"] == cid].iloc[0]

            for i in range(count):
                aug_img = augment_template(tpl_path, target_size=(64, 64))
                fname = f"synth_{cid}_{split}_{i}.jpg"
                dest = c_dir / fname
                aug_img.save(dest, quality=92)

                synthetic_records.append({
                    "crop_rel_path": f"processed/{split}/{cid}/{fname}",
                    "split": split,
                    "class_id": cid,
                    "sign_code": row["sign_code"],
                    "sign_name_vi": row["sign_name_vi"],
                    "category": row["category"],
                    "crop_w": 64,
                    "crop_h": 64
                })
    print(f"[+] Generated {len(synthetic_records)} realistic synthetic samples!")

    # Step 2: Extract real street crops from Kaggle Vietnam Traffic Sign dataset
    real_records = []
    max_real = {"train": 20, "val": 4, "test": 4}
    
    if DETECTION_ROOT.exists():
        print(f"\n[+] Extracting real-world Vietnamese street crops from {DETECTION_ROOT}...")
        for split in ["train", "val", "test"]:
            split_dir = DETECTION_ROOT / split
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if not lbl_dir.exists():
                continue

            lbl_files = list(lbl_dir.glob("*.txt"))
            random.shuffle(lbl_files)
            counts_map = {cid: 0 for cid in range(250)}
            max_p_cls = max_real[split]

            for lbl_path in tqdm(lbl_files, desc=f"Real crops {split}"):
                img_path = img_dir / f"{lbl_path.stem}.jpg"
                if not img_path.exists():
                    img_path = img_dir / f"{lbl_path.stem}.png"
                if not img_path.exists():
                    continue

                try:
                    im = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                w, h = im.size
                text = lbl_path.read_text().strip()
                boxes = []
                if text:
                    for line in text.splitlines():
                        p = line.strip().split()
                        if len(p) >= 5:
                            det_cid = int(p[0])
                            xc, yc, bw, bh = map(float, p[1:5])
                            xmin = max(0, int((xc - bw/2) * w))
                            ymin = max(0, int((yc - bh/2) * h))
                            xmax = min(w, int((xc + bw/2) * w))
                            ymax = min(h, int((yc + bh/2) * h))
                            boxes.append((det_cid, xmin, ymin, xmax, ymax))

                for det_cid, xmin, ymin, xmax, ymax in boxes:
                    bw_px = xmax - xmin
                    bh_px = ymax - ymin
                    if bw_px < 18 or bh_px < 18:
                        continue
                    if det_cid not in MAP_82_TO_250:
                        continue
                    target_cid = MAP_82_TO_250[det_cid]
                    if target_cid >= 250:
                        continue

                    if counts_map[target_cid] >= max_p_cls:
                        continue

                    pad_x = int(bw_px * 0.1)
                    pad_y = int(bh_px * 0.1)
                    px_min = max(0, xmin - pad_x)
                    py_min = max(0, ymin - pad_y)
                    px_max = min(w, xmax + pad_x)
                    py_max = min(h, ymax + pad_y)

                    crop = im.crop((px_min, py_min, px_max, py_max)).resize((64, 64), Image.Resampling.LANCZOS)
                    fname = f"real_{target_cid}_{split}_{counts_map[target_cid]}.jpg"
                    dest = PROCESSED_DIR / split / str(target_cid) / fname
                    crop.save(dest, quality=95)

                    counts_map[target_cid] += 1
                    row = df_labels[df_labels["class_id"] == target_cid].iloc[0]
                    real_records.append({
                        "crop_rel_path": f"processed/{split}/{target_cid}/{fname}",
                        "split": split,
                        "class_id": target_cid,
                        "sign_code": row["sign_code"],
                        "sign_name_vi": row["sign_name_vi"],
                        "category": row["category"],
                        "crop_w": 64,
                        "crop_h": 64
                    })

        print(f"[+] Successfully extracted {len(real_records)} real street crops!")
    else:
        print(f"[!] Warning: DETECTION_ROOT not found at {DETECTION_ROOT}")

    # Combine all records
    all_records = synthetic_records + real_records
    df_all = pd.DataFrame(all_records)
    summary_path = PROCESSED_DIR / "dataset_summary.csv"
    df_all.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=================================================================")
    print(f"  DATASET CREATION COMPLETE: {len(df_all)} TOTAL SAMPLES")
    print(f"  - Train: {len(df_all[df_all['split'] == 'train'])}")
    print(f"  - Val:   {len(df_all[df_all['split'] == 'val'])}")
    print(f"  - Test:  {len(df_all[df_all['split'] == 'test'])}")
    print(f"  - Classes covered: {df_all['class_id'].nunique()}/250")
    print("=================================================================")

if __name__ == "__main__":
    main()
