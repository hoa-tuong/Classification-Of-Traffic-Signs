import sys
import shutil
import random
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DETECTION_ROOT = Path(r"D:\Code\Traffic-sign-detection-VietNam\traffic-sign-detection-vietnam\dataset")

# Map 82 detection classes to the 250 official QCVN classes
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


def main():
    random.seed(42)
    print("=================================================================")
    print("  REBUILDING DATASET FOR EXACTLY 250 OFFICIAL VIETNAM CLASSES")
    print("=================================================================")

    labels_csv = DATA_DIR / "class_labels.csv"
    df_labels = pd.read_csv(labels_csv)
    assert len(df_labels) == 250, f"Expected 250 classes, got {len(df_labels)}"
    print(f"[+] Verified {len(df_labels)} official classes (0 to 249).")

    # Step 1: Remove any folders >= 250 in processed splits
    print("[+] Removing any class folders >= 250 in processed splits...")
    for split in ["train", "val", "test"]:
        s_dir = PROCESSED_DIR / split
        if s_dir.exists():
            for folder in s_dir.iterdir():
                if folder.is_dir() and folder.name.isdigit():
                    if int(folder.name) >= 250:
                        shutil.rmtree(folder)
                        print(f"    - Removed {folder}")

    # Remove previous real_* and bg_* crops from classes 0 to 249
    print("[+] Removing old real_* and bg_* files...")
    del_count = 0
    for p in PROCESSED_DIR.glob("*/*/*"):
        if p.is_file() and (p.name.startswith("real_") or p.name.startswith("bg_")):
            p.unlink()
            del_count += 1
    print(f"[+] Removed {del_count} old crop files.")

    # Step 2: Load base synthetic samples for classes 0 to 249
    base_records = []
    for split in ["train", "val", "test"]:
        for cid in range(250):
            c_dir = PROCESSED_DIR / split / str(cid)
            if not c_dir.exists():
                continue
            row = df_labels[df_labels["class_id"] == cid].iloc[0]
            for img_file in c_dir.glob("*.jpg"):
                if not (img_file.name.startswith("real_") or img_file.name.startswith("bg_")):
                    base_records.append({
                        "crop_rel_path": str(img_file.relative_to(DATA_DIR)),
                        "split": split,
                        "class_id": cid,
                        "sign_code": row["sign_code"],
                        "sign_name_vi": row["sign_name_vi"],
                        "category": row["category"],
                        "crop_w": 64,
                        "crop_h": 64
                    })
    print(f"[+] Loaded {len(base_records)} canonical synthetic samples across 250 classes.")

    # Step 3: Extract real-world street crops from Traffic-sign-detection-VietNam
    max_real_train = 25
    max_real_val = 5
    max_real_test = 5

    counts_train = {cid: 0 for cid in range(250)}
    counts_val = {cid: 0 for cid in range(250)}
    counts_test = {cid: 0 for cid in range(250)}

    real_records = []

    for split in ["train", "val", "test"]:
        split_dir = DETECTION_ROOT / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not lbl_dir.exists():
            continue

        lbl_files = list(lbl_dir.glob("*.txt"))
        random.shuffle(lbl_files)

        counts_map = counts_train if split == "train" else (counts_val if split == "val" else counts_test)
        max_per_class = max_real_train if split == "train" else (max_real_val if split == "val" else max_real_test)

        print(f"\n[+] Extracting real street crops for split '{split}' ({len(lbl_files)} images)...")
        for lbl_path in tqdm(lbl_files, desc=f"Cropping {split}"):
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

                if counts_map[target_cid] >= max_per_class:
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
                row_label = df_labels[df_labels["class_id"] == target_cid].iloc[0]
                real_records.append({
                    "crop_rel_path": str(dest.relative_to(DATA_DIR)),
                    "split": split,
                    "class_id": target_cid,
                    "sign_code": row_label["sign_code"],
                    "sign_name_vi": row_label["sign_name_vi"],
                    "category": row_label["category"],
                    "crop_w": 64,
                    "crop_h": 64
                })

    print(f"\n[+] Created {len(real_records)} real-world street crops!")

    # Step 4: Save dataset_summary.csv
    all_records = base_records + real_records
    df_all_summary = pd.DataFrame(all_records)
    summary_csv = PROCESSED_DIR / "dataset_summary.csv"
    df_all_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[+] Saved {summary_csv}: Total {len(df_all_summary)} samples across exactly {df_all_summary['class_id'].nunique()} official classes!")
    print(f"    - Train: {len(df_all_summary[df_all_summary['split'] == 'train'])}")
    print(f"    - Val:   {len(df_all_summary[df_all_summary['split'] == 'val'])}")
    print(f"    - Test:  {len(df_all_summary[df_all_summary['split'] == 'test'])}")

    # Step 5: Update configs/config.yaml num_classes: 250
    config_yaml = PROJECT_ROOT / "configs" / "config.yaml"
    cfg_text = config_yaml.read_text(encoding="utf-8")
    for old_num in ["268", "251"]:
        cfg_text = cfg_text.replace(f"num_classes: {old_num}", "num_classes: 250")
    config_yaml.write_text(cfg_text, encoding="utf-8")
    print("[+] Updated config.yaml: num_classes = 250!")


if __name__ == "__main__":
    main()
