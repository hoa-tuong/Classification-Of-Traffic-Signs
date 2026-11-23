import os
import random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_CSV = DATA_DIR / "class_labels.csv"

def ensure_rgb(image: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, bg_color)
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return image.convert("RGB")

def augment_image(img: Image.Image, max_angle: float = 20.0) -> Image.Image:
    # Random rotation (-max_angle to +max_angle degrees) with white background
    angle = random.uniform(-max_angle, max_angle)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
    
    # Random brightness (0.85 to 1.15)
    enh_b = ImageEnhance.Brightness(img)
    img = enh_b.enhance(random.uniform(0.85, 1.15))
    
    # Random contrast (0.85 to 1.15)
    enh_c = ImageEnhance.Contrast(img)
    img = enh_c.enhance(random.uniform(0.85, 1.15))
    
    return img

def main():
    random.seed(42)
    df_labels = pd.read_csv(LABELS_CSV)
    print(f"[+] Loaded {len(df_labels)} classes from {LABELS_CSV}")

    # Map class_id to template path
    tpl_map = {}
    for f in TEMPLATES_DIR.glob("*.png"):
        cid = int(f.stem.split("_")[0])
        tpl_map[cid] = f

    splits_config = {
        "train": 12, # 12 samples per class with rich rotations (-20 to +20)
        "val": 2,    # 2 samples per class (1 upright, 1 rotated)
        "test": 2    # 2 samples per class (1 upright, 1 rotated)
    }

    records = []
    for split, count in splits_config.items():
        print(f"[+] Generating {count} samples per class for split '{split}'...")
        for cid in range(250):
            row = df_labels[df_labels["class_id"] == cid].iloc[0]
            tpl_path = tpl_map[cid]
            base_img = ensure_rgb(Image.open(tpl_path))

            dest_dir = PROCESSED_DIR / split / str(cid)
            dest_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                if i == 0:
                    # Clean upright sample
                    aug_img = base_img.copy()
                elif i == 1:
                    # Mild rotation (-8 to +8)
                    aug_img = augment_image(base_img.copy(), max_angle=8.0)
                else:
                    # Stronger rotation (-20 to +20)
                    aug_img = augment_image(base_img.copy(), max_angle=20.0)

                aug_img = aug_img.resize((64, 64), Image.Resampling.LANCZOS)
                fname = f"{cid}_{split}_{i}.jpg"
                dest_file = dest_dir / fname
                aug_img.save(dest_file, quality=92)

                rel_path = f"processed/{split}/{cid}/{fname}"
                records.append({
                    "crop_rel_path": rel_path,
                    "split": split,
                    "class_id": cid,
                    "sign_code": row["sign_code"],
                    "sign_name_vi": row["sign_name_vi"],
                    "category": row["category"],
                    "crop_w": 64,
                    "crop_h": 64
                })

    df_summary = pd.DataFrame(records)
    summary_path = PROCESSED_DIR / "dataset_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[+] Saved {len(df_summary)} samples to {summary_path}")
    print(f"    - Train: {len(df_summary[df_summary['split'] == 'train'])}")
    print(f"    - Val:   {len(df_summary[df_summary['split'] == 'val'])}")
    print(f"    - Test:  {len(df_summary[df_summary['split'] == 'test'])}")

if __name__ == "__main__":
    main()
