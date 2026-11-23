import os
import random
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_CSV = DATA_DIR / "class_labels.csv"

# Pre-generate diverse natural background types (Sky, Road, Trees, Buildings, Gradients)
def generate_natural_background(w: int, h: int) -> np.ndarray:
    bg_type = random.choice(["sky", "asphalt", "greenery", "building", "gradient", "noisy"])
    
    if bg_type == "sky":
        # Sky: blue / white / gray gradient
        c1 = np.array([random.randint(180, 255), random.randint(190, 240), random.randint(110, 180)], dtype=np.float32) # BGR
        c2 = np.array([random.randint(220, 255), random.randint(220, 255), random.randint(200, 255)], dtype=np.float32)
        grad = np.linspace(0, 1, h)[:, None, None]
        bg = (c1 * (1 - grad) + c2 * grad).astype(np.uint8)
        bg = np.tile(bg, (1, w, 1))

    elif bg_type == "asphalt":
        # Road / concrete: grayish with slight noise
        gray_val = random.randint(90, 160)
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        bg = np.clip(gray_val + noise, 0, 255).astype(np.uint8)

    elif bg_type == "greenery":
        # Trees / foliage: greens and dark yellows
        base_g = np.array([random.randint(20, 60), random.randint(80, 150), random.randint(20, 60)], dtype=np.int16) # BGR
        noise = np.random.randint(-25, 25, (h, w, 3), dtype=np.int16)
        bg = np.clip(base_g + noise, 0, 255).astype(np.uint8)

    elif bg_type == "building":
        # Buildings: beige, cream, brick, glass tones
        base_b = np.array([random.randint(140, 200), random.randint(150, 210), random.randint(160, 220)], dtype=np.int16)
        noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
        bg = np.clip(base_b + noise, 0, 255).astype(np.uint8)

    elif bg_type == "gradient":
        # Random multi-color gradient
        c1 = np.random.randint(80, 220, 3).astype(np.float32)
        c2 = np.random.randint(80, 220, 3).astype(np.float32)
        grad = np.linspace(0, 1, w)[None, :, None]
        bg = (c1 * (1 - grad) + c2 * grad).astype(np.uint8)
        bg = np.tile(bg, (h, 1, 1))

    else: # noisy
        bg = np.random.randint(80, 200, (h, w, 3), dtype=np.uint8)

    # Slight blur to background so it looks naturally out-of-focus
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    return bg

def get_template_alpha_mask(bgr_img: np.ndarray) -> np.ndarray:
    """Find the outside black/background corners using 4-corner flood-fill."""
    h, w = bgr_img.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Check corner pixels to determine background threshold
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    corner_colors = [bgr_img[y, x] for x, y in corners]
    
    # If corners are near black (< 25)
    for cx, cy in corners:
        if np.max(bgr_img[cy, cx]) < 25:
            cv2.floodFill(bgr_img.copy(), flood_mask, (cx, cy), 0,
                          loDiff=(20, 20, 20), upDiff=(20, 20, 20))
            
    outer_bg = flood_mask[1:-1, 1:-1]
    # Invert: sign pixels are 255, background is 0
    sign_mask = np.where(outer_bg > 0, 0, 255).astype(np.uint8)
    return sign_mask

def apply_realistic_perspective(img_pil: Image.Image, max_distortion: float = 0.3) -> Image.Image:
    """Applies realistic 3D perspective distortion (viewing sign from below / side)."""
    pt = transforms.RandomPerspective(distortion_scale=max_distortion, p=1.0, fill=(128, 128, 128))
    return pt(img_pil)

def apply_random_shadow(img_bgr: np.ndarray) -> np.ndarray:
    """Simulates realistic diagonal shadow from tree branch or overhang."""
    if random.random() > 0.4:
        return img_bgr
    h, w = img_bgr.shape[:2]
    # Create diagonal shadow mask
    x1, y1 = random.randint(0, w), 0
    x2, y2 = random.randint(0, w), h
    mask = np.ones((h, w), dtype=np.float32)
    # Gradient shadow
    for y in range(h):
        for x in range(w):
            val = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if val > 0:
                mask[y, x] = random.uniform(0.55, 0.75)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    shadowed = (img_bgr.astype(np.float32) * mask[:, :, None]).clip(0, 255).astype(np.uint8)
    return shadowed

def augment_template_to_realistic(template_path: Path, target_size=(64, 64), is_eval=False) -> Image.Image:
    bgr = cv2.imread(str(template_path))
    h, w = bgr.shape[:2]
    
    # 1. Extract sign mask
    sign_mask = get_template_alpha_mask(bgr)
    
    # 2. Generate natural background
    natural_bg = generate_natural_background(w, h)
    
    # 3. Composite sign over natural background
    alpha = (sign_mask[:, :, None] / 255.0).astype(np.float32)
    composite_bgr = (bgr.astype(np.float32) * alpha + natural_bg.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    
    # 4. Apply random shadow
    if not is_eval:
        composite_bgr = apply_random_shadow(composite_bgr)

    # Convert to PIL RGB
    rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)

    # 5. Apply scale and rotation
    if is_eval:
        rot_angle = random.uniform(-15, 15)
    else:
        rot_angle = random.uniform(-25, 25)

    img_pil = img_pil.rotate(rot_angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))

    # 6. Apply 3D Perspective distortion
    if not is_eval or random.random() < 0.5:
        dist_scale = random.uniform(0.15, 0.35) if not is_eval else 0.2
        img_pil = apply_realistic_perspective(img_pil, max_distortion=dist_scale)

    # 7. Apply random blur (Motion blur / Camera blur)
    if random.random() < (0.4 if not is_eval else 0.2):
        blur_radius = random.uniform(0.5, 1.5)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 8. Color jitter (Brightness, contrast, saturation)
    enh_b = ImageEnhance.Brightness(img_pil)
    img_pil = enh_b.enhance(random.uniform(0.75, 1.25) if not is_eval else random.uniform(0.9, 1.1))

    enh_c = ImageEnhance.Contrast(img_pil)
    img_pil = enh_c.enhance(random.uniform(0.8, 1.25))

    # 9. Resize to target size
    img_pil = img_pil.resize(target_size, Image.Resampling.LANCZOS)
    return img_pil

def main():
    import pandas as pd
    from tqdm import tqdm
    
    random.seed(42)
    np.random.seed(42)
    df_labels = pd.read_csv(LABELS_CSV)
    print(f"[+] Loaded {len(df_labels)} official classes.")

    tpl_map = {}
    for f in TEMPLATES_DIR.glob("*.png"):
        cid = int(f.stem.split("_")[0])
        tpl_map[cid] = f

    splits_config = {
        "train": 14, # 14 realistic synthetic variations per class
        "val": 2,    # 2 validation variations per class
        "test": 2    # 2 test variations per class
    }

    records = []
    for split, count in splits_config.items():
        print(f"[+] Generating realistic samples for split '{split}' ({count} per class)...")
        for cid in tqdm(range(250), desc=f"Classes ({split})"):
            row = df_labels[df_labels["class_id"] == cid].iloc[0]
            tpl_path = tpl_map[cid]
            dest_dir = PROCESSED_DIR / split / str(cid)
            dest_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                is_eval = (split != "train")
                img_real = augment_template_to_realistic(tpl_path, target_size=(64, 64), is_eval=is_eval)
                
                fname = f"synth_{cid}_{split}_{i}.jpg"
                dest_file = dest_dir / fname
                img_real.save(dest_file, quality=92)

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
    print(f"[+] Saved {len(df_summary)} realistic samples to {summary_path}")

if __name__ == "__main__":
    main()
