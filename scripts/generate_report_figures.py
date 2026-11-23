import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path("c:/ML/Classification-Of-Traffic-Signs")
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate Model Comparison Bar Chart
plt.figure(figsize=(10, 5.5), dpi=300)
models = ['Template Matching\n(MSE Baseline)', 'Custom Deep CNN\n(From Scratch)', 'ResNet-18\n(Transfer Learning)', 'MobileNetV3-Small\n(Đồ án lựa chọn)']
acc = [35.0, 41.2, 76.8, 81.56]
f1 = [28.4, 36.8, 74.5, 79.31]
latency = [5.0, 12.0, 28.0, 8.0] # ms

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
rects1 = ax1.bar(x - width/2, acc, width, label='Độ chính xác Test Acc (%)', color='#2b5c8f', edgecolor='black', linewidth=0.8)
rects2 = ax1.bar(x + width/2, f1, width, label='Macro F1-Score (%)', color='#4ba3e3', edgecolor='black', linewidth=0.8)

ax1.set_ylabel('Tỷ lệ (%)', fontsize=12, fontweight='bold', color='#1e3d59')
ax1.set_title('SO SÁNH HIỆU SUẤT THỰC NGHIỆM GIỮA CÁC MÔ HÌNH (250 LỚP BIỂN BÁO)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Add text labels on bars
for r in rects1:
    h = r.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
for r in rects2:
    h = r.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

# Second axis for latency
ax2 = ax1.twinx()
line = ax2.plot(x, latency, color='#d9534f', marker='o', linewidth=2.5, markersize=8, label='Độ trễ CPU (ms/ảnh)')
ax2.set_ylabel('Thời gian suy luận CPU (ms)', fontsize=12, fontweight='bold', color='#d9534f')
ax2.set_ylim(0, 40)
for i, txt in enumerate(latency):
    ax2.annotate(f'{txt:.0f} ms', (x[i], latency[i]), xytext=(0, 8), textcoords="offset points",
                 ha='center', fontsize=10, fontweight='bold', color='#d9534f')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

plt.tight_layout()
chart_path = FIG_DIR / "model_comparison_bar.png"
plt.savefig(chart_path)
plt.close()
print(f"Saved: {chart_path}")

# 2. Generate Data Augmentation & Processing Stages Demo Image
# We show 5 stages:
# 1: Original template (with black corners)
# 2: Flood-fill alpha mask
# 3: Natural background composite
# 4: 3D perspective & rotation
# 5: Real street image crop
plt.figure(figsize=(15, 3.8), dpi=300)

# Sample template: P.102 (Cấm đi ngược chiều) or P.101
tpl_path = PROJECT_ROOT / "data" / "templates" / "1_P.102.png"
if not tpl_path.exists():
    tpl_path = list((PROJECT_ROOT / "data" / "templates").glob("*.png"))[0]

img_tpl = cv2.imread(str(tpl_path))
img_tpl = cv2.cvtColor(img_tpl, cv2.COLOR_BGR2RGB)
h, w, _ = img_tpl.shape

# Stage 2: Flood-fill mask
gray = cv2.cvtColor(img_tpl, cv2.COLOR_RGB2GRAY)
mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
diff = 12
for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
    if gray[seed[1], seed[0]] < 30:
        cv2.floodFill(gray, mask, seed, 0, diff, diff, cv2.FLOODFILL_MASK_ONLY | (255 << 8))
fg_mask = (mask[1:h+1, 1:w+1] == 0).astype(np.uint8) * 255

# Stage 3: Natural background composite
bg_sample = np.zeros((h, w, 3), dtype=np.uint8)
# Sky gradient
for i in range(h):
    alpha = i / h
    bg_sample[i, :] = [int(135 * (1 - alpha) + 200 * alpha), int(206 * (1 - alpha) + 220 * alpha), int(235 * (1 - alpha) + 240 * alpha)]
composite = np.where(fg_mask[:, :, None] > 0, img_tpl, bg_sample)

# Stage 4: 3D perspective & rotation
pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts2 = np.float32([[w*0.1, h*0.05], [w*0.9, h*0.12], [w*0.02, h*0.95], [w*0.85, h*0.88]])
M_persp = cv2.getPerspectiveTransform(pts1, pts2)
skewed = cv2.warpPerspective(composite, M_persp, (w, h), borderValue=(135, 206, 235))

# Stage 5: Real street image
real_files = list((PROJECT_ROOT / "data" / "processed" / "test").rglob("real_*.jpg"))
if real_files:
    img_real = cv2.imread(str(real_files[0]))
    img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
    img_real = cv2.resize(img_real, (w, h))
else:
    img_real = composite

stages = [
    (img_tpl, "(1) Ảnh Mẫu Gốc\n(Cạm bẫy 4 góc đen)"),
    (fg_mask, "(2) Tách Biên Flood-fill\n(Khử đặc trưng góc giả)"),
    (composite, "(3) Ghép Nền Tự Nhiên\n(Mô phỏng bầu trời / ngoại cảnh)"),
    (skewed, "(4) Biến Dạng Phối Cảnh 3D\n(Xoay & Nghiêng góc nhìn xe)"),
    (img_real, "(5) Ảnh Đường Phố Thực Tế\n(Dung hợp 1,703 mẫu thật)")
]

for idx, (img, title) in enumerate(stages):
    plt.subplot(1, 5, idx + 1)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(title, fontsize=10, fontweight='bold', pad=8)
    plt.axis('off')

plt.tight_layout()
demo_path = FIG_DIR / "data_processing_demo.png"
plt.savefig(demo_path)
plt.close()
print(f"Saved: {demo_path}")
