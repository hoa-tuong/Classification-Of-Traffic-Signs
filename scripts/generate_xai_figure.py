import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path("c:/ML/Classification-Of-Traffic-Signs")
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set font family
plt.rcParams['font.family'] = 'Segoe UI'

fig = plt.figure(figsize=(15, 8.5), dpi=300, facecolor='#0d1117')
ax = fig.add_subplot(111)
ax.set_facecolor('#0d1117')

# Title
fig.text(0.5, 0.95, "TÍNH NĂNG ĐỘT PHÁ: TRỢ LÝ PHÁP LUẬT GIAO THÔNG THÔNG MINH (EXPLAINABLE AI)", 
         ha='center', va='top', fontsize=16, fontweight='bold', color='#58a6ff')
fig.text(0.5, 0.91, "Không chỉ nhận diện mã nhãn, hệ thống trích xuất toàn văn quy chuẩn pháp lý, phạm vi hiệu lực, ví dụ thực tế và mẹo ghi nhớ nhanh", 
         ha='center', va='top', fontsize=11, color='#8b949e')

# --- LEFT PANEL: AI PREDICTION & IMAGE ---
# Panel background
rect_left = patches.FancyBboxPatch((0.02, 0.05), 0.32, 0.82, boxstyle="round,pad=0.015", 
                                   fc='#161b22', ec='#30363d', lw=1.5, transform=fig.transFigure)
fig.patches.append(rect_left)

fig.text(0.18, 0.84, "KẾT QUẢ SUY LUẬN AI", ha='center', fontsize=13, fontweight='bold', color='#f0f6fc')

# Load P.102 template image
tpl_path = PROJECT_ROOT / "data" / "templates" / "1_P.102.png"
if tpl_path.exists():
    img_tpl = cv2.imread(str(tpl_path))
    img_tpl = cv2.cvtColor(img_tpl, cv2.COLOR_BGR2RGB)
else:
    img_tpl = np.zeros((150, 150, 3), dtype=np.uint8)

# Show image in an inset axes
ax_img = fig.add_axes([0.08, 0.52, 0.20, 0.26])
ax_img.imshow(img_tpl)
ax_img.set_title("Biển báo cần phân loại", color='#c9d1d9', fontsize=11, pad=6)
ax_img.axis('off')

# Prediction box
fig.text(0.18, 0.46, "Dự đoán Top-1:", ha='center', fontsize=11, color='#8b949e')
fig.text(0.18, 0.42, "[P.102] CẤM ĐI NGƯỢC CHIỀU", ha='center', fontsize=12, fontweight='bold', color='#ff7b72')

# Confidence bar
fig.text(0.18, 0.36, "Độ tin cậy: 99.81%", ha='center', fontsize=12, fontweight='bold', color='#3fb950')
ax_bar = fig.add_axes([0.05, 0.31, 0.26, 0.03])
ax_bar.set_facecolor('#21262d')
ax_bar.barh([0], [99.81], color='#238636', height=0.6)
ax_bar.set_xlim(0, 100)
ax_bar.axis('off')

# Other metrics
info_text = (
    "• Mô hình: MobileNetV3-Small (Transfer Learning)\n"
    "• Thời gian suy luận: 8 ms trên CPU\n"
    "• Trạng thái: Nhận diện chính xác tuyệt đối\n"
    "• Phân loại: Nhóm Biển Báo Cấm (QCVN 41:2019)"
)
fig.text(0.05, 0.12, info_text, fontsize=10.5, color='#c9d1d9', linespacing=1.6)

# --- RIGHT PANEL: EXPLAINABLE LEGAL ASSISTANT ---
rect_right = patches.FancyBboxPatch((0.36, 0.05), 0.62, 0.82, boxstyle="round,pad=0.015", 
                                    fc='#161b22', ec='#1f6feb', lw=1.8, transform=fig.transFigure)
fig.patches.append(rect_right)

# Header Right
fig.text(0.39, 0.84, "QUY CHUẨN KỸ THUẬT QUỐC GIA QCVN 41:2019/BGTVT", fontsize=13, fontweight='bold', color='#58a6ff')
fig.text(0.39, 0.805, "Biển số P.102: CẤM ĐI NGƯỢC CHIỀU  |  Nhóm: Biển báo cấm", fontsize=11.5, fontweight='bold', color='#e6edf3')

# Section 1: Y nghia
fig.text(0.39, 0.74, "1. Ý NGHĨA QUY ĐỊNH CỦA BIỂN BÁO:", fontsize=11.5, fontweight='bold', color='#f0883e')
sec1 = (
    "• Đặc điểm: Biển tròn nền đỏ, vạch ngang màu trắng ở giữa.\n"
    "• Mục đích: Báo cấm tất cả các loại phương tiện (cơ giới và thô sơ) đi vào theo chiều đặt biển;\n"
    "  ngăn ngừa xung đột trực diện tại đường một chiều, nhánh ra/vào, hầm chui, đường hẹp.\n"
    "• Ngoại lệ: Xe được quyền ưu tiên theo Luật Giao thông (Cứu hỏa, Công an, Quân sự, Cứu thương\n"
    "  khi đang làm nhiệm vụ khẩn cấp) được phép đi vào theo chỉ đạo nghiệp vụ.\n"
    "• Người đi bộ: Vẫn được phép đi trên vỉa hè hoặc lề đường, không đi tràn xuống lòng đường."
)
fig.text(0.40, 0.58, sec1, fontsize=10, color='#c9d1d9', linespacing=1.45)

# Section 2: Pham vi tac dung
fig.text(0.39, 0.54, "2. PHẠM VI TÁC DỤNG & VỊ TRÍ ĐẶT BIỂN:", fontsize=11.5, fontweight='bold', color='#a371f7')
sec2 = (
    "• Hiệu lực: Bắt đầu ngay tại vị trí đặt biển và áp dụng cho hướng tiến thẳng qua biển vào đoạn đường phía trước.\n"
    "• Chiều áp dụng: Chỉ cấm hành vi 'đi vào' từ phía đặt biển. Xe lưu thông đúng chiều không bị ảnh hưởng.\n"
    "• Vị trí cắm: Đầu vào của đường một chiều; lối ra bãi đỗ xe, trạm xăng, đường hẹp, nhánh ra cao tốc.\n"
    "• Không cần biển 'Hết cấm' vì đây là biển cấm theo hướng vào; hiệu lực áp dụng tại miệng lối vào."
)
fig.text(0.40, 0.40, sec2, fontsize=10, color='#c9d1d9', linespacing=1.45)

# Section 3: Vi du thuc te
fig.text(0.39, 0.36, "3. VÍ DỤ TÌNH HUỐNG THỰC TẾ TRÊN ĐƯỜNG PHỐ:", fontsize=11.5, fontweight='bold', color='#79c0ff')
sec3 = (
    "Bạn điều khiển xe đến ngã ba/ngã tư và thấy biển P.102 gắn ở đầu một con phố.\n"
    "→ Điều đó có nghĩa: Bạn tuyệt đối không được rẽ vào con phố đó từ hướng bạn đang đi tới!\n"
    "→ Xử lý: Hãy tiếp tục đi thẳng hoặc rẽ hướng khác. Các xe đang chạy từ trong con phố đó đi ra vẫn di chuyển\n"
    "bình thường, còn chiều đi vào của bạn đã bị cấm hoàn toàn."
)
fig.text(0.40, 0.24, sec3, fontsize=10, color='#c9d1d9', linespacing=1.45)

# Section 4: Ghi nho nhanh
fig.text(0.39, 0.20, "4. MẸO GHI NHỚ NHANH DÀNH CHO TÀI XẾ:", fontsize=11.5, fontweight='bold', color='#7ee787')
sec4 = (
    "• Thấy biển tròn đỏ gạch trắng P.102 = 'ĐỪNG ĐI VÀO LỐI NÀY - ĐƯỜNG MỘT CHIỀU!'.\n"
    "• Thường đi kèm đối ứng với biển R.407a 'Đường một chiều' được cắm ở đầu còn lại của tuyến đường."
)
fig.text(0.40, 0.12, sec4, fontsize=10.5, fontweight='bold', color='#7ee787', linespacing=1.45)

ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

out_fig = FIG_DIR / "explainable_ai_demo.png"
plt.savefig(out_fig, facecolor='#0d1117', edgecolor='none')
plt.close()
print(f"Saved: {out_fig}")
