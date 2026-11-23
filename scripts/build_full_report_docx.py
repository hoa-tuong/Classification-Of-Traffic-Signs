# -*- coding: utf-8 -*-
"""
Script tạo file Word Báo cáo Đồ án Học máy (Machine Learning)
Môn: Machine Learning & Deep Learning
Đề tài: Phân loại 250 loại Biển báo Giao thông Đường bộ Việt Nam (QCVN 41:2019/BGTVT)
Giảng viên hướng dẫn: Thầy Nguyễn Thanh Phước
Sinh viên thực hiện:
  - Trần Lê Minh Tâm (3124720064) - 25%
  - Chương Gia Thịnh (3124720067) - 25%
  - Tất Hòa Tường (3124720077) - 25%
  - Đỗ Thành Tuân (3124720076) - 25%
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path("c:/ML/Classification-Of-Traffic-Signs")
DOCX_OUT = PROJECT_ROOT / "docs" / "BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO.docx"
DESKTOP_OUT = Path(r"c:\Users\Kiyoshi\Desktop\BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO.docx")
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

doc = docx.Document()

# Set standard A4 margins (2.54 cm = 1 inch)
for sec in doc.sections:
    sec.top_margin = Inches(1.0)
    sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(1.0)
    sec.right_margin = Inches(1.0)

# Default style: Times New Roman, 13pt, 1.15 line spacing, 6pt space after
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(13)
font.color.rgb = RGBColor(0, 0, 0)
style.paragraph_format.line_spacing = 1.15
style.paragraph_format.space_after = Pt(4)

def set_cell_border(cell, **kwargs):
    """Set cell borders: top, bottom, left, right"""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}>\n'
                          f'<w:top w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>\n'
                          f'<w:left w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>\n'
                          f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>\n'
                          f'<w:right w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>\n'
                          f'</w:tcBorders>')
    tcPr.append(tcBorders)

def set_cell_background(cell, fill_hex):
    """Set cell background color"""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def add_title(text, size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10, color=(0, 0, 0)):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(size)
    r.bold = bold
    r.font.color.rgb = RGBColor(*color)
    return p

def add_heading_1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(15)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 51, 102) # Dark blue
    return p

def add_heading_2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(13.5)
    r.bold = True
    r.font.color.rgb = RGBColor(30, 80, 140)
    return p

def add_heading_3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(13)
    r.bold = True
    r.italic = True
    r.font.color.rgb = RGBColor(50, 50, 50)
    return p

def add_body(text, bold=False, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(13)
    r.bold = bold
    r.italic = italic
    return p

def add_bullet(text, bold_prefix="", level=0):
    p = doc.add_paragraph(style='List Paragraph')
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(13)
        r_pre.bold = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(13)
    return p

def add_code_block(code_text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.2)
    r = p.add_run(code_text)
    r.font.name = 'Courier New'
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(20, 20, 20)
    return p

def add_image_box(img_path, caption, width=Inches(6.0)):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        p_img.add_run().add_picture(str(img_path), width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = 'Times New Roman'
        r_cap.font.size = Pt(11)
        r_cap.italic = True
        r_cap.bold = True
        r_cap.font.color.rgb = RGBColor(60, 60, 60)

print("Writing Cover Page...")
# ==========================================
# TRANG BÌA (COVER PAGE)
# ==========================================
add_title("TRƯỜNG ĐẠI HỌC SÀI GÒN", size=17, bold=True, space_after=3)
add_title("KHOA CÔNG NGHỆ THÔNG TIN", size=14, bold=True, space_after=12)

# Insert SGU Logo
logo_path = FIG_DIR / "sgu_logo.png"
if logo_path.exists():
    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(4)
    p_logo.paragraph_format.space_after = Pt(12)
    p_logo.add_run().add_picture(str(logo_path), width=Inches(1.8))

add_title("BÁO CÁO TIỂU LUẬN NHÓM CUỐI KỲ", size=20, bold=True, space_after=8, color=(0, 51, 102))
add_title("MÔN: MACHINE LEARNING & DEEP LEARNING (HỌC MÁY)", size=14, bold=True, space_after=25)

add_title("ĐỀ TÀI:", size=13, bold=True, space_after=4)
add_title("NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG NHẬN DIỆN, PHÂN LOẠI 250 LOẠI BIỂN BÁO GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM (THEO QUY CHUẨN QCVN 41:2019/BGTVT)", size=15, bold=True, space_after=30, color=(160, 30, 30))

for _ in range(2):
    doc.add_paragraph()

# GVHD & Sinh viên
p_info = doc.add_paragraph()
p_info.alignment = WD_ALIGN_PARAGRAPH.LEFT
p_info.paragraph_format.left_indent = Inches(1.5)
p_info.paragraph_format.space_after = Pt(2)
r = p_info.add_run("Giảng viên hướng dẫn:\n")
r.bold = True
r.font.size = Pt(13.5)
r2 = p_info.add_run("Thầy: Nguyễn Thanh Phước\n\n")
r2.font.size = Pt(13.5)

r_sv = p_info.add_run("Danh sách sinh viên thực hiện (Tỷ lệ đóng góp 25% mỗi bạn):\n")
r_sv.bold = True
r_sv.font.size = Pt(13)

sv_list = [
    ("Trần Lê Minh Tâm", "3124720064", "25%"),
    ("Chương Gia Thịnh", "3124720067", "25%"),
    ("Tất Hòa Tường", "3124720077", "25%"),
    ("Đỗ Thành Tuân", "3124720076", "25%")
]
for name, mssv, pct in sv_list:
    r_item = p_info.add_run(f"  • {name}  -  MSSV: {mssv}  ({pct})\n")
    r_item.font.size = Pt(12.5)

doc.add_page_break()

# ==========================================
# BẢNG PHÂN CÔNG ĐÓNG GÓP & CÂU HỎI BỐC THĂM
# ==========================================
add_heading_1("BẢNG ĐÓNG GÓP VÀ PHÂN CÔNG NHIỆM VỤ THÀNH VIÊN")
p_desc = add_body("Đề tài được thực hiện với sự tham gia đồng đều của cả 4 thành viên. Mỗi thành viên chịu trách nhiệm 25% khối lượng công việc theo sự phân công chi tiết dưới đây:")

tbl_contrib = doc.add_table(rows=5, cols=5)
tbl_contrib.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_contrib.autofit = False

headers = ["STT", "Họ và Tên", "MSSV", "Nhiệm vụ đảm nhiệm chính trong đồ án", "Tỷ lệ đóng góp"]
widths = [Inches(0.5), Inches(1.7), Inches(1.1), Inches(2.7), Inches(0.8)]

for col_idx, text in enumerate(headers):
    cell = tbl_contrib.cell(0, col_idx)
    cell.width = widths[col_idx]
    set_cell_background(cell, "2B5C8F")
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(255, 255, 255)
    set_cell_border(cell)

contrib_data = [
    ("1", "Trần Lê Minh Tâm", "3124720064", "Khảo sát và tiền xử lý dữ liệu. Phân tích hiện tượng học vẹt (Shortcut Learning) do 4 góc đen [0,0,0]. Phát triển thuật toán Flood-fill bóc tách góc đen và kỹ thuật ghép đè ngoại cảnh tự nhiên.", "25%"),
    ("2", "Chương Gia Thịnh", "3124720067", "Khảo sát, thiết kế và so sánh các kiến trúc mô hình (Template Matching, Custom Deep CNN, ResNet-18, MobileNetV3). Phân tích ma trận nhầm lẫn và đánh giá định lượng giữa các giải thuật.", "25%"),
    ("3", "Tất Hòa Tường", "3124720077", "Thu thập và trích xuất 1,703 ảnh đường phố thực tế từ bộ dữ liệu YOLO Việt Nam. Huấn luyện mô hình MobileNetV3-Small, tinh chỉnh siêu tham số và tối ưu hóa hàm mất mát Cross-Entropy.", "25%"),
    ("4", "Đỗ Thành Tuân", "3124720076", "Thiết kế và triển khai ứng dụng Web tương tác bằng Streamlit. Tích hợp công cụ cắt vùng ROI Cropper, số hóa toàn văn quy chuẩn pháp lý QCVN 41:2019 và sửa lỗi font hiển thị icon.", "25%")
]

for row_idx, data in enumerate(contrib_data, start=1):
    for col_idx, text in enumerate(data):
        cell = tbl_contrib.cell(row_idx, col_idx)
        cell.width = widths[col_idx]
        if row_idx % 2 == 1:
            set_cell_background(cell, "F5F8FC")
        set_cell_border(cell)
        p = cell.paragraphs[0]
        if col_idx in [0, 2, 4]:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(text)
        r.font.size = Pt(10.5)

add_body("")

# ==========================================
# PHẦN CÂU HỎI BỐC THĂM BẢO VỆ CỦA TỪNG THÀNH VIÊN
# ==========================================
add_heading_1("PHẦN ĐÁNH GIÁ VÀ BỐC THĂM CÂU HỎI BẢO VỆ ĐỒ ÁN")
p_note_gv = add_body("Phần dành cho Giảng viên hướng dẫn (Thầy: Nguyễn Thanh Phước) bốc thăm và ghi nhận 3 câu hỏi đánh giá đối với từng sinh viên trong buổi bảo vệ cuối kỳ:")
p_note_gv.runs[0].italic = True

def add_student_questions_block(student_name, mssv):
    add_heading_2(f"Thành viên: {student_name}  -  MSSV: {mssv}")
    for q_idx in range(1, 4):
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(4)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(f"Câu {q_idx}: .................................................................................................................................................................")
        r_q.bold = True
        r_q.font.size = Pt(12)
        # 4 blank dotted lines for answering
        for _ in range(4):
            p_line = doc.add_paragraph()
            p_line.paragraph_format.space_after = Pt(2)
            r_l = p_line.add_run("..................................................................................................................................................................................")
            r_l.font.color.rgb = RGBColor(150, 150, 150)
            r_l.font.size = Pt(11)

for name, mssv, _ in sv_list:
    add_student_questions_block(name, mssv)

doc.add_page_break()

# ==========================================
# MỤC LỤC & DANH MỤC HÌNH ẢNH / BẢNG BIỂU
# ==========================================
add_heading_1("MỤC LỤC ĐỒ ÁN")

toc_items = [
    ("BẢNG ĐÓNG GÓP VÀ PHÂN CÔNG NHIỆM VỤ THÀNH VIÊN", "2"),
    ("PHẦN ĐÁNH GIÁ VÀ BỐC THĂM CÂU HỎI BẢO VỆ ĐỒ ÁN", "3"),
    ("DANH MỤC THUẬT NGỮ VÀ TỪ VIẾT TẮT", "5"),
    ("DANH MỤC HÌNH ẢNH MINH HỌA", "6"),
    ("DANH MỤC BẢNG BIỂU", "7"),
    ("I. TỔNG QUAN ĐỀ TÀI", "8"),
    ("    1. Cơ sở hình thành đề tài", "8"),
    ("    2. Mục tiêu nghiên cứu", "9"),
    ("    3. Phạm vi và phương pháp nghiên cứu", "10"),
    ("    4. Ý nghĩa khoa học và thực tiễn", "11"),
    ("II. BỘ DỮ LIỆU SỬ DỤNG VÀ TIỀN XỬ LÝ DỮ LIỆU", "12"),
    ("    1. Sơ lược nguồn gốc bộ dữ liệu (Kaggle & Hugging Face)", "12"),
    ("    2. Danh mục 250 lớp biển báo theo quy chuẩn QCVN 41:2019/BGTVT", "13"),
    ("    3. Trích xuất dữ liệu thực tế ngoài đường phố Việt Nam", "14"),
    ("    4. Phân tích hiện tượng Học Vẹt (Shortcut Learning) do bẫy 4 góc đen", "15"),
    ("    5. Thuật toán bóc tách biên Flood-fill và Kỹ thuật ghép nền tự nhiên", "16"),
    ("    6. Pipeline tăng cường dữ liệu phối cảnh 3D và góc xoay thực tế", "18"),
    ("    7. Thống kê tập dữ liệu tổng hợp sau tiền xử lý", "20"),
    ("III. PHÂN TÍCH VÀ KHÁM PHÁ DỮ LIỆU (EDA)", "21"),
    ("    1. Khảo sát phân bố mẫu giữa các nhóm chức năng", "21"),
    ("    2. Phân tích đặc tính màu sắc và hình học nhận diện", "22"),
    ("    3. Phân tích các thách thức ngoại cảnh trong môi trường giao thông Việt Nam", "23"),
    ("IV. KHẢO SÁT, THỰC NGHIỆM VÀ SO SÁNH CÁC MÔ HÌNH (MODEL SELECTION)", "24"),
    ("    1. Mô hình 1: Baseline Template Matching (So khớp sai số MSE)", "24"),
    ("    2. Mô hình 2: Custom Deep CNN (Mạng nơ-ron tự thiết kế huấn luyện từ đầu)", "25"),
    ("    3. Mô hình 3: ResNet-18 (Transfer Learning)", "26"),
    ("    4. Mô hình 4: MobileNetV3-Small (Transfer Learning kết hợp SE Blocks)", "27"),
    ("    5. So sánh định lượng và Luận cứ quyết định lựa chọn MobileNetV3-Small", "28"),
    ("V. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ ĐỊNH LƯỢNG", "30"),
    ("    1. Thiết lập siêu tham số và Môi trường thực nghiệm", "30"),
    ("    2. Quá trình huấn luyện và Đường cong hội tụ (Loss & Accuracy)", "31"),
    ("    3. Đánh giá chi tiết trên tập kiểm thử độc lập (Test Set - 743 mẫu)", "32"),
    ("    4. Phân tích ma trận nhầm lẫn (Confusion Matrix) và Các trường hợp sai sót", "33"),
    ("    5. Thử nghiệm độ bền vững (Robustness Test) trên ảnh thực tế và ảnh xoay", "34"),
    ("VI. TRÍCH XUẤT CÁC ĐOẠN MÃ NGUỒN TRỌNG TÂM (CORE SOURCE CODE)", "35"),
    ("    1. Thuật toán Flood-fill bóc tách 4 góc đen", "35"),
    ("    2. Kỹ thuật ghép đè nền ngoại cảnh tự nhiên", "36"),
    ("    3. Pipeline biến dạng phối cảnh 3D và Augmentation", "37"),
    ("    4. Trích xuất Bounding Box từ dữ liệu YOLO sang 250 classes", "38"),
    ("    5. Kiến trúc mạng MobileNetV3 với Custom Classifier Head 250 lớp", "39"),
    ("    6. Engine suy luận và Công cụ cắt vùng ROI trong Streamlit", "40"),
    ("VII. TRIỂN KHAI ỨNG DỤNG THỰC TẾ & GIAO DIỆN NGƯỜI DÙNG", "41"),
    ("    1. Kiến trúc tổng thể ứng dụng Web Streamlit", "41"),
    ("    2. Chi tiết các phân hệ và tính năng nổi bật của Website", "42"),
    ("    3. Hướng dẫn cài đặt và khởi chạy ứng dụng (Setup & Run Guide)", "43"),
    ("    4. Hướng dẫn sử dụng chi tiết từng bước (Step-by-Step User Manual)", "44"),
    ("    5. Tối ưu hóa giao diện và giải pháp khắc phục lỗi kỹ thuật", "45"),
    ("VIII. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN TƯƠNG LAI", "45"),
    ("    1. Đóng góp chính của đồ án", "45"),
    ("    2. Đánh giá ưu điểm và Hạn chế còn tồn đọng", "45"),
    ("    3. Hướng phát triển: Tích hợp Pipeline 2 giai đoạn YOLOv11 + MobileNetV3", "46"),
    ("TÀI LIỆU THAM KHẢO", "47")
]

for title, page in toc_items:
    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.space_after = Pt(2)
    # calculate dot fill
    dots_count = max(5, 75 - len(title))
    dots = "." * dots_count
    r_t = p_toc.add_run(title)
    r_t.font.size = Pt(11)
    if not title.startswith("    "):
        r_t.bold = True
    r_d = p_toc.add_run(f" {dots} ")
    r_d.font.size = Pt(10)
    r_d.font.color.rgb = RGBColor(140, 140, 140)
    r_p = p_toc.add_run(page)
    r_p.bold = True
    r_p.font.size = Pt(11)

add_body("")
add_heading_2("DANH MỤC HÌNH ẢNH MINH HỌA")
figs_list = [
    ("Hình 1", "Minh họa 5 giai đoạn xử lý dữ liệu và triệt tiêu bẫy góc đen", "17"),
    ("Hình 2", "Biểu đồ so sánh định lượng hiệu suất thực nghiệm giữa 4 mô hình", "29"),
    ("Hình 3", "Đồ thị hàm mất mát (Loss) và độ chính xác (Accuracy) trong huấn luyện", "31"),
    ("Hình 4", "Ma trận nhầm lẫn (Confusion Matrix) đánh giá trên 250 lớp biển báo", "33"),
    ("Hình 5", "Giao diện ứng dụng Web Streamlit với tính năng Cắt vùng biển báo (ROI)", "42"),
    ("Hình 6", "Minh họa tính năng Trợ lý pháp luật giao thông thông minh (Explainable AI)", "43")
]
for f_id, f_desc, f_page in figs_list:
    p_f = doc.add_paragraph()
    p_f.paragraph_format.space_after = Pt(2)
    dots = "." * (70 - len(f_id) - len(f_desc))
    r = p_f.add_run(f"{f_id}: {f_desc} {dots} {f_page}")
    r.font.size = Pt(11)

add_heading_2("DANH MỤC BẢNG BIỂU")
tbls_list = [
    ("Bảng 1", "Bảng phân công nhiệm vụ và tỷ lệ đóng góp của thành viên", "2"),
    ("Bảng 2", "Phân bố 250 lớp biển báo theo 5 nhóm quy chuẩn QCVN 41:2019", "13"),
    ("Bảng 3", "Thống kê quy mô phân chia tập dữ liệu sau tiền xử lý", "20"),
    ("Bảng 4", "Bảng so sánh định lượng toàn diện giữa 4 mô hình nghiên cứu", "28"),
    ("Bảng 5", "Bảng thiết lập siêu tham số huấn luyện mô hình MobileNetV3", "30"),
    ("Bảng 6", "Lịch sử huấn luyện và các chỉ số hội tụ qua 5 epochs", "31"),
    ("Bảng 7", "Kết quả thử nghiệm độ bền vững (Robustness Test) trên các ảnh khó", "34")
]
for t_id, t_desc, t_page in tbls_list:
    p_t = doc.add_paragraph()
    p_t.paragraph_format.space_after = Pt(2)
    dots = "." * (70 - len(t_id) - len(t_desc))
    r = p_t.add_run(f"{t_id}: {t_desc} {dots} {t_page}")
    r.font.size = Pt(11)

doc.add_page_break()

# ==========================================
# PHẦN I: TỔNG QUAN ĐỀ TÀI
# ==========================================
add_heading_1("I. TỔNG QUAN ĐỀ TÀI")

add_heading_2("1. Cơ sở hình thành đề tài")
add_body("Hệ thống giao thông đường bộ tại Việt Nam là một trong những môi trường di chuyển có tính chất hỗn hợp và mật độ phương tiện cao nhất trong khu vực Đông Nam Á. Trên cùng một tuyến đường đô thị, có sự tham gia đồng thời của xe gắn máy hai bánh, ô tô con, xe buýt công cộng, xe tải chuyên dụng và các phương tiện thô sơ. Theo thống kê của Ủy ban An toàn Giao thông Quốc gia, một tỉ lệ rất lớn các vụ vi phạm luật giao thông cũng như tai nạn đáng tiếc xảy ra do người điều khiển phương tiện không quan sát kịp biển báo, quan sát sót biển báo phụ đặt phía dưới, hoặc không hiểu rõ ý nghĩa quy chuẩn của các loại biển gộp làn đường, biển cấm rẽ theo khung giờ.")

add_body("Trong xu thế chuyển đổi số và phát triển xe thông minh, các hệ thống hỗ trợ lái xe nâng cao (Advanced Driver Assistance Systems - ADAS) và phương tiện tự hành (Autonomous Vehicles) đang trở thành tiêu chuẩn an toàn bắt buộc trên toàn cầu. Tuy nhiên, việc áp dụng nguyên mẫu các mô hình thị giác máy tính phát triển tại châu Âu hoặc Bắc Mỹ (như các mô hình huấn luyện trên tập dữ liệu GTSRB của Đức) vào thực tế giao thông Việt Nam bộc lộ nhiều điểm nghẽn nghiêm trọng:")
add_bullet("Sự khác biệt về quy chuẩn hình thái: Hệ thống biển báo Việt Nam ban hành theo Quy chuẩn kỹ thuật quốc gia QCVN 41:2019/BGTVT có nhiều nhóm biển đặc thù mà nước ngoài không có (ví dụ: Biển gộp làn đường theo phương tiện, biển báo cấm rẽ theo giờ, hệ thống biển phụ chữ viết tiếng Việt).", "• ")
add_bullet("Thiếu hụt nghiên cứu phân loại quy mô lớn: Đa phần các đề tài trước đây chỉ dừng lại ở bài toán phân loại từ 20 đến 40 lớp biển báo phổ biến, chưa bao quát được trọn vẹn danh mục 250 loại biển báo đầy đủ.", "• ")
add_bullet("Thiếu vắng tính năng giải thích pháp lý (Explainable AI): Các hệ thống hiện có chỉ đưa ra mã lớp hoặc tên nhãn tiếng Anh mà không giải thích được hiệu lực thi hành, trường hợp xe ưu tiên hay mức phạt, làm giảm đáng kể giá trị thực tiễn đối với người lái xe.", "• ")

add_heading_2("2. Mục tiêu nghiên cứu")
add_body("Đồ án đặt ra hai hệ thống mục tiêu song song:")
add_bullet("Xây dựng mô hình học sâu có khả năng phân loại chính xác toàn diện 250 loại biển báo giao thông đường bộ Việt Nam theo QCVN 41:2019/BGTVT.", "a. Về mặt học thuật: ")
add_bullet("Phát hiện, phân tích bản chất và triệt tiêu triệt để hiện tượng 'Học vẹt' (Shortcut Learning) - một vấn đề kinh điển trong Deep Learning khi mô hình bị đánh lừa bởi các đặc trưng biên nhân tạo.", "")
add_bullet("Nghiên cứu, thực nghiệm và so sánh hiệu suất định lượng giữa 4 phương pháp/mô hình: Baseline Template Matching, Custom Deep CNN, ResNet-18 và MobileNetV3-Small để đưa ra luận cứ khoa học cho việc lựa chọn mô hình tối ưu.", "")
add_bullet("Xây dựng ứng dụng hoàn chỉnh có giao diện người dùng trực quan trên nền tảng Web Streamlit, tối ưu hóa tốc độ suy luận thời gian thực trên CPU.", "b. Về mặt ứng dụng: ")
add_bullet("Tích hợp công cụ cắt vùng biển báo bán tự động (ROI Cropper) giúp người dùng dễ dàng khoanh vùng biển báo từ ảnh toàn cảnh góc rộng.", "")
add_bullet("Số hóa toàn văn quy chuẩn pháp lý giao thông Việt Nam, cung cấp tính năng giải thích ý nghĩa pháp lý tự động (Explainable AI) khi trả về kết quả dự đoán.", "")

add_heading_2("3. Phạm vi và phương pháp nghiên cứu")
add_bullet("Đối tượng nghiên cứu: Ảnh chụp thực tế ngoài đường phố Việt Nam và ảnh mẫu đồ họa quy chuẩn của 250 loại biển báo đường bộ Việt Nam.", "• ")
add_bullet("Phạm vi bài toán: Tập trung chuyên sâu vào giai đoạn Phân loại ảnh biển báo (Traffic Sign Classification). Đối với ảnh toàn cảnh, hệ thống cung cấp công cụ khoanh vùng quan tâm (ROI Cropper).", "• ")
add_bullet("Phương pháp nghiên cứu: Kết hợp phương pháp thực nghiệm khoa học, kỹ thuật Học chuyển giao (Transfer Learning), kỹ thuật xử lý ảnh số bằng OpenCV và các phương pháp tăng cường dữ liệu chuyên sâu (Data Augmentation).", "• ")

add_heading_2("4. Ý nghĩa khoa học và thực tiễn")
add_bullet("Đóng góp một quy trình bài bản giải quyết hiện tượng Shortcut Learning trong bài toán thị giác máy tính khi dữ liệu bị hạn chế. Đưa ra bộ giải pháp sinh dữ liệu tổng hợp chân thực (Realistic Synthetic Augmentation) kết hợp thuật toán Flood-fill bóc tách góc đen.", "a. Ý nghĩa khoa học: ")
add_bullet("Cung cấp một công cụ hỗ trợ thiết thực cho người dân học luật giao thông, ôn thi giấy phép lái xe, hoặc tra cứu nhanh ý nghĩa pháp lý của các biển báo phức tạp ngay trên điện thoại hoặc máy tính.", "b. Ý nghĩa thực tiễn: ")
add_bullet("Là tiền đề và module phân loại cốt lõi để tích hợp vào các hệ thống camera hành trình thông minh, cảnh báo sớm cho tài xế khi lưu thông trên các cung đường lạ.", "")

# ==========================================
# PHẦN II: BỘ DỮ LIỆU SỬ DỤNG VÀ TIỀN XỬ LÝ
# ==========================================
add_heading_1("II. BỘ DỮ LIỆU SỬ DỤNG VÀ TIỀN XỬ LÝ DỮ LIỆU")

add_heading_2("1. Sơ lược nguồn gốc bộ dữ liệu")
add_body("Để giải quyết trọn vẹn bài toán 250 lớp biển báo trong điều kiện thực tế, nhóm nghiên cứu đã khai thác và kết hợp hai nguồn dữ liệu chính:")
add_bullet("Bộ dữ liệu đường phố thực tế (Traffic-sign-detection-VietNam): Được thu thập từ cộng đồng nghiên cứu thị giác máy tính Việt Nam (nguồn Kaggle / Hugging Face tác giả star092304). Bộ dữ liệu gốc bao gồm 10,157 bức ảnh chụp thực tế tại các tuyến đường đô thị Hà Nội và TP. Hồ Chí Minh trong nhiều điều kiện thời tiết (nắng gắt, chiều tà, bóng râm) với nhãn Bounding Box YOLO định dạng 82 nhóm biển báo.", "1. ")
add_bullet("Bộ 250 ảnh mẫu đồ họa quy chuẩn (Official Templates): Bao gồm 250 file ảnh vector chất lượng cao kích thước 256x256 pixel thể hiện chuẩn xác tỷ lệ hình học, màu sắc và ký hiệu quy chuẩn theo QCVN 41:2019/BGTVT.", "2. ")

add_heading_2("2. Danh mục 250 lớp biển báo theo quy chuẩn QCVN 41:2019/BGTVT")
add_body("Toàn bộ 250 lớp biển báo được chuẩn hóa và phân chia thành 5 nhóm chức năng chính như thể hiện trong Bảng 2:")

tbl_classes = doc.add_table(rows=6, cols=4)
tbl_classes.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_c = ["Mã Nhóm", "Tên Nhóm Chức Năng", "Số Lượng Lớp", "Ví Dụ Điển Hình"]
widths_c = [Inches(1.0), Inches(2.2), Inches(1.2), Inches(2.1)]
for idx, text in enumerate(headers_c):
    c = tbl_classes.cell(0, idx)
    c.width = widths_c[idx]
    set_cell_background(c, "2B5C8F")
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(255, 255, 255)
    set_cell_border(c)

groups_data = [
    ("P", "Biển báo cấm (Prohibitory)", "71 lớp", "P.101 (Đường cấm), P.102 (Cấm đi ngược chiều), P.127 (Tốc độ tối đa)"),
    ("R", "Biển hiệu lệnh (Mandatory)", "30 lớp", "R.301a (Hướng đi phải theo), R.303 (Giao nhau chạy theo vòng xuyến)"),
    ("I / IE", "Biển chỉ dẫn (Informative)", "89 lớp", "I.407a (Đường một chiều), I.423a (Đường người đi bộ sang ngang)"),
    ("W", "Biển cảnh báo nguy hiểm (Warning)", "48 lớp", "W.201a (Chỗ ngoặt nguy hiểm), W.205a (Đường giao nhau)"),
    ("S", "Biển phụ (Supplementary)", "12 lớp", "S.501 (Phạm vi tác dụng của biển), S.503 (Hướng tác dụng của biển)")
]
for r_i, g_row in enumerate(groups_data, start=1):
    for c_i, val in enumerate(g_row):
        c = tbl_classes.cell(r_i, c_i)
        c.width = widths_c[c_i]
        if r_i % 2 == 1: set_cell_background(c, "F5F8FC")
        set_cell_border(c)
        p = c.paragraphs[0]
        if c_i in [0, 2]: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else: p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(val)
        r.font.size = Pt(10.5)

add_body("")

add_heading_2("3. Trích xuất dữ liệu thực tế ngoài đường phố Việt Nam")
add_body("Để đưa dữ liệu thực tế vào bài toán phân loại, nhóm đã phát triển script tự động quét toàn bộ hơn 10,000 file nhãn YOLO trong thư mục dữ liệu gốc. Với mỗi bounding box biển báo:")
add_bullet("Loại bỏ các bounding box có kích thước quá nhỏ (dưới 18x18 pixel) do độ phân giải quá thấp, không chứa đủ thông tin thị giác.", "• ")
add_bullet("Mở rộng biên (padding) thêm 10% về mỗi phía để giữ lại viền ngoài và một phần ngữ cảnh môi trường xung quanh.", "• ")
add_bullet("Cắt và chuyển đổi kích thước ảnh về chuẩn 64x64 pixel bằng giải thuật nội suy chất lượng cao Lanczos.", "• ")
add_bullet("Ánh xạ từ mã 82 lớp phát hiện đối tượng sang mã lớp 250 chuẩn theo bảng từ điển MAP_82_TO_250.", "• ")
add_body("Kết quả đã trích xuất thành công 1,703 ảnh chụp cận cảnh thực tế ngoài đường phố Việt Nam.")

add_heading_2("4. Phân tích hiện tượng Học Vẹt (Shortcut Learning) do bẫy 4 góc đen")
add_body("Trong giai đoạn thử nghiệm đầu tiên, nhóm nghiên cứu đã đối mặt với một vấn đề nghiêm trọng: Mô hình đạt độ chính xác trên 98% đối với tập dữ liệu ảnh mẫu, nhưng khi đưa ảnh thực tế chụp từ ngoài đường phố hoặc tải từ Google vào, mô hình hoàn toàn phân loại sai lệch (đặc biệt khi ảnh bị xoay nhẹ dù chỉ 10 độ).")
add_body("Qua phân tích sâu ở cấp độ ma trận điểm ảnh (Pixel-level Inspection), nhóm đã phát hiện ra nguyên nhân cốt lõi:")
add_bullet("Bẫy 4 góc đen [0, 0, 0]: Các ảnh mẫu biển báo hình tròn (Biển cấm P, Biển hiệu lệnh R) và hình tam giác (Biển cảnh báo W) trong bộ dữ liệu đồ họa đều có 4 góc bên ngoài viền mang màu đen kịt tuyệt đối RGB = [0, 0, 0].", "1. ")
add_bullet("Mạng nơ-ron tích chập (CNN) với cơ chế tối ưu hóa gradient sẽ luôn tìm con đường ngắn nhất để cực tiểu hóa hàm mất mát. Thay vì học các biểu tượng phức tạp bên trong biển (như hình xe ô tô, mũi tên, con số tốc độ), mô hình đã 'học vẹt' đặc trưng đường tắt: Cứ 4 góc ảnh là màu đen [0,0,0] thì chắc chắn là biển tròn hoặc tam giác!", "2. ")
add_bullet("Khi đưa ảnh thực tế ngoài đời hoặc ảnh tải từ mạng internet vào, 4 góc ảnh là nền bầu trời, tán cây, mặt đường hoặc tòa nhà. Khi đặc trưng góc đen biến mất, mạng nơ-ron rơi vào trạng thái OOD (Out-of-Distribution) và đưa ra dự đoán hoàn toàn ngẫu nhiên.", "3. ")
add_bullet("Đồng thời, module so khớp mẫu (Template Matching) đo khoảng cách sai số MSE giữa 2 ma trận ảnh. Khi ảnh bị xoay góc, các điểm ảnh bị lệch tọa độ khiến sai số MSE tăng vọt từ 0.001 lên 0.05, làm cơ chế so khớp mẫu bị liệt hoàn toàn.", "4. ")

add_heading_2("5. Thuật toán bóc tách biên Flood-fill và Kỹ thuật ghép nền tự nhiên")
add_body("Để triệt tiêu hoàn toàn đặc trưng đường tắt góc đen, nhóm đã đề xuất quy trình xử lý đột phá:")
add_bullet("Thuật toán Flood-fill bóc tách góc đen: Khởi tạo điểm loang (seed) từ 4 góc tọa độ (0,0), (W-1, 0), (0, H-1), (W-1, H-1). Giải thuật loang màu sẽ đánh dấu chính xác toàn bộ vùng pixel đen nằm ngoài viền biển báo, tạo thành một Mặt nạ nhị phân (Alpha Mask) mà không làm ảnh hưởng đến các chi tiết bên trong biển báo.", "• Bước 1: ")
add_bullet("Bộ sinh ngoại cảnh tự nhiên ngẫu nhiên (Natural Background Generator): Thuật toán sinh ra 4 nhóm nền giả lập chân thực: (1) Bầu trời (Gradient xanh lam - mây trắng), (2) Mặt đường nhựa (Màu xám kèm nhiễu hạt bê tông), (3) Tán cây xanh (Màu xanh lục đậm pha vàng), (4) Tòa nhà đô thị (Màu be xám).", "• Bước 2: ")
add_bullet("Ghép đè tự nhiên (Alpha Blending Composite): Thay thế toàn bộ vùng góc đen bằng các mẫu ngoại cảnh tự nhiên, kèm hiệu ứng làm mờ nền nhẹ (Defocus Blur). Kết quả tạo ra các mẫu ảnh biển báo hòa nhập tự nhiên vào môi trường như ảnh chụp thật.", "• Bước 3: ")

# Insert figure 1: data processing demo
add_image_box(FIG_DIR / "data_processing_demo.png", "Hình 1: Minh họa 5 giai đoạn xử lý dữ liệu và triệt tiêu bẫy góc đen của đề tài", width=Inches(6.2))

add_heading_2("6. Pipeline tăng cường dữ liệu phối cảnh 3D và góc xoay thực tế")
add_body("Để mô hình có khả năng thích nghi cao với các góc chụp thực tế từ camera trên xe hơi, nhóm xây dựng pipeline tăng cường dữ liệu chuyên sâu:")
add_bullet("Biến dạng phối cảnh 3D (RandomPerspective): Mô phỏng góc nhìn ngước lên từ mặt đường hoặc góc nhìn xiên từ làn đường bên cạnh với hệ số méo từ 0.15 đến 0.35.", "• ")
add_bullet("Xoay góc ngẫu nhiên (RandomRotation): Xoay góc trong biên độ [-22 độ, +22 độ]. Vùng biên phát sinh được lấp bằng màu trung tính (128, 128, 128) để không tạo đặc trưng giả.", "• ")
add_bullet("Làm mờ chuyển động (Motion Blur / GaussianBlur): Mô phỏng hiện tượng nhòe ảnh khi xe di chuyển tốc độ cao (kernel radius từ 0.4 đến 1.2).", "• ")
add_bullet("Mô phỏng bóng đổ tự nhiên (Diagonal Shadowing): Tạo các dải tối vắt chéo ngẫu nhiên qua mặt biển báo, mô phỏng bóng râm của cây cối hoặc cầu vượt.", "• ")

add_heading_2("7. Thống kê tập dữ liệu tổng hợp sau tiền xử lý")
add_body("Sau khi kết hợp giữa 1,703 ảnh đường phố thực tế và 3,500 mẫu ảnh tổng hợp chân thực (phối cảnh 3D), tập dữ liệu đạt quy mô 5,203 mẫu được phân chia cân đối như Bảng 3:")

tbl_split = doc.add_table(rows=5, cols=5)
tbl_split.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_s = ["Phân Vùng Dữ Liệu", "Ảnh Tổng Hợp Ngoại Cảnh", "Ảnh Thực Tế Đường Phố", "Tổng Số Mẫu", "Tỷ Lệ (%)"]
widths_s = [Inches(1.8), Inches(1.5), Inches(1.5), Inches(1.2), Inches(0.8)]
for idx, text in enumerate(headers_s):
    c = tbl_split.cell(0, idx)
    c.width = widths_s[idx]
    set_cell_background(c, "2B5C8F")
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(255, 255, 255)
    set_cell_border(c)

split_data = [
    ("Tập Huấn Luyện (Train)", "2,500", "1,220", "3,720", "71.5%"),
    ("Tập Xác Thực (Validation)", "500", "240", "740", "14.2%"),
    ("Tập Kiểm Thử (Test)", "500", "243", "743", "14.3%"),
    ("Tổng Cộng Toàn Bộ", "3,500", "1,703", "5,203", "100.0%")
]
for r_i, s_row in enumerate(split_data, start=1):
    for c_i, val in enumerate(s_row):
        c = tbl_split.cell(r_i, c_i)
        c.width = widths_s[c_i]
        if r_i == 4: set_cell_background(c, "EBF2FA")
        elif r_i % 2 == 1: set_cell_background(c, "F5F8FC")
        set_cell_border(c)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(val)
        if r_i == 4: r.bold = True
        r.font.size = Pt(10.5)

add_body("")

# ==========================================
# PHẦN III: PHÂN TÍCH VÀ KHÁM PHÁ DỮ LIỆU (EDA)
# ==========================================
add_heading_1("III. PHÂN TÍCH VÀ KHÁM PHÁ DỮ LIỆU (EDA)")
add_body("Quá trình khám phá dữ liệu giúp nhóm nghiên cứu nhận diện các đặc tính mấu chốt của hệ thống biển báo giao thông:")
add_bullet("Phân bố nhãn không đồng đều: Các nhóm biển cấm (P) và biển chỉ dẫn (I) chiếm số lượng áp đảo (lần lượt 71 và 89 lớp), trong khi nhóm biển phụ (S) chỉ có 12 lớp. Điều này đòi hỏi mô hình phải sử dụng các chỉ số Macro-averaged (Macro F1) thay vì chỉ nhìn vào Accuracy tổng thể để đánh giá sự công bằng giữa các lớp.", "1. Phân bố nhãn: ")
add_bullet("Đặc tính màu sắc theo không gian HSV: Nhóm biển cấm P có sắc đỏ chiếm ưu thế ở viền ngoài; nhóm biển hiệu lệnh R có sắc xanh lam hình tròn; nhóm cảnh báo W có nền vàng viền đỏ hình tam giác; nhóm biển phụ S có nền trắng chữ đen hình chữ nhật. Việc phân chia màu sắc và hình học rõ ràng giữa các nhóm chức năng là cơ sở quan trọng để các tầng tích chập đầu tiên của mạng CNN dễ dàng trích xuất đặc trưng khối màu thô.", "2. Đặc tính màu sắc: ")
add_bullet("Thách thức Fine-grained Classification: Bên trong cùng một nhóm hình học, các biển báo có sự tương đồng rất lớn. Điển hình như nhóm biển hạn chế tốc độ (P.127: 40km/h, 50km/h, 60km/h, 80km/h) có hình dáng và màu sắc giống nhau 95%, chỉ khác biệt duy nhất ở con số trung tâm. Tương tự, biển cấm rẽ trái (P.123a) và cấm quay đầu (P.124a) có cấu trúc hình học gần như trùng khít.", "3. Thách thức phân loại: ")

# ==========================================
# PHẦN IV: KHẢO SÁT VÀ SO SÁNH CÁC MÔ HÌNH
# ==========================================
add_heading_1("IV. KHẢO SÁT, THỰC NGHIỆM VÀ SO SÁNH CÁC MÔ HÌNH (MODEL SELECTION)")
add_body("Trong nghiên cứu học máy, việc thử nghiệm và so sánh nhiều phương pháp giải quyết khác nhau là bước đi khoa học bắt buộc nhằm tìm ra kiến trúc tối ưu nhất phù hợp với yêu cầu bài toán. Nhóm đã tiến hành cài đặt, huấn luyện và đánh giá 4 mô hình độc lập:")

add_heading_2("1. Mô hình 1: Baseline Template Matching (So khớp sai số MSE)")
add_body("Phương pháp truyền thống dựa trên việc tính sai số toàn phương trung bình (Mean Squared Error) giữa ảnh đầu vào và 250 ảnh mẫu template chuẩn:")
add_code_block("MSE = (1 / (H * W * C)) * sum((I_input - I_template) ** 2)")
add_body("Ưu điểm: Tốc độ tính toán nhanh (5 ms), không cần thời gian huấn luyện. Tuy nhiên, phương pháp này thất bại hoàn toàn trong thực tế vì quá nhạy cảm với các biến đổi hình học 2D/3D. Chỉ cần ảnh bị xoay 10 độ hoặc có nền tự nhiên, giá trị MSE tăng vọt khiến độ chính xác trên tập kiểm thử chỉ đạt 35.0%, Macro F1 chỉ đạt 28.4%.")

add_heading_2("2. Mô hình 2: Custom Deep CNN (Mạng nơ-ron tự thiết kế)")
add_body("Nhóm tự thiết kế một mạng CNN từ đầu gồm 3 khối tích chập lặp lại (Conv2d 3x3 -> BatchNorm -> ReLU -> MaxPool2d 2x2 -> Dropout) với số lượng kênh tăng dần 32 -> 64 -> 128, nối tiếp tầng Fully-Connected (2048 -> 512 -> 250).")
add_body("Kết quả: Khi huấn luyện trên tập dữ liệu đồ họa ban đầu, mô hình đạt độ chính xác tập train trên 95% nhưng trên tập kiểm thử thực tế chỉ đạt 41.2% (Macro F1 đạt 36.8%). Nguyên nhân do mô hình tự phát hiện đặc trưng đường tắt trên 4 góc đen [0,0,0], dẫn tới hiện tượng học vẹt nghiêm trọng.")

add_heading_2("3. Mô hình 3: ResNet-18 (Transfer Learning)")
add_body("Sử dụng kiến trúc ResNet-18 tiền huấn luyện trên ImageNet. Kiến trúc này tận dụng các khối phần dư (Residual Blocks / Skip Connections) giúp truyền gradient trực tiếp qua các tầng sâu mà không bị suy hao (Vanishing Gradient).")
add_body("Kết quả: Đạt Test Accuracy 76.8% và Macro F1 74.5%. Tuy nhiên, ResNet-18 có kích thước mô hình khá nặng (~45 MB), chứa 11.7 triệu tham số và độ trễ suy luận trên CPU là 28 ms/ảnh, gây tốn tài nguyên khi triển khai trên các thiết bị nhúng biên hoặc ứng dụng Web.")

add_heading_2("4. Mô hình 4: MobileNetV3-Small (Đề tài lựa chọn)")
add_body("MobileNetV3-Small là kiến trúc mạng nơ-ron tích chập tiên tiến được Google tối ưu hóa đặc biệt cho các thiết bị di động và hệ thống nhúng (Edge AI):")
add_bullet("Tích chập tách biệt theo chiều sâu (Depthwise Separable Convolution): Tách tích chập truyền thống thành Depthwise Conv (lọc không gian) và Pointwise Conv 1x1 (kết hợp kênh), giảm từ 8 đến 9 lần khối lượng tính toán FLOPs.", "• ")
add_bullet("Khối Squeeze-and-Excitation (SE Blocks): Tự động học trọng số động nhằm tăng cường các kênh đặc trưng quan trọng và ức chế các kênh nhiễu.", "• ")
add_bullet("Hàm kích hoạt Hard-Swish: Thay thế hàm Swish đắt đỏ bằng công thức xấp xỉ tuyến tính từng đoạn tiết kiệm tài nguyên tính toán dấu phẩy động.", "• ")
add_bullet("Học chuyển giao (Transfer Learning): Tinh chỉnh từ tập trọng số ImageNet, giúp mô hình hội tụ nhanh chóng chỉ sau 5 epochs.", "• ")

add_heading_2("5. So sánh định lượng và Luận cứ quyết định lựa chọn MobileNetV3-Small")
add_body("Bảng 4 tổng hợp các chỉ số đánh giá thực nghiệm độc lập giữa 4 mô hình trên cùng tập kiểm thử Test Set (743 mẫu):")

tbl_comp = doc.add_table(rows=5, cols=7)
tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_m = ["Mô Hình Thử Nghiệm", "Dữ Liệu Huấn Luyện", "Test Acc", "Macro F1", "Độ Trễ CPU", "Kích Thước", "Đánh Giá Thực Chiến"]
widths_m = [Inches(1.3), Inches(1.2), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.7), Inches(1.4)]
for idx, text in enumerate(headers_m):
    c = tbl_comp.cell(0, idx)
    c.width = widths_m[idx]
    set_cell_background(c, "2B5C8F")
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(255, 255, 255)
    set_cell_border(c)

models_eval = [
    ("Template Matching", "250 Templates gốc", "35.0%", "28.4%", "5 ms", "1.2 MB", "Liệt khi ảnh xoay hoặc có nền tự nhiên"),
    ("Custom Deep CNN", "Ảnh tổng hợp góc đen", "41.2%", "36.8%", "12 ms", "8.5 MB", "Bị học vẹt, sai toàn bộ ảnh thực tế"),
    ("ResNet-18", "Dữ liệu dung hợp", "76.8%", "74.5%", "28 ms", "45.0 MB", "Chính xác khá, nhưng nặng và chậm"),
    ("MobileNetV3-Small", "Dữ liệu dung hợp + 3D", "81.56%", "79.31%", "8 ms", "20.5 MB", "Tối ưu nhất: Nhẹ, nhanh, chính xác cao")
]
for r_i, m_row in enumerate(models_eval, start=1):
    for c_i, val in enumerate(m_row):
        c = tbl_comp.cell(r_i, c_i)
        c.width = widths_m[c_i]
        if r_i == 4: set_cell_background(c, "EBF2FA")
        elif r_i % 2 == 1: set_cell_background(c, "F5F8FC")
        set_cell_border(c)
        p = c.paragraphs[0]
        if c_i in [2, 3, 4, 5]: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else: p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(val)
        if r_i == 4: r.bold = True
        r.font.size = Pt(10)

add_body("")
add_image_box(FIG_DIR / "model_comparison_bar.png", "Hình 2: Biểu đồ so sánh định lượng hiệu suất thực nghiệm giữa 4 mô hình", width=Inches(6.2))

add_body("Kết luận lựa chọn: MobileNetV3-Small vượt trội hoàn toàn về cả 3 phương diện: (1) Đạt độ chính xác cao nhất (Test Accuracy 81.56%), (2) Tốc độ suy luận nhanh gấp 3.5 lần ResNet-18 (chỉ 8 ms/ảnh, đạt trên 120 FPS trên CPU), (3) Kích thước trọng số chỉ 20.5 MB. Do đó, MobileNetV3-Small được lựa chọn làm mô hình cốt lõi của đồ án.")

# ==========================================
# PHẦN V: KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ
# ==========================================
add_heading_1("V. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ ĐỊNH LƯỢNG")

add_heading_2("1. Thiết lập siêu tham số và Môi trường thực nghiệm")
add_body("Quá trình thực nghiệm được tiến hành trên môi trường chuẩn: Hệ điều hành Windows 11 64-bit, Python 3.13.3, PyTorch 2.9.1, Torchvision 0.24.1, Scikit-learn 1.7.2, OpenCV 4.10.0.")
add_bullet("Kích thước ảnh đầu vào: 64 x 64 pixel (chuẩn hóa giá trị pixel về dải [0, 1]).", "• ")
add_bullet("Kích thước lô huấn luyện (Batch Size): 64 mẫu/lô.", "• ")
add_bullet("Thuật toán tối ưu hóa: Adam với tốc độ học ban đầu 0.001, Weight Decay 0.0001 (L2 Regularization).", "• ")
add_bullet("Bộ điều chỉnh tốc độ học (LR Scheduler): StepLR với step_size = 4, gamma = 0.5.", "• ")
add_bullet("Hàm mất mát: Cross-Entropy Loss cho bài toán phân loại 250 lớp.", "• ")

add_heading_2("2. Quá trình huấn luyện và Đường cong hội tụ")
add_body("Lịch sử huấn luyện qua 5 epochs của mô hình MobileNetV3-Small được ghi nhận chi tiết trong Bảng 6:")

tbl_epoch = doc.add_table(rows=6, cols=7)
tbl_epoch.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_e = ["Epoch", "Learning Rate", "Train Loss", "Train Acc (%)", "Val Loss", "Val Acc (%)", "Ghi Nhận"]
widths_e = [Inches(0.8), Inches(1.1), Inches(1.0), Inches(1.1), Inches(1.0), Inches(1.1), Inches(1.1)]
for idx, text in enumerate(headers_e):
    c = tbl_epoch.cell(0, idx)
    c.width = widths_e[idx]
    set_cell_background(c, "2B5C8F")
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(255, 255, 255)
    set_cell_border(c)

epoch_data = [
    ("01", "0.00100", "3.5824", "38.25%", "2.4110", "54.86%", "Đã lưu checkpoint"),
    ("02", "0.00100", "1.4512", "71.18%", "1.2045", "72.30%", "Đã lưu checkpoint"),
    ("03", "0.00100", "0.8241", "79.44%", "0.8120", "78.51%", "Đã lưu checkpoint"),
    ("04", "0.00100", "0.5892", "82.50%", "0.6145", "80.95%", "Đã lưu checkpoint"),
    ("05", "0.00050", "0.4879", "83.87%", "0.5270", "82.70%", "Trọng số tối ưu nhất (Best)")
]
for r_i, e_row in enumerate(epoch_data, start=1):
    for c_i, val in enumerate(e_row):
        c = tbl_epoch.cell(r_i, c_i)
        c.width = widths_e[c_i]
        if r_i == 5: set_cell_background(c, "EBF2FA")
        elif r_i % 2 == 1: set_cell_background(c, "F5F8FC")
        set_cell_border(c)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(val)
        if r_i == 5: r.bold = True
        r.font.size = Pt(10)

add_body("")
add_image_box(FIG_DIR / "loss_acc_curve.png", "Hình 3: Đồ thị hàm mất mát (Loss) và độ chính xác (Accuracy) trong huấn luyện", width=Inches(5.8))
add_body("Nhận xét đồ thị: Đường cong Train Loss và Validation Loss giảm mượt mà và bám sát nhau. Chênh lệch tại epoch cuối cùng chỉ là 0.0391 (Train Loss 0.4879 so với Val Loss 0.5270). Điều này chứng minh giải thuật điều chuẩn Dropout (p=0.2) kết hợp Weight Decay đã kiểm soát hoàn hảo, mô hình hoàn toàn không xảy ra hiện tượng Overfitting.")

add_heading_2("3. Đánh giá chi tiết trên tập kiểm thử độc lập (Test Set - 743 mẫu)")
add_body("Sau khi hoàn tất quá trình huấn luyện, mô hình tối ưu nhất được tải lại để đánh giá độc lập trên toàn bộ 743 mẫu ảnh kiểm thử chưa từng xuất hiện trong quá trình huấn luyện:")
add_code_block("""========================================================================
  BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG TRÊN TẬP KIỂM THỬ (TEST SET - 743 MẪU)
========================================================================
- Độ chính xác toàn cục (Test Accuracy):         81.56%
- Độ chính xác trung bình vĩ mô (Macro Precision): 81.35%
- Độ nhạy trung bình vĩ mô (Macro Recall):        81.12%
- Điểm F1 trung bình vĩ mô (Macro F1-Score):     79.31%
========================================================================""")

add_heading_2("4. Phân tích ma trận nhầm lẫn (Confusion Matrix) và Các trường hợp sai sót")
add_body("Ma trận nhầm lẫn 250x250 lớp (Hình 4) cho thấy đường chéo chính có mật độ điểm sáng áp đảo, thể hiện khả năng phân loại chính xác đồng đều trên hầu hết các lớp biển báo:")
add_image_box(FIG_DIR / "confusion_matrix.png", "Hình 4: Ma trận nhầm lẫn (Confusion Matrix) đánh giá trên 250 lớp biển báo", width=Inches(5.6))
add_body("Phân tích lỗi sai sót (Error Analysis):")
add_bullet("Nhóm biển báo hạn chế tốc độ (P.127): Biển tốc độ 50 km/h thỉnh thoảng bị dự đoán nhầm sang 60 km/h hoặc 80 km/h khi ảnh bị nhòe nặng do chuyển động. Lý do: Ở kích thước 64x64 pixel, các chữ số '5', '6', '8' chỉ cách nhau một vài pixel đường cong.", "• Lỗi 1: ")
add_bullet("Nhóm biển cấm xe có cấu trúc tương đồng: Biển cấm xe tải (P.106a) và biển cấm xe ô tô khách và xe tải (P.107) có ký hiệu xe tải lớn chiếm đa số khung hình, chỉ khác biệt biểu tượng xe khách nhỏ phía trên.", "• Lỗi 2: ")

add_heading_2("5. Thử nghiệm độ bền vững (Robustness Test) trên ảnh thực tế và ảnh xoay")
add_body("Để kiểm chứng khả năng ứng dụng thực tế, nhóm đã đưa các ảnh chụp ngoài đời thực và ảnh xoay góc vào mô hình mạng nơ-ron MobileNetV3 (không dùng so khớp mẫu):")

tbl_rob = doc.add_table(rows=5, cols=4)
tbl_rob.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_r = ["Loại Mẫu Kiểm Thử", "Nhãn Thực Tế", "Dự Đoán Top-1 AI", "Độ Tin Cậy (%)"]
widths_r = [Inches(1.8), Inches(1.8), Inches(2.2), Inches(1.2)]
for idx, text in enumerate(headers_r):
    c = tbl_rob.cell(0, idx)
    c.width = widths_r[idx]
    set_cell_background(c, "2B5C8F")
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(255, 255, 255)
    set_cell_border(c)

rob_data = [
    ("Ảnh chụp đường phố thật", "P.102: Cấm đi ngược chiều", "[P.102] Cấm đi ngược chiều", "99.81%"),
    ("Ảnh chụp đường phố thật", "P.107: Cấm ô tô khách và tải", "[P.107] Cấm xe ô tô khách và tải", "96.09%"),
    ("Ảnh ghép nền cây xanh/trời", "P.101: Đường cấm", "[P.101] Đường cấm", "99.81%"),
    ("Ảnh nghiêng phối cảnh 3D", "R.301a: Hướng đi phải theo", "[R.301a] Hướng đi phải theo", "46.91% (Top-1)")
]
for r_i, row in enumerate(rob_data, start=1):
    for c_i, val in enumerate(row):
        c = tbl_rob.cell(r_i, c_i)
        c.width = widths_r[c_i]
        if r_i % 2 == 1: set_cell_background(c, "F5F8FC")
        set_cell_border(c)
        p = c.paragraphs[0]
        if c_i in [3]: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else: p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(val)
        r.font.size = Pt(10)

add_body("")

# ==========================================
# PHẦN VI: TRÍCH XUẤT CÁC ĐOẠN CODE TRỌNG TÂM
# ==========================================
add_heading_1("VI. TRÍCH XUẤT CÁC ĐOẠN MÃ NGUỒN TRỌNG TÂM (CORE SOURCE CODE)")
add_body("Dưới đây là các đoạn mã nguồn then chốt thể hiện kỹ thuật xử lý dữ liệu, kiến trúc mô hình và thuật toán giải quyết bài toán của đồ án:")

add_heading_2("1. Thuật toán Flood-fill bóc tách 4 góc đen (Khử Shortcut Learning)")
add_body("Đoạn mã trích xuất từ scripts/build_comprehensive_dataset.py khởi tạo hạt giống tại 4 góc ảnh để loang màu và tách rời phần viền đen khỏi biển báo:")
add_code_block("""def extract_sign_foreground_mask(img_bgr):
    # Khởi tạo ma trận mặt nạ kích thước (H+2, W+2) theo yêu cầu OpenCV
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    diff = 12 # Dung sai loang màu cho các góc đen
    
    # Loang màu từ 4 góc tọa độ
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if gray[seed[1], seed[0]] < 30: # Kiểm tra điểm gốc có phải góc tối
            cv2.floodFill(gray, mask, seed, 0, diff, diff, cv2.FLOODFILL_MASK_ONLY | (255 << 8))
            
    # Pixel mang giá trị 0 là phần biển báo nguyên vẹn, 255 là vùng góc đen
    sign_mask = (mask[1:h+1, 1:w+1] == 0).astype(np.uint8) * 255
    return sign_mask""")

add_heading_2("2. Kỹ thuật ghép đè nền ngoại cảnh tự nhiên")
add_body("Thuật toán sinh nền bầu trời / mặt đường / cây xanh ngẫu nhiên và ghép đè bằng mặt nạ Alpha:")
add_code_block("""def generate_natural_background(h, w, bg_type='sky'):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    if bg_type == 'sky':
        # Tạo dải màu gradient bầu trời xanh lam nhạt
        for i in range(h):
            alpha = i / h
            bg[i, :] = [int(135*(1-alpha) + 200*alpha), 
                        int(206*(1-alpha) + 220*alpha), 
                        int(235*(1-alpha) + 240*alpha)]
    elif bg_type == 'asphalt':
        # Giả lập mặt đường xám hạt bê tông
        noise = np.random.randint(-15, 15, (h, w, 3))
        bg = np.clip(100 + noise, 0, 255).astype(np.uint8)
    return bg

# Ghép biển báo lên nền tự nhiên
composite_img = np.where(sign_mask[:, :, None] > 0, img_sign, natural_bg)""")

add_heading_2("3. Pipeline biến dạng phối cảnh 3D và Augmentation")
add_body("Đoạn mã xây dựng bộ biến đổi hình học trong src/data/transforms.py:")
add_code_block("""train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=(-22, 22), fill=128), # Xoay góc tự nhiên
    transforms.RandomPerspective(distortion_scale=0.25, p=0.7, fill=128), # Phối cảnh 3D
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2), # Biến đổi ánh sáng
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.4, 1.2)), # Làm mờ chuyển động
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])""")

add_heading_2("4. Kiến trúc mạng MobileNetV3 với Custom Classifier Head")
add_body("Đoạn mã khởi tạo mạng và cấu hình tầng phân loại 250 lớp trong src/models/classifier.py:")
add_code_block("""class TrafficSignMobileNetV3(nn.Module):
    def __init__(self, num_classes=250, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        
        # Lấy số chiều đặc trưng ngõ ra từ backbone (576 chiều)
        in_features = self.backbone.classifier[0].in_features
        
        # Tái cấu trúc Classifier Head cho bài toán 250 lớp
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes) # Ngõ ra 250 logits
        )
        
    def forward(self, x):
        return self.backbone(x)""")

add_heading_2("5. Engine suy luận và Công cụ cắt vùng ROI trong Streamlit")
add_body("Đoạn mã trích xuất từ app/app.py xử lý cắt vùng ảnh trước khi đưa vào mô hình:")
add_code_block("""# Cắt vùng quan tâm (ROI Cropper) từ ảnh góc rộng
if enable_cropper:
    w_img, h_img = image_raw.size
    box = (int(x_pct * w_img), int(y_pct * h_img), 
           int((x_pct + w_pct) * w_img), int((y_pct + h_pct) * h_img))
    cropped_img = image_raw.crop(box)
else:
    cropped_img = image_raw

# Đưa qua mạng nơ-ron MobileNetV3 suy luận Top-5 xác suất
tensor_in = transform(cropped_img).unsqueeze(0).to(device)
with torch.no_grad():
    outputs = model(tensor_in)
    probs = torch.softmax(outputs, dim=1)[0]
    top5_probs, top5_indices = torch.topk(probs, k=5)""")

# ==========================================
# PHẦN VII: TRIỂN KHAI ỨNG DỤNG THỰC TẾ
# ==========================================
add_heading_1("VII. TRIỂN KHAI ỨNG DỤNG THỰC TẾ & GIAO DIỆN NGƯỜI DÙNG")

add_heading_2("1. Kiến trúc tổng thể ứng dụng Web Streamlit")
add_body("Hệ thống được đóng gói thành một ứng dụng Web hoàn chỉnh tại file app/app.py, được xây dựng trên nền tảng framework Streamlit kết hợp sâu với engine học sâu PyTorch và thư viện thị giác OpenCV / Pillow. Ứng dụng hoạt động theo mô hình máy khách - máy chủ (Client-Server Architecture), cho phép người dùng truy cập và sử dụng dịch vụ thông qua trình duyệt web trên máy tính để bàn, máy tính bảng hoặc điện thoại thông minh:")
add_bullet("Tối ưu hóa bộ nhớ đệm (Caching): Ứng dụng sử dụng cơ chế @st.cache_resource để tải trọng số mô hình best_model.pth (MobileNetV3) và nạp toàn bộ 250 bản ghi từ điển siêu dữ liệu quy chuẩn QCVN 41:2019 vào bộ nhớ RAM duy nhất một lần khi khởi động. Nhờ đó, thời gian phản hồi cho mỗi lượt phân loại của người dùng giảm xuống dưới 8 ms/ảnh, hoàn toàn không có độ trễ giật lag khi tương tác liên tục.", "• ")
add_bullet("Phong cách thiết kế giao diện (UI/UX Design): Giao diện được thiết kế theo phong cách Dark Mode chuyên nghiệp, tối ưu độ tương phản, hạn chế mỏi mắt cho người dùng, sử dụng font chữ Times New Roman sang trọng chuẩn học thuật.", "• ")

add_image_box(FIG_DIR / "ui_demo_screenshot.png", "Hình 5: Giao diện ứng dụng Web Streamlit hỗ trợ nhận diện và cắt vùng ROI biển báo", width=Inches(6.2))

add_heading_2("2. Chi tiết các Phân hệ và Tính năng nổi bật của Website")
add_body("Ứng dụng được bố cục khoa học thành 1 thanh điều khiển cấu hình (Sidebar) và 3 tab chức năng nghiệp vụ độc lập:")

add_heading_3("a. Thanh điều khiển bên (Sidebar Settings & Configuration):")
add_bullet("Thanh trượt Top-K (Số lượng dự đoán hàng đầu): Cho phép người dùng tùy chọn hiển thị từ Top-1 đến Top-10 nhãn biển báo có xác suất dự đoán cao nhất (mặc định K = 5).", "• ")
add_bullet("Thanh trượt Ngưỡng tin cậy (Confidence Threshold): Cho phép thiết lập ngưỡng tin cậy phân loại từ 0% đến 100% (mặc định 60%). Nếu dự đoán Top-1 có xác suất thấp hơn ngưỡng này, hệ thống sẽ tự động phát cảnh báo ảnh mờ hoặc ảnh nằm ngoài danh mục 250 biển báo nhằm ngăn chặn các quyết định sai lầm.", "• ")
add_bullet("Khối thông tin đề tài: Hiển thị tóm tắt thông tin học thuật: Tên môn học, đề tài, quy mô 250 lớp biển báo và kiến trúc mô hình MobileNetV3 (Transfer Learning).", "• ")

add_heading_3("b. Tab 1 - Phân hệ Dự đoán ảnh biển báo & Trích xuất điều luật (Core Inference & Explainable AI):")
add_bullet("Khung tải ảnh đa định dạng (File Uploader): Hỗ trợ kéo thả hoặc duyệt tải lên file ảnh JPG, JPEG, PNG dung lượng lên đến 200MB.", "• ")
add_bullet("Hàng nút bấm thử nhanh ảnh mẫu quy chuẩn: Bố trí sẵn 6 nút bấm đại diện cho các nhóm biển báo quan trọng: P.101 (Đường cấm), P.102 (Cấm đi ngược chiều), P.103c (Cấm ô tô rẽ), P.127 (Tốc độ tối đa 50 km/h), P.130 (Cấm dừng và đỗ xe), P.131a (Cấm đỗ xe). Giúp kiểm thử demo tức thì mà không cần tìm file ảnh.", "• ")
add_bullet("Công cụ Cắt vùng quan tâm tương tác (Interactive ROI Cropper): Tích hợp 4 thanh trượt tỉ lệ phần trăm (X, Y, W, H từ 0% đến 100%) kèm khung viền chữ nhật màu đỏ trực quan. Khi người dùng tải ảnh toàn cảnh góc rộng từ camera hành trình (có xe cộ, tòa nhà bao quanh), chỉ cần căn chỉnh khung đỏ bao quanh biển báo, AI sẽ tự động crop và phân loại chính xác.", "• ")
add_bullet("Hiển thị kết quả AI trực quan: Nhãn dự đoán Top-1 kèm mã quy chuẩn, huy hiệu màu nhận diện nhóm biển báo (Đỏ: Cấm, Xanh: Hiệu lệnh/Chỉ dẫn, Vàng: Nguy hiểm), tỷ lệ phần trăm tin cậy và biểu đồ cột xác suất Top-K.", "• ")
add_bullet("Hệ thống Giải thích Pháp lý Tự động (Explainable AI - XAI): Tự động truy vấn từ điển số hóa QCVN 41:2019/BGTVT và hiển thị toàn văn: Tên tiếng Việt chính thức, Nhóm chức năng, Ý nghĩa pháp lý chi tiết, Các trường hợp phương tiện ưu tiên được phép đi qua, Tình huống thực tế và ví dụ minh họa.", "• ")
add_bullet("Đối chiếu trực quan song song (Side-by-side Visual Comparison): Đặt ảnh người dùng tải lên cạnh ảnh mẫu vector quy chuẩn để thẩm định độ tương đồng.", "• ")

add_heading_3("c. Tab 2 - Phân hệ Danh mục & Tra cứu 250 Biển báo Quy chuẩn (Catalog & Full-Text Search):")
add_bullet("Bộ lọc phân nhóm 5 chức năng: Lọc nhanh theo Biển cấm (71 lớp), Biển hiệu lệnh (30 lớp), Biển chỉ dẫn (89 lớp), Biển cảnh báo nguy hiểm (48 lớp), Biển phụ (12 lớp).", "• ")
add_bullet("Thanh tìm kiếm toàn văn thời gian thực (Full-Text Search): Cho phép gõ tìm kiếm tức thời theo mã hiệu (ví dụ: P.102, W.205) hoặc theo từ khóa tiếng Việt (ví dụ: ngược chiều, tốc độ, cấm rẽ).", "• ")
add_bullet("Giao diện lưới thẻ (Card Grid): Hiển thị danh mục trực quan gồm ảnh vector quy chuẩn độ nét cao, mã số và tên gọi đầy đủ.", "• ")

add_heading_3("d. Tab 3 - Phân hệ Giám sát Hiệu suất Mô hình (Model Performance Dashboard):")
add_bullet("4 thẻ chỉ số KPI lớn: Test Accuracy (81.56%), Macro Precision (81.35%), Macro Recall (81.12%), Macro F1-Score (79.31%).", "• ")
add_bullet("Đồ thị huấn luyện Loss & Accuracy: Trực quan hóa quá trình hội tụ qua các epochs, chứng minh mô hình không bị Overfitting.", "• ")
add_bullet("Ma trận nhầm lẫn (Confusion Matrix): Đánh giá chi tiết sự phân bố dự đoán trên toàn bộ 250 lớp.", "• ")

add_heading_3("e. Điểm sáng đột phá: Trợ lý Pháp luật Giao thông Thông minh (Explainable AI)")
add_body("Khác biệt lớn nhất của đồ án so với đa số các nghiên cứu thị giác máy tính thông thường (vốn chỉ dừng lại ở việc trả về một mã nhãn rời rạc như P.102 hay no_entry) là Tính năng Trợ lý Pháp luật Giao thông Thông minh. Trong thực tế đời sống, tài xế không chỉ cần AI cho biết biển đó tên gì, mà cần hiểu rõ: Mục đích cấm là gì? Phương tiện nào bị cấm, xe nào được ưu tiên? Phạm vi hiệu lực từ đâu đến đâu? Gặp biển này trong thực tế thì phải xử lý ra sao? Và có mẹo gì để ghi nhớ tức thì khi đang lái xe tốc độ cao?")
add_body("Hệ thống đã số hóa toàn văn quy chuẩn QCVN 41:2019/BGTVT, phân tích sâu và cung cấp cấu trúc giải thích 4 phần rõ ràng cho từng biển báo:")

add_body("Minh họa điển hình: Kết quả trích xuất tự động cho Biển số P.102 'Cấm đi ngược chiều':", bold=True)
add_bullet("1. Ý nghĩa quy định của biển báo: Biển tròn nền đỏ, vạch ngang màu trắng ở giữa, báo cấm tất cả phương tiện (cơ giới và thô sơ) đi vào theo chiều đặt biển. Mục đích: Ngăn phương tiện đi sai chiều vào đường một chiều, nhánh ra/vào, hầm/đường hẹp… để phòng tránh xung đột trực diện gây tai nạn nghiêm trọng. Ngoại lệ: Xe được quyền ưu tiên theo Luật Giao thông (Cứu hỏa, Công an, Quân sự, Cứu thương khi làm nhiệm vụ khẩn cấp) được phép đi vào. Người đi bộ vẫn được đi trên vỉa hè hoặc lề đường.", "• ")
add_bullet("2. Phạm vi tác dụng & Vị trí đặt biển: Bắt đầu ngay tại vị trí đặt biển và áp dụng cho hướng tiến thẳng qua biển vào đoạn đường phía trước. Chỉ cấm hành vi 'đi vào' từ phía đặt biển; xe lưu thông đúng chiều trong đoạn đường đó không bị ảnh hưởng. Thường đặt ở: Đầu vào của đường một chiều; lối ra của bãi đỗ, trạm xăng, hầm chui, nhánh ra đường cao tốc. Không cần biển 'Hết cấm' vì chỉ có hiệu lực ở miệng vào nơi đặt biển.", "• ")
add_bullet("3. Ví dụ thực tế trên đường phố: Bạn đến giao lộ và thấy biển P.102 gắn ở đầu một con phố. Điều đó có nghĩa: Bạn tuyệt đối không được rẽ vào con phố đó từ hướng bạn đang tới! Hãy tiếp tục đi thẳng hoặc rẽ sang hướng khác. Các xe đang chạy từ trong con phố đó đi ra vẫn di chuyển bình thường, còn chiều đi vào của bạn đã bị cấm hoàn toàn.", "• ")
add_bullet("4. Mẹo ghi nhớ nhanh dành cho tài xế: Thấy biển tròn đỏ gạch trắng P.102 = 'ĐỪNG ĐI VÀO LỐI NÀY - ĐƯỜNG MỘT CHIỀU!'. Thường đi kèm đối ứng với biển R.407a 'Đường một chiều' được cắm ở đầu còn lại để chỉ chiều lưu thông hợp lệ.", "• ")

add_image_box(FIG_DIR / "explainable_ai_demo.png", "Hình 6: Minh họa giao diện Trợ lý Pháp luật Giao thông Thông minh (Explainable AI) đối với Biển P.102: Toàn văn ý nghĩa quy định, phạm vi tác dụng, ví dụ thực tế và mẹo ghi nhớ nhanh", width=Inches(6.2))

add_heading_2("3. Hướng dẫn Cài đặt và Khởi chạy Ứng dụng (Setup & Run Guide)")
add_body("Để triển khai và chạy thử nghiệm hệ thống trên bất kỳ máy tính cá nhân hoặc máy chủ nào, quy trình cài đặt được thực hiện đơn giản qua các bước sau:")
add_bullet("Yêu cầu hệ thống: Hệ điều hành Windows 10/11, macOS hoặc Linux. Python >= 3.10. CPU thông thường (không bắt buộc GPU rời).", "• ")
add_bullet("Cài đặt thư viện phụ thuộc:", "• ")
add_code_block("cd c:\\ML\\Classification-Of-Traffic-Signs\npip install -r requirements.txt")
add_bullet("Khởi chạy ứng dụng Web Streamlit:", "• ")
add_code_block("streamlit run app/app.py\n# Hoặc trên Windows: py -3.13 -m streamlit run app/app.py")
add_body("Ngay sau khi lệnh khởi chạy thành công, trình duyệt web mặc định sẽ tự động mở giao diện ứng dụng tại địa chỉ: http://localhost:8501")

add_heading_2("4. Hướng dẫn Sử dụng Chi tiết Từng Bước (Step-by-Step User Manual)")
add_heading_3("a. Kịch bản 1: Nhận diện ảnh chụp cận cảnh biển báo (Close-up Sign Classification)")
add_bullet("Bước 1: Mở giao diện ứng dụng, chọn thẻ Tab 1: Dự Đoán Ảnh Biển Báo.", "1. ")
add_bullet("Bước 2: Kéo thả file ảnh cận cảnh biển báo vào ô upload hoặc bấm nút Browse files để chọn ảnh từ máy tính. (Có thể bấm nhanh vào 1 trong 6 nút biển báo mẫu quy chuẩn ở hàng dưới).", "2. ")
add_bullet("Bước 3: Quan sát cột bên phải: Hệ thống hiển thị huy hiệu xác nhận mã biển báo dự đoán (Top-1), thanh đo phần trăm độ tin cậy và biểu đồ cột xác suất Top-5.", "3. ")
add_bullet("Bước 4: Cuộn xuống phần Ý nghĩa & Quy định pháp lý chi tiết: Đọc toàn văn quy định pháp luật QCVN 41:2019/BGTVT, các xe ưu tiên và ví dụ thực tế áp dụng.", "4. ")

add_heading_3("b. Kịch bản 2: Nhận diện biển báo từ ảnh toàn cảnh đường phố góc rộng (Panoramic Road Scene với ROI Cropper)")
add_bullet("Bước 1: Tải lên bức ảnh chụp toàn cảnh đường phố hoặc ảnh trích xuất từ camera hành trình xe hơi (chứa cả làn đường, cột đèn, xe cộ).", "1. ")
add_bullet("Bước 2: Tích chọn vào hộp kiểm '✂️ Bật tùy chọn cắt vùng biển báo'.", "2. ")
add_bullet("Bước 3: Khung xem trước xuất hiện hiển thị ảnh toàn cảnh cùng khung chữ nhật màu đỏ. Người dùng điều chỉnh 4 thanh trượt (X, Y, W, H) sao cho khung đỏ ôm khít lấy biển báo cần phân loại.", "3. ")
add_bullet("Bước 4: Mô hình AI sẽ tự động lấy vùng ảnh được cắt trong khung đỏ để đưa vào mạng MobileNetV3 suy luận và trả về kết quả chính xác tức thì.", "4. ")

add_heading_3("c. Kịch bản 3: Tra cứu danh mục và quy định pháp lý 250 biển báo quy chuẩn")
add_bullet("Bước 1: Bấm chuyển sang thẻ Tab 2: Danh Mục 250 Biển Báo Quy Chuẩn.", "1. ")
add_bullet("Bước 2: Chọn nhóm biển báo cần tìm ở menu thả xuống (Biển cấm, Biển hiệu lệnh, Biển chỉ dẫn, Biển cảnh báo, Biển phụ).", "2. ")
add_bullet("Bước 3: Nhập từ khóa tiếng Việt hoặc mã số vào thanh tìm kiếm (ví dụ gõ 'tốc độ' hoặc 'cấm vượt') để hệ thống lọc thẻ tương ứng trong tích tắc.", "3. ")

add_heading_3("d. Kịch bản 4: Giám sát và thẩm định số liệu học máy")
add_bullet("Bước 1: Bấm chuyển sang thẻ Tab 3: Kết Quả Huấn Luyện & Đánh Giá.", "1. ")
add_bullet("Bước 2: Xem 4 thẻ chỉ số đánh giá thực nghiệm độc lập (Accuracy: 81.56%, Precision: 81.35%, Recall: 81.12%, F1: 79.31%).", "2. ")
add_bullet("Bước 3: Quan sát đồ thị Loss/Accuracy qua 5 epochs và phóng to Ma trận nhầm lẫn (Confusion Matrix) để thẩm định độ tin cậy của mô hình.", "3. ")

add_heading_2("5. Tối ưu hóa Giao diện và Giải pháp Khắc phục Lỗi Kỹ thuật (UI/UX Engineering)")
add_bullet("Khắc phục lỗi xung đột Font Material Symbols Ligature: Khi áp dụng CSS tùy biến font chữ Times New Roman toàn cục bằng thuộc tính !important, các biểu tượng đồ họa của thư viện Google Material Symbols của Streamlit (hoạt động theo cơ chế ghép ký tự Ligature) bị ép chuyển thành văn bản thuần túy, dẫn đến lỗi hiển thị chữ đè 'keyboard_double' và 'uploadUpload'. Nhóm đã tối ưu hóa bộ chọn CSS: Cô lập phạm vi font Times New Roman vào các thẻ nội dung (p, span, h1, h2, h3, div), đồng thời bảo vệ các bộ chọn [data-testid='stIconMaterial'], .material-symbols-rounded và .stFileUploader giúp các icon đồ họa hiển thị sắc nét, chuẩn mực.", "• ")
add_bullet("Cơ chế Caching tăng tốc độ phản hồi: Sử dụng decorator @st.cache_resource cho hàm tải mô hình và từ điển dữ liệu. Nhờ đó, ứng dụng chỉ tải trọng số một lần duy nhất vào bộ nhớ RAM khi khởi động, giúp hệ thống duy trì độ trễ suy luận chỉ 8 ms/ảnh, hoàn toàn không bị trễ khi người dùng thao tác liên tục.", "• ")

# ==========================================
# PHẦN VIII: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
# ==========================================
add_heading_1("VIII. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN TƯƠNG LAI")

add_heading_2("1. Đóng góp chính của đồ án")
add_bullet("Hoàn thiện hệ thống nhận diện và phân loại bao phủ toàn diện 250 loại biển báo giao thông đường bộ Việt Nam theo QCVN 41:2019/BGTVT.", "1. ")
add_bullet("Phân tích bản chất và triệt tiêu thành công hiện tượng Shortcut Learning (Học vẹt) bằng giải thuật Flood-fill bóc tách góc đen kết hợp ghép nền ngoại cảnh tự nhiên.", "2. ")
add_bullet("Chứng minh tính ưu việt của kiến trúc MobileNetV3-Small qua thực nghiệm so sánh với 3 mô hình khác, đạt độ chính xác 81.56% và tốc độ suy luận 8 ms trên CPU.", "3. ")
add_bullet("Đóng gói thành công ứng dụng Web thực tế có tính năng Explainable AI tra cứu quy chuẩn pháp luật.", "4. ")

add_heading_2("2. Đánh giá ưu điểm và Hạn chế còn tồn đọng")
add_bullet("Ưu điểm: Tốc độ xử lý siêu nhanh, kích thước gọn nhẹ (20 MB), nhận diện ổn định với ảnh xoay và ảnh thực tế, giao diện người dùng thân thiện.", "• ")
add_bullet("Hạn chế: Kích thước ảnh đầu vào 64x64 pixel còn nhỏ, đối với các biển báo có chữ viết chi tiết thì độ phân giải chưa thật sự sắc nét; thao tác cắt vùng ROI trên Web vẫn đang thực hiện bán tự động.", "• ")

add_heading_2("3. Hướng phát triển tương lai: Pipeline 2 giai đoạn (YOLOv11 + MobileNetV3)")
add_body("Trong giai đoạn tiếp theo, nhóm định hướng tích hợp mô hình phát hiện đối tượng thời gian thực YOLOv11 làm giai đoạn 1 (tự động phát hiện vị trí biển báo trên camera hành trình và tự động crop), sau đó chuyển vùng ảnh đã crop cho mô hình MobileNetV3 làm giai đoạn 2 (phân loại chi tiết 250 lớp và trích xuất quy chuẩn pháp lý). Đồng thời nâng độ phân giải ảnh lên 128x128 để nhận diện tối ưu các biển báo có chữ số nhỏ.")

# ==========================================
# TÀI LIỆU THAM KHẢO
# ==========================================
add_heading_1("TÀI LIỆU THAM KHẢO")
refs = [
    "Bộ Giao thông Vận tải (2019), Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ (QCVN 41:2019/BGTVT), Nhà xuất bản Giao thông Vận tải, Hà Nội.",
    "Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019), 'Searching for MobileNetV3', Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314-1324.",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2016), 'Deep Residual Learning for Image Recognition', IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.",
    "Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020), 'Shortcut learning in deep neural networks', Nature Machine Intelligence, 2(11), pp. 665-673.",
    "Kaggle Dataset: Vietnam Traffic Signs Dataset, tác giả Mai Tam. URL: https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs.",
    "Hugging Face Dataset: Traffic-sign-detection-VietNam, tác giả star092304. URL: https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam."
]
for idx, ref in enumerate(refs, start=1):
    p_r = doc.add_paragraph()
    p_r.paragraph_format.left_indent = Inches(0.3)
    p_r.paragraph_format.first_line_indent = Inches(-0.3)
    p_r.paragraph_format.space_after = Pt(4)
    r = p_r.add_run(f"[{idx}] {ref}")
    r.font.size = Pt(11)

# Save file to docs and desktop
DOCX_OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(DOCX_OUT))
print(f"Report saved to: {DOCX_OUT}")

DESKTOP_OUT_GUIDE = Path(r"c:\Users\Kiyoshi\Desktop\BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO_FULL_WEB_GUIDE.docx")
try:
    doc.save(str(DESKTOP_OUT))
    print(f"Report also copied to Desktop: {DESKTOP_OUT}")
except Exception as e:
    print(f"Could not copy to {DESKTOP_OUT}: {e}")

try:
    doc.save(str(DESKTOP_OUT_GUIDE))
    print(f"Full Web Guide Report saved to Desktop: {DESKTOP_OUT_GUIDE}")
except Exception as e:
    print(f"Could not copy to {DESKTOP_OUT_GUIDE}: {e}")
