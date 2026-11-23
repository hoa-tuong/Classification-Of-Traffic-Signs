# SLIDE THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN MACHINE LEARNING
## ĐỀ TÀI: PHÂN LOẠI 250 LOẠI BIỂN BÁO GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM (QCVN 41:2019/BGTVT)

---

### SLIDE 1: TRANG TIÊU ĐỀ
* **Tên đề tài:** Nghiên cứu và Xây dựng Hệ thống Nhận diện, Phân loại 250 Loại Biển Báo Giao Thông Đường Bộ Việt Nam Theo Quy Chuẩn QCVN 41:2019/BGTVT.
* **Môn học:** Machine Learning & Deep Learning
* **Nhóm thực hiện:** [Tên Nhóm / Tên Sinh Viên]
* **Giảng viên hướng dẫn:** [Tên Giảng Viên]
* **Nền tảng mã nguồn:** GitHub: `hoa-tuong/Classification-Of-Traffic-Signs`

---

### SLIDE 2: ĐẶT VẤN ĐỀ & TÍNH CẤP THIẾT
* **Bối cảnh:**
  - Hệ thống giao thông đường bộ Việt Nam phức tạp, mật độ phương tiện cao.
  - Người lái xe dễ vi phạm hoặc gặp nguy hiểm do không quan sát kịp biển báo (đặc biệt biển cấm theo giờ, biển gộp làn đường).
  - Các hệ thống hỗ trợ lái xe ADAS / Xe tự hành đang rất cần module thị giác máy tính đặc thù cho Việt Nam.
* **Hạn chế của các nghiên cứu trước:**
  - Đa số chỉ dùng tập dữ liệu nước ngoài (GTSRB của Đức) hoặc chỉ giới hạn ở 20-40 biển báo cơ bản.
  - Thiếu hệ thống giải thích ý nghĩa pháp lý đi kèm.
* **Mục tiêu đồ án:**
  - Xây dựng mô hình AI nhận diện trọn vẹn **250 lớp biển báo** theo chuẩn QCVN 41:2019.
  - Xử lý tốt ảnh chụp thực tế ngoài đường phố và ảnh tải từ mạng internet.

---

### SLIDE 3: BÀI HỌC KINH NGHIỆM: PHÁT HIỆN LỖI "HỌC VẸT" (SHORTCUT LEARNING)
* **Vấn đề ban đầu của nhóm:**
  - Mô hình đạt độ chính xác rất cao trên tập dữ liệu mẫu, nhưng **thất bại hoàn toàn** khi xoay nghiêng hoặc tải ảnh thực tế từ Google.
* **Nguyên nhân sâu xa (Root Cause):**
  1. **Cạm bẫy góc đen:** Ảnh mẫu biển tròn/tam giác có 4 góc màu đen tuyệt đối $[0, 0, 0]$. CNN học đường tắt: *Góc đen = Biển cấm/hiệu lệnh*. Khi ảnh thực tế có nền bầu trời/cây cối $\rightarrow$ Mô hình mất phương hướng.
  2. **Lỗi Template Matching:** Thuật toán đo sai số pixel MSE ($<0.015$) bị liệt hoàn toàn khi ảnh bị lệch góc dù chỉ $10^\circ$.

---

### SLIDE 4: ĐỘT PHÁ GIẢI PHÁP: PIPELINE CHỐNG HỌC VẸT
* **1. Thuật toán Flood-fill bóc tách góc đen:**
  - Loang màu từ 4 góc ảnh để nhận diện chính xác phần biên ngoài biển báo mà không làm mất chi tiết bên trong.
* **2. Ghép đè nền tự nhiên (Natural Background Replacement):**
  - Thay thế góc đen bằng 4 loại ngoại cảnh: Bầu trời (Sky Gradient), Mặt đường (Asphalt), Cây xanh (Foliage), Tòa nhà (Urban).
* **3. Tăng cường biến dạng phối cảnh 3D:**
  - `RandomPerspective` (độ méo 0.2 - 0.35) mô phỏng góc nhìn camera xe.
  - Xoay ngẫu nhiên $\pm 22^\circ$, làm mờ chuyển động, đổ bóng ngang.
* **4. Trích xuất 1,703 ảnh thực tế:**
  - Cắt từ hơn 10,000 ảnh đường phố Việt Nam (Hà Nội, TP.HCM) và ánh xạ sang chuẩn 250 lớp.

---

### SLIDE 5: KIẾN TRÚC MÔ HÌNH HỌC SÂU (MOBILENETV3-SMALL)
* **Lý do chọn MobileNetV3-Small:**
  - Thiết kế siêu nhẹ (chỉ khoảng 20 MB trọng số).
  - Tối ưu cho thiết bị biên (Edge AI / Embedded Systems) với tốc độ suy luận $\sim 8\text{ ms}$ trên CPU thường.
* **Kỹ thuật cốt lõi:**
  - **Depthwise Separable Convolution:** Giảm 8-9 lần khối lượng tính toán so với tích chập truyền thống.
  - **Squeeze-and-Excitation (SE Blocks):** Tự động học tầm quan trọng của từng kênh đặc trưng.
  - **Hard-Swish Activation:** Tăng tốc độ tính toán phần cứng.
  - **Transfer Learning:** Khởi tạo từ ImageNet weights, tinh chỉnh tầng Classifier Head 250 lớp.

---

### SLIDE 6: KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG
* **Quy mô tập dữ liệu sau cải tiến:** 5,203 mẫu (Train: 3,720 | Val: 740 | Test: 743).
* **Chỉ số đánh giá trên tập kiểm thử độc lập (Test Set):**
  - **Độ chính xác toàn cục (Accuracy):** **81.56%**
  - **Độ chính xác trung bình vĩ mô (Macro Precision):** **81.35%**
  - **Độ nhạy trung bình vĩ mô (Macro Recall):** **81.12%**
  - **Điểm F1 trung bình vĩ mô (Macro F1-Score):** **79.31%**
* **Độ ổn định huấn luyện:**
  - Train Loss: $0.4879$ | Val Loss: $0.5270$ (chênh lệch cực nhỏ, hoàn toàn không bị Overfitting).

---

### SLIDE 7: SO SÁNH TRƯỚC VÀ SAU CẢI TIẾN

| Tiêu chí đánh giá | Trước cải tiến (Baseline) | Sau cải tiến (Đề tài hoàn thiện) |
| :--- | :---: | :---: |
| **Nhận diện ảnh mẫu thẳng** | 100% | 99.8% |
| **Nhận diện ảnh xoay nghiêng ($\pm 20^\circ$)** | **0% (Liệt hoàn toàn)** | **81.56%** |
| **Nhận diện ảnh mạng / ảnh thực tế** | **Sai lệch hoàn toàn** | **Ổn định cao ($>80\%$)** |
| **Cơ chế hoạt động** | So khớp pixel cục bộ (Học vẹt) | Học biểu diễn đặc trưng không gian sâu (CNN) |
| **Số lượng lớp bao phủ** | 250 lớp lý thuyết | 250 lớp thực chiến |

---

### SLIDE 8: ỨNG DỤNG THỰC TẾ & GIAO DIỆN STREAMLIT
* **Tab 1: Nhận diện & Cắt vùng (ROI Cropper):**
  - Hỗ trợ tải ảnh hoặc chọn nhanh ảnh mẫu.
  - Tích hợp 4 thanh trượt ($X, Y, W, H$) giúp cắt chính xác biển báo từ ảnh góc rộng trước khi nhận diện.
  - Hiển thị Top-5 dự đoán kèm độ tin cậy và trích dẫn toàn văn điều luật quy chuẩn.
* **Tab 2: Từ điển số hóa 250 biển báo:**
  - Tra cứu nhanh theo mã (`P.102`, `W.201...`) hoặc tên tiếng Việt.
* **Tab 3: Bảng chỉ số thực nghiệm:**
  - Hiển thị đầy đủ Confusion Matrix và đồ thị Loss/Accuracy.

---

### SLIDE 9: KẾT LUẬN & HƯỚNG PHÁT TRIỂN TƯƠNG LAI
* **Đóng góp của đồ án:**
  - Giải quyết bài toán phân loại toàn diện 250 lớp biển báo giao thông Việt Nam theo QCVN 41:2019.
  - Phân tích và triệt tiêu thành công hiện tượng Shortcut Learning trong học sâu.
  - Đóng gói ứng dụng trực quan, có tính năng giải thích pháp luật (Explainable AI).
* **Hướng phát triển tiếp theo:**
  - **End-to-End ADAS:** Ghép nối mô hình phát hiện đối tượng **YOLOv11** (tự động phát hiện và cắt bounding box) với mô hình **MobileNetV3** (phân loại chi tiết).
  - Tăng kích thước ảnh đầu vào ($128 \times 128$) để nhận diện tốt hơn các biển báo có chữ viết nhỏ.

---

### SLIDE 10: PHIÊN HỎI ĐÁP (Q&A) & CÂU HỎI PHẢN BIỆN THƯỜNG GẶP
1. **Câu hỏi:** *Tại sao không huấn luyện mạng nơ-ron từ đầu (From Scratch) mà dùng Transfer Learning?*
   - **Trả lời:** 250 lớp là bài toán rất lớn. Nhiều lớp biển báo hiếm gặp không có đủ hàng ngàn ảnh mẫu thực tế. Transfer Learning từ ImageNet giúp mạng thừa hưởng sẵn các bộ lọc phát hiện hình học cơ bản, giúp mô hình hội tụ nhanh chỉ sau 5 epochs mà không bị quá khớp.
2. **Câu hỏi:** *Nếu ảnh đầu vào chụp cả xe cộ và người đi đường thì mô hình xử lý ra sao?*
   - **Trả lời:** Đề tài tập trung vào bài toán Classification (phân loại vùng ảnh biển báo). Để giải quyết ảnh toàn cảnh, ứng dụng cung cấp công cụ ROI Cropper; trong tương lai nhóm sẽ tích hợp thêm YOLO làm bước tiền xử lý tự động phát hiện bounding box.
