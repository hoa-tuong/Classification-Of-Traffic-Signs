# Đồ Án Môn Học Machine Learning: Phân Loại Biển Báo Giao Thông Việt Nam

Hệ thống ứng dụng mô hình học sâu **Convolutional Neural Network (Custom Deep CNN)** và các kiến trúc **Transfer Learning (MobileNetV3 / ResNet18)** để nhận diện và phân loại toàn diện **250 loại biển báo giao thông quy chuẩn đường bộ Việt Nam** (theo QCVN 41:2019/BGTVT).

---

## 1. Tổng Quan Bộ Dữ Liệu (Dataset Overview)

- **Danh mục phân loại quy chuẩn (`data/class_labels.csv`):** 250 lớp biển báo đầy đủ 5 nhóm chức năng chính theo QCVN 41:2019/BGTVT:
  - **Nhóm P (Biển cấm):** Cấm đi ngược chiều, cấm dừng đỗ, cấm rẽ, giới hạn tốc độ, cấm các loại phương tiện...
  - **Nhóm I / IE (Biển chỉ dẫn):** Chỉ dẫn hướng đi, vị trí quay xe, đường ưu tiên, trạm kiểm tra tải trọng, đường cao tốc...
  - **Nhóm R (Biển hiệu lệnh):** Hướng đi phải theo, tuyến đường dành riêng cho xe buýt/người đi bộ, vòng xuyến...
  - **Nhóm W (Biển cảnh báo nguy hiểm):** Khúc cua nguy hiểm, giao nhau cùng cấp/ưu tiên, đường trơn trượt, công trường...
  - **Nhóm S (Biển phụ):** Biển chỉ hướng hiệu lực, thời gian áp dụng, loại xe áp dụng...
- **Kho ảnh mẫu quy chuẩn (`data/templates/`):** 250 ảnh mẫu vector/PNG sắc nét ($256 \times 256$) làm mốc đối sánh chuẩn.
- **Tập dữ liệu huấn luyện phân loại (`data/processed/`):** Ảnh biển báo đã tiền xử lý, chuẩn hóa và tăng cường dữ liệu (*Data Augmentation*) kích thước $64 \times 64$:
  - **Tập Huấn luyện (Train):** 3,000 - 5,000 ảnh.
  - **Tập Xác thực (Validation):** 500 - 750 ảnh.
  - **Tập Kiểm thử (Test):** 500 - 750 ảnh.
  - **Bảng tổng hợp dữ liệu:** `data/processed/dataset_summary.csv`.

---

## 2. Cấu Trúc Dự Án Hoàn Chỉnh

Dự án được tổ chức theo kiến trúc module hóa chuẩn công nghiệp (*Clean Modular Architecture*):

```text
Classification-Of-Traffic-Signs/
│
├── configs/
│   └── config.yaml                     # Quản lý tập trung siêu tham số (epochs, lr, batch_size, model, paths...)
│
├── data/
│   ├── raw/                            # Thư mục lưu dữ liệu ảnh thực tế thô
│   ├── templates/                      # 250 ảnh mẫu biển báo quy chuẩn sắc nét (PNG)
│   ├── processed/                      # Dữ liệu ảnh đã tiền xử lý và phân chia theo class
│   │   ├── train/                      # Tập huấn luyện (các folder từ 0 đến 249)
│   │   ├── val/                        # Tập xác thực
│   │   ├── test/                       # Tập kiểm thử
│   │   └── dataset_summary.csv         # Bảng tổng hợp đường dẫn, nhãn và metadata của toàn bộ tập dữ liệu
│   ├── class_labels.csv                # Từ điển 250 lớp: class_id, mã biển, tên tiếng Việt, nhóm & quy định
│   └── dataset_cache.pkl               # Cache embedding phục vụ tìm kiếm & đối chiếu mẫu siêu tốc
│
├── notebooks/                          # 5 Jupyter Notebooks phục vụ chuỗi thực nghiệm khoa học
│   ├── 01_exploratory_data_analysis.ipynb      # Khám phá dữ liệu (EDA), phân bố class, kiểm tra mất cân bằng
│   ├── 02_preprocessing_augmentation.ipynb     # Thử nghiệm & trực quan hóa Data Augmentation (xoay, đổi màu, 3D)
│   ├── 03_baseline_model.ipynb                 # Thiết kế mạng Custom Deep CNN, huấn luyện baseline, vẽ Loss/Acc
│   ├── 04_transfer_learning.ipynb              # Thử nghiệm MobileNetV3 / ResNet-18, đo benchmark Latency & FPS
│   ├── 05_model_evaluation.ipynb               # Đánh giá toàn diện, Confusion Matrix, phân tích lỗi & Top-5 Confidence
│   └── README.md                               # Hướng dẫn chi tiết từng notebook
│
├── src/                                # Mã nguồn chính đã module hóa độc lập
│   ├── __init__.py
│   ├── config.py                       # Hàm đọc và xác thực cấu hình config.yaml
│   │
│   ├── data/                           # Module nạp & tiền xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── dataset.py                  # PyTorch TrafficSignDataset tùy biến
│   │   ├── transforms.py               # Pipeline Data Augmentation & Normalization
│   │   └── dataloader.py               # Hàm khởi tạo DataLoader cho Train / Val / Test
│   │
│   ├── models/                         # Module kiến trúc mạng nơ-ron
│   │   ├── __init__.py
│   │   ├── custom_cnn.py               # Mạng Custom Deep CNN (3 blocks + BatchNorm + Dropout)
│   │   ├── transfer_models.py          # MobileNetV3-Small / ResNet-18 Transfer Learning
│   │   └── loss.py                     # Hàm mất mát nâng cao: Focal Loss, Label Smoothing Loss
│   │
│   ├── training/                       # Module huấn luyện & đánh giá
│   │   ├── __init__.py
│   │   ├── train.py                    # Vòng lặp huấn luyện (train loop) & xác thực (val loop)
│   │   └── evaluate.py                 # Đánh giá độc lập trên Test Set & xuất báo cáo chi tiết
│   │
│   └── utils/                          # Module tiện ích bổ trợ
│       ├── __init__.py
│       ├── metrics.py                  # Tính Accuracy, Macro Precision, Recall, F1-score
│       ├── visualizer.py               # Vẽ Loss/Acc curves và Ma trận nhầm lẫn Confusion Matrix
│       ├── logger.py                   # Cấu hình logging chuẩn ghi ra console và file log
│       └── dataset_matcher.py          # Bộ đối sánh ảnh mẫu quy chuẩn cho ứng dụng Web
│
├── scripts/                            # Các script tiện ích hỗ trợ sinh dữ liệu & báo cáo
│   ├── prepare_dataset.py              # Script trích xuất và tăng cường dữ liệu
│   ├── generate_synthetic_dataset.py   # Script sinh nhanh tập ảnh huấn luyện từ 250 templates
│   ├── generate_realistic_dataset.py   # Script sinh dữ liệu mô phỏng nền thực tế phức tạp
│   └── download_real_dataset.py        # Tự động tải tập dữ liệu đường phố thực tế từ Hugging Face
│
├── checkpoints/                        # Lưu trữ trọng số mô hình tối ưu
│   └── best_model.pth                  # Trọng số tốt nhất đã huấn luyện (MobileNetV3 / Custom CNN)
│
├── outputs/                            # Kết quả xuất bản từ quá trình chạy
│   ├── figures/                        # Lưu biểu đồ Loss/Accuracy, Confusion Matrix, Grad-CAM demo
│   ├── classification_report.txt       # Báo cáo chi tiết F1-score, Precision, Recall từng lớp
│   └── logs/                           # File ghi nhật ký huấn luyện
│
├── app/                                # Ứng dụng Web Demo tương tác (Streamlit)
│   ├── app.py                          # Giao diện Web Dark Theme cao cấp
│   └── static/                         # Thư mục chứa tài nguyên tĩnh
│
├── docs/                               # Tài liệu học thuật đồ án
│   ├── BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO.docx # Báo cáo đồ án đầy đủ định dạng Word
│   ├── BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO.md   # Bản thảo báo cáo định dạng Markdown
│   └── SLIDES_THUYET_TRINH_DO_AN.md          # Dàn ý slide thuyết trình bảo vệ đồ án
│
├── .gitignore                          # Cấu hình bỏ qua cache, file checkpoint nặng và venv
├── requirements.txt                    # Danh sách các thư viện Python cần thiết
├── train.py                            # Entrypoint chạy huấn luyện mô hình từ dòng lệnh
├── predict.py                          # Entrypoint dự đoán ảnh bất kỳ từ dòng lệnh
└── README.md
```

---

## 3. Hướng Dẫn Sử Dụng & Thứ Tự Chạy Dự Án

### Bước 1: Cài đặt môi trường & thư viện
Mở Terminal / PowerShell tại thư mục dự án:
```powershell
cd Classification-Of-Traffic-Signs
python -m pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu huấn luyện
Nếu trong thư mục `data/processed/` chưa có dữ liệu ảnh và file `dataset_summary.csv`, hãy chạy script sinh dữ liệu:
```powershell
python scripts/generate_synthetic_dataset.py
```
*Script sẽ tự động tạo các tập `train`, `val`, `test` với đầy đủ 250 lớp biển báo và xuất file `data/processed/dataset_summary.csv`.*

### Bước 3: Thực nghiệm nghiên cứu qua chuỗi 5 Notebooks
Mở VS Code hoặc Jupyter Lab và chạy lần lượt 5 notebook trong thư mục `notebooks/`:
1. **`01_exploratory_data_analysis.ipynb`**: Khám phá phân bố dữ liệu 250 lớp, độ sáng, tỷ lệ tập train/val/test.
2. **`02_preprocessing_augmentation.ipynb`**: Trực quan hóa các phép xoay, đổi màu, phối cảnh 3D; giải thích quy tắc không dùng lật ngang để bảo toàn nhãn.
3. **`03_baseline_model.ipynb`**: Xây dựng mạng Custom CNN, chạy huấn luyện thử nghiệm, vẽ biểu đồ Loss và Accuracy.
4. **`04_transfer_learning.ipynb`**: Thử nghiệm MobileNetV3 và ResNet-18, đo benchmark tốc độ suy luận (FPS & Latency) và so sánh hiệu năng.
5. **`05_model_evaluation.ipynb`**: Đánh giá toàn diện trên Test Set, phân tích Ma trận nhầm lẫn (Confusion Matrix), mổ xẻ các trường hợp đoán sai và test hàm dự đoán Top-5 xác suất.

### Bước 4: Huấn luyện mô hình từ dòng lệnh (CLI Training)
Bạn có thể cấu hình tham số trong file `configs/config.yaml` hoặc truyền trực tiếp qua tham số dòng lệnh:

* **Huấn luyện mô hình Transfer Learning MobileNetV3 (Khuyên dùng - tối ưu nhất):**
  ```powershell
  python train.py --model mobilenet_v3 --epochs 8
  ```
* **Hoặc huấn luyện mô hình Custom CNN tự thiết kế:**
  ```powershell
  python train.py --model custom_cnn --epochs 8
  ```
*Sau khi huấn luyện hoàn tất, trọng số mô hình tốt nhất được lưu tại `checkpoints/best_model.pth`, các biểu đồ học tập tự động xuất tại `outputs/figures/`.*

### Bước 5: Dự đoán ảnh đơn lẻ bằng dòng lệnh (CLI Inference)
Dự đoán nhanh kết quả trên một bức ảnh biển báo bất kỳ:
```powershell
python predict.py --image "data/templates/0_P.101.png" --top_k 5
```
*Kết quả in ra gồm: Tên biển báo tiếng Việt, Mã hiệu quy chuẩn QCVN, Độ tự tin (%), Giải thích ý nghĩa luật giao thông chi tiết và Bảng Top-5 dự đoán.*

### Bước 6: Khởi chạy ứng dụng Web Demo (Streamlit Interactive App)
Khởi động giao diện web tương tác thời gian thực:
```powershell
python -m streamlit run app/app.py
```
* Trình duyệt sẽ tự động mở giao diện tại: **`http://localhost:8501`**.
* **Tính năng nổi bật:**
  - Giao diện Dark Theme hiện đại, font chữ chuẩn học thuật.
  - Cho phép tải ảnh cá nhân từ máy tính hoặc chọn nhanh từ kho ảnh mẫu quy chuẩn.
  - Dự đoán thời gian thực với thanh đo độ tự tin Top-5.
  - Công cụ cắt vùng biển báo (ROI Cropper) từ ảnh đường phố góc rộng.
  - Tự động tra cứu quy chuẩn, ý nghĩa và mức phạt giao thông liên quan.

---

## 4. Bảng So Sánh Hiệu Năng Các Kiến Trúc Mô Hình

Dưới đây là kết quả thực nghiệm tổng hợp trên tập kiểm thử độc lập (Test Set) gồm 250 lớp biển báo Việt Nam:

| Tiêu chí đánh giá | Custom Deep CNN | MobileNetV3-Small (Lựa chọn) | ResNet-18 |
| :--- | :---: | :---: | :---: |
| **Độ chính xác (Accuracy)** | 94.67% | **98.27%** | **98.93%** |
| **Macro F1-Score** | 93.80% | **98.15%** | **98.85%** |
| **Tổng số tham số (Params)** | ~1.30M | **~1.78M** | ~11.20M |
| **Dung lượng file weights** | **~5.12 MB** | **~6.24 MB** | ~43.68 MB |
| **Thời gian trễ suy luận (CPU)** | ~12.4 ms | **~9.8 ms** | ~24.6 ms |
| **Tốc độ khung hình (FPS)** | ~80.6 FPS | **~102.0 FPS** | ~40.6 FPS |
| **Mục đích sử dụng tối ưu** | Mô hình cơ sở (Baseline) | **Web Demo & Camera hành trình nhúng** | Đòi hỏi độ chính xác tuyệt đối |

---

## 5. Tài Liệu Báo Cáo & Thuyết Trình

* **Báo cáo đồ án chi tiết (Word):** [`docs/BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO.docx`](docs/BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO.docx).
* **Bản thảo báo cáo (Markdown):** [`docs/BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO.md`](docs/BAO_CAO_DO_AN_PHAN_LOAI_BIEN_BAO.md).
* **Dàn ý Slide thuyết trình bảo vệ đề tài:** [`docs/SLIDES_THUYET_TRINH_DO_AN.md`](docs/SLIDES_THUYET_TRINH_DO_AN.md).
