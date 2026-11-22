# Đồ Án Môn Học Machine Learning: Phân Loại Biển Báo Giao Thông Việt Nam

Hệ thống ứng dụng mô hình học sâu **Convolutional Neural Network (Custom Deep CNN)** và các mô hình **Transfer Learning (MobileNetV3 / ResNet18)** để nhận diện và phân loại toàn diện **250 loại biển báo giao thông quy chuẩn đường bộ Việt Nam** (theo QCVN 41:2019/BGTVT).

---

## 1. Tổng Quan Bộ Dữ Liệu (Dataset Overview)
- **Tập dữ liệu gốc (`vietnam_traffic_sign`):** 250 biển báo quy chuẩn giao thông Việt Nam bao gồm đầy đủ 5 nhóm:
  - **Biển cấm (P):** Cấm đi ngược chiều, cấm dừng đỗ, cấm rẽ, cấm các loại phương tiện...
  - **Biển chỉ dẫn (I / IE):** Chỉ dẫn hướng đi, vị trí quay xe, đường ưu tiên...
  - **Biển hiệu lệnh (R):** Hướng đi phải theo, tuyến đường dành riêng...
  - **Biển cảnh báo nguy hiểm (W):** Cảnh báo khúc cua nguy hiểm, giao nhau, người đi bộ...
  - **Biển phụ (S):** Biển chỉ hướng, thời gian áp dụng, loại phương tiện...
- **Tập dữ liệu huấn luyện phân loại (`data/processed`):** Tổng cộng **6,500 ảnh biển báo** được tiền xử lý và tăng cường (Data Augmentation) kích thước $64 \times 64$:
  - **Tập Huấn luyện (Train):** 5,000 ảnh (20 ảnh/lớp trên toàn bộ 250 lớp)
  - **Tập Xác thực (Validation):** 750 ảnh (3 ảnh/lớp trên toàn bộ 250 lớp)
  - **Tập Kiểm thử (Test):** 750 ảnh (3 ảnh/lớp trên toàn bộ 250 lớp)
- **Danh mục phân loại (`data/class_labels.csv`):** 250 lớp với đầy đủ thông tin: ID, Mã hiệu quy chuẩn, Tên Tiếng Việt, Nhóm biển báo, Mô tả ý nghĩa chi tiết & đường dẫn ảnh mẫu quy chuẩn.

---

## 2. Cấu Trúc Dự Án Hoàn Chỉnh

```text
traffic_sign_classification/
│
├── configs/
│   └── config.yaml                     # File cấu hình tham số (epochs, lr, batch_size, model...)
│
├── data/
│   ├── raw/                            # Thư mục chứa dữ liệu thô (nếu có)
│   ├── templates/                      # 250 ảnh mẫu biển báo quy chuẩn sắc nét (256x256)
│   ├── processed/                      # 6,500 ảnh biển báo đã tiền xử lý
│   │   ├── train/                      # Dữ liệu huấn luyện theo từng class (0 đến 249)
│   │   ├── val/                        # Dữ liệu validation
│   │   ├── test/                       # Dữ liệu test
│   │   └── dataset_summary.csv         # Bảng tổng hợp toàn bộ 6,500 mẫu
│   └── class_labels.csv                # Ánh xạ 250 lớp, mã biển, tên tiếng Việt, nhóm & quy định
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb # Notebook EDA khám phá & trực quan hóa dữ liệu
│   └── README.md                       # Hướng dẫn sử dụng các notebook
│
├── src/                                # Toàn bộ mã nguồn đã module hóa
│   ├── config.py                       # Quản lý cấu hình & đường dẫn động
│   ├── data/
│   │   ├── dataset.py                  # PyTorch Dataset tùy biến
│   │   └── transforms.py               # Tiền xử lý & Data Augmentation (xoay, đổi màu, làm mờ, chuẩn hóa)
│   ├── models/
│   │   ├── custom_cnn.py               # Kiến trúc Custom Deep CNN (3 blocks + BatchNorm + Dropout)
│   │   └── transfer_models.py          # MobileNetV3 / ResNet18 Transfer Learning
│   ├── training/
│   │   └── train.py                    # Vòng lặp huấn luyện (train loop) & đánh giá (val loop)
│   └── utils/
│       ├── metrics.py                  # Tính toán Accuracy, Precision, Recall, Macro F1-score
│       └── visualizer.py               # Vẽ biểu đồ Loss/Acc và ma trận nhầm lẫn Confusion Matrix
│
├── scripts/
│   └── prepare_dataset.py              # Script tự động trích xuất, tăng cường và phân chia 250 lớp
│
├── checkpoints/
│   └── best_model.pth                  # Trọng số mô hình tối ưu nhất (250 lớp)
│
├── outputs/
│   ├── figures/                        # Lưu biểu đồ Loss/Accuracy và Confusion Matrix
│   ├── classification_report.txt       # Báo cáo chi tiết F1-score, Precision, Recall từng lớp
│   └── logs/                           # Log file quá trình huấn luyện
│
├── app/
│   └── app.py                          # Giao diện Web tương tác Dark Theme bằng Streamlit (Font Times New Roman)
│
├── docs/                               # Chứa tài liệu thuyết trình và báo cáo đồ án
├── train.py                            # Entrypoint huấn luyện mô hình
├── predict.py                          # Dự đoán nhanh trên ảnh bất kỳ từ dòng lệnh
├── requirements.txt                    # Danh sách các thư viện cần thiết
└── README.md
```

---

## 3. Hướng Dẫn Sử Dụng & Chạy Dự Án

### Bước 1: Cài đặt thư viện
```powershell
cd d:\Code\traffic_sign_classification
pip install -r requirements.txt
```

### Bước 2: Huấn luyện mô hình (Training)
```powershell
# Chạy huấn luyện mô hình Custom Deep CNN trên 250 lớp biển báo:
python train.py --epochs 8

# Hoặc thử nghiệm với mô hình Transfer Learning MobileNetV3:
python train.py --model mobilenet_v3 --epochs 8
```
Sau khi huấn luyện:
- Checkpoint mô hình tốt nhất được lưu tại: `checkpoints/best_model.pth`.
- Đồ thị quá trình học lưu tại: `outputs/figures/loss_acc_curve.png`.
- Báo cáo phân loại và Confusion Matrix lưu tại: `outputs/figures/confusion_matrix.png`.

### Bước 3: Chạy ứng dụng Web Demo (Streamlit)
```powershell
streamlit run app/app.py
```
- Giao diện Dark Theme hiện đại, đồng bộ toàn bộ font **Times New Roman**.
- Hỗ trợ công cụ **Cắt vùng biển báo (ROI Cropper)** từ ảnh toàn cảnh đường phố.
- Tự động đối chiếu với **Ảnh mẫu biển báo quy chuẩn** và hiển thị giải thích quy chuẩn đường bộ.

### Bước 4: Dự đoán ảnh đơn lẻ bằng dòng lệnh (CLI)
```powershell
python predict.py --image "data/processed/test/4/4_test_0.jpg" --top_k 5
```
