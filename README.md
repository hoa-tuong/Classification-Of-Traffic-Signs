# Đồ Án Môn Học: Phân Loại Biển Báo Giao Thông (Traffic Sign Classification)

Dự án áp dụng các kỹ thuật **Machine Learning & Deep Learning (CNN)** để nhận diện và phân loại các loại biển báo giao thông phục vụ hỗ trợ lái xe an toàn và hệ thống ADAS / xe tự hành.

---

## 1. Cấu Trúc Dự Án

```text
traffic_sign_classification/
├── data/
│   ├── raw/                 # Dữ liệu ảnh gốc tải về
│   ├── processed/           # Dữ liệu sau khi xử lý / chia tập train-val-test
│   └── class_labels.csv     # Bảng ánh xạ ID sang tên biển báo
├── notebooks/               # Các notebook thử nghiệm EDA, Augment, Model so sánh
├── src/                     # Source code module hóa (data, models, training, utils)
├── configs/
│   └── config.yaml          # File cấu hình siêu tham số (epochs, lr, batch_size,...)
├── checkpoints/             # Nơi lưu file trọng số mô hình tốt nhất (.pth)
├── outputs/                 # Biểu đồ đánh giá (Loss/Accuracy curve, Confusion matrix)
├── app/
│   └── app.py               # Ứng dụng giao diện Web demo (Streamlit)
├── train.py                 # File thực thi huấn luyện mô hình
├── predict.py               # File test dự đoán trên ảnh đơn lẻ
├── requirements.txt         # Danh sách thư viện Python
└── README.md
```

---

## 2. Hướng Dẫn Cài Đặt Môi Trường

### Bước 1: Tạo và kích hoạt môi trường ảo (khuyến nghị)
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt trên Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 3. Chuẩn Bị Dữ Liệu
Bạn có thể sử dụng bộ dữ liệu chuẩn như **GTSRB (German Traffic Sign Recognition Benchmark)** hoặc tập dữ liệu biển báo Việt Nam.
Cấu trúc thư mục dữ liệu trong `data/raw/` hoặc `data/processed/`:
```text
data/raw/
├── 0/
│   ├── 00001.png
│   └── ...
├── 1/
│   ├── 00002.png
│   └── ...
└── ...
```

---

## 4. Huấn Luyện Mô Hình (Training)
Chỉnh sửa tham số huấn luyện nếu cần trong file `configs/config.yaml`, sau đó chạy:
```bash
python train.py
```
Quá trình huấn luyện sẽ tự động:
- Huấn luyện và kiểm thử trên tập Validation.
- Lưu lại checkpoint tốt nhất vào `checkpoints/best_model.pth`.
- Xuất biểu đồ Loss & Accuracy vào `outputs/figures/loss_acc_curve.png`.

---

## 5. Dự Đoán Ảnh Đơn Lẻ (CLI Inference)
```bash
python predict.py --image path/to/your/image.jpg --top_k 3
```

---

## 6. Chạy Ứng Dụng Demo (Giao Diện Web Streamlit)
```bash
streamlit run app/app.py
```
Trình duyệt sẽ tự động mở trang web cho phép bạn tải ảnh biển báo bất kỳ lên để mô hình nhận diện và hiển thị xác suất tin cậy.

