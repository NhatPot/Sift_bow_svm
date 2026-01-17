# Traffic Classification - SIFT + BoW + SVM

Dự án phân loại phương tiện giao thông sử dụng SIFT feature extraction, Bag of Words (BoW), và SVM classifier.

## 📋 Mô tả

Mô hình này phân loại 5 loại đối tượng giao thông:
- **Bus** (Xe buýt)
- **Car** (Xe ô tô)
- **Moto** (Xe máy)
- **Pedestrian** (Người đi bộ)
- **Truck** (Xe tải)

### Quy trình hoạt động:
1. Trích xuất đặc trưng SIFT từ ảnh đầu vào
2. Xây dựng Bag of Words (BoW) dictionary sử dụng KMeans clustering (100 clusters)
3. Vector hóa mỗi ảnh thành vector 100 chiều dựa trên BoW dictionary
4. Huấn luyện mô hình SVM để phân loại

## 🛠️ Yêu cầu hệ thống

- Python 3.7 hoặc 3.8
- Anaconda hoặc Miniconda
- Windows/Linux/MacOS

## 📦 Cài đặt với Anaconda

### Bước 1: Tạo môi trường Anaconda mới

Mở **Anaconda Prompt** và chạy các lệnh sau:

```bash
# Tạo môi trường mới với Python 3.8
conda create -n traffic_classify python=3.8 -y

# Kích hoạt môi trường
conda activate traffic_classify
```

### Bước 2: Cài đặt OpenCV với SIFT support

**Lưu ý quan trọng:** Code sử dụng `cv2.xfeatures2d.SIFT_create()` nên cần cài đặt `opencv-contrib-python` phiên bản 3.4.x.

```bash
# Cài đặt opencv-contrib-python (bao gồm SIFT)
pip install opencv-contrib-python==3.4.18.65
```

**Tại sao dùng phiên bản 3.4.18.65?**
- Code hiện tại sử dụng `cv2.xfeatures2d.SIFT_create()` - cú pháp của OpenCV 3.4.x
- Phiên bản 3.4.18.65 là phiên bản ổn định cuối cùng của dòng 3.4.x
- SIFT từng là thuật toán có bản quyền, từ OpenCV 4.4.0 trở đi đã được đưa trở lại với cú pháp khác: `cv2.SIFT_create()` thay vì `cv2.xfeatures2d.SIFT_create()`
- Nếu muốn dùng OpenCV 4.x, bạn cần sửa code thay `cv2.xfeatures2d.SIFT_create()` thành `cv2.SIFT_create()`

### Bước 3: Cài đặt các thư viện khác

```bash
# Cài đặt NumPy
conda install numpy -y

# Cài đặt Matplotlib
conda install matplotlib -y

# Cài đặt scikit-learn
conda install scikit-learn -y

# Cài đặt SciPy
conda install scipy -y
```

### Bước 4: Kiểm tra cài đặt

```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__); print('SIFT available:', hasattr(cv2.xfeatures2d, 'SIFT_create'))"
```

Kết quả mong đợi:
```
OpenCV version: 3.4.18
SIFT available: True
```

## 📂 Cấu trúc thư mục

```
Sift_bow_svm/
│
├── Traffic_Classify.py          # File code chính
├── README.md                     # File hướng dẫn này
│
└── Traffic-Data/
    ├── trainingset/              # Thư mục dữ liệu huấn luyện
    │   ├── bus/                  # Ảnh xe buýt
    │   ├── car/                  # Ảnh xe ô tô
    │   ├── moto/                 # Ảnh xe máy
    │   ├── pedestrian/           # Ảnh người đi bộ
    │   └── truck/                # Ảnh xe tải
    │
    ├── image_test/               # Thư mục ảnh test
    │   └── car.png               # Ảnh test mẫu
    │
    └── bow_dictionary150.pkl     # BoW dictionary đã train (tạo tự động)
```

## 🚀 Chạy chương trình

### Chạy toàn bộ quy trình (training + testing)

```bash
# Đảm bảo đã kích hoạt môi trường
conda activate traffic_classify

# Di chuyển vào thư mục dự án
cd "c:\Users\MinhNhat\Desktop\Hoc Tap\Thac Si\Vision\Sift_bow_svm"

# Chạy chương trình
python Traffic_Classify.py
```

### Kết quả mong đợi

Chương trình sẽ:
1. Đọc dữ liệu từ thư mục `Traffic-Data/trainingset/`
2. Trích xuất đặc trưng SIFT từ tất cả ảnh
3. Tạo BoW dictionary (nếu chưa tồn tại file `bow_dictionary150.pkl`)
4. Tạo vector đặc trưng cho mỗi ảnh
5. Chia dữ liệu thành tập train (80%) và test (20%)
6. Huấn luyện mô hình SVM
7. Test với ảnh `Traffic-Data/image_test/car.png`
8. In ra kết quả dự đoán và độ chính xác (accuracy)
9. Hiển thị ảnh test

**Output mẫu:**
```
[3]
Your prediction:  car
0.85
```

## 🔧 Tùy chỉnh

### Thay đổi số lượng clusters trong BoW

Mở file `Traffic_Classify.py` và sửa dòng 56:

```python
num_clusters = 100  # Thay đổi giá trị này (50, 150, 200, ...)
```

**Lưu ý:** Khi thay đổi `num_clusters`, bạn cần xóa file `bow_dictionary150.pkl` để tạo lại dictionary.

### Thay đổi tham số SVM

Mở file `Traffic_Classify.py` và sửa dòng 89:

```python
svm = sklearn.svm.SVC(C=10)  # Thay đổi tham số C, kernel, gamma, ...
```

Ví dụ:
```python
svm = sklearn.svm.SVC(C=10, kernel='rbf', gamma='auto')
```

### Test với ảnh khác

Thay đổi đường dẫn ảnh test ở dòng 93:

```python
img_test = cv2.imread('Traffic-Data/image_test/car.png')  # Đổi thành đường dẫn ảnh của bạn
```

## ❗ Xử lý lỗi thường gặp

### Lỗi: `AttributeError: module 'cv2' has no attribute 'xfeatures2d'`

**Nguyên nhân:** Cài đặt `opencv-python` thay vì `opencv-contrib-python`

**Giải pháp:**
```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python==3.4.18.65
```

### Lỗi: `FileNotFoundError: Traffic-Data/trainingset`

**Nguyên nhân:** Chưa có dữ liệu training hoặc đường dẫn sai

**Giải pháp:**
- Đảm bảo thư mục `Traffic-Data/trainingset/` tồn tại
- Đảm bảo có các thư mục con: `bus/`, `car/`, `moto/`, `pedestrian/`, `truck/`
- Đảm bảo mỗi thư mục có ít nhất vài ảnh

### Lỗi: `cv2.imshow()` không hiển thị ảnh

**Nguyên nhân:** Thiếu `cv2.waitKey()`

**Giải pháp:** Đã được xử lý trong code (dòng 109)

### Lỗi: Accuracy quá thấp

**Giải pháp:**
- Tăng số lượng dữ liệu training
- Tăng số lượng clusters (ví dụ: 200, 300)
- Thử các tham số SVM khác nhau (C, kernel, gamma)
- Kiểm tra chất lượng ảnh training

## 📊 Thông tin thêm

### Các thư viện được sử dụng

| Thư viện | Phiên bản đề xuất | Mục đích |
|----------|-------------------|----------|
| opencv-contrib-python | 3.4.18.65 | Trích xuất SIFT features |
| numpy | latest | Xử lý ma trận và vector |
| scikit-learn | latest | KMeans clustering và SVM |
| scipy | latest | Tính khoảng cách Euclidean |
| matplotlib | latest | Vẽ đồ thị (nếu cần) |

### Tham số mô hình mặc định

- **Số clusters (BoW):** 100
- **SVM C parameter:** 10
- **SVM kernel:** RBF (mặc định)
- **Train/Test split:** 80/20
- **Random state:** 42

## 📝 Ghi chú

- File `bow_dictionary150.pkl` sẽ được tự động tạo ra sau lần chạy đầu tiên
- Quá trình tạo BoW dictionary có thể mất vài phút tùy thuộc vào số lượng ảnh
- Khi thêm dữ liệu mới, nên xóa file `.pkl` để train lại dictionary

## 🔄 Cập nhật môi trường

Nếu cần cài đặt lại hoặc xuất môi trường:

### Xuất danh sách packages

```bash
conda activate traffic_classify
conda list --export > requirements.txt
```

### Tạo file environment.yml

```bash
conda env export > environment.yml
```

### Cài đặt từ environment.yml

```bash
conda env create -f environment.yml
```

## 📧 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Phiên bản Python (nên dùng 3.7 hoặc 3.8)
2. Phiên bản OpenCV (phải là opencv-contrib-python 3.4.18.65)
3. Cấu trúc thư mục dữ liệu
4. Đường dẫn file trong code

---

**Chúc bạn thành công! 🎉**
