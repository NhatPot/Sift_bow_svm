# 🚶 Pedestrian Detection System
## SIFT + Bag of Words + SVM + Sliding Window + Image Pyramid

Hệ thống nhận diện người đi bộ sử dụng Computer Vision cổ điển (SIFT features, BoW, SVM) với sliding window và multi-scale detection.

---

## 📋 Tổng quan

### Đối tượng được nhận diện:
-  **🚶Pedestrian** (Người đi bộ)

### Kiến trúc hệ thống:
```
Input Image → Image Pyramid (Multi-scale)
                ↓
         Sliding Window
                ↓
         SIFT Extraction
                ↓
         BoW Feature Vector
                ↓
         SVM Prediction (with confidence)
                ↓
         Confidence Filtering
                ↓
         Non-Maximum Suppression (NMS)
                ↓
         Bounding Boxes
```

### Quy trình hoạt động:
1. **Training Phase**:
   - Trích xuất SIFT descriptors từ training images
   - Clustering với KMeansạo BoW dictionary (100 clusters)
   - Tạo feature vectors cho mỗi ảnh
   - Train SVM classifier với probability enabled

2. **Detection Phase**:
   - Resize ảnh ở nhiều scales (Image Pyramid)
   - Sliding window trên mỗi scale
   - Extract SIFT + BoW cho mỗi window
   - SVM prediction với confidence threshold
   - Non-Maximum Suppression để loại bỏ duplicate boxes

---

## 🛠️ Yêu cầu hệ thống

- **Python**: 3.8
- **OS**: Windows/Linux/MacOS
- **RAM**: Tối thiểu 4GB (8GB đề xuất)
- **Anaconda/Miniconda** (khuyến nghị)

---

## 📦 Cài đặt

### Bước 1: Tạo môi trường Anaconda

```bash
# Tạo môi trường mới
conda create -n traffic_classify python=3.8 -y

# Kích hoạt
conda activate traffic_classify
```

### Bước 2: Cài đặt OpenCV với SIFT

⚠️ **QUAN TRỌNG**: Phải dùng `opencv-contrib-python==3.4.18.65` để hỗ trợ `cv2.xfeatures2d.SIFT_create()`

```bash
pip install opencv-contrib-python==3.4.18.65
```

**Tại sao phiên bản 3.4.18.65?**
- Code sử dụng `cv2.xfeatures2d.SIFT_create()` (cú pháp OpenCV 3.4.x)
- SIFT đã được chuyển về main repo từ OpenCV 4.4.0 với cú pháp mới: `cv2.SIFT_create()`
- Để dùng OpenCV 4.x, cần sửa tất cả `cv2.xfeatures2d.SIFT_create()` → `cv2.SIFT_create()`

### Bước 3: Cài đặt các thư viện khác

```bash
# Core libraries
conda install numpy scikit-learn scipy matplotlib -y

# For web app (optional)
pip install streamlit
```

### Bước 4: Kiểm tra cài đặt

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__); print('SIFT:', hasattr(cv2.xfeatures2d, 'SIFT_create'))"
```

Kết quả mong đợi:
```
OpenCV: 3.4.18
SIFT: True
```

---

## 📂 Cấu trúc thư mục

```
Sift_bow_svm/
│
├── Traffic_Classify.py           # Training script
├── Traffic_Detection_Demo.py     # CLI detection demo
├── app.py                         # Streamlit web app
├── README.md                      # Documentation
│
└── Traffic-Data/
    ├── trainingset/               # Training images
    │   ├── bus/                   # Bus images
    │   ├── car/                   # Car images
    │   ├── moto/                  # Motorcycle images
    │   ├── pedestrian/            # Pedestrian images
    │   └── truck/                 # Truck images
    │
    ├── image_test/                # Test images
    │   ├── test_image.jpg         # Car test image
    │   └── test_pedestrian.png    # Pedestrian test image
    │
    ├── bow_dictionary150.pkl      # BoW dictionary (auto-generated)
    └── svm_model.pkl              # Trained SVM model (auto-generated)
```

---

## 🚀 Sử dụng

### 1️⃣ Training Model

Train model với dữ liệu trong `Traffic-Data/trainingset/`:

```bash
python Traffic_Classify.py
```

**Output:**
- `Traffic-Data/bow_dictionary150.pkl` - BoW dictionary
- `Traffic-Data/svm_model.pkl` - Trained SVM model

**Kết quả mẫu:**
```
Training SVM with probability enabled...
Saved SVM model to 'Traffic-Data/svm_model.pkl'
[3]
Your prediction: car
Accuracy: 0.85
```

**Lưu ý:**
- Mỗi folder trong `trainingset/` phải có ít nhất 20-30 ảnh
- Khi thêm/sửa dữ liệu, cần train lại model
- Training images sẽ được resize về **64x128** (cho pedestrian) hoặc **64x80** (cho vehicles)

---

### 2️⃣ CLI Detection Demo

Chạy detection với sliding window trên ảnh test:

```bash
python Traffic_Detection_Demo.py
```

**Cấu hình trong code** (dòng 337-348):
```python
TEST_IMAGE_PATH = 'Traffic-Data/image_test/test_pedestrian.png'
TARGET_LABEL = 'pedestrian'  # or 'car', 'bus', 'truck', 'moto'
WINDOW_WIDTH = 64
WINDOW_HEIGHT = 128
STEP_SIZE = 5
```

**Tính năng:**
- ✅ Multi-scale detection (Image Pyramid)
- ✅ Sliding window với configurable step size
- ✅ Confidence thresholding (default: 0.4)
- ✅ Non-Maximum Suppression (NMS threshold: 0.15)
- ✅ Real-time progress indicators

**Output:**
```
Loading test image: Traffic-Data/image_test/test_pedestrian.png
Image size: 187x336
Detecting 'pedestrian' (ID: 0)
Window size: 64x128, Step size: 5
Using Image Pyramid (scale: 1.3)

  Pyramid level 1: 187x336 (ratio: 1.00)
    → Found 3 detections at this level

  Pyramid level 2: 143x258 (ratio: 1.30)
    → Found 5 detections at this level

Total pyramid levels: 5
Processed 558 windows across all scales
Raw detections: 12
Applying Non-Maximum Suppression...
Final detections after NMS: 1

Displaying result...
Result saved to: Traffic-Data/detection_result.jpg
```

---

### 3️⃣ Web App (Streamlit)

Chạy interactive web interface:

```bash
streamlit run app.py
```

Web app mở tại: **http://localhost:8501**

**Tính năng:**
- 📁 Upload ảnh trực tiếp
- 🎯 Chọn target object (pedestrian, car, bus, truck, moto)
- 🎚️ Điều chỉnh Confidence Threshold (0.3 - 0.95)
- 🔄 Điều chỉnh NMS Threshold (0.1 - 0.5)
- 🔧 Advanced settings: window size, step size
- 📊 Side-by-side comparison (original vs result)
- ⏳ Progress bar khi processing
- 💾 Model caching với `@st.cache_resource`

**Screenshot:**
```
┌─────────────────────┬─────────────────────┐
│  Original Image     │  Detected Objects   │
│  [Upload ảnh]       │  [Kết quả với box]  │
└─────────────────────┴─────────────────────┘
       Detection Count: 2
```

---

## ⚙️ Tham số quan trọng

### Training Parameters (`Traffic_Classify.py`)

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `num_clusters` | 100 | Số clusters cho BoW dictionary |
| `SVM C` | 10 | Regularization parameter |
| `probability` | True | Enable predict_proba() |
| `train_test_split` | 0.8/0.2 | Train/test ratio |
| `resize` | (64, 128) | Training image size |

### Detection Parameters (`Traffic_Detection_Demo.py`)

| Tham số | Pedestrian | Car/Vehicle | Mô tả |
|---------|------------|-------------|-------|
| `WINDOW_WIDTH` | 64 | 64-80 | Chiều rộng cửa sổ |
| `WINDOW_HEIGHT` | 128 | 80 | Chiều cao cửa sổ |
| `STEP_SIZE` | 5 | 8-15 | Bước nhảy sliding window |
| `confidence_thresh` | 0.4 | 0.6 | Ngưỡng confidence |
| `nms_thresh` | 0.15 | 0.2 | Ngưỡng NMS |
| `min_sift` | 8 | 12 | Minimum SIFT descriptors |
| `pyramid_scale` | 1.3 | 1.3 | Tỉ lệ thu nhỏ pyramid |

---

## 🎯 Điều chỉnh cho từng loại object

### Detect Pedestrians (Người đi bộ)

**Đặc điểm**: Chữ nhật đứng (1:2), ít texture (quần đen)

```python
# Traffic_Detection_Demo.py
TARGET_LABEL = 'pedestrian'
WINDOW_WIDTH = 64
WINDOW_HEIGHT = 128  # Tỉ lệ 1:2
STEP_SIZE = 5         # Quét kỹ
confidence > 0.4      # Threshold thấp
min_sift >= 8         # Cho low-texture
```

### Detect Cars/Vehicles

**Đặc điểm**: Hình chữ nhật, nhiều chi tiết

```python
TARGET_LABEL = 'car'  # or 'bus', 'truck'
WINDOW_WIDTH = 64
WINDOW_HEIGHT = 80    # Gần vuông hơn
STEP_SIZE = 10        # Nhanh hơn
confidence > 0.6      # Threshold cao hơn
min_sift >= 12        # Nhiều features
```

---

## ❗ Xử lý lỗi thường gặp

### 1. `AttributeError: module 'cv2' has no attribute 'xfeatures2d'`

**Nguyên nhân**: Cài đặt `opencv-python` thay vì `opencv-contrib-python`

**Giải pháp**:
```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python==3.4.18.65
```

### 2. `cv2.error: resize() - Assertion failed !ssize.empty()`

**Nguyên nhân**: File ảnh corrupt hoặc không load được

**Giải pháp**: 
- Script đã tự động skip corrupt files
- Kiểm tra log để tìm file lỗi và xóa/thay thế

### 3. `predict_proba is not available when probability=False`

**Nguyên nhân**: SVM được train không có `probability=True`

**Giải pháp**: Train lại model
```bash
python Traffic_Classify.py
```

### 4. Detection count = 0 (Không detect được)

**Nguyên nhân**: 
- Model chưa train lại sau khi thay đổi resize dimensions
- Confidence threshold quá cao
- Target object không có trong ảnh

**Giải pháp**:
1. Train lại model: `python Traffic_Classify.py`
2. Giảm confidence threshold xuống 0.4-0.5
3. Giảm min_sift xuống 8
4. Tăng step_size lên 8-10 (quét nhanh hơn để test)

### 5. Quá nhiều false positives

**Giải pháp**:
- Tăng confidence threshold lên 0.7-0.8
- Giảm NMS threshold xuống 0.1-0.15
- Tăng min_sift lên 15-20

---

## 📊 Performance Tips

### Tăng tốc detection:
- Tăng `STEP_SIZE` (trade-off: có thể bỏ sót)
- Tăng `pyramid_scale` từ 1.3 → 1.5 (ít levels hơn)
- Tăng `min_sift` để skip windows sớm

### Tăng accuracy:
- Giảm `STEP_SIZE` xuống 3-5
- Giảm `pyramid_scale` xuống 1.2 (nhiều levels hơn)
- Tăng số lượng training data
- Tăng `num_clusters` lên 150-200

### Giảm false positives:
- Tăng `confidence_thresh`
- Tăng `min_sift`
- Giảm `nms_thresh`
- Train với dữ liệu background

---

## 📝 Workflow Development

### 1. Chuẩn bị dữ liệu
```bash
Traffic-Data/trainingset/
├── pedestrian/  # 50+ images
├── car/         # 50+ images
└── ...
```

### 2. Train model
```bash
python Traffic_Classify.py
```

### 3. Test với CLI
```bash
python Traffic_Detection_Demo.py
```

### 4. Fine-tune parameters
- Điều chỉnh confidence, NMS, step_size
- Test lại

### 5. Deploy web app
```bash
streamlit run app.py
```

---

## 🔧 Advanced Customization

### Thay đổi BoW clusters

File: `Traffic_Classify.py`, line ~56
```python
num_clusters = 150  # Từ 100 → 150
```
→ Xóa `bow_dictionary150.pkl` và train lại

### Thay đổi SVM kernel

File: `Traffic_Classify.py`, line ~95
```python
svm = sklearn.svm.SVC(C=10, kernel='linear', probability=True)
```

### Thêm custom label

1. Tạo folder trong `trainingset/` (ví dụ: `bicycle/`)
2. Thêm vào `label2id`:
```python
label2id = {'pedestrian':0, 'moto':1, 'truck':2, 
            'car':3, 'bus':4, 'bicycle':6, 'background':5}
```
3. Train lại model

---

## 📚 Technical Details

### Algorithms Used:
- **SIFT**: Scale-Invariant Feature Transform
- **BoW**: Bag of Words với KMeans clustering
- **SVM**: Support Vector Machine với RBF kernel
- **NMS**: Non-Maximum Suppression (IoU-based)
- **Image Pyramid**: Multi-scale representation

### Libraries:
- `opencv-contrib-python==3.4.18.65` - SIFT features
- `numpy` - Array operations
- `scikit-learn` - KMeans, SVM
- `scipy` - Distance calculations
- `streamlit` - Web interface

---

## 📖 References

- [SIFT Paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) - Lowe, 2004
- [Bag of Words in Computer Vision](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)
- [OpenCV SIFT Tutorial](https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html)
- [Sliding Window Detection](https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more training data
- [ ] Implement HOG features
- [ ] Add deep learning comparison
- [ ] Mobile deployment
- [ ] Real-time video detection

---

## 📧 Support

Nếu gặp vấn đề:
1. Kiểm tra phiên bản Python (3.8)
2. Kiểm tra OpenCV version (3.4.18.65)
3. Đảm bảo có đủ training data
4. Chạy lại training sau khi thay đổi code

---

