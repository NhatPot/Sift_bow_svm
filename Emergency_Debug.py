"""
EMERGENCY DEBUG - Tại sao không detect được?
Chạy file này để xem chi tiết từng bước
"""

import os
import numpy as np
import cv2
import pickle
from scipy.spatial.distance import cdist

print("="*70)
print("EMERGENCY DEBUG - PEDESTRIAN DETECTION")
print("="*70)

# 1. Load models
print("\n1. Loading models...")
try:
    BoW = pickle.load(open('Traffic-Data/bow_dictionary150.pkl', 'rb'))
    svm = pickle.load(open('Traffic-Data/svm_model.pkl', 'rb'))
    print(f"   ✓ BoW: {len(BoW)} clusters")
    print(f"   ✓ SVM: {len(svm.classes_)} classes = {svm.classes_}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# 2. Load test image
print("\n2. Loading test image...")
TEST_IMAGE = 'Traffic-Data/image_test/test_pedestrian.png'
image = cv2.imread(TEST_IMAGE)
if image is None:
    print(f"   ✗ Could not load {TEST_IMAGE}")
    print("   Available images:")
    for f in os.listdir('Traffic-Data/image_test/'):
        print(f"     - {f}")
    exit(1)

print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]}")

# 3. Test với window ở giữa ảnh
print("\n3. Testing CENTER window (64x128)...")
h, w = image.shape[:2]
center_y = h // 2 - 64
center_x = w // 2 - 32
window = image[center_y:center_y+128, center_x:center_x+64]

print(f"   Window position: ({center_x}, {center_y})")

# Extract SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(window, None)

if des is None:
    print("   ✗ NO SIFT features found!")
    exit(1)

print(f"   ✓ SIFT descriptors: {len(des)}")

# Create BoW
features = np.array([0] * len(BoW))
distance = cdist(des, BoW)
argmin = np.argmin(distance, axis=1)
for j in argmin:
    features[j] += 1

print(f"   ✓ BoW vector (sum={np.sum(features)})")

# Predict
bow_feature = features.reshape(1, -1)
probs = svm.predict_proba(bow_feature)[0]

print("\n4. SVM Predictions:")
label2id = {'pedestrian':0, 'moto':1, 'truck':2, 'car':3, 'bus':4, 'background':5}
id2label = {v:k for k,v in label2id.items()}

for class_id in range(len(probs)):
    label_name = id2label.get(class_id, f"class_{class_id}")
    prob = probs[class_id]
    marker = " ← TARGET" if class_id == 0 else ""
    print(f"   {label_name:12s}: {prob:.4f} ({prob*100:.1f}%){marker}")

# 5. Check thresholds
print("\n5. Threshold Analysis:")
max_prob_idx = np.argmax(probs)
max_prob = probs[max_prob_idx]
max_label = id2label.get(max_prob_idx, f"class_{max_prob_idx}")

print(f"   Predicted class: {max_label} (ID: {max_prob_idx})")
print(f"   Confidence: {max_prob:.4f} ({max_prob*100:.1f}%)")

if max_prob_idx == 0:  # pedestrian
    print(f"   ✓ CORRECT prediction!")
    if max_prob > 0.6:
        print(f"   ✓ Confidence {max_prob:.2f} > 0.6 → WOULD DETECT")
    else:
        print(f"   ✗ Confidence {max_prob:.2f} < 0.6 → TOO LOW")
        print(f"   → GIẢI PHÁP: Giảm threshold xuống {max_prob-0.05:.2f}")
else:
    print(f"   ✗ WRONG prediction! Predicted {max_label} instead of pedestrian")
    print(f"   → Pedestrian confidence: {probs[0]:.4f}")
    print(f"   → GIẢI PHÁP: Cần TRAIN LẠI model với nhiều ảnh pedestrian hơn")

# 6. Test multiple positions
print("\n6. Testing MULTIPLE positions...")
positions = [
    (w//4, h//4, "Top-left quarter"),
    (w//2, h//4, "Top-center"),
    (3*w//4, h//4, "Top-right quarter"),
    (w//4, h//2, "Middle-left"),
    (w//2, h//2, "Center"),
    (3*w//4, h//2, "Middle-right"),
]

pedestrian_count = 0
for px, py, desc in positions:
    if px+64 > w or py+128 > h:
        continue
    
    win = image[py:py+128, px:px+64]
    kp, des = sift.detectAndCompute(win, None)
    
    if des is None or len(des) < 12:
        continue
    
    # BoW
    features = np.array([0] * len(BoW))
    distance = cdist(des, BoW)
    argmin = np.argmin(distance, axis=1)
    for j in argmin:
        features[j] += 1
    
    # Predict
    bow_f = features.reshape(1, -1)
    p = svm.predict_proba(bow_f)[0]
    pred = np.argmax(p)
    conf = p[pred]
    
    if pred == 0 and conf > 0.5:  # pedestrian với threshold thấp
        pedestrian_count += 1
        print(f"   {desc:20s}: pedestrian {conf:.2f}")

print(f"\n   → Found {pedestrian_count}/6 positions predict pedestrian with >50% conf")

# 7. Kết luận
print("\n" + "="*70)
print("DIAGNOSTIC CONCLUSION:")
print("="*70)

if max_prob_idx == 0 and max_prob > 0.5:
    print("✓ Model CAN detect pedestrian!")
    if max_prob < 0.6:
        print(f"⚠ BUT confidence {max_prob:.2f} < 0.6")
        print(f"\nGIẢI PHÁP:")
        print(f"1. Trong Traffic_Detection_Demo.py dòng ~410, sửa:")
        print(f"   if prediction == target_id and confidence > {max_prob-0.05:.2f}:")
    else:
        print("✓ Confidence is good!")
        print("\nVấn đề có thể là:")
        print("1. Window không trúng vị trí người")
        print("2. STEP_SIZE quá lớn, thử giảm xuống 5-8")
else:
    print("✗ Model CANNOT detect pedestrian correctly!")
    print(f"✗ It predicts '{max_label}' instead")
    print(f"\nGIẢI PHÁP:")
    print(f"1. BẮT BUỘC train lại:")
    print(f"   python Traffic_Classify.py")
    print(f"2. Đảm bảo folder Traffic-Data/trainingset/pedestrian/ có ảnh")
    print(f"3. Kiểm tra ảnh training có resize đúng 64x128 không")

print("="*70)
