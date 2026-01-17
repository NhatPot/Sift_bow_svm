"""
DEBUG SCRIPT - Kiểm tra vấn đề detection
Chạy file này để tìm nguyên nhân tại sao không detect được
"""

import os
import numpy as np
import cv2
import pickle
from scipy.spatial.distance import cdist

print("="*70)
print("DEBUG TRAFFIC DETECTION")
print("="*70)

# Load models
print("\n1. Loading models...")
try:
    with open('Traffic-Data/bow_dictionary150.pkl', 'rb') as f:
        BoW = pickle.load(f)
    print(f"   ✓ BoW dictionary loaded: {len(BoW)} clusters")
except Exception as e:
    print(f"   ✗ Error loading BoW: {e}")
    exit(1)

try:
    with open('Traffic-Data/svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    print(f"   ✓ SVM model loaded")
    print(f"   - Model type: {type(svm)}")
    print(f"   - Number of classes: {len(svm.classes_)}")
    print(f"   - Classes: {svm.classes_}")
except Exception as e:
    print(f"   ✗ Error loading SVM: {e}")
    exit(1)

# Check label mapping
print("\n2. Checking label mappings...")
label2id_train = {'pedestrian':0, 'moto':1, 'truck':2, 'car':3, 'bus':4, 'background':5}
label2id_demo = {'pedestrian':0, 'moto':1, 'truck':2, 'car':3, 'bus':4}

print(f"   Training labels: {label2id_train}")
print(f"   Demo labels: {label2id_demo}")
print(f"   SVM expects {len(svm.classes_)} classes")

if len(label2id_train) != len(svm.classes_):
    print("   ⚠ WARNING: Mismatch between training labels and SVM classes!")
    print("   → Model was trained BEFORE 'background' was added")
    print("   → Please retrain: python Traffic_Classify.py")

# Load and check test image
print("\n3. Checking test image...")
TEST_IMAGE = 'Traffic-Data/image_test/test_image.jpg'

if not os.path.exists(TEST_IMAGE):
    print(f"   ✗ Test image not found: {TEST_IMAGE}")
    print(f"   Available images in Traffic-Data/image_test/:")
    if os.path.exists('Traffic-Data/image_test/'):
        for f in os.listdir('Traffic-Data/image_test/'):
            print(f"     - {f}")
    exit(1)

img = cv2.imread(TEST_IMAGE)
if img is None:
    print(f"   ✗ Could not load image")
    exit(1)

print(f"   ✓ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

# Check one training image size for reference
print("\n4. Checking training image sizes (sample from 'car' folder)...")
car_folder = 'Traffic-Data/trainingset/car'
if os.path.exists(car_folder):
    sample_files = os.listdir(car_folder)[:5]
    sizes = []
    for fname in sample_files:
        try:
            train_img = cv2.imread(os.path.join(car_folder, fname))
            if train_img is not None:
                sizes.append((train_img.shape[1], train_img.shape[0]))
        except:
            pass
    
    if sizes:
        print(f"   Sample training image sizes: {sizes}")
        avg_w = int(np.mean([s[0] for s in sizes]))
        avg_h = int(np.mean([s[1] for s in sizes]))
        print(f"   Average size: {avg_w}x{avg_h}")
        print(f"   → Recommended window size: {avg_w}x{avg_h}")

# Test SIFT extraction on different window sizes
print("\n5. Testing SIFT extraction on test image...")
sift = cv2.xfeatures2d.SIFT_create()

window_sizes = [(50, 50), (100, 100), (150, 150), (200, 200)]
for (w, h) in window_sizes:
    # Try top-left corner
    if img.shape[0] >= h and img.shape[1] >= w:
        window = img[0:h, 0:w]
        kp, des = sift.detectAndCompute(window, None)
        desc_count = len(des) if des is not None else 0
        print(f"   Window {w}x{h}: {desc_count} SIFT descriptors")

# Test one window with full pipeline
print("\n6. Testing full pipeline on one window...")
WINDOW_SIZE = (100, 100)
winW, winH = WINDOW_SIZE

if img.shape[0] >= winH and img.shape[1] >= winW:
    # Get center window
    center_y = img.shape[0] // 2
    center_x = img.shape[1] // 2
    y1 = max(0, center_y - winH//2)
    x1 = max(0, center_x - winW//2)
    y2 = y1 + winH
    x2 = x1 + winW
    
    window = img[y1:y2, x1:x2]
    print(f"   Testing window at ({x1},{y1}) size {window.shape[1]}x{window.shape[0]}")
    
    # Extract SIFT
    kp, des = sift.detectAndCompute(window, None)
    if des is not None and len(des) > 0:
        print(f"   ✓ Found {len(des)} SIFT descriptors")
        
        # Create BoW feature
        features = np.array([0] * len(BoW))
        distance = cdist(des, BoW)
        argmin = np.argmin(distance, axis=1)
        for j in argmin:
            features[j] += 1
        
        print(f"   ✓ Created BoW feature vector (sum: {np.sum(features)})")
        
        # Predict with SVM
        bow_feature = features.reshape(1, -1)
        
        # Try predict
        prediction = svm.predict(bow_feature)[0]
        print(f"   - Simple predict(): {prediction}")
        
        # Try predict_proba
        try:
            probs = svm.predict_proba(bow_feature)[0]
            pred_id = np.argmax(probs)
            confidence = probs[pred_id]
            print(f"   - predict_proba(): class {pred_id} with confidence {confidence:.3f}")
            print(f"   - All probabilities: {probs}")
            
            # Map to label
            id2label_demo = {0:'pedestrian', 1:'moto', 2:'truck', 3:'car', 4:'bus'}
            if pred_id in id2label_demo:
                print(f"   → Predicted: '{id2label_demo[pred_id]}'")
            
            # Check confidence threshold
            if confidence > 0.7:
                print(f"   ✓ Confidence > 0.7 → Would be detected")
            elif confidence > 0.1:
                print(f"   ⚠ Confidence {confidence:.3f} between 0.1-0.7 → Depends on threshold")
            else:
                print(f"   ✗ Confidence {confidence:.3f} < 0.1 → Would NOT be detected")
        except Exception as e:
            print(f"   ✗ Error in predict_proba: {e}")
    else:
        print(f"   ✗ No SIFT descriptors found in this window")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
print("""
Possible issues:
1. **Label mismatch**: Training has 'background':5 but model was trained before it was added
   → Solution: Re-run 'python Traffic_Classify.py' to retrain the model

2. **Window size**: If training images are much larger/smaller than 100x100
   → Solution: Adjust WINDOW_WIDTH/HEIGHT in demo to match training size

3. **Confidence threshold too high**: Current 0.1 is very low, but model might predict with low confidence
   → Solution: Check probabilities above, adjust threshold if needed

4. **No SIFT features in windows**: Some windows might not have enough keypoints
   → Solution: Use smaller step size or different window positions

5. **Test image doesn't match training domain**: If test image is very different from training
   → Solution: Use similar images to those in training set
""")
print("="*70)
