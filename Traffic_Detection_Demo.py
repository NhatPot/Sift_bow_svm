import os
import numpy as np
import cv2
import pickle
from scipy.spatial.distance import cdist


# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================

# Load BoW dictionary
print("Loading BoW dictionary...")
with open('Traffic-Data/bow_dictionary150.pkl', 'rb') as f:
    BoW = pickle.load(f)
num_clusters = len(BoW)
print(f"BoW dictionary loaded with {num_clusters} clusters")

# Load trained SVM model
print("Loading SVM model...")
with open('Traffic-Data/svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)
print("SVM model loaded successfully")

# Label mapping
label2id = {'pedestrian':0, 'moto':1, 'truck':2, 'car':3, 'bus':4}
id2label = {v: k for k, v in label2id.items()}


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_sift_features(img):
    """Extract SIFT features from a single image window"""
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return des


def create_bow_feature(descriptors, bow_dict, num_clusters):
    """Create BoW feature vector from SIFT descriptors"""
    features = np.array([0] * num_clusters)
    
    if descriptors is not None and len(descriptors) > 0:
        distance = cdist(descriptors, bow_dict)
        argmin = np.argmin(distance, axis=1)
        for j in argmin:
            features[j] += 1
    
    return features


# ============================================================================
# IMAGE PYRAMID FUNCTION (MULTI-SCALE DETECTION)
# ============================================================================

def pyramid(image, scale=1.3, minSize=(64, 128)):  # Scale 1.3 cho nhiều levels hơn
    """
    Tạo ra các phiên bản ảnh nhỏ dần (Image Pyramid) để detect vật thể ở nhiều scale
    
    Args:
        image: Input image
        scale: Tỉ lệ thu nhỏ mỗi lần (1.3 = mỗi lần nhỏ đi 1.3 lần, nhiều levels hơn)
        minSize: (width, height) Kích thước tối thiểu, nhỏ hơn thì dừng
                 (64, 128) for pedestrian | (64, 80) for cars
    
    Yields:
        Ảnh đã resize
    """
    # Yield ảnh gốc đầu tiên
    yield image
    
    # Vòng lặp thu nhỏ dần
    while True:
        # Tính kích thước mới
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        
        # Resize ảnh
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        
        # Nếu ảnh nhỏ hơn kích thước window thì dừng
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image


# ============================================================================
# SLIDING WINDOW FUNCTION
# ============================================================================

def sliding_window(image, stepSize, windowSize):
    """
    Slide a window across the image
    
    Args:
        image: Input image
        stepSize: Step size for sliding
        windowSize: (width, height) of the window
    
    Yields:
        (x, y, window): coordinates and window crop
    """
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# ============================================================================
# NON-MAXIMUM SUPPRESSION (NMS)
# ============================================================================

def non_max_suppression(boxes, overlapThresh=0.5):
    """
    IMPROVED NMS: Sort by area, use Intersection over Min Area
    Aggressively eliminates nested boxes (legs/torso inside full body)
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by AREA (largest first)
    idxs = np.argsort(area)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersection = w * h
        other_areas = area[idxs[:last]]
        min_area = other_areas
        overlap = intersection / min_area
        
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int").tolist()
    """
    Apply non-maximum suppression to eliminate redundant overlapping boxes
    
    Args:
        boxes: List of bounding boxes as (x1, y1, x2, y2)
        overlapThresh: IoU threshold for suppression
    
    Returns:
        List of boxes after NMS
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy array
    boxes = np.array(boxes)
    
    # Initialize the list of picked indexes
    pick = []
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # IMPROVED: Use min area to detect nested boxes
        intersection = w * h
        min_area = np.minimum(area[i], area[idxs[:last]])
        overlap = intersection / min_area
        
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int").tolist()


# ============================================================================
# OBJECT DETECTION WITH SLIDING WINDOW
# ============================================================================

def detect_objects(image_path, target_label='car', winW=64, winH=80, stepSize=15, pyramid_scale=1.2):
    """
    Detect objects in an image using MULTI-SCALE sliding window with Image Pyramid
    
    Args:
        image_path: Path to test image
        target_label: Label to detect (e.g., 'car', 'bus', 'pedestrian')
        winW: Window width
        winH: Window height
        stepSize: Step size for sliding window
        pyramid_scale: Scale factor for image pyramid (1.2 = each layer is 1.2x smaller)
    
    Returns:
        Image with detected bounding boxes
    """
    # Load test image
    print(f"\nLoading test image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Get target label ID
    if target_label not in label2id:
        print(f"Error: Unknown label '{target_label}'")
        print(f"Available labels: {list(label2id.keys())}")
        return None
    
    target_id = label2id[target_label]
    print(f"Detecting '{target_label}' (ID: {target_id})")
    print(f"Window size: {winW}x{winH}, Step size: {stepSize}")
    print(f"Using Image Pyramid (scale: {pyramid_scale})")
    
    # List to store detected bounding boxes
    detected_boxes = []
    
    # Counters
    total_windows = 0
    pyramid_level = 0
    
    # ========================================================================
    # IMAGE PYRAMID LOOP - QUAN TRỌNG!
    # ========================================================================
    for resized in pyramid(image, scale=pyramid_scale, minSize=(winW, winH)):
        pyramid_level += 1
        
        # Tính tỉ lệ resize so với ảnh gốc (để quy đổi tọa độ sau)
        resize_ratio = image.shape[1] / float(resized.shape[1])
        
        print(f"\n  Pyramid level {pyramid_level}: {resized.shape[1]}x{resized.shape[0]} (ratio: {resize_ratio:.2f})")
        
        # Counter for this pyramid level
        level_detections = 0
        
        # Slide window across the RESIZED image
        for (x, y, window) in sliding_window(resized, stepSize, (winW, winH)):
            total_windows += 1
            
            # Skip if window is not the right size
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            
            # Extract SIFT features
            try:
                sift_descriptors = extract_sift_features(window)
                
                # Skip if no SIFT features detected
                if sift_descriptors is None or len(sift_descriptors) == 0:
                    continue
                
                # Filter: Skip windows with too few SIFT features
                # EMERGENCY: Giảm xuống 8 để catch low-texture (quần đen)
                if len(sift_descriptors) < 8:  # EMERGENCY: 12 → 8
                    continue
                
                # Create BoW feature vector
                bow_feature = create_bow_feature(sift_descriptors, BoW, num_clusters)
                
                # Reshape for SVM prediction
                bow_feature = bow_feature.reshape(1, -1)
                
                # --- PREDICTION WITH CONFIDENCE ---
                # Lấy xác suất của tất cả các lớp
                probs = svm.predict_proba(bow_feature)[0]
                
                # Lấy nhãn có xác suất cao nhất
                prediction = np.argmax(probs)
                confidence = probs[prediction]

                # ĐIỀU KIỆN LỌC (EMERGENCY - Rất thấp):
                # 1. Phải đúng nhãn target
                # 2. Confidence > 0.4 (40%) - EMERGENCY: Rất thấp!
                # 3. Không được là background
                if prediction == target_id and confidence > 0.4:  # EMERGENCY: 0.6 → 0.4
                    # Kiểm tra thêm: không phải background nếu label 'background' tồn tại
                    if 'background' in label2id and prediction == label2id['background']:
                        continue
                    
                    # ===== QUAN TRỌNG: QUY ĐỔI TỌA ĐỘ VỀ ẢNH GỐC =====
                    # Vì đang tìm trên ảnh resize, phải nhân tọa độ với resize_ratio
                    startX = int(x * resize_ratio)
                    startY = int(y * resize_ratio)
                    endX = int((x + winW) * resize_ratio)
                    endY = int((y + winH) * resize_ratio)
                    
                    detected_boxes.append([startX, startY, endX, endY])
                    level_detections += 1
            
            except Exception as e:
                # Handle any errors during feature extraction
                continue
        
        print(f"    → Found {level_detections} detections at this level")
    
    # ========================================================================
    # END OF PYRAMID LOOP
    # ========================================================================
    
    print(f"\nTotal pyramid levels: {pyramid_level}")
    print(f"Processed {total_windows} windows across all scales")
    print(f"Raw detections: {len(detected_boxes)}")
    
    # Clone image for display
    clone = image.copy()
    
    # Apply Non-Maximum Suppression
    if len(detected_boxes) > 0:
        print("Applying Non-Maximum Suppression...")
        final_boxes = non_max_suppression(detected_boxes, overlapThresh=0.15)  # Giảm từ 0.3 → 0.15 để loại bỏ overlap tốt hơn
        print(f"Final detections after NMS: {len(final_boxes)}")
        
        # Draw final bounding boxes
        for (x1, y1, x2, y2) in final_boxes:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Smart label positioning to avoid edge clipping
            text_y = y1 - 10 if y1 > 25 else y1 + 20
            cv2.putText(clone, target_label, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("No objects detected!")
    
    return clone








# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION FOR PEDESTRIAN DETECTION
    # ========================================================================
    
    # 1. Test image with pedestrians
    TEST_IMAGE_PATH = 'Traffic-Data/image_test/test_pedestrian.png'
    
    # 2. Target label
    TARGET_LABEL = 'pedestrian'  # Đổi từ 'car' sang 'pedestrian'
    
    # 3. Window size - PHẢI KHỚP với kích thước train (64x128)
    WINDOW_WIDTH = 64      # Chữ nhật đứng cho người
    WINDOW_HEIGHT = 128    # Tỉ lệ 1:2
    
    # 4. Step size - Nhỏ hơn vì người nhỏ hơn xe
    STEP_SIZE = 5          # EMERGENCY: Giảm từ 10 → 5 để quét kỹ hơn
    
    # NOTE: Để detect xe (car/bus/truck), đổi lại:
    # - TARGET_LABEL = 'car'
    # - WINDOW_WIDTH = 64, WINDOW_HEIGHT = 80
    # - STEP_SIZE = 15
    
    print("=" * 70)
    print("TRAFFIC OBJECT DETECTION DEMO")
    print("Using SIFT + BoW + SVM with Sliding Window and NMS")
    print("=" * 70)
    
    # Run detection
    result_image = detect_objects(
        image_path=TEST_IMAGE_PATH,
        target_label=TARGET_LABEL,
        winW=WINDOW_WIDTH,
        winH=WINDOW_HEIGHT,
        stepSize=STEP_SIZE
    )
    
    # Display result
    if result_image is not None:
        print("\nDisplaying result... Press any key to exit.")
        cv2.imshow("Object Detection Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Optional: Save result
        output_path = 'Traffic-Data/detection_result.jpg'
        cv2.imwrite(output_path, result_image)
        print(f"\nResult saved to: {output_path}")
    
    print("\nDemo completed!")
