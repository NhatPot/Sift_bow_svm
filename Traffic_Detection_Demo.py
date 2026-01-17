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

def non_max_suppression(boxes, overlapThresh=0.3):
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
    
    # Compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort the bounding boxes by the bottom-right y-coordinate
    idxs = np.argsort(y2)
    
    # Keep looping while some indexes still remain
    while len(idxs) > 0:
        # Get the last index in the indexes list and add to picked
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Compute the ratio of overlap (IoU)
        overlap = (w * h) / area[idxs[:last]]
        
        # Delete indexes with overlap greater than threshold
        idxs = np.delete(idxs, np.concatenate(([last], 
                         np.where(overlap > overlapThresh)[0])))
    
    # Return only the picked bounding boxes
    return boxes[pick].astype("int").tolist()


# ============================================================================
# OBJECT DETECTION WITH SLIDING WINDOW
# ============================================================================

def detect_objects(image_path, target_label='car', winW=100, winH=100, stepSize=20):
    """
    Detect objects in an image using sliding window and SVM
    
    Args:
        image_path: Path to test image
        target_label: Label to detect (e.g., 'car', 'bus', 'pedestrian')
        winW: Window width
        winH: Window height
        stepSize: Step size for sliding window
    
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
    
    # List to store detected bounding boxes
    detected_boxes = []
    
    # Clone image for display
    clone = image.copy()
    
    # Counter for windows processed
    window_count = 0
    detected_count = 0
    
    # Slide window across the image
    for (x, y, window) in sliding_window(image, stepSize, (winW, winH)):
        window_count += 1
        
        # Skip if window is not the right size
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        # Extract SIFT features
        try:
            sift_descriptors = extract_sift_features(window)
            
            # Skip if no SIFT features detected
            if sift_descriptors is None or len(sift_descriptors) == 0:
                continue
            
            # Create BoW feature vector
            bow_feature = create_bow_feature(sift_descriptors, BoW, num_clusters)
            
            # Reshape for SVM prediction
            bow_feature = bow_feature.reshape(1, -1)
            
            # --- PREDICTION WITH CONFIDENCE (THAY VÌ PHÁN BỪA) ---
            # Lấy xác suất của tất cả các lớp
            probs = svm.predict_proba(bow_feature)[0]
            
            # Lấy nhãn có xác suất cao nhất
            prediction = np.argmax(probs)
            confidence = probs[prediction]

            # ĐIỀU KIỆN LỌC KHẮT KHE HƠN:
            # 1. Phải đúng nhãn target (ví dụ 'car')
            # 2. Độ tự tin (confidence) phải lớn hơn 0.7 (70%)
            # 3. Không được là nhãn background (ID: 5) - nếu có trong model
            if prediction == target_id and confidence > 0.7:
                # Kiểm tra thêm: không phải background nếu label 'background' tồn tại
                if 'background' in label2id and prediction == label2id['background']:
                    continue
                    
                detected_count += 1
                box = [x, y, x + winW, y + winH]
                detected_boxes.append(box)
                
                # In ra để debug xem nó tự tin bao nhiêu
                print(f"Found {target_label} at ({x},{y}) with confidence: {confidence:.2f}")
                
                # Optional: draw all detections before NMS (for debugging)
                # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 1)
        
        except Exception as e:
            # Handle any errors during feature extraction
            continue
    
    print(f"\nProcessed {window_count} windows")
    print(f"Raw detections: {detected_count}")
    
    # Apply Non-Maximum Suppression
    if len(detected_boxes) > 0:
        print("Applying Non-Maximum Suppression...")
        final_boxes = non_max_suppression(detected_boxes, overlapThresh=0.3)
        print(f"Final detections after NMS: {len(final_boxes)}")
        
        # Draw final bounding boxes
        for (x1, y1, x2, y2) in final_boxes:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label text
            cv2.putText(clone, target_label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("No objects detected!")
    
    return clone


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    TEST_IMAGE_PATH = 'Traffic-Data/image_test/test_image.jpg'
    TARGET_LABEL = 'car'  # Change this to detect different objects
    WINDOW_WIDTH = 100    # Adjust based on your training image size
    WINDOW_HEIGHT = 100   # Adjust based on your training image size
    STEP_SIZE = 20        # Smaller = more thorough but slower
    
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
