"""
Traffic Detection Web App
Using SIFT + BoW + SVM + Sliding Window + Image Pyramid

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cdist
from PIL import Image
import io

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Traffic Object Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODELS WITH CACHING
# ============================================================================

@st.cache_resource
def load_models():
    """Load BoW dictionary and SVM model once and cache them"""
    try:
        with open('Traffic-Data/bow_dictionary150.pkl', 'rb') as f:
            bow_dict = pickle.load(f)
        
        with open('Traffic-Data/svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        
        return bow_dict, svm_model, len(bow_dict)
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please train the model first.")
        st.stop()

# Load models
BoW, svm, num_clusters = load_models()

# Label mapping
label2id = {'pedestrian': 0, 'moto': 1, 'truck': 2, 'car': 3, 'bus': 4, 'background': 5}
id2label = {v: k for k, v in label2id.items()}

# ============================================================================
# CORE ALGORITHM FUNCTIONS (Copied from Traffic_Detection_Demo.py)
# ============================================================================

def extract_sift_features(img):
    """Extract SIFT features from image window"""
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


def pyramid(image, scale=1.3, minSize=(64, 128)):
    """Generate image pyramid for multi-scale detection"""
    yield image
    
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image


def sliding_window(image, stepSize, windowSize):
    """Slide window across image"""
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def non_max_suppression(boxes, overlapThresh=0.5):
    """
    IMPROVED NMS: "Cá lớn nuốt cá bé"
    - Sort by AREA (largest first) instead of y2
    - Use Intersection over Min Area
    - Higher default threshold (0.5) to aggressively eliminate nested boxes
    
    This eliminates part-based detections (leg, torso boxes inside full body box)
    """
    if len(boxes) == 0:
        return []
    
    # Convert to float for accurate computation
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # KEY CHANGE: Sort by AREA (largest first) instead of y2
    # This prioritizes keeping the largest (full body) box
    idxs = np.argsort(area)
    
    while len(idxs) > 0:
        # Pick the largest box
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find intersection with remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Intersection over Min Area
        intersection = w * h
        other_areas = area[idxs[:last]]
        
        # Since sorting by area, current box is largest
        # Min area is the other (smaller) boxes
        min_area = other_areas
        
        overlap = intersection / min_area
        
        # Higher threshold (0.5): if small box is 50%+ covered by large box, remove it
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int").tolist()
    """
    Apply IMPROVED NMS using Intersection over Min Area
    This eliminates nested boxes effectively (e.g., small box inside large box)
    """
    if len(boxes) == 0:
        return []
    
    # Convert to float for accurate computation
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by y2 (prefer larger/lower boxes)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find intersection with remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # --- KEY IMPROVEMENT: Use min area instead of union ---
        # This detects when small box is nested inside large box
        intersection = w * h
        min_area = np.minimum(area[i], area[idxs[:last]])
        overlap = intersection / min_area
        
        # Remove boxes with high overlap relative to smaller box
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int").tolist()


def detect_objects_web(image, target_label, confidence_thresh, nms_thresh, 
                       winW=64, winH=128, stepSize=5, progress_bar=None):
    """
    Detect objects using SIFT + BoW + SVM with Sliding Window & Pyramid
    
    Args:
        image: Input image (BGR format from OpenCV)
        target_label: Target class to detect
        confidence_thresh: Minimum confidence threshold
        nms_thresh: NMS overlap threshold
        winW, winH: Window size
        stepSize: Sliding window step
        progress_bar: Streamlit progress bar (optional)
    
    Returns:
        Image with bounding boxes drawn
        Number of detections
    """
    if target_label not in label2id:
        return image, 0
    
    target_id = label2id[target_label]
    detected_boxes = []
    
    total_windows = 0
    pyramid_levels = 0
    
    # Multi-scale detection with pyramid
    for resized in pyramid(image, scale=1.3, minSize=(winW, winH)):
        pyramid_levels += 1
        resize_ratio = image.shape[1] / float(resized.shape[1])
        
        # Update progress if provided
        if progress_bar:
            progress_bar.progress(min(pyramid_levels * 15, 95))
        
        for (x, y, window) in sliding_window(resized, stepSize, (winW, winH)):
            total_windows += 1
            
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            
            try:
                sift_descriptors = extract_sift_features(window)
                
                if sift_descriptors is None or len(sift_descriptors) < 8:
                    continue
                
                bow_feature = create_bow_feature(sift_descriptors, BoW, num_clusters)
                bow_feature = bow_feature.reshape(1, -1)
                
                probs = svm.predict_proba(bow_feature)[0]
                prediction = np.argmax(probs)
                confidence = probs[prediction]
                
                if prediction == target_id and confidence > confidence_thresh:
                    if 'background' in label2id and prediction == label2id['background']:
                        continue
                    
                    # Scale coordinates back to original image
                    startX = int(x * resize_ratio)
                    startY = int(y * resize_ratio)
                    endX = int((x + winW) * resize_ratio)
                    endY = int((y + winH) * resize_ratio)
                    
                    detected_boxes.append([startX, startY, endX, endY])
            
            except Exception:
                continue
    
    # Apply NMS
    clone = image.copy()
    final_count = 0
    
    if len(detected_boxes) > 0:
        final_boxes = non_max_suppression(detected_boxes, overlapThresh=nms_thresh)
        final_count = len(final_boxes)
        
        # Draw boxes with improved text positioning
        for (x1, y1, x2, y2) in final_boxes:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Smart text positioning: avoid edge clipping
            text_y = y1 - 10 if y1 > 25 else y1 + 20
            cv2.putText(clone, target_label, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if progress_bar:
        progress_bar.progress(100)
    
    return clone, final_count


# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.title("🚶 Pedestrian Object Detection")
st.markdown("### SIFT + Bag of Words + SVM + Sliding Window")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a traffic image to detect objects"
    )
    
    st.markdown("---")
    
    # Target label selection
    available_labels = ['pedestrian', 'car', 'bus', 'truck', 'moto']  # Pedestrian first
    target_label = st.selectbox(
        "🎯 Target Object",
        available_labels,
        index=0,  # Default: pedestrian
        help="Select which type of object to detect"
    )
    
    # Confidence threshold
    confidence_thresh = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.3,
        max_value=0.95,
        value=0.6,
        step=0.05,
        help="Minimum confidence score (higher = stricter)"
    )
    
    # NMS threshold
    nms_thresh = st.slider(
        "🔄 NMS Threshold",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Overlap threshold for Non-Maximum Suppression (lower = less overlap)"
    )
    
    st.markdown("---")
    
    # Window size (advanced)
    with st.expander("🔧 Advanced Settings"):
        if target_label == 'pedestrian':
            default_w, default_h = 64, 128
        else:
            default_w, default_h = 64, 80
        
        win_width = st.number_input("Window Width", value=default_w, min_value=32, max_value=256)
        win_height = st.number_input("Window Height", value=default_h, min_value=32, max_value=256)
        step_size = st.number_input("Step Size", value=5, min_value=3, max_value=20)
    
    st.markdown("---")
    st.info("💡 **Tip**: Lower confidence = more detections but more false positives")

# Main area
if uploaded_file is None:
    # Display instructions
    st.info("👈 **Please upload an image from the sidebar to start detection**")
    
    st.markdown("""
    ### How to use:
    1. Upload a traffic image using the sidebar
    2. Select the target object type (car, pedestrian, etc.)
    3. Adjust confidence and NMS thresholds
    4. Wait for detection results
    
    ### Supported objects:
    - 🚗 Cars
    - 🚌 Buses
    - 🚚 Trucks
    - 🚶 Pedestrians
    - 🏍️ Motorcycles
    """)

else:
    # Process uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("❌ Failed to load image. Please try another file.")
    else:
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_container_width=True)
            st.caption(f"Size: {image.shape[1]}x{image.shape[0]} pixels")
        
        with col2:
            st.subheader(f"🎯 Detected {target_label.title()}s")
            
            # Detection with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner(f"🔍 Detecting {target_label}s..."):
                status_text.text("Running multi-scale sliding window detection...")
                
                result_image, detection_count = detect_objects_web(
                    image=image,
                    target_label=target_label,
                    confidence_thresh=confidence_thresh,
                    nms_thresh=nms_thresh,
                    winW=win_width,
                    winH=win_height,
                    stepSize=step_size,
                    progress_bar=progress_bar
                )
                
                status_text.text("✅ Detection complete!")
            
            # Convert BGR to RGB for display
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, use_container_width=True)
            
            # Show results
            if detection_count > 0:
                st.success(f"✅ Found **{detection_count}** {target_label}(s)!")
            else:
                st.warning(f"⚠️ No {target_label}s detected. Try lowering the confidence threshold.")
        
        # Additional info
        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("Detection Count", detection_count)
        
        with col_info2:
            st.metric("Confidence Threshold", f"{confidence_thresh:.0%}")
        
        with col_info3:
            st.metric("NMS Threshold", f"{nms_thresh:.0%}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Traffic Object Detection using SIFT + BoW + SVM</p>
    <p><small>Built with Streamlit | Computer Vision Project</small></p>
</div>
""", unsafe_allow_html=True)
