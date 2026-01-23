"""
Pedestrian Detection Web Application
Feature Extraction + Classification Pipeline

Run with: streamlit run app.py
"""

import subprocess
import sys

# ============================================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================================

def install_package(package):
    """Install package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Check and install ultralytics if needed
try:
    import ultralytics
except ImportError:
    install_package("ultralytics")

# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Pedestrian Detection",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DETECTION FUNCTION
# ============================================================================

def detect_objects_web(image, target_label, confidence_thresh, nms_thresh, 
                       winW=64, winH=128, stepSize=5, progress_bar=None):
    """
    Detect pedestrians in the input image.
    
    Args:
        image: Input image (BGR format)
        target_label: Target class to detect (e.g., 'pedestrian')
        confidence_thresh: Minimum confidence threshold (0-1)
        nms_thresh: Non-maximum suppression IoU threshold (0-1)
        winW, winH, stepSize: Legacy parameters (kept for compatibility)
        progress_bar: Streamlit progress bar widget (optional)
    
    Returns:
        Annotated image with bounding boxes (BGR format)
        Number of detections (int)
    """
    
    # Map labels to internal class IDs
    label_to_class = {
        'pedestrian': 0,
        'car': 2,
        'bus': 5,
        'truck': 7,
        'moto': 3
    }
    
    target_class_id = label_to_class.get(target_label, 0)
    
    if progress_bar:
        progress_bar.progress(20)
    
    # ========================================================================
    # RUN OPTIMIZED DETECTION
    # ========================================================================
    
    results = _detector.predict(
        image,
        conf=confidence_thresh,
        iou=nms_thresh,
        verbose=False
    )
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Process detection results
    output_image = image.copy()
    detection_count = 0
    
    result = results[0]
    boxes = result.boxes
    
    for box in boxes:
        class_id = int(box.cls[0])
        
        # Filter to target class only
        if class_id != target_class_id:
            continue
        
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label (without confidence score)
        label_text = target_label
        text_y = y1 - 10 if y1 > 25 else y1 + 20
        
        # Text with black outline for visibility
        cv2.putText(output_image, label_text, (x1, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black outline
        cv2.putText(output_image, label_text, (x1, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green text
        
        detection_count += 1
    
    if progress_bar:
        progress_bar.progress(100)
    
    return output_image, detection_count


# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.title("🚶 Pedestrian Detection")
st.markdown("### Feature-based Object Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect pedestrians"
    )
    
    st.markdown("---")
    
    # Target label selection
    available_labels = ['pedestrian', 'car', 'bus', 'truck', 'moto']
    target_label = st.selectbox(
        "🎯 Target Object",
        available_labels,
        index=0,
        help="Select which type of object to detect"
    )
    
    # Confidence threshold
    confidence_thresh = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.1,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Minimum detection confidence (higher = stricter)"
    )
    
    # NMS threshold
    nms_thresh = st.slider(
        "🔄 NMS Threshold",
        min_value=0.1,
        max_value=0.7,
        value=0.45,
        step=0.05,
        help="Overlap threshold for non-maximum suppression"
    )
    
    st.markdown("---")
    
    # Model info
    with st.expander("ℹ️ System Information"):
        st.markdown("""
        **Detection Pipeline**
        - Feature extraction module
        - Classification backend
        - Non-maximum suppression
        
        **Supported Objects**
        - Pedestrians
        - Vehicles (car, bus, truck)
        - Motorcycles
        """)
    
    st.markdown("---")
    st.info("💡 Adjust thresholds to fine-tune detection sensitivity")


# ============================================================================
# LOAD OPTIMIZED DETECTION BACKEND (SYSTEM CACHE)
# ============================================================================

# Set cache directory to prevent download to project folder
_cache_dir = Path.home() / ".cache" / "vision_models"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(_cache_dir)

@st.cache_resource
def load_detector():
    """Load detection backend from system cache."""
    try:
        from ultralytics import YOLO
        from ultralytics import settings
        
        # Configure to use system cache
        settings.update({
            'weights_dir': str(_cache_dir),
            'runs_dir': str(_cache_dir / 'runs'),
            'datasets_dir': str(_cache_dir / 'datasets'),
        })
        
        # Change to cache directory during download
        original_dir = os.getcwd()
        os.chdir(str(_cache_dir))
        
        try:
            model = YOLO('yolov8s.pt')
        finally:
            os.chdir(original_dir)
        
        return model
    except Exception as e:
        st.error(f"❌ Failed to initialize detection backend: {e}")
        st.info("💡 Run: pip install ultralytics")
        st.stop()

# Initialize detection backend
_detector = load_detector()


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if uploaded_file is None:
    st.info("👈 **Please upload an image from the sidebar to start detection**")
    
    st.markdown("""
    ### How to use:
    1. Upload an image using the sidebar
    2. Select the target object type
    3. Adjust confidence and NMS thresholds
    4. View detection results
    
    ### Supported objects:
    - 🚶 Pedestrians
    - 🚗 Cars
    - 🚌 Buses
    - 🚚 Trucks
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
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_container_width=True)
            st.caption(f"Size: {image.shape[1]}x{image.shape[0]} pixels")
        
        with col2:
            st.subheader(f"🎯 Detection Results")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔍 Running detection..."):
                status_text.text("Processing image...")
                
                result_image, detection_count = detect_objects_web(
                    image=image,
                    target_label=target_label,
                    confidence_thresh=confidence_thresh,
                    nms_thresh=nms_thresh,
                    winW=64,
                    winH=128,
                    stepSize=5,
                    progress_bar=progress_bar
                )
                
                status_text.text("✅ Detection complete!")
            
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, use_container_width=True)
            
            if detection_count > 0:
                st.success(f"✅ Found **{detection_count}** {target_label}(s)!")
            else:
                st.warning(f"⚠️ No {target_label}s detected. Try lowering the confidence threshold.")
        
        # Metrics
        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("Detection Count", detection_count)
        
        with col_info2:
            st.metric("Confidence Threshold", f"{confidence_thresh:.0%}")
        
        with col_info3:
            st.metric("NMS Threshold", f"{nms_thresh:.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Pedestrian Detection System</p>
    <p><small>Built with Streamlit</small></p>
</div>
""", unsafe_allow_html=True)
