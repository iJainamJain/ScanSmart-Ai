import sys
from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.detection.contours import find_document_contour
from src.detection.edges import denoise, to_grayscale
from src.enhancement.basic import enhance_document
from src.enhancement.contrast import adjust_brightness_contrast
from src.enhancement.sharpen import sharpen
from src.morphology.operations import closing, opening
from src.pdf.export import export_single_page_pdf, export_searchable_pdf
from src.perspective.transform import four_point_transform
from src.preprocessing.loader import resize_image
from src.segmentation.threshold import adaptive_threshold, clean_mask, segment_paper

st.set_page_config(page_title="SmartScan AI", layout="centered")
st.title("SmartScan AI")
st.write("Upload a document photo or capture one with your camera to scan it.")

# Initialize session state for manual corners
if 'manual_corners' not in st.session_state:
    st.session_state.manual_corners = []

if 'last_click' not in st.session_state:
    st.session_state.last_click = None

# Input selection
input_option = st.radio("Select Image Source", ["Upload Image", "Camera"])
image_file = None
if input_option == "Upload Image":
    image_file = st.file_uploader("Upload a document", type=['jpg', 'jpeg', 'png'])
else:
    image_file = st.camera_input("Take a photo")

if image_file is not None:
    # Reset corners if a new image is loaded
    if 'current_image_name' not in st.session_state or st.session_state.current_image_name != image_file.name:
        st.session_state.current_image_name = image_file.name
        st.session_state.manual_corners = []
        st.session_state.last_click = None

    # Load image
    bytes_data = image_file.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 1. Preprocess & Detect
    resized, _scale = resize_image(cv_img)
    gray = to_grayscale(resized)
    blurred = denoise(gray)
    paper_mask = segment_paper(blurred)
    cleaned_mask = clean_mask(paper_mask)
    image_area = resized.shape[0] * resized.shape[1]
    
    auto_corners = find_document_contour(cleaned_mask, image_area, gray)
    
    st.subheader("1. Boundary Detection")
    st.write("If the red boundary is incorrect, click 4 points on the image (in any order) to manually set the corners.")
    
    # Determine which corners to use
    corners_to_use = auto_corners
    if len(st.session_state.manual_corners) == 4:
        corners_to_use = np.array(st.session_state.manual_corners, dtype=np.float32)
    elif len(st.session_state.manual_corners) > 0:
        corners_to_use = None  # User is in the middle of clicking

    # Draw corners on a preview copy
    preview = resized.copy()
    
    # Draw established corners
    if corners_to_use is not None:
        cv2.drawContours(preview, [corners_to_use.astype(int)], -1, (0, 0, 255), 3)
        for pt in corners_to_use.astype(int):
            cv2.circle(preview, tuple(pt), 10, (255, 0, 0), -1)

    # Draw clicks in progress
    for pt in st.session_state.manual_corners:
        cv2.circle(preview, tuple(pt), 10, (0, 255, 0), -1)

    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    
    # Capture clicks
    value = streamlit_image_coordinates(Image.fromarray(preview_rgb), key="preview")
    
    if value is not None:
        point = (value["x"], value["y"])
        # Prevent re-adding the same point if streamlit re-runs without a new click
        if st.session_state.last_click != point:
            st.session_state.last_click = point
            if len(st.session_state.manual_corners) < 4:
                st.session_state.manual_corners.append(point)
                st.rerun()

    # Controls for manual override
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reset Manual Corners"):
            st.session_state.manual_corners = []
            st.session_state.last_click = None
            st.rerun()
    with col2:
        if len(st.session_state.manual_corners) > 0 and len(st.session_state.manual_corners) < 4:
            st.warning(f"Clicked {len(st.session_state.manual_corners)}/4 points...")

    # 2. Process & Export
    st.subheader("2. Final Processing")
    ocr_enabled = st.checkbox("Make PDF Searchable (OCR with Tesseract)", value=True)
    
    if corners_to_use is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Flattening and enhancing..."):
                # Run the rest of the pipeline
                flattened = four_point_transform(resized, corners_to_use)
                contrast_adjusted = adjust_brightness_contrast(flattened, brightness=10, contrast=1.15)
                sharpened = sharpen(contrast_adjusted)
                enhanced = enhance_document(sharpened)
                adaptive = adaptive_threshold(enhanced)
                morph_cleaned = closing(opening(adaptive, kernel_size=3), kernel_size=3)
                
                # Show result
                st.image(morph_cleaned, caption="Binarized Scan", use_column_width=True, channels="GRAY")
                
                # Create PDF in temp dir
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir_path = Path(temp_dir)
                    img_path = temp_dir_path / "temp.png"
                    pdf_path = temp_dir_path / "scan.pdf"
                    
                    cv2.imwrite(str(img_path), morph_cleaned)
                    
                    success = False
                    if ocr_enabled:
                        # Streamlit toast won't block, but spinner will hide it. We just run it.
                        success = export_searchable_pdf(img_path, pdf_path)
                        if not success:
                            st.warning("OCR failed or Tesseract is missing. Falling back to standard PDF.")
                            
                    if not success:
                        export_single_page_pdf(img_path, pdf_path)
                    
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                        
                    st.success("Processing complete!")
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name="smartscan_output.pdf",
                        mime="application/pdf"
                    )
    else:
        st.info("Please set all 4 corners to proceed.")
