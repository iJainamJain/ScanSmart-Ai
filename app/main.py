import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pdf.export import (
    export_multi_page_pdf,
    export_searchable_multi_page_pdf,
)
from src.pipeline import detect_document, scan_page
from src.preprocessing.loader import resize_image

st.set_page_config(
    page_title="SmartScan AI",
    page_icon="app/static/icon-192.png",
    layout="centered",
)

# Installable-to-home-screen support. Streamlit owns the page <head>, so these
# tags are injected into the DOM instead; browsers honour a manifest link found
# there. This gives the icon and a chrome-free standalone window - it is not a
# full PWA (no service worker, so no offline use), which Streamlit's architecture
# does not allow without patching its served HTML.
st.markdown(
    """
    <link rel="manifest" href="app/static/manifest.json">
    <meta name="theme-color" content="#1a1a2e">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="SmartScan">
    <link rel="apple-touch-icon" href="app/static/icon-192.png">
    """,
    unsafe_allow_html=True,
)

st.title("SmartScan AI")
st.write("Scan document photos and export them as a single PDF.")

if "manual_corners" not in st.session_state:
    st.session_state.manual_corners = []
if "last_click" not in st.session_state:
    st.session_state.last_click = None
if "pages" not in st.session_state:
    st.session_state.pages = []  # list of {"name": str, "bw": ndarray}

# ---------------------------------------------------------------- capture
st.subheader("1. Add a page")
source = st.radio("Image source", ["Upload Image", "Camera"], horizontal=True)
image_file = (
    st.file_uploader("Upload a document", type=["jpg", "jpeg", "png"])
    if source == "Upload Image"
    else st.camera_input("Take a photo")
)

if image_file is not None:
    if st.session_state.get("current_image_name") != image_file.name:
        st.session_state.current_image_name = image_file.name
        st.session_state.manual_corners = []
        st.session_state.last_click = None

    data = np.frombuffer(image_file.getvalue(), np.uint8)
    resized, _ = resize_image(cv2.imdecode(data, cv2.IMREAD_COLOR))
    auto_corners = detect_document(resized)

    st.write("If the red boundary is wrong, click 4 points to set the corners yourself.")

    corners = auto_corners
    if len(st.session_state.manual_corners) == 4:
        corners = np.array(st.session_state.manual_corners, dtype=np.float32)
    elif st.session_state.manual_corners:
        corners = None  # part-way through clicking

    preview = resized.copy()
    if corners is not None:
        cv2.drawContours(preview, [corners.astype(int)], -1, (0, 0, 255), 3)
        for point in corners.astype(int):
            cv2.circle(preview, tuple(point), 10, (255, 0, 0), -1)
    for point in st.session_state.manual_corners:
        cv2.circle(preview, tuple(point), 10, (0, 255, 0), -1)

    click = streamlit_image_coordinates(
        Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)), key="preview"
    )
    if click is not None:
        point = (click["x"], click["y"])
        if st.session_state.last_click != point and len(st.session_state.manual_corners) < 4:
            st.session_state.last_click = point
            st.session_state.manual_corners.append(point)
            st.rerun()

    left, right = st.columns([1, 3])
    with left:
        if st.button("Reset corners"):
            st.session_state.manual_corners = []
            st.session_state.last_click = None
            st.rerun()
    with right:
        pending = len(st.session_state.manual_corners)
        if 0 < pending < 4:
            st.warning(f"Clicked {pending}/4 points...")

    flatten_lighting = st.checkbox(
        "Remove shadows and uneven lighting",
        value=True,
        help="Estimates the illumination field and divides it out. Removes cast "
        "shadows and lighting gradients, and recovers faint text the plain "
        "pipeline misses.",
    )

    if corners is None:
        st.info("Set all 4 corners to continue.")
    elif st.button("Add page to document", type="primary"):
        with st.spinner("Flattening and enhancing..."):
            _, final_bw = scan_page(resized, corners, flatten_lighting=flatten_lighting)
        st.session_state.pages.append({"name": image_file.name, "bw": final_bw})
        st.session_state.manual_corners = []
        st.session_state.last_click = None
        st.rerun()

# ------------------------------------------------------------------ pages
st.subheader(f"2. Document ({len(st.session_state.pages)} page(s))")

if not st.session_state.pages:
    st.caption("No pages yet - add one above.")
else:
    for index, page in enumerate(st.session_state.pages):
        thumb, controls = st.columns([1, 2])
        with thumb:
            st.image(page["bw"], width=110, channels="GRAY")
        with controls:
            st.caption(f"Page {index + 1} - {page['name']}")
            up, down, delete = st.columns(3)
            pages = st.session_state.pages
            if up.button("Up", key=f"up{index}", disabled=index == 0):
                pages[index - 1], pages[index] = pages[index], pages[index - 1]
                st.rerun()
            if down.button("Down", key=f"dn{index}", disabled=index == len(pages) - 1):
                pages[index + 1], pages[index] = pages[index], pages[index + 1]
                st.rerun()
            if delete.button("Delete", key=f"del{index}"):
                pages.pop(index)
                st.rerun()

    # ---------------------------------------------------------------- export
    st.subheader("3. Export")
    use_ocr = st.checkbox("Make PDF searchable (OCR with Tesseract)", value=True)

    if st.button("Build PDF", type="primary"):
        with st.spinner("Building PDF..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                page_files = []
                for index, page in enumerate(st.session_state.pages):
                    path = temp / f"page_{index:03d}.png"
                    cv2.imwrite(str(path), page["bw"])
                    page_files.append(path)

                pdf_path = temp / "smartscan.pdf"
                built = False
                if use_ocr:
                    built = export_searchable_multi_page_pdf(page_files, pdf_path)
                    if not built:
                        st.warning("Tesseract unavailable - exporting an image-only PDF instead.")
                if not built:
                    export_multi_page_pdf(page_files, pdf_path)

                pdf_bytes = pdf_path.read_bytes()

        st.success(f"PDF ready ({len(st.session_state.pages)} page(s)).")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="smartscan_output.pdf",
            mime="application/pdf",
        )

    if st.button("Clear all pages"):
        st.session_state.pages = []
        st.rerun()
