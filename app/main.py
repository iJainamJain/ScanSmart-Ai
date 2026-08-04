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

from src.pdf.export import export_multi_page_pdf, export_searchable_multi_page_pdf
from src.pipeline import MODE_BW, MODE_COLOR, MODE_GRAY, detect_document, render_page
from src.perspective.transform import order_points
from src.preprocessing.loader import resize_image

MODE_LABELS = {MODE_COLOR: "Color", MODE_GRAY: "Grayscale", MODE_BW: "Black & white"}
HANDLE_SELECT_RADIUS = 90  # px, in resized-image coordinates

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
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="SmartScan">
    <link rel="apple-touch-icon" href="app/static/icon-192.png">
    <style>
    /* Streamlit's default block padding is generous on desktop; on a phone
    screen it eats a meaningful fraction of the viewport. */
    @media (max-width: 640px) {
        .block-container { padding: 1rem 0.75rem; }
        div[data-testid="stImage"] img { max-height: 60vh; object-fit: contain; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SmartScan AI")
st.write("Scan document photos and export them as a single PDF.")

# ---------------------------------------------------------------- state
state = st.session_state
state.setdefault("pages", [])  # each: name, resized, corners, mode, flatten, _cache_key, _output
state.setdefault("processed_uploads", set())
state.setdefault("camera_capture_count", 0)
state.setdefault("editing_index", None)
state.setdefault("editor_corners", None)
state.setdefault("editor_selected", None)
state.setdefault("editor_last_click", None)
state.setdefault("default_mode", MODE_BW)
state.setdefault("default_flatten", True)


def _decode(uploaded_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(uploaded_bytes, np.uint8)
    resized, _ = resize_image(cv2.imdecode(data, cv2.IMREAD_COLOR))
    return resized


def _add_page(name: str, resized: np.ndarray) -> None:
    corners = detect_document(resized)
    state.pages.append(
        {
            "name": name,
            "resized": resized,
            "corners": corners,
            "mode": state.default_mode,
            "flatten": state.default_flatten,
            "_cache_key": None,
            "_output": None,
        }
    )


def _page_output(page: dict) -> np.ndarray:
    """Render (and cache) a page's output; only recomputes when its settings change."""
    corners_key = tuple(page["corners"].round(1).flatten()) if page["corners"] is not None else None
    key = (corners_key, page["mode"], page["flatten"])
    if page["_cache_key"] != key:
        page["_output"] = render_page(page["resized"], page["corners"], page["mode"], page["flatten"])
        page["_cache_key"] = key
    return page["_output"]


def _corners_or_full_frame(resized: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    if corners is not None:
        return order_points(corners)
    h, w = resized.shape[:2]
    margin = int(min(h, w) * 0.03)
    return np.array(
        [[margin, margin], [w - margin, margin], [w - margin, h - margin], [margin, h - margin]],
        dtype=np.float32,
    )


# ------------------------------------------------------------ corner editor
def render_corner_editor(resized: np.ndarray) -> None:
    """Interactive corner picker: click a handle to pick it up, click again to
    place it. Replaces the old "click 4 points in strict order, no do-overs"
    flow - any corner can be adjusted, in any order, as many times as needed,
    with the quad and a live label showing which handle (if any) is selected.
    """
    if state.editor_corners is None:
        state.editor_corners = _corners_or_full_frame(resized, None)

    corners = state.editor_corners
    preview = resized.copy()
    cv2.polylines(preview, [corners.astype(int)], True, (0, 0, 255), 3)
    for i, point in enumerate(corners.astype(int)):
        selected = state.editor_selected == i
        color = (0, 220, 255) if selected else (255, 60, 60)
        radius = 26 if selected else 20
        cv2.circle(preview, tuple(point), radius, color, -1)
        cv2.circle(preview, tuple(point), radius, (255, 255, 255), 2)
        cv2.putText(
            preview, str(i + 1), (point[0] - 8, point[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2,
        )

    if state.editor_selected is not None:
        st.caption(f"Corner {state.editor_selected + 1} selected - tap where it should go.")
    else:
        st.caption("Tap a corner handle to pick it up, then tap where it should go.")

    click = streamlit_image_coordinates(
        Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)), key="editor_click"
    )
    if click is not None:
        point = (click["x"], click["y"])
        if state.editor_last_click != point:
            state.editor_last_click = point
            if state.editor_selected is None:
                distances = np.linalg.norm(corners - np.array(point), axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] < HANDLE_SELECT_RADIUS:
                    state.editor_selected = nearest
                    st.rerun()
            else:
                corners[state.editor_selected] = point
                state.editor_corners = corners
                state.editor_selected = None
                st.rerun()

    reset_auto, reset_full, deselect = st.columns(3)
    if reset_auto.button("Auto-detect", key="editor_auto"):
        state.editor_corners = _corners_or_full_frame(resized, detect_document(resized))
        state.editor_selected = None
        st.rerun()
    if reset_full.button("Use full image", key="editor_full"):
        state.editor_corners = _corners_or_full_frame(resized, None)
        state.editor_selected = None
        st.rerun()
    if deselect.button("Deselect", key="editor_deselect", disabled=state.editor_selected is None):
        state.editor_selected = None
        st.rerun()


# ---------------------------------------------------------------- add pages
if state.editing_index is None:
    st.subheader("1. Add pages")

    mode_col, flatten_col = st.columns([2, 1])
    with mode_col:
        state.default_mode = st.radio(
            "Output mode for new pages", list(MODE_LABELS), format_func=lambda m: MODE_LABELS[m],
            horizontal=True, index=list(MODE_LABELS).index(state.default_mode),
        )
    with flatten_col:
        state.default_flatten = st.checkbox(
            "Remove shadows", value=state.default_flatten,
            help="Estimates the illumination field and divides it out - removes cast shadows "
            "and lighting gradients, and recovers faint text a plain scan misses.",
        )

    source = st.radio("Image source", ["Upload Image", "Camera"], horizontal=True)

    if source == "Upload Image":
        uploads = st.file_uploader(
            "Upload document photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
            help="Select several at once - each is auto-detected and added immediately. "
            "Fix any wrong crop afterward with Edit below.",
        )
        new_count = 0
        for uploaded in uploads or []:
            signature = f"{uploaded.name}:{uploaded.size}"
            if signature in state.processed_uploads:
                continue
            _add_page(uploaded.name, _decode(uploaded.getvalue()))
            state.processed_uploads.add(signature)
            new_count += 1
        if new_count:
            st.success(f"Added {new_count} page(s).")
    else:
        camera_key = f"camera_{state.camera_capture_count}"
        photo = st.camera_input("Take a photo", key=camera_key)
        st.caption(
            "Each photo is added automatically - the camera resets right away so you can "
            "keep shooting pages back to back."
        )
        if photo is not None:
            _add_page(photo.name or f"camera_{state.camera_capture_count}.jpg", _decode(photo.getvalue()))
            state.camera_capture_count += 1
            st.rerun()
else:
    page = state.pages[state.editing_index]
    st.subheader(f"Editing page {state.editing_index + 1} - {page['name']}")
    render_corner_editor(page["resized"])

    mode_col, flatten_col = st.columns([2, 1])
    with mode_col:
        edit_mode = st.radio(
            "Output mode", list(MODE_LABELS), format_func=lambda m: MODE_LABELS[m],
            horizontal=True, index=list(MODE_LABELS).index(page["mode"]), key="editor_mode",
        )
    with flatten_col:
        edit_flatten = st.checkbox("Remove shadows", value=page["flatten"], key="editor_flatten")

    save_col, cancel_col = st.columns(2)
    if save_col.button("Save changes", type="primary"):
        page["corners"] = state.editor_corners
        page["mode"] = edit_mode
        page["flatten"] = edit_flatten
        state.editing_index = None
        state.editor_corners = None
        state.editor_selected = None
        st.rerun()
    if cancel_col.button("Cancel"):
        state.editing_index = None
        state.editor_corners = None
        state.editor_selected = None
        st.rerun()

# ------------------------------------------------------------------ pages
st.subheader(f"2. Document ({len(state.pages)} page(s))")

if not state.pages:
    st.caption("No pages yet - add one above.")
else:
    apply_col, _ = st.columns([2, 1])
    if apply_col.button("Apply current output mode + shadow setting to all pages"):
        for page in state.pages:
            page["mode"] = state.default_mode
            page["flatten"] = state.default_flatten
        st.rerun()

    for index, page in enumerate(state.pages):
        if index == state.editing_index:
            continue  # shown in the editor panel above instead
        thumb, controls = st.columns([1, 2])
        with thumb:
            output = _page_output(page)
            st.image(output, use_container_width=True, channels="GRAY" if output.ndim == 2 else "BGR")
        with controls:
            st.caption(f"Page {index + 1} - {page['name']} - {MODE_LABELS[page['mode']]}")
            up, down, edit, delete = st.columns(4)
            pages = state.pages
            if up.button("Up", key=f"up{index}", disabled=index == 0):
                pages[index - 1], pages[index] = pages[index], pages[index - 1]
                st.rerun()
            if down.button("Down", key=f"dn{index}", disabled=index == len(pages) - 1):
                pages[index + 1], pages[index] = pages[index], pages[index + 1]
                st.rerun()
            if edit.button("Edit", key=f"ed{index}"):
                state.editing_index = index
                state.editor_corners = _corners_or_full_frame(page["resized"], page["corners"])
                state.editor_selected = None
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
                for index, page in enumerate(state.pages):
                    path = temp / f"page_{index:03d}.png"
                    cv2.imwrite(str(path), _page_output(page))
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

        st.success(f"PDF ready ({len(state.pages)} page(s)).")
        st.download_button(
            "Download PDF", data=pdf_bytes, file_name="smartscan_output.pdf", mime="application/pdf",
        )

    if st.button("Clear all pages"):
        state.pages = []
        state.processed_uploads = set()
        st.rerun()
