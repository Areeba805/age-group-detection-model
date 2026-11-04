import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Age Group Detection - YOLOv8 FYP",
    page_icon="🧑‍🤝‍🧑",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"

# Custom CSS for a clean look
st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0;}
    .sub-title {color: #6b7280; margin-top: 0; margin-bottom: 1.2rem;}
    .metric-card {
        background: #f3f4f6; border-radius: 12px; padding: 14px 18px;
        border-left: 5px solid #2563eb; margin-bottom: 8px; color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(path: str):
    return YOLO(path)


def find_image(*names):
    """Return the first existing image path from a list of candidate names."""
    for n in names:
        p = BASE_DIR / n
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📌 Project Info")
    st.markdown(
        """
        **Project:** Age Group Detection  
        **Model:** YOLOv8 (custom trained)  
        **Task:** Object detection of age groups  
        **Framework:** Ultralytics + Streamlit  
        """
    )
    st.divider()
    st.subheader("⚙️ Settings")
    conf_thres = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    iou_thres = st.slider("IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.05)
    st.divider()
    st.caption("Final Year Project • BSCS")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">Age Group Detection - YOLOv8 FYP Project</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload an image and the model will detect faces/people and predict their age group.</p>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("❌ `best.pt` not found. Place it in the same folder as `app.py`.")
    st.stop()

model = load_model(str(MODEL_PATH))

tab1, tab2 = st.tabs(["🔍 Detection", "📊 Model Performance"])

# ---------------------------------------------------------------------------
# Tab 1: Detection
# ---------------------------------------------------------------------------
with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)

        with st.spinner("Running detection..."):
            results = model.predict(img_np, conf=conf_thres, iou=iou_thres, verbose=False)
        result = results[0]

        # result.plot() returns a BGR image -> convert to RGB
        annotated = result.plot()[:, :, ::-1]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Detection Result")
            st.image(annotated, use_container_width=True)

        st.subheader("Detected Age Groups")
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            st.warning(
                "No detections found. Try lowering the confidence threshold in the sidebar "
                "or use a clearer image."
            )
        else:
            names = model.names  # class names come directly from the trained model
            rows = []
            for i, b in enumerate(boxes, start=1):
                cls_id = int(b.cls[0])
                rows.append(
                    {
                        "#": i,
                        "Age Group": names[cls_id],
                        "Confidence": round(float(b.conf[0]), 3),
                    }
                )
            df = pd.DataFrame(rows)

            m1, m2 = st.columns([1, 2])
            with m1:
                st.markdown(
                    f'<div class="metric-card"><b>Total detections:</b> {len(df)}</div>',
                    unsafe_allow_html=True,
                )
                summary = df["Age Group"].value_counts().rename_axis("Age Group").reset_index(name="Count")
                st.dataframe(summary, hide_index=True, use_container_width=True)
            with m2:
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                        )
                    },
                )
    else:
        st.info("👆 Upload a jpg/png image to start.")

# ---------------------------------------------------------------------------
# Tab 2: Model Performance
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Training Results")
    results_img = find_image("results.png", "results__1_.png", "results (1).png")
    if results_img:
        st.image(str(results_img), use_container_width=True)
    else:
        st.warning("`results.png` not found in the app folder.")

    st.subheader("Confusion Matrix")
    cm_img = find_image("confusion_matrix.png")
    if cm_img:
        st.image(str(cm_img), use_container_width=True)
    else:
        st.warning("`confusion_matrix.png` not found in the app folder.")

    with st.expander("Model classes"):
        st.write(list(model.names.values()))
