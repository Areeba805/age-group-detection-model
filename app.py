import base64
import io
import re
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Age Group Detection - YOLOv8 FYP Project",
    page_icon="🌿",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"

# One soft colour per age group, young to old (mint -> blue -> violet -> rose -> peach)
PALETTE = [
    "#9FD3C7", "#8CC7D8", "#86B3E3", "#8E9DE6", "#A294E4",
    "#BB92DB", "#D791C6", "#E79AA9", "#EDAE92",
]
INK = (31, 30, 51)

# The CSS forces a light look with dark text, so the app is readable even when
# the visitor's browser or Streamlit is set to dark mode.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

:root {
  color-scheme: light;
  --bg: #F2F1F8;
  --card: #FFFFFF;
  --ink: #1F1E33;
  --muted: #5B5970;
  --line: #DAD8EA;
  --accent: #5B66C4;
  --soft: #E8EAF8;
}
html, body { color-scheme: light; }

/* Base: light surfaces, dark text everywhere */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg) !important; }
.stApp, .stApp :where(p, span, label, li, h1, h2, h3, h4, h5, small, a) { color: var(--ink) !important; }
.stApp :where(p, label, li, button, input, small) { font-family: 'Plus Jakarta Sans', sans-serif; }
.block-container { max-width: 1080px; padding-top: 2.4rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }

/* Top bar and its icons (star, pencil, GitHub, share) */
[data-testid="stHeader"] { background: var(--bg) !important; }
.stApp [data-testid="stHeader"] :where(button, a, span, p, svg) { color: var(--ink) !important; fill: var(--ink) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebarContent"] { background: transparent !important; }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.ag-side-title { font-size: .98rem; font-weight: 600; margin: 0 0 .6rem 0; }
.stApp .ag-side-text { font-size: .9rem; line-height: 1.65; color: var(--muted) !important; margin: 0 0 1rem 0; }
.ag-meta { margin-bottom: 1.8rem; }
.ag-meta div { padding: 9px 0; border-top: 1px solid var(--line); }
.stApp .ag-meta span { display: block; font-size: .76rem; color: var(--muted) !important; }
.ag-meta b { display: block; font-size: .92rem; font-weight: 500; color: var(--ink); }
.ag-legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1.8rem; }
.ag-legend span { border-radius: 99px; padding: 4px 11px; font-size: .78rem; font-weight: 600; }
[data-testid="stWidgetLabel"] p { font-size: .88rem; font-weight: 500; }

/* Hero */
.ag-hero h1 {
  font-family: 'Instrument Serif', serif; font-weight: 400;
  font-size: clamp(2.6rem, 5.4vw, 4rem); line-height: 1.05;
  letter-spacing: -0.01em; margin: 0 0 .9rem 0; padding: 0;
}
.stApp .ag-hero p { font-size: 1.06rem; line-height: 1.65; color: var(--muted) !important; max-width: 34em; margin: 0 0 1.1rem 0; }
.ag-strip { display: flex; width: 150px; height: 6px; border-radius: 99px; overflow: hidden; margin-bottom: 2rem; }
.ag-strip i { flex: 1; }

/* Main tabs: outlined pills, the active one is filled dark */
.stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: none !important; margin-bottom: 1.1rem; background: transparent !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; height: 0 !important; }
.stTabs button[data-baseweb="tab"] {
  height: 44px !important; border-radius: 99px !important; padding: 0 24px !important;
  background: var(--card) !important; border: 1px solid var(--line) !important;
}
.stApp .stTabs button[data-baseweb="tab"] p { font-size: .95rem; font-weight: 500; color: var(--ink) !important; }
.stTabs button[aria-selected="true"] { background: var(--ink) !important; border-color: var(--ink) !important; }
.stApp .stTabs button[aria-selected="true"] p { color: #FFFFFF !important; }

/* Photo source switch (Upload / Camera): two pill buttons, the active one is dark */
.stApp [data-testid="stBaseButton-secondary"], .stApp [data-testid="baseButton-secondary"],
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="baseButton-primary"] {
  border-radius: 99px; padding: .55rem 1.5rem; border: 1px solid var(--line); background: var(--card) !important;
}
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="baseButton-primary"] {
  background: var(--ink) !important; border-color: var(--ink);
}
.stApp [data-testid="stBaseButton-primary"] *, .stApp [data-testid="baseButton-primary"] * { color: #FFFFFF !important; }

/* Upload and camera */
.stApp [data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important; border: 1.5px dashed #9FA5DA; border-radius: 24px; padding: 2.4rem 1.5rem;
}
.stApp [data-testid="stFileUploaderDropzone"] button {
  border-radius: 99px; border: 1px solid var(--ink); background: var(--card) !important;
}
.stApp [data-testid="stFileUploaderDropzoneInstructions"] :where(span, small) { color: var(--muted) !important; }
.stApp [data-testid="stCameraInput"] { border-radius: 24px; overflow: hidden; }
.stApp [data-testid="stCameraInput"] button { border-radius: 99px; border: none; background: var(--ink) !important; }
.stApp [data-testid="stCameraInput"] button * { color: #FFFFFF !important; }

/* Buttons */
.stApp .stDownloadButton button {
  border-radius: 99px; border: none; background: var(--ink) !important; padding: .6rem 1.5rem;
}
.stApp .stDownloadButton button * { color: #FFFFFF !important; font-weight: 500; }

/* Cards */
.ag-card {
  background: var(--card); border: 1px solid var(--line); border-radius: 24px; padding: 1.5rem 1.6rem;
  box-shadow: 0 1px 2px rgba(31,30,51,.05), 0 14px 34px -16px rgba(31,30,51,.16);
  margin-bottom: 1.2rem;
}
.ag-card h3 { font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 1.75rem; margin: 0 0 .35rem 0; padding: 0; }
.stApp .ag-note { font-size: .93rem; line-height: 1.6; color: var(--muted) !important; margin: 0 0 1rem 0; max-width: 42em; }
.ag-photo { width: 100%; display: block; border-radius: 16px; }
.stApp .ag-cap { font-size: .86rem; color: var(--muted) !important; margin: .7rem 0 0 .2rem; }

/* Empty state */
.ag-empty { margin-top: 1.2rem; padding: 1.3rem 1.7rem; border-radius: 20px; background: var(--soft); }
.ag-empty p { margin: 0 0 .5rem 0; font-weight: 600; font-size: .98rem; }
.ag-empty ul { margin: 0; padding-left: 1.1rem; font-size: .93rem; line-height: 1.85; }
.stApp .ag-empty li { color: var(--muted) !important; }

/* Results */
.ag-count { font-family: 'Instrument Serif', serif; font-size: 1.95rem; margin: 1.4rem 0 .3rem 0; color: var(--ink); }
.ag-spectrum { display: grid; grid-template-columns: repeat(auto-fit, minmax(86px, 1fr)); gap: 8px; margin: .8rem 0 1.4rem 0; }
.ag-chip { border-radius: 14px; padding: 10px 6px; text-align: center; color: var(--ink); }
.ag-chip b { display: block; font-size: .93rem; font-weight: 600; }
.stApp .ag-chip span { font-size: .78rem; color: rgba(31,30,51,.75) !important; }
.ag-row { display: flex; align-items: center; gap: 14px; padding: 12px 0; border-top: 1px solid var(--line); }
.ag-row:first-child { border-top: none; }
.ag-dot { width: 12px; height: 12px; border-radius: 99px; flex: none; }
.ag-grp { width: 64px; font-weight: 600; font-size: .95rem; flex: none; }
.ag-bar { flex: 1; height: 8px; border-radius: 99px; background: var(--soft); overflow: hidden; }
.ag-bar i { display: block; height: 100%; border-radius: 99px; }
.stApp .ag-pct { width: 46px; text-align: right; font-size: .9rem; color: var(--muted) !important; flex: none; }
.ag-hint { margin: 1rem 0; padding: 12px 16px; border-radius: 14px; background: #FBF1DE; color: #5E4A2A; font-size: .92rem; line-height: 1.55; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def age_key(name: str) -> int:
    """Sort key so '6-10' comes before '11-15' and '60-' comes last."""
    m = re.match(r"\d+", str(name))
    return int(m.group()) if m else 999


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def to_data_uri(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def file_data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return YOLO(path)


def find_image(*names):
    for n in names:
        p = BASE_DIR / n
        if p.exists():
            return p
    return None


def annotate(image: Image.Image, boxes, names, color_map) -> Image.Image:
    """Draw soft rounded boxes with a colour-coded label pill for each detection."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    stroke = max(4, int(min(w, h) * 0.006))
    fs = max(14, int(min(w, h) * 0.026))
    font = load_font(fs)

    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        name = names[int(b.cls[0])]
        conf = float(b.conf[0])
        col = hex_to_rgb(color_map[name])
        radius = int(min(stroke * 3, (x2 - x1) / 2, (y2 - y1) / 2))

        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, outline=(255, 255, 255, 235), width=stroke + 2)
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, outline=col + (255,), width=stroke)

        label = f"{name}  {conf:.0%}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        padx, pady = int(fs * 0.6), int(fs * 0.4)
        pw, ph = tw + 2 * padx, th + 2 * pady
        px = min(max(x1, 0), max(w - pw, 0))
        py = y1 - ph - 4
        if py < 0:
            py = y1 + 4
        draw.rounded_rectangle([px, py, px + pw, py + ph], radius=ph // 2, fill=col + (255,))
        draw.text((px + padx - tb[0], py + pady - tb[1]), label, font=font, fill=INK + (255,))
    return img


def spectrum_html(ordered, counts, color_map) -> str:
    chips = []
    for n in ordered:
        c = counts.get(n, 0)
        bg = color_map[n] if c else color_map[n] + "40"
        chips.append(f'<div class="ag-chip" style="background:{bg}"><b>{n}</b><span>{c if c else "none"}</span></div>')
    return '<div class="ag-spectrum">' + "".join(chips) + "</div>"


def rows_html(dets, color_map) -> str:
    out = []
    for name, conf in dets:
        col = color_map[name]
        out.append(
            f'<div class="ag-row"><span class="ag-dot" style="background:{col}"></span>'
            f'<span class="ag-grp">{name}</span>'
            f'<div class="ag-bar"><i style="width:{conf * 100:.0f}%;background:{col}"></i></div>'
            f'<span class="ag-pct">{conf:.0%}</span></div>'
        )
    return "".join(out)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error("best.pt was not found. Put it in the same folder as app.py.")
    st.stop()

model = load_model(str(MODEL_PATH))
names = model.names
ordered = sorted(names.values(), key=age_key)
color_map = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(ordered)}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="ag-side-title">About this project</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="ag-side-text">Finds people in a photo and estimates an age range for each one.</p>'
        '<div class="ag-meta">'
        "<div><span>Model</span><b>YOLOv8, custom trained</b></div>"
        "<div><span>Built with</span><b>Ultralytics and Streamlit</b></div>"
        "<div><span>Created by</span><b>Areeba Luqman</b></div>"
        "<div><span>Project</span><b>BSCS Final Year Project</b></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="ag-side-title">Age groups</p>', unsafe_allow_html=True)
    legend = "".join(f'<span style="background:{color_map[n]}">{n}</span>' for n in ordered)
    st.markdown(f'<div class="ag-legend">{legend}</div>', unsafe_allow_html=True)

    st.markdown('<p class="ag-side-title">Detection settings</p>', unsafe_allow_html=True)
    conf_thres = st.slider(
        "Minimum confidence", 0.05, 0.90, 0.25, 0.05,
        help="Lower it to catch more faces. Raise it to keep only the surest predictions.",
    )
    iou_thres = st.slider(
        "Overlap filter", 0.10, 0.90, 0.45, 0.05,
        help="Controls how overlapping boxes around the same face are merged.",
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
strip = "".join(f'<i style="background:{color_map[n]}"></i>' for n in ordered)
st.markdown(
    '<div class="ag-hero"><h1>Age Group Detection</h1>'
    "<p>Upload a photo or take one with your camera. The model finds each person and estimates an age range.</p>"
    f'<div class="ag-strip">{strip}</div></div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Detection", "Model performance"])

# ---------------------------------------------------------------------------
# Tab 1: Detection
# ---------------------------------------------------------------------------
with tab1:
    if "mode" not in st.session_state:
        st.session_state.mode = "upload"

    def set_mode(m):
        st.session_state.mode = m

    b1, b2, _ = st.columns([1.3, 1.3, 4], gap="small")
    with b1:
        st.button(
            "Upload a photo", key="mode_upload", on_click=set_mode, args=("upload",),
            type="primary" if st.session_state.mode == "upload" else "secondary",
        )
    with b2:
        st.button(
            "Use camera", key="mode_camera", on_click=set_mode, args=("camera",),
            type="primary" if st.session_state.mode == "camera" else "secondary",
        )
    mode = "Use camera" if st.session_state.mode == "camera" else "Upload a photo"

    if mode == "Use camera":
        source = st.camera_input("Take a photo", label_visibility="collapsed")
    else:
        source = st.file_uploader(
            "Upload a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

    if source is None:
        extra = "<li>Allow camera access when your browser asks</li>" if mode == "Use camera" else ""
        st.markdown(
            '<div class="ag-empty"><p>Add a photo to begin</p><ul>'
            "<li>Clear, front-facing faces give the best results</li>"
            "<li>Good lighting helps the model tell age groups apart</li>"
            f"{extra}"
            "<li>Your photo is processed in memory and never saved</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
        except Exception:
            st.error("This file could not be read as an image. Try another JPG or PNG.")
            st.stop()
        image.thumbnail((1600, 1600))

        with st.spinner("Looking for faces..."):
            result = model.predict(image, conf=conf_thres, iou=iou_thres, verbose=False)[0]

        boxes = result.boxes
        n = 0 if boxes is None else len(boxes)
        annotated = annotate(image, boxes, names, color_map) if n else image

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown(
                f'<div class="ag-card"><img class="ag-photo" src="{to_data_uri(image)}">'
                '<p class="ag-cap">Your photo</p></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f'<div class="ag-card"><img class="ag-photo" src="{to_data_uri(annotated)}">'
                '<p class="ag-cap">Detected age groups</p></div>',
                unsafe_allow_html=True,
            )

        if n == 0:
            st.markdown(
                '<div class="ag-hint">No faces were found. Try a clearer photo, or lower '
                "Minimum confidence in the sidebar.</div>",
                unsafe_allow_html=True,
            )
        else:
            dets = sorted(
                [(names[int(b.cls[0])], float(b.conf[0])) for b in boxes],
                key=lambda d: d[1],
                reverse=True,
            )
            counts = {}
            for name, _ in dets:
                counts[name] = counts.get(name, 0) + 1

            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            st.download_button(
                "Download result",
                data=buf.getvalue(),
                file_name="age-detection-result.png",
                mime="image/png",
            )

            st.markdown(
                f'<div class="ag-count">{n} detection{"s" if n != 1 else ""}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(spectrum_html(ordered, counts, color_map), unsafe_allow_html=True)
            st.markdown(
                f'<div class="ag-card">{rows_html(dets, color_map)}</div>',
                unsafe_allow_html=True,
            )
            if max(c for _, c in dets) < 0.4:
                st.markdown(
                    '<div class="ag-hint">Confidence is low for this photo. A clearer, front-facing '
                    "image usually improves the result.</div>",
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Tab 2: Model performance
# ---------------------------------------------------------------------------
with tab2:
    results_img = find_image("results.png", "results__1_.png", "results (1).png")
    cm_img = find_image("confusion_matrix.png")

    if results_img:
        st.markdown(
            '<div class="ag-card"><h3>Training results</h3>'
            '<p class="ag-note">Each chart tracks one measure across training. Loss curves going down and '
            "precision, recall and mAP going up show the model kept learning.</p>"
            f'<img class="ag-photo" src="{file_data_uri(str(results_img))}"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("results.png was not found in the app folder.")

    if cm_img:
        st.markdown(
            '<div class="ag-card"><h3>Confusion matrix</h3>'
            '<p class="ag-note">Columns are the real age group and rows are what the model predicted. '
            "A strong diagonal means correct predictions. Neighbouring groups, such as 31-40 and 41-50, "
            "are the most commonly mixed up.</p>"
            f'<img class="ag-photo" src="{file_data_uri(str(cm_img))}"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("confusion_matrix.png was not found in the app folder.")
