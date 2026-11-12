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
INK = (38, 37, 58)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #F5F4F9;
  --card: #FFFFFF;
  --ink: #26253A;
  --muted: #767489;
  --line: #E6E4F0;
  --accent: #6F7AD1;
  --soft: #ECEEFA;
}

.stApp { background: var(--bg); color: var(--ink); }
.stApp p, .stApp label, .stApp li, .stApp button, .stApp input,
.stApp [data-baseweb="tab"] { font-family: 'Plus Jakarta Sans', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1080px; padding-top: 2.6rem; padding-bottom: 4rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.ag-side-title { font-size: 0.95rem; font-weight: 600; color: var(--ink); margin: 0 0 .6rem 0; }
.ag-side-text { font-size: .88rem; line-height: 1.7; color: var(--muted); margin: 0 0 1.6rem 0; }
.ag-side-text b { color: var(--ink); font-weight: 500; }
[data-testid="stWidgetLabel"] p { font-size: .86rem; color: var(--ink); }

/* Hero */
.ag-hero h1 {
  font-family: 'Instrument Serif', serif; font-weight: 400;
  font-size: clamp(2.6rem, 5.4vw, 4rem); line-height: 1.05;
  letter-spacing: -0.01em; margin: 0 0 .9rem 0; color: var(--ink); padding: 0;
}
.ag-hero p { font-size: 1.05rem; line-height: 1.65; color: var(--muted); max-width: 34em; margin: 0 0 1.1rem 0; }
.ag-strip { display: flex; width: 150px; height: 6px; border-radius: 99px; overflow: hidden; margin-bottom: 2rem; }
.ag-strip i { flex: 1; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; margin-bottom: 1rem; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab"] {
  height: 42px; border-radius: 99px; padding: 0 20px; background: transparent; color: var(--muted);
}
.stTabs [data-baseweb="tab"] p { font-size: .95rem; font-weight: 500; }
.stTabs [aria-selected="true"] {
  background: var(--card); color: var(--ink);
  box-shadow: 0 1px 2px rgba(38,37,58,.06), 0 6px 18px -8px rgba(38,37,58,.14);
}

/* Upload */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card); border: 1.5px dashed #C7CAE8; border-radius: 24px; padding: 2.2rem 1.5rem;
}
[data-testid="stFileUploaderDropzone"] button {
  border-radius: 99px; border: 1px solid var(--line); background: var(--soft); color: var(--ink);
}

/* Cards */
.ag-card {
  background: var(--card); border: 1px solid var(--line); border-radius: 24px; padding: 1.5rem 1.6rem;
  box-shadow: 0 1px 2px rgba(38,37,58,.04), 0 14px 34px -16px rgba(38,37,58,.14);
  margin-bottom: 1.2rem;
}
.ag-card h3 { font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 1.7rem; margin: 0 0 .35rem 0; color: var(--ink); padding: 0; }
.ag-note { font-size: .92rem; line-height: 1.6; color: var(--muted); margin: 0 0 1rem 0; max-width: 42em; }
.ag-photo { width: 100%; display: block; border-radius: 16px; }
.ag-cap { font-size: .85rem; color: var(--muted); margin: .7rem 0 0 .2rem; }

/* Empty state */
.ag-empty { margin-top: 1.2rem; padding: 1.2rem 1.6rem; border-radius: 20px; background: var(--soft); color: var(--ink); }
.ag-empty p { margin: 0 0 .5rem 0; font-weight: 600; font-size: .95rem; }
.ag-empty ul { margin: 0; padding-left: 1.1rem; color: var(--muted); font-size: .92rem; line-height: 1.8; }

/* Results */
.ag-count { font-family: 'Instrument Serif', serif; font-size: 1.9rem; margin: 1.6rem 0 .3rem 0; color: var(--ink); }
.ag-spectrum { display: grid; grid-template-columns: repeat(auto-fit, minmax(86px, 1fr)); gap: 8px; margin: .8rem 0 1.4rem 0; }
.ag-chip { border-radius: 14px; padding: 10px 6px; text-align: center; color: var(--ink); }
.ag-chip b { display: block; font-size: .92rem; font-weight: 600; }
.ag-chip span { font-size: .78rem; color: rgba(38,37,58,.7); }
.ag-row { display: flex; align-items: center; gap: 14px; padding: 12px 0; border-top: 1px solid var(--line); }
.ag-row:first-child { border-top: none; }
.ag-dot { width: 12px; height: 12px; border-radius: 99px; flex: none; }
.ag-grp { width: 64px; font-weight: 600; font-size: .95rem; flex: none; }
.ag-bar { flex: 1; height: 8px; border-radius: 99px; background: var(--soft); overflow: hidden; }
.ag-bar i { display: block; height: 100%; border-radius: 99px; }
.ag-pct { width: 46px; text-align: right; font-size: .9rem; color: var(--muted); flex: none; }
.ag-hint { margin-top: 1rem; padding: 12px 16px; border-radius: 14px; background: #FBF3E6; color: #6B5636; font-size: .9rem; line-height: 1.55; }

/* Legend (sidebar) */
.ag-legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1.6rem; }
.ag-legend span { border-radius: 99px; padding: 4px 11px; font-size: .78rem; font-weight: 500; color: var(--ink); }
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
        '<p class="ag-side-text"><b>Age Group Detection</b><br>'
        "A YOLOv8 model trained on custom data to find people in a photo and estimate an age range for each.<br><br>"
        "<b>Model</b> YOLOv8, custom trained<br>"
        "<b>Built with</b> Ultralytics and Streamlit<br>"
        "<b>By</b> Areeba Luqman, BSCS Final Year Project</p>",
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
    "<p>Upload a photo. The model finds each person in it and estimates an age range.</p>"
    f'<div class="ag-strip">{strip}</div></div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Detection", "Model performance"])

# ---------------------------------------------------------------------------
# Tab 1: Detection
# ---------------------------------------------------------------------------
with tab1:
    uploaded = st.file_uploader(
        "Upload a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown(
            '<div class="ag-empty"><p>Add a photo to begin</p><ul>'
            "<li>Clear, front-facing faces give the best results</li>"
            "<li>Good lighting helps the model tell age groups apart</li>"
            "<li>Your photo is processed in memory and never saved</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        except Exception:
            st.error("This file could not be read as an image. Try another JPG or PNG.")
            st.stop()
        image.thumbnail((1600, 1600))

        with st.spinner("Looking for faces..."):
            result = model.predict(image, conf=conf_thres, iou=iou_thres, verbose=False)[0]

        boxes = result.boxes
        n = 0 if boxes is None else len(boxes)
        annotated = annotate(image, boxes, names, color_map) if n else image

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
