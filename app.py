import base64
import io
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
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
INK = (27, 26, 47)

# The CSS forces a light look with dark text, so the app stays readable even when
# the visitor's browser or Streamlit is set to dark mode. Selectors avoid
# Streamlit's generated class names so they keep working across versions.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

:root {
  color-scheme: light;
  --bg: #F3F2F9;
  --ink: #1B1A2F;
  --muted: #4F4D66;
  --line: #D9D7EA;
  --edge: #C9C7DE;
  --accent: #5B66C4;
  --soft: #E8EAF8;
  --glass: rgba(255,255,255,.82);
}
html, body { color-scheme: light; }

/* Canvas: calm gradient wash behind everything */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent !important; }
.stApp {
  background:
    radial-gradient(54rem 36rem at 92% -8%, rgba(162,148,228,.30), transparent 60%),
    radial-gradient(46rem 34rem at -8% 8%, rgba(159,211,199,.34), transparent 58%),
    radial-gradient(44rem 30rem at 70% 108%, rgba(237,174,146,.24), transparent 60%),
    var(--bg) !important;
  background-attachment: fixed !important;
}

/* Text: dark everywhere */
.stApp, .stApp :where(p, span, label, li, h1, h2, h3, h4, h5, small, a, summary) { color: var(--ink) !important; }
.stApp :where(p, label, li, button, input, small, summary) { font-family: 'Plus Jakarta Sans', sans-serif; }
.block-container { max-width: 1080px; padding-top: 2.4rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }

/* Top bar and its icons */
[data-testid="stHeader"] { background: transparent !important; }
.stApp [data-testid="stHeader"] :where(button, a, span, p, svg) { color: var(--ink) !important; fill: var(--ink) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: rgba(255,255,255,.88) !important; backdrop-filter: blur(18px);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebarContent"] { background: transparent !important; }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.ag-side-title { font-size: .98rem; font-weight: 600; margin: 0 0 .6rem 0; }
.stApp .ag-side-text { font-size: .9rem; line-height: 1.65; color: var(--muted) !important; margin: 0 0 1rem 0; }
.ag-meta { margin-bottom: 1.8rem; }
.ag-meta div { padding: 9px 0; border-top: 1px solid var(--line); }
.stApp .ag-meta span { display: block; font-size: .76rem; color: var(--muted) !important; }
.ag-meta b { display: block; font-size: .92rem; font-weight: 500; color: var(--ink); line-height: 1.45; }
.ag-legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1.8rem; }
.ag-legend span { border-radius: 99px; padding: 4px 11px; font-size: .78rem; font-weight: 600; }
[data-testid="stWidgetLabel"] p { font-size: .88rem; font-weight: 500; }

/* Hero */
.ag-hero { padding: .6rem 0 .4rem 0; }
.ag-hero h1 {
  font-family: 'Instrument Serif', serif; font-weight: 400;
  font-size: clamp(3rem, 7vw, 5rem); line-height: 1; letter-spacing: -0.015em;
  margin: 0 0 1rem 0; padding: 0;
}
.stApp .ag-hero p { font-size: 1.1rem; line-height: 1.65; color: var(--muted) !important; max-width: 33em; margin: 0 0 1.3rem 0; }
.ag-strip { display: flex; width: 170px; height: 7px; border-radius: 99px; overflow: hidden; margin-bottom: 2.2rem; }
.ag-strip i { flex: 1; }

/* Tabs: outlined pills, the active one is filled dark */
.stApp [data-baseweb="tab-list"] { gap: 10px; border-bottom: none !important; background: transparent !important; margin-bottom: 1.2rem; }
.stApp [data-baseweb="tab-highlight"], .stApp [data-baseweb="tab-border"] { display: none !important; height: 0 !important; }
.stApp [data-baseweb="tab"], .stApp [role="tab"] {
  height: 46px !important; border-radius: 99px !important; padding: 0 26px !important;
  background: rgba(255,255,255,.9) !important; border: 1px solid var(--edge) !important;
  box-shadow: 0 1px 2px rgba(27,26,47,.05);
}
.stApp [data-baseweb="tab"] p, .stApp [role="tab"] p { font-size: .95rem; font-weight: 500; color: var(--ink) !important; white-space: nowrap; }
.stApp [data-baseweb="tab"][aria-selected="true"], .stApp [role="tab"][aria-selected="true"] {
  background: var(--ink) !important; border-color: var(--ink) !important;
}
.stApp [data-baseweb="tab"][aria-selected="true"] p, .stApp [role="tab"][aria-selected="true"] p { color: #FFFFFF !important; }

/* Buttons: Photo / Video switch */
.stApp [data-testid="stBaseButton-secondary"], .stApp [data-testid="baseButton-secondary"],
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="baseButton-primary"] {
  border-radius: 99px; padding: .6rem 1.7rem; min-height: 46px; white-space: nowrap;
  border: 1px solid var(--edge); background: rgba(255,255,255,.92) !important;
  box-shadow: 0 1px 2px rgba(27,26,47,.06);
  transition: transform .15s ease, box-shadow .15s ease;
}
.stApp [data-testid="stBaseButton-secondary"]:hover, .stApp [data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-1px); box-shadow: 0 8px 20px -10px rgba(27,26,47,.35);
}
.stApp [data-testid="stBaseButton-secondary"] p, .stApp [data-testid="stBaseButton-primary"] p,
.stApp [data-testid="baseButton-secondary"] p, .stApp [data-testid="baseButton-primary"] p {
  white-space: nowrap; overflow: visible; text-overflow: clip; font-weight: 500;
}
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="baseButton-primary"] {
  background: var(--ink) !important; border-color: var(--ink);
}
.stApp [data-testid="stBaseButton-primary"] *, .stApp [data-testid="baseButton-primary"] * { color: #FFFFFF !important; }

/* Upload */
.stApp [data-testid="stFileUploaderDropzone"] {
  background: var(--glass) !important; border: 1.5px dashed #9FA5DA; border-radius: 28px; padding: 2.6rem 1.6rem;
}
.stApp [data-testid="stFileUploaderDropzone"] button {
  border-radius: 99px; border: 1px solid var(--ink); background: #FFFFFF !important;
}
.stApp [data-testid="stFileUploaderDropzoneInstructions"] :where(span, small) { color: var(--muted) !important; }

/* Download button */
.stApp [data-testid="stDownloadButton"] button, .stApp .stDownloadButton button {
  border-radius: 99px; border: none; background: var(--ink) !important; padding: .65rem 1.7rem; min-height: 46px;
}
.stApp [data-testid="stDownloadButton"] button *, .stApp .stDownloadButton button * { color: #FFFFFF !important; font-weight: 500; }

/* Glass cards */
@keyframes agRise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.ag-card, .ag-step, .ag-metric, .ag-count, .ag-spectrum { animation: agRise .5s ease both; }
@media (prefers-reduced-motion: reduce) { .ag-card, .ag-step, .ag-metric, .ag-count, .ag-spectrum { animation: none; } }
.ag-card {
  background: var(--glass); backdrop-filter: blur(16px) saturate(1.2);
  border: 1px solid rgba(255,255,255,.95); border-radius: 28px; padding: 1.5rem 1.6rem;
  box-shadow: 0 1px 2px rgba(27,26,47,.05), 0 26px 50px -26px rgba(27,26,47,.26);
  margin-bottom: 1.2rem;
}
.ag-card h3 { font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 1.8rem; margin: 0 0 .35rem 0; padding: 0; }
.stApp .ag-note { font-size: .93rem; line-height: 1.6; color: var(--muted) !important; margin: 0 0 1rem 0; max-width: 42em; }
.ag-photo { width: 100%; display: block; border-radius: 18px; }
.stApp .ag-cap { font-size: .86rem; color: var(--muted) !important; margin: .8rem 0 0 .2rem; }

/* Empty state: three steps */
.ag-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 1.4rem 0 1rem 0; }
.ag-step {
  background: var(--glass); backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,.95); border-radius: 24px; padding: 1.3rem 1.3rem 1.2rem 1.3rem;
  box-shadow: 0 1px 2px rgba(27,26,47,.05), 0 20px 40px -24px rgba(27,26,47,.24);
}
.ag-ico { width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center; margin-bottom: .9rem; }
.ag-ico svg { width: 22px; height: 22px; stroke: #1B1A2F; fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.ag-step b { display: block; font-size: 1rem; font-weight: 600; margin-bottom: .25rem; }
.stApp .ag-step span { display: block; font-size: .9rem; line-height: 1.55; color: var(--muted) !important; }
.stApp .ag-fine { font-size: .88rem; line-height: 1.6; color: var(--muted) !important; margin: .6rem 0 0 .2rem; }
.ag-help {
  margin-top: 1rem; padding: 1rem 1.3rem; border-radius: 20px;
  background: rgba(255,255,255,.7); border: 1px solid var(--line);
}
.ag-help summary { cursor: pointer; font-weight: 600; font-size: .95rem; }
.ag-help ol { margin: .7rem 0 0 0; padding-left: 1.2rem; font-size: .92rem; line-height: 1.8; }
.stApp .ag-help li { color: var(--muted) !important; }

/* Results */
.ag-count { font-family: 'Instrument Serif', serif; font-size: 2.1rem; margin: 1.5rem 0 .3rem 0; color: var(--ink); }
.ag-spectrum { display: grid; grid-template-columns: repeat(auto-fit, minmax(86px, 1fr)); gap: 8px; margin: .8rem 0 1.4rem 0; }
.ag-chip { border-radius: 16px; padding: 11px 6px; text-align: center; color: var(--ink); }
.ag-chip b { display: block; font-size: .93rem; font-weight: 600; }
.stApp .ag-chip span { font-size: .78rem; color: rgba(27,26,47,.75) !important; }
.ag-row { display: flex; align-items: center; gap: 14px; padding: 13px 0; border-top: 1px solid var(--line); }
.ag-row:first-child { border-top: none; }
.ag-dot { width: 12px; height: 12px; border-radius: 99px; flex: none; }
.ag-grp { width: 64px; font-weight: 600; font-size: .95rem; flex: none; }
.ag-bar { flex: 1; height: 9px; border-radius: 99px; background: var(--soft); overflow: hidden; }
.ag-bar i { display: block; height: 100%; border-radius: 99px; }
.stApp .ag-pct { width: 46px; text-align: right; font-size: .9rem; color: var(--muted) !important; flex: none; }
.ag-hint { margin: 1rem 0; padding: 13px 17px; border-radius: 16px; background: #FBF1DE; color: #5E4A2A; font-size: .92rem; line-height: 1.55; }

/* Model metrics */
.ag-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 1.2rem; }
.ag-metric {
  background: var(--glass); backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,.95); border-radius: 24px; padding: 1.1rem 1.3rem;
  box-shadow: 0 1px 2px rgba(27,26,47,.05), 0 20px 40px -24px rgba(27,26,47,.24);
}
.stApp .ag-metric span { display: block; font-size: .82rem; color: var(--muted) !important; }
.ag-metric b { display: block; font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 2.4rem; line-height: 1.15; }

.stApp video { width: 100%; border-radius: 18px; }
.stApp .ag-footer { text-align: center; font-size: .82rem; color: var(--muted) !important; margin-top: 2.5rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ICON_UPLOAD = '<svg viewBox="0 0 24 24"><path d="M12 16V5"/><path d="M7.5 9.5 12 5l4.5 4.5"/><path d="M5 19h14"/></svg>'
ICON_SCAN = (
    '<svg viewBox="0 0 24 24"><path d="M4 8V6a2 2 0 0 1 2-2h2"/><path d="M16 4h2a2 2 0 0 1 2 2v2"/>'
    '<path d="M20 16v2a2 2 0 0 1-2 2h-2"/><path d="M8 20H6a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/></svg>'
)
ICON_CHART = '<svg viewBox="0 0 24 24"><path d="M5 19v-8"/><path d="M12 19V5"/><path d="M19 19v-6"/></svg>'

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

@lru_cache(maxsize=16)
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

def get_metrics(model):
    """Read the validation metrics saved inside best.pt, if present."""
    try:
        ckpt = getattr(model, "ckpt", None) or {}
        tm = ckpt.get("train_metrics") or {}
        keys = [
            ("Precision", "metrics/precision(B)"),
            ("Recall", "metrics/recall(B)"),
            ("mAP50", "metrics/mAP50(B)"),
            ("mAP50-95", "metrics/mAP50-95(B)"),
        ]
        return [(label, float(tm[k])) for label, k in keys if k in tm]
    except Exception:
        return []

def draw_boxes(image: Image.Image, items, color_map) -> Image.Image:
    """Draw soft rounded boxes with a colour-coded label pill.
    items: list of (x1, y1, x2, y2, age_group_name, confidence)."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    stroke = max(3, int(min(w, h) * 0.006))
    fs = max(13, int(min(w, h) * 0.026))
    font = load_font(fs)

    for x1, y1, x2, y2, name, conf in items:
        col = hex_to_rgb(color_map[name])
        radius = int(max(0, min(stroke * 3, (x2 - x1) / 2, (y2 - y1) / 2)))

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

def annotate(image: Image.Image, boxes, names, color_map) -> Image.Image:
    items = []
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        items.append((x1, y1, x2, y2, names[int(b.cls[0])], float(b.conf[0])))
    return draw_boxes(image, items, color_map)

def spectrum_html(ordered, counts, color_map, suffix="") -> str:
    chips = []
    for n in ordered:
        c = counts.get(n, 0)
        bg = color_map[n] if c else color_map[n] + "40"
        label = f"{c}{suffix}" if c else "none"
        chips.append(f'<div class="ag-chip" style="background:{bg}"><b>{n}</b><span>{label}</span></div>')
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

def steps_html(color_map, ordered, kind="photo") -> str:
    c = [color_map[ordered[i]] for i in (0, 4, 7)] if len(ordered) >= 8 else ["#9FD3C7", "#A294E4", "#E79AA9"]
    if kind == "video":
        texts = [
            ("Add a video", "Upload a short clip. Up to 20 seconds works best."),
            ("Detect", "Every frame is scanned and each person is followed, so labels stay steady."),
            ("Get the video", "Watch it here or download a copy with age groups drawn on every person."),
        ]
    else:
        texts = [
            ("Add a photo", "Upload a JPG or PNG from your device."),
            ("Detect", "The model finds each person in the photo and marks them with a box."),
            ("Read the result", "See the age group and a confidence score for every detection."),
        ]
    icons = [ICON_UPLOAD, ICON_SCAN, ICON_CHART]
    cards = "".join(
        f'<div class="ag-step"><div class="ag-ico" style="background:{col}">{icon}</div><b>{t}</b><span>{d}</span></div>'
        for icon, col, (t, d) in zip(icons, c, texts)
    )
    return f'<div class="ag-steps">{cards}</div>'

# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
MAX_SECONDS = 20   # longer videos are trimmed to keep processing quick
MAX_SIDE = 720     # frames are scaled down so their longest side is at most this
# --- STABILITY FIX: low-confidence votes ko ignore karo ---
MIN_VOTE_CONF = 0.40   # 40% se kam wali prediction vote nahi karegi
TRACK_MAX_MISSED = 30  # banda 30 frames tak ghayab ho to bhi usi ID pe rakho

def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

def dedupe(dets, thr=0.5):
    """Keep one box per face: drop a box that overlaps a more confident one."""
    kept = []
    for d in sorted(dets, key=lambda d: d[2], reverse=True):
        if all(iou_xyxy(d[0], k[0]) < thr for k in kept):
            kept.append(d)
    return kept

class SimpleTracker:
    """Follows each person across frames. Every person (track) collects confidence-weighted
    votes for an age group, so the final label reflects the whole video, not one frame."""

    def __init__(self, min_iou=0.10, max_missed=TRACK_MAX_MISSED):
        self.min_iou, self.max_missed = min_iou, max_missed
        self.tracks, self.done, self._next = [], [], 1

    def _score(self, t_box, d_box):
        v = iou_xyxy(t_box, d_box)
        if v >= self.min_iou:
            return 1.0 + v
        # fast movers: also accept a box whose centre is close to the previous one
        dx = ((t_box[0] + t_box[2]) - (d_box[0] + d_box[2])) / 2
        dy = ((t_box[1] + t_box[3]) - (d_box[1] + d_box[3])) / 2
        dist = (dx * dx + dy * dy) ** 0.5
        limit = 0.75 * max(t_box[2] - t_box[0], t_box[3] - t_box[1], 1.0)
        return 1.0 - dist / limit if dist < limit else 0.0

    def update(self, dets):
        """dets: list of (box, class_id, confidence). Returns list of (box, track_id, confidence)."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                sc = self._score(t["box"], d[0])
                if sc > 0:
                    pairs.append((sc, ti, di))
        pairs.sort(reverse=True)

        used_t, assigned = set(), {}
        for _, ti, di in pairs:
            if ti in used_t or di in assigned:
                continue
            used_t.add(ti)
            assigned[di] = self.tracks[ti]
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["missed"] += 1

        out = []
        for di, (box, cls, conf) in enumerate(dets):
            t = assigned.get(di)
            if t is None:
                t = {"id": self._next, "box": box, "hits": 0, "missed": 0, "votes": {}, "n": {}}
                self._next += 1
                self.tracks.append(t)
            t["box"], t["missed"] = box, 0
            t["hits"] += 1
            # FIX: sirf sure predictions hi vote karengi, 28% wali nahi
            if conf >= MIN_VOTE_CONF:
                t["votes"][cls] = t["votes"].get(cls, 0.0) + conf
                t["n"][cls] = t["n"].get(cls, 0) + 1
            out.append((box, t["id"], conf))

        alive = []
        for t in self.tracks:
            (self.done if t["missed"] > self.max_missed else alive).append(t)
        self.tracks = alive
        return out

    def final(self, min_hits=3):
        """track id -> (winning age group id, its average confidence). Very short tracks are ignored."""
        res = {}
        for t in self.tracks + self.done:
            if t["hits"] >= min_hits and t["votes"]:
                cls = max(t["votes"], key=t["votes"].get)
                res[t["id"]] = (cls, t["votes"][cls] / t["n"][cls])
        return res

class VideoOut:
    """Writes an H.264 mp4 that browsers can play. Falls back to mp4v (download only)."""

    def __init__(self, path, fps, size):
        self.path, self.fps, self.size = path, fps, size
        self.playable, self._w, self._cv = True, None, None
        try:
            import imageio.v2 as imageio
            self._w = imageio.get_writer(
                path, format="FFMPEG", mode="I", fps=fps, codec="libx264", quality=7,
                pixelformat="yuv420p", macro_block_size=1, ffmpeg_log_level="error",
                ffmpeg_params=["-movflags", "+faststart"],
            )
        except Exception:
            self._fallback()

    def _fallback(self):
        self._w, self.playable = None, False
        self._cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)

    def write(self, rgb):
        if self._w is not None:
            try:
                self._w.append_data(rgb)
                return
            except Exception:
                self._fallback()
        self._cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self._w is not None:
            self._w.close()
        if self._cv is not None:
            self._cv.release()

def read_frames(path, max_frames):
    """Yield (index, BGR frame) scaled down, with even width and height."""
    cap = cv2.VideoCapture(path)
    try:
        idx = 0
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                return
            h, w = frame.shape[:2]
            scale = min(1.0, MAX_SIDE / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            yield idx, frame[: h - h % 2, : w - w % 2]      # H.264 needs even sizes
            idx += 1
    finally:
        cap.release()

def process_video(in_path, out_path, model, names, color_map, conf, iou, stride, on_progress=None):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError("This video could not be opened. Try an MP4 file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    max_frames = int(fps * MAX_SECONDS)
    target = min(total, max_frames) if total > 0 else max_frames

    def report(frac):
        if on_progress:
            on_progress(min(frac, 1.0))

    # Pass 1: find faces and follow every person through the video.
    tracker, per_frame, current, n = SimpleTracker(), [], [], 0
    for idx, frame in read_frames(in_path, max_frames):
        if idx % stride == 0:
            res = model.predict(frame, conf=conf, iou=iou, agnostic_nms=True, verbose=False)[0]  # frame is BGR, as YOLO expects
            dets = []
            if res.boxes is not None:
                for b in res.boxes:
                    box = tuple(float(v) for v in b.xyxy[0].tolist())
                    dets.append((box, int(b.cls[0]), float(b.conf[0])))
            current = [(box, tid) for box, tid, _ in tracker.update(dedupe(dets))]
        per_frame.append(current)
        n = idx + 1
        if idx % 3 == 0:
            report(0.8 * n / max(target, 1))
    if n == 0:
        raise ValueError("No frames could be read from this video.")

    # Each person gets ONE age group for the whole video: the one with the most confidence votes.
    checks = -(-n // stride)
    final = tracker.final(min_hits=3 if checks >= 8 else 1)

    # Pass 2: draw the steady labels and write the video.
    out = None
    try:
        for idx, frame in read_frames(in_path, n):
            if idx >= len(per_frame):
                break
            items = []
            for box, tid in per_frame[idx]:
                if tid in final:
                    cls, cf = final[tid]
                    items.append((*box, names[cls], cf))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            drawn = np.asarray(draw_boxes(Image.fromarray(rgb), items, color_map))
            if out is None:
                out = VideoOut(out_path, fps, (frame.shape[1], frame.shape[0]))
            out.write(drawn)
            if idx % 3 == 0:
                report(0.8 + 0.2 * (idx + 1) / n)
    finally:
        if out is not None:
            out.close()
    if out is None:
        raise ValueError("No frames could be read from this video.")

    groups = {}
    for cls, _ in final.values():
        groups[names[cls]] = groups.get(names[cls], 0) + 1
    return {
        "people": sum(groups.values()), "groups": groups, "seconds": n / fps,
        "trimmed": (total > max_frames) if total > 0 else n >= max_frames,
        "playable": out.playable,
    }

def render_video_panel():
    vid = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed", key="video_up"
    )
    if vid is None:
        st.markdown(steps_html(color_map, ordered, "video"), unsafe_allow_html=True)
        st.markdown(
            '<p class="ag-fine">Clips up to 20 seconds work best, longer videos are trimmed. '
            "Processing runs on a shared CPU, so a short clip can take about a minute. "
            "Your video is deleted from the server as soon as processing ends.</p>",
            unsafe_allow_html=True,
        )
        return

    key = (vid.name, vid.size, conf_thres, iou_thres, stride)
    run = st.button("Detect ages in video", type="primary", key="run_video")

    if run:
        suffix = Path(vid.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(vid.getbuffer())
        tmp.close()
        out_path = tmp.name + "_out.mp4"
        bar = st.progress(0.0, text="Starting...")
        try:
            summary = process_video(
                tmp.name, out_path, model, names, color_map, conf_thres, iou_thres, stride,
                on_progress=lambda f: bar.progress(f, text=f"Detecting ages... {int(f * 100)}%"),
            )
            st.session_state.video_result = {
                "key": key, "summary": summary, "data": Path(out_path).read_bytes(),
            }
        except Exception as e:
            st.session_state.video_result = None
            st.error(f"Could not process this video. {e}")
        finally:
            bar.empty()
            for f in (tmp.name, out_path):
                try:
                    os.remove(f)
                except OSError:
                    pass

    res = st.session_state.get("video_result")
    ready = bool(res) and res["key"] == key

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        st.video(vid)
        st.markdown('<p class="ag-cap">Your video</p>', unsafe_allow_html=True)
    with right:
        if ready and res["summary"]["playable"]:
            st.video(res["data"], format="video/mp4")
            st.markdown('<p class="ag-cap">Detected age groups</p>', unsafe_allow_html=True)
        elif ready:
            st.markdown(
                '<div class="ag-hint">Your result is ready, but this server cannot preview it. '
                "Use the download button to watch it.</div>",
                unsafe_allow_html=True,
            )
        else:
            msg = ("Settings changed. Press Detect ages in video to run it again."
                   if res else "Press Detect ages in video to see the result here.")
            st.markdown(f'<div class="ag-hint" style="margin-top:0">{msg}</div>', unsafe_allow_html=True)

    if ready:
        s = res["summary"]
        st.download_button(
            "Download video", data=res["data"], file_name="age-detection-result.mp4",
            mime="video/mp4", key="video_dl",
        )
        if s["people"] == 0:
            st.markdown(
                '<div class="ag-hint">No people were found. Try a clearer clip, or lower '
                "Minimum confidence in the sidebar.</div>",
                unsafe_allow_html=True,
            )
        else:
            n = s["people"]
            st.markdown(
                f'<div class="ag-count">About {n} {"person" if n == 1 else "people"}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(spectrum_html(ordered, s["groups"], color_map), unsafe_allow_html=True)
        note = f"Processed {s['seconds']:.0f} seconds of video."
        if s["trimmed"]:
            note += f" Only the first {MAX_SECONDS} seconds were used."
        st.markdown(f'<p class="ag-fine">{note}</p>', unsafe_allow_html=True)

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
        "<div><span>Created by</span><b>Areeba Luqman and Mubashira Shahid</b></div>"
        "<div><span>Project</span><b>BSCS Final Year Project</b></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="ag-side-title">Age groups</p>', unsafe_allow_html=True)
    legend = "".join(f'<span style="background:{color_map[n]}">{n}</span>' for n in ordered)
    st.markdown(f'<div class="ag-legend">{legend}</div>', unsafe_allow_html=True)

    st.markdown('<p class="ag-side-title">Detection settings</p>', unsafe_allow_html=True)
    conf_thres = st.slider(
        "Minimum confidence", 0.05, 0.90, 0.40, 0.05,
        help="Lower it to catch more faces. Raise it to keep only the surest predictions.",
    )
    iou_thres = st.slider(
        "Overlap filter", 0.10, 0.90, 0.45, 0.05,
        help="Controls how overlapping boxes around the same face are merged.",
    )
    stride = st.slider(
        "Video: check every N frames", 1, 5, 2, 1,
        help="Higher is faster, but boxes update less often. Only used for videos.",
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
strip = "".join(f'<i style="background:{color_map[n]}"></i>' for n in ordered)
st.markdown(
    '<div class="ag-hero"><h1>Age Group Detection</h1>'
    "<p>Upload a photo or a short video. The model finds each person and estimates an age range.</p>"
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

    if st.session_state.mode not in ("upload", "video"):
        st.session_state.mode = "upload"

    b1, b2, _ = st.columns([2, 2, 3], gap="small")
    for col, mode_id, label in ((b1, "upload", "Upload a photo"), (b2, "video", "Upload a video")):
        with col:
            st.button(
                label, key=f"mode_{mode_id}", on_click=set_mode, args=(mode_id,),
                type="primary" if st.session_state.mode == mode_id else "secondary",
            )
    use_video = st.session_state.mode == "video"

    source = None
    if use_video:
        render_video_panel()
    else:
        source = st.file_uploader(
            "Upload a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="photo_up"
        )

    if use_video:
        pass
    elif source is None:
        st.markdown(steps_html(color_map, ordered), unsafe_allow_html=True)
        st.markdown(
            '<p class="ag-fine">Clear, front-facing faces in good light work best. '
            "Photos are processed in memory and never saved.</p>",
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
            result = model.predict(image, conf=conf_thres, iou=iou_thres, agnostic_nms=True, verbose=False)[0]

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
    metrics = get_metrics(model)
    if metrics:
        cards = "".join(f'<div class="ag-metric"><span>{label}</span><b>{value:.2f}</b></div>' for label, value in metrics)
        st.markdown(f'<div class="ag-metrics">{cards}</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="ag-fine" style="margin-bottom:1.2rem">Validation scores saved with the model at its best training epoch.</p>',
            unsafe_allow_html=True,
        )

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

st.markdown(
    '<p class="ag-footer">Age Group Detection, a BSCS Final Year Project</p>',
    unsafe_allow_html=True,
)
