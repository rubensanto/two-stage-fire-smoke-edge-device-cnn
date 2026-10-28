"""
One PDF per premature alarm: the video frame is embedded as a raster at native resolution,
bounding boxes and labels are vector graphics (matplotlib patches/text, TrueType fonts).
Output: paper/figs/premature_alarms/vXX_<cause>.pdf
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configurations import MODELS, PAPER, VIDEOS  # noqa: E402

VIDS = VIDEOS / "positive"            # data/videos/positive or SAFERFIRE_VIDEOS/positive
RPI = MODELS / "saferfire"            # exact files used on the RPi5
OUT = PAPER / "figs" / "premature_alarms"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {"light": YOLO(str(RPI / "YOLOv8n_best.onnx"), task="detect"),
          "heavy": YOLO(str(RPI / "YOLOv12m_best.onnx"), task="detect")}
COLORS = {"fire": "#FF6A00", "smoke": "#1E6FFF"}

# (video id, file, trigger frame, model, conf used for drawing, output name)
# Videos 21 and 54 score 0.18/0.22 on x86 for the box the RPi5 run flagged above 0.25, so they are drawn at 0.15.
CASES = [
    (2,  "personal10",     0,    "heavy", 0.25, "v02_orange_objects"),
    (6,  "personal17",     50,   "light", 0.25, "v06_blurred_object"),
    (16, "poss(3)_crop1",  104,  "light", 0.25, "v16_turnout_gear"),
    (17, "poss(3)_crop2",  95,   "light", 0.25, "v17_turnout_gear"),
    (21, "poss(7)_crop2",  40,   "light", 0.15, "v21_decorative_lamp"),
    (23, "poss(9)_crop",   117,  "light", 0.25, "v23_table_lamp"),
    (35, "poss(17)_crop1", 2477, "light", 0.25, "v35_glass_glare"),
    (47, "poss(26)_crop",  915,  "heavy", 0.25, "v47_glass_reflection"),
    (51, "poss(30)_crop",  168,  "light", 0.25, "v51_scba_cylinder"),
    (54, "poss(32)_crop1", 104,  "light", 0.15, "v54_glass_reflection"),
]

for vid, name, frame, kind, conf, out in CASES:
    cap = cv2.VideoCapture(str(VIDS / f"{name}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    cap.release()
    assert ok, (vid, frame)
    r = MODELS[kind].predict(bgr, conf=conf, imgsz=640, verbose=False)[0]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # crop the black letterbox bars contained in the video itself; boxes are shifted accordingly
    gray = rgb.mean(axis=2)
    rows = (gray.mean(axis=1) > 12).nonzero()[0]
    cols = (gray.mean(axis=0) > 12).nonzero()[0]
    oy, ox = int(rows[0]), int(cols[0])
    rgb = rgb[oy:int(rows[-1]) + 1, ox:int(cols[-1]) + 1]
    h, w = rgb.shape[:2]

    fig = plt.figure(figsize=(w / 100, h / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="none")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    lw = max(1.5, w / 320)
    fs = max(9, w / 55)
    dets = []
    for b in r.boxes:
        cls = r.names[int(b.cls)]
        c = float(b.conf)
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1, x2 = max(0, x1 - ox), min(w, x2 - ox)
        y1, y2 = max(0, y1 - oy), min(h, y2 - oy)
        col = COLORS.get(cls, "#FFFFFF")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, linewidth=lw))
        # label above the box, or inside it when the box touches the top edge
        ty, va = (y1 - lw, "bottom") if y1 > fs * 1.6 else (y1 + lw, "top")
        text = f"{cls} {c:.2f}"
        # approx. label width in pixels (figure dpi 100 -> 100/72 px per pt); right-align near the right edge
        tw = len(text) * fs * 0.62 * 100 / 72
        tx, ha = (x1, "left") if x1 + tw < w else (x2, "right")
        ax.text(tx, ty, text, color="white", fontsize=fs, fontweight="bold", va=va, ha=ha,
                bbox=dict(facecolor=col, edgecolor="none", pad=1.5))
        dets.append(f"{cls} {c:.2f}")
    fig.savefig(OUT / f"{out}.pdf")
    plt.close(fig)
    print(f"V{vid:02d} frame {frame} {kind}@{conf}: {dets} -> {out}.pdf")
