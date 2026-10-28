"""
One PDF per negative-video false alarm, in the same format as make_pdf_figures.py (raster frame at
native resolution, vector boxes and labels). Inference with onnxruntime on the exact RPi5 models,
with the ultralytics pre/post-processing (letterbox 640, RGB/255, NMS IoU 0.7 per class), so no torch
is needed.

The frame drawn is, within the window that ends at the recorded alarm frame and starts one inference
time (plus 0.5 s) earlier, the frame with the highest score of the model that raised the alarm.
Output: paper/figs/negative_false_alarms/nXX_<cause>.pdf

Usage: python make_neg_false_alarm_figures.py [--check]   (--check reproduces two positive-video panels)
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import onnxruntime as ort
import pandas as pd

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configurations import DATA, MODELS, NEGATIVE_RESULTS, PAPER, VIDEOS  # noqa: E402

# Videos are not in the repository: data/videos/{positive,negative}/ or the folder set in SAFERFIRE_VIDEOS.
NEG = VIDEOS / "negative"
POS = VIDEOS / "positive"
OUT = PAPER / "figs" / "negative_false_alarms"
DETAILS = NEGATIVE_RESULTS
MANIFEST = DATA / "negative_videos.csv"

SESS = {"light": ort.InferenceSession(str(MODELS / "saferfire" / "YOLOv8n_best.onnx")),
        "heavy": ort.InferenceSession(str(MODELS / "saferfire" / "YOLOv12m_best.onnx"))}
T_INF = {"light": 0.184, "heavy": 1.371}
NAMES = {0: "smoke", 1: "fire"}
COLORS = {"fire": "#FF6A00", "smoke": "#1E6FFF"}


def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def nms(boxes, scores, iou=0.7):
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ov = inter / (a(boxes[i:i + 1]) + a(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][ov < iou]
    return keep


def detect(kind, bgr, conf):
    img, r, left, top = letterbox(bgr)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
    out = SESS[kind].run(None, {"images": x})[0][0].T           # (8400, 4 + nc)
    cls_scores = out[:, 4:]
    cls = cls_scores.argmax(1)
    sc = cls_scores.max(1)
    m = sc >= conf
    if not m.any():
        return []
    b, sc, cls = out[m, :4], sc[m], cls[m]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    dets = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for k in nms(xyxy[idx] + c * 7680, sc[idx]):
            j = idx[k]
            x1, y1, x2, y2 = xyxy[j]
            dets.append((NAMES[int(c)], float(sc[j]),
                         ((x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r)))
    return sorted(dets, key=lambda d: -d[1])


def read(path, frame):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    cap.release()
    assert ok, (path, frame)
    return bgr


def draw(bgr, dets, out):
    """Same drawing as make_pdf_figures.py."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
    for cls, c, (x1, y1, x2, y2) in dets:
        x1, x2 = max(0, x1 - ox), min(w, x2 - ox)
        y1, y2 = max(0, y1 - oy), min(h, y2 - oy)
        col = COLORS.get(cls, "#FFFFFF")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, linewidth=lw))
        ty, va = (y1 - lw, "bottom") if y1 > fs * 1.6 else (y1 + lw, "top")
        text = f"{cls} {c:.2f}"
        tw = len(text) * fs * 0.62 * 100 / 72
        tx, ha = (x1, "left") if x1 + tw < w else (x2, "right")
        ax.text(tx, ty, text, color="white", fontsize=fs, fontweight="bold", va=va, ha=ha,
                bbox=dict(facecolor=col, edgecolor="none", pad=1.5))
    fig.savefig(out)
    plt.close(fig)


# (N, configuration (sheet name) in saferfire_false_alarms.xlsx, model that raised the alarm, its threshold, output name)
CASES = [
    (57, "TS_cL0.25_cH0.25_W1T1", "heavy", 0.25, "n57_steam_geothermal"),
    (32, "TS_cL0.25_cH0.25_W1T1", "heavy", 0.25, "n32_dust_vehicle"),
    (65, "TS_cL0.25_cH0.25_W1T1", "heavy", 0.25, "n65_backlit_clouds"),
    (36, "BO_c0.75", "heavy", 0.75, "n36_fog_headlights"),
    (59, "BO_c0.75", "heavy", 0.75, "n59_smokestack_plume"),
    (4, "LO_c0.25_W1T1", "light", 0.25, "n04_warm_object"),
]


def main():
    if "--check" in sys.argv:
        for name, frame, kind in [("poss(3)_crop1", 104, "light"), ("poss(26)_crop", 915, "heavy")]:
            print(name, [(c, round(s, 2)) for c, s, _ in detect(kind, read(POS / f"{name}.mp4", frame), 0.25)])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    det = pd.read_excel(DETAILS, "details")
    files = {int(r.id): r.file for r in pd.read_csv(MANIFEST).itertuples()}
    for n, exp, kind, conf, name in CASES:
        f = files[n]
        row = det[(det.experiment == exp) & (det.video == f)].iloc[0]
        assert bool(row.alarm), (n, exp)
        end = int(row.alarm_frame)
        start = max(0, int(end - (T_INF[kind] + 0.5) * row.fps))
        best = (-1, None, None)
        for fr in range(start, end + 1):
            d = detect(kind, read(NEG / f, fr), conf)
            if d and d[0][1] > best[0]:
                best = (d[0][1], fr, d)
        assert best[1] is not None, f"N{n}: no detection >= {conf} in frames {start}-{end}"
        draw(read(NEG / f, best[1]), best[2], OUT / f"{name}.pdf")
        print(f"N{n} {exp}: frame {best[1]} (alarm {end}) {[(c, round(s, 2)) for c, s, _ in best[2]]} -> {name}.pdf")


if __name__ == "__main__":
    main()
