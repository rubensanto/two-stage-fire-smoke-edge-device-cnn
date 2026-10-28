"""
Ideal latency of Light Only, Best Only and Two-Step on the positive videos, without running any model.

The simulated live stream is the one of experiments/saferfire_positive.py (virtual clock advanced by the inference
time, frame = floor(t * fps), latency = (alarm frame - onset frame) / fps), but the models are replaced by ideal
ones: a frame is detected if and only if it is at or after the annotated onset frame, and every inference takes the
mean time of Table 6 of the paper (YOLOv8n: t_L = 0.184 s, YOLOv12m: t_H = 1.371 s). There are therefore no
premature alarms, the confidence thresholds play no role, and only the temporal rule (W,T) matters. The result is
deterministic, one latency per video and approach (Table 8 and Figure 16c of the paper).

Usage: python experiments/ideal_latency.py [VIDEOS] [--onsets DIR] [--t-light 0.184] [--t-heavy 1.371] [--out CSV]
VIDEOS defaults to data/videos/positive (or $SAFERFIRE_VIDEOS/positive), DIR to data/onset_frames (onset frame of
each video in <video name>.txt) and CSV to results/positive_videos/ideal_latency.csv. Videos without an onset file
(negative videos) are skipped. The videos are only opened to read FPS and frame count.
"""
import argparse
import csv
import math
import os
import re
import statistics
from collections import deque
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
VIDEOS = Path(os.environ.get("SAFERFIRE_VIDEOS", REPO / "data" / "videos")) / "positive"
ONSETS = REPO / "data" / "onset_frames"
OUT_CSV = REPO / "results" / "positive_videos" / "ideal_latency.csv"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm", ".mpg", ".mpeg"}
WT_RULES = [(1, 1), (3, 2), (5, 3)]
APPROACHES = ([("LO", w, t) for w, t in WT_RULES] + [("BO", 1, 1)] + [("TS", w, t) for w, t in WT_RULES])


def approach_name(kind, w, t):
    return "Best Only" if kind == "BO" else f"{'Light Only' if kind == 'LO' else 'Two-Step'} ({w},{t})"


def frame_from_time(virtual_time, fps):
    return max(0, int(math.floor(virtual_time * fps)))


def simulate(kind, w, t, fps, frame_count, onset, t_light, t_heavy):
    """Alarm time (s) of one approach on one video, or None if the video ends first. Same loop as the RPi script."""
    virtual_time = 0.0
    history = deque([False] * w, maxlen=w)
    heavy_pending = False
    while True:
        current_frame = frame_from_time(virtual_time, fps)
        if current_frame >= frame_count:
            return None
        detected = current_frame >= onset   # ideal model

        if kind == "BO":
            if detected:
                return virtual_time + t_heavy
            virtual_time += t_heavy
        elif kind == "LO":
            history.append(detected)
            if sum(history) >= t:
                return virtual_time + t_light
            virtual_time += t_light
        else:   # Two-Step: the heavy model verifies the frame after the one that satisfied the (W,T) rule
            if heavy_pending:
                virtual_time += t_heavy
                heavy_pending = False
                if detected:
                    return virtual_time
                continue
            history.append(detected)
            virtual_time += t_light
            if sum(history) >= t:
                heavy_pending = True


def expected_latency(kind, t, t_light, t_heavy):
    """Eq. (expected_latency) of the paper: uniform waiting time for the first post-onset frame."""
    if kind == "BO":
        return 1.5 * t_heavy
    return (t + 0.5) * t_light + (t_heavy if kind == "TS" else 0.0)


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.stem)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="?", default=VIDEOS, type=Path)
    parser.add_argument("--onsets", type=Path, help="folder with <video name>.txt (default: data/onset_frames, or "
                                                    "VIDEOS/firstframe when VIDEOS is given and has that folder)")
    parser.add_argument("--t-light", type=float, default=0.184, help="inference time of YOLOv8n (s)")
    parser.add_argument("--t-heavy", type=float, default=1.371, help="inference time of YOLOv12m (s)")
    parser.add_argument("--out", default=OUT_CSV, type=Path)
    args = parser.parse_args()

    folder = args.videos
    onsets = args.onsets or (folder / "firstframe" if (folder / "firstframe").is_dir() else ONSETS)
    videos = sorted((p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS), key=natural_key)
    names = [approach_name(*a) for a in APPROACHES]
    rows, skipped = [], []

    for video in videos:
        onset_file = onsets / f"{video.stem}.txt"
        if not onset_file.exists():
            skipped.append(video.name)
            continue
        onset = int(re.search(r"-?\d+", onset_file.read_text()).group())
        cap = cv2.VideoCapture(str(video))
        fps, frame_count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0:
            raise RuntimeError(f"Invalid FPS for {video}: {fps}")

        latencies = []
        for kind, w, t in APPROACHES:
            alarm_time = simulate(kind, w, t, fps, frame_count, onset, args.t_light, args.t_heavy)
            latencies.append(None if alarm_time is None else (frame_from_time(alarm_time, fps) - onset) / fps)
        rows.append({"id": f"P{len(rows) + 1}", "video": video.name, "fps": fps, "frame_count": frame_count,
                     "onset_frame": onset, "onset_s": onset / fps, "latencies": latencies})

    if not rows:
        raise SystemExit(f"No videos with an onset file found in {folder.resolve()}")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "video", "fps", "frame_count", "onset_frame", "onset_s"] + names)
        for r in rows:
            writer.writerow([r["id"], r["video"], f"{r['fps']:.3f}", r["frame_count"], r["onset_frame"],
                             f"{r['onset_s']:.2f}"] + ["" if x is None else f"{x:.3f}" for x in r["latencies"]])

    print(f"t_L = {args.t_light} s, t_H = {args.t_heavy} s | {len(rows)} positive videos"
          + (f" | skipped (no onset file): {len(skipped)}" if skipped else ""))
    print(f"FPS values: {sorted({round(r['fps'], 3) for r in rows})}\n")
    print(f"{'Approach':<18}{'alarms':>8}{'median':>9}{'mean':>9}{'min':>9}{'max':>9}{'Eq. (s)':>10}")
    for i, (kind, w, t) in enumerate(APPROACHES):
        values = [r["latencies"][i] for r in rows if r["latencies"][i] is not None]
        stats = (f"{statistics.median(values):9.3f}{statistics.mean(values):9.3f}{min(values):9.3f}{max(values):9.3f}"
                 if values else f"{'-':>9}" * 4)
        print(f"{names[i]:<18}{len(values):>5}/{len(rows):<2}{stats}"
              f"{expected_latency(kind, t, args.t_light, args.t_heavy):10.3f}")

    # how often Two-Step beats Best Only, per video
    bo = names.index("Best Only")
    for i, name in enumerate(names):
        if name.startswith("Two-Step"):
            pairs = [(r["latencies"][i], r["latencies"][bo]) for r in rows
                     if None not in (r["latencies"][i], r["latencies"][bo])]
            faster = sum(a < b for a, b in pairs)
            equal = sum(a == b for a, b in pairs)
            print(f"{name} vs Best Only: faster in {faster}, equal in {equal}, slower in {len(pairs) - faster - equal}")
    print(f"\nPer-video latencies (s) written to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
