"""
Fire/smoke experiment on a simulated live stream (Raspberry Pi 5).

Pipeline under test: arXiv 2607.03131, sec. III-G
(YOLOv10 + 5-level validation cascade). Only this pipeline is run.

NEGATIVE videos (no fire): same pipeline, configurations and simulated
live stream as dumitru_positive.py. Each video is streamed until the
first alarm or the end of the video.
Result = 1 if a (false) alarm is raised anywhere in the clip, 0 otherwise
(the clip-level false-alarm metric of the paper, FIRESENSE test).
"""

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import psutil
from openpyxl import load_workbook
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Paths (relative to the repository, independent of the folder the script is launched from):
#   models/                          models (Git LFS)
#   data/videos/positive/*.mp4       positive videos  (not versioned; or set SAFERFIRE_VIDEOS)
#   data/videos/negative/*.mp4       negative videos
#   data/onset_frames/<video>.txt    annotated onset frame of each positive video
#   runs/<script name>/              output of the runs (not versioned); add it to results/ with
#                                    analysis/import_runs.py
REPO = Path(__file__).resolve().parents[1]
VIDEOS = Path(os.environ.get("SAFERFIRE_VIDEOS", REPO / "data" / "videos"))
DATASET_DIR = VIDEOS / "negative"

VIDEO_DIR = DATASET_DIR

# Output
OUTPUT_DIR = REPO / "runs" / "dumitru_negative"
EXCEL_PATH = OUTPUT_DIR / "results.xlsx"
PARTIAL_EXCEL_PATH = OUTPUT_DIR / "partial_results.xlsx"

# Single model, as in arXiv:2607.03131, sec. III-G ("pre-trained YOLOv10";
# variant and training data not given: YOLOv10n trained on our data)
MODEL_PATH = str(REPO / "models" / "dumitru" / "YOLOv10n_best.pt")

# Number of repetitions for each video/configuration
# (1 run, as for the other methods on the negative videos)
N_RUNS = 1
RUNS_PER_BATCH = 1

# YOLO image size
IMG_SIZE = 640

# None = automatic
# Examples:
# DEVICE = "cpu"
# DEVICE = "cuda:0"
DEVICE = None


# ------------------------------------------------------------
# PAPER PIPELINE  (arXiv 2607.03131, sec. III-G)
#
#   1) per-class confidence   (paper: fire 0.40, smoke 0.55)
#   2) size plausibility      (paper: 0.3% .. 85% of frame area)
#   3) HSV colour check       (paper: "heuristic", NO thresholds given)
#   4) IoU-tracking temporal confirmation
#                             (paper: "several consecutive frames",
#                              N not given)
#   5) per-location alert cooldown
#
# "Consecutive frames" = consecutive INSPECTED frames. With a slow
# model on the Pi, two inspected frames can be several video frames
# apart, so K frames cost K * (inference time) of stream time.
# ------------------------------------------------------------

PAPER_DEFAULTS = {
    "fire_conf": 0.40,         # level 1 (from the paper)
    "smoke_conf": 0.55,        # level 1 (from the paper)
    "use_size": True,          # level 2
    "min_area": 0.003,         # level 2 (from the paper: 0.3% of frame)
    "max_area": 0.85,          # level 2 (from the paper: 85% of frame)
    "use_hsv": True,           # level 3
    "confirm_frames": 3,       # level 4 (NOT in the paper -> sweep it)
    "iou_thr": 0.30,           # level 4 (NOT in the paper)
    "max_missed": 0,           # level 4: 0 = strictly consecutive
    "cooldown_s": 10.0,        # level 5 (NOT in the paper)
}

# Level-3 HSV thresholds. THESE ARE PLACEHOLDERS: the paper only says
# "heuristic colour check in HSV". Calibrate them on your own videos.
# OpenCV scale: H in [0,179], S and V in [0,255].
#   fire : fraction of pixels that are red/orange/yellow and saturated/bright
#   smoke: fraction of pixels that are low-saturation (grey/white) and
#          neither almost black nor saturated-white
HSV_PARAMS = {
    "fire": {"h_max": 35, "h_min2": 170, "s_min": 70, "v_min": 120, "min_frac": 0.10},
    "smoke": {"s_max": 80, "v_min": 50, "v_max": 240, "min_frac": 0.50},
}
HSV_CROP_SIZE = 64  # box crop is resized to NxN before the HSV test (constant, tiny cost)

# Progressive ablation study.
# Every key not listed here comes from PAPER_DEFAULTS.
# The label is the sheet name (results/README.md): levels on + K.
PAPER_CONFIGS = [
    {"label": "raw_K1",      "use_size": False, "use_hsv": False, "confirm_frames": 1},
    {"label": "size_K1",     "use_size": True,  "use_hsv": False, "confirm_frames": 1},
    {"label": "size_hsv_K1", "use_size": True,  "use_hsv": True,  "confirm_frames": 1},
    {"label": "size_K3",     "use_size": True,  "use_hsv": False, "confirm_frames": 3},
    {"label": "full_K3",     "use_size": True,  "use_hsv": True,  "confirm_frames": 3},
    {"label": "full_K5",     "use_size": True,  "use_hsv": True,  "confirm_frames": 5},
]


# ----------------------------------------------------------------------
# SYSTEM LOG (temperature, memory, CPU %, GPU clock)
# ----------------------------------------------------------------------
SYSTEM_LOG_PATH = OUTPUT_DIR / "system_log.csv"

LOG_INTERVAL = 60 * 30
_last_log_time = time.time()
_log_measurement_number = 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Write header once
if not SYSTEM_LOG_PATH.exists():
    with SYSTEM_LOG_PATH.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["measurement", "timestamp", "temp_c", "process_memory_mib", "system_memory_mib", "system_memory_percent", "cpu_percent", "gpu_mhz"]
        )

_system_log_file = SYSTEM_LOG_PATH.open(
    "a",
    newline="",
    encoding="utf-8",
)

_system_log_writer = csv.writer(
    _system_log_file
)


def read_temperature() -> float:
    """Return the Pi's CPU temperature in °C using vcgencmd."""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return float("nan")


def current_memory_mib() -> float:
    """Resident‑set‑size of this Python process in MiB."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 2)


def current_system_memory_mib() -> float:
    """ Return total system RAM currently in use, in MiB"""
    memory = psutil.virtual_memory()
    return memory.used / (1024 ** 2)


def current_system_memory_percent() -> float:
    """Return the percentage of total system RAM currently in use."""
    return psutil.virtual_memory().percent


def read_cpu_percent() -> float:
    """
    Return the total CPU utilisation (0‑100 %).
    Using ``interval=None`` makes the value refer to the time since the
    previous call – exactly what we need for a periodic logger.
    """
    return psutil.cpu_percent(interval=None)


def read_gpu_clock_mhz() -> float:
    """
    Return the VideoCore‑GPU clock in MHz.
    ``vcgencmd measure_clock core`` yields Hz, e.g. ``core=500000000``.
    """
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_clock", "core"]
        ).decode().strip()
        hz = int(out.split("=")[1])
        return hz / 1_000_000   # MHz
    except Exception:
        return float("nan")


def _maybe_log_metrics():
    """Write a CSV line if LOG_INTERVAL seconds have passed."""
    global _last_log_time, _log_measurement_number

    now = time.time()
    if now - _last_log_time < LOG_INTERVAL:
        return                # not time yet

    _log_measurement_number += 1
    # ------------------------------------------------------------------
    # 1️⃣  Gather the metrics
    # ------------------------------------------------------------------
    temp_c = read_temperature()
    mem_proc_mib = current_memory_mib()
    mem_sys_mib = current_system_memory_mib()
    mem_sys_prct = current_system_memory_percent()
    cpu_pct = read_cpu_percent()
    gpu_mhz = read_gpu_clock_mhz()

    print(
        "\n" + "=" * 40 +
        "\nSystem Metrics"
        "\n" + "-" * 40 +
        f"\nTemperature = {temp_c:.2f} °C"
        f"\nProcess Memory = {mem_proc_mib:.2f} MiB"
        f"\nSystem Memory = {mem_sys_mib:.2f} MiB"
        f"\nSystem Memory = {mem_sys_prct:.2f} %"
        f"\nCPU Usage = {cpu_pct:.1f} %"
        f"\nGPU Clock = {gpu_mhz:.1f} MHz"
        "\n" + "=" * 40
    )

    # ------------------------------------------------------------------
    # 2️⃣  Write a line to system_log.csv
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    _system_log_writer.writerow(
        [
            _log_measurement_number,
            timestamp,
            f"{temp_c:.2f}",
            f"{mem_proc_mib:.2f}",
            f"{mem_sys_mib:.2f}",
            f"{mem_sys_prct:.2f}",
            f"{cpu_pct:.1f}",
            f"{gpu_mhz:.1f}",
        ]
    )
    _system_log_file.flush()

    try:
        os.fsync(_system_log_file.fileno())
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3️⃣  Keep the original timestamp for the next interval
    # ------------------------------------------------------------------
    _last_log_time = now


# ============================================================
# RESULT CLASS
# ============================================================

@dataclass
class SimulationResult:
    alarm: bool
    alarm_frame: int | None
    alarm_time: float | None

    real_frame: int
    fps: float

    latency_seconds: float | None

    number_of_inferences: int
    total_inference_time: float     # YOLO predict() only
    postprocess_time: float         # parsing + levels 2-5 (filters, tracking)


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a folder/file name."""
    return re.sub(r"[^\w\-.]+", "_", name)


def get_video_files() -> list[Path]:
    """Find all MP4 videos in the dataset folder."""

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        raise FileNotFoundError(
            f"No .mp4 files found in {VIDEO_DIR.resolve()}"
        )

    return videos


def get_real_frame(video_path: Path) -> int:
    """
    Negative videos have no fire, so there is no ground-truth frame.

    Returning 0 means latency_seconds becomes the time (in seconds
    since the start of the video) at which a FALSE alarm was raised.
    """

    return 0


def get_video_info(video_path: Path):
    """
    Get FPS, number of frames, width and height.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise ValueError(
            f"Invalid FPS for {video_path}: {fps}"
        )

    return fps, frame_count, width, height


def frame_from_time(
    virtual_time: float,
    fps: float,
) -> int:
    """
    Convert simulated live-stream time to frame number.
    """

    return max(
        0,
        int(math.floor(virtual_time * fps)),
    )


# ============================================================
# YOLO INFERENCE
# ============================================================

def run_yolo_raw(
    model: YOLO,
    frame,
    confidence: float,
):
    """
    Run one YOLO inference and return the raw Ultralytics result
    (boxes, classes, confidences).
    """

    start = time.perf_counter()

    kwargs = {
        "source": frame,
        "conf": confidence,
        "imgsz": IMG_SIZE,
        "verbose": False,
    }

    if DEVICE is not None:
        kwargs["device"] = DEVICE

    results = model.predict(**kwargs)

    inference_time = time.perf_counter() - start

    return results[0], inference_time


# ============================================================
# PAPER PIPELINE
# ============================================================

@dataclass
class Detection:
    cls: str                                   # "fire" or "smoke"
    conf: float
    box: tuple[float, float, float, float]     # x1, y1, x2, y2 (pixels)


@dataclass
class Track:
    cls: str
    box: tuple[float, float, float, float]
    hits: int = 1      # consecutive inspected frames in which it was seen
    missed: int = 0


def canonical_class(name: str) -> str | None:
    """
    Map the model's class name to "fire" / "smoke".
    """
    n = name.lower()
    if "smoke" in n:
        return "smoke"
    if "fire" in n or "flame" in n:
        return "fire"
    return None


def iou_xyxy(a, b) -> float:
    """Compute IoU between two bounding boxes in xyxy format."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )

    return inter / union if union > 0 else 0.0


class IoUTracker:
    """
    Greedy IoU tracker (same class only).
    """

    def __init__(self, iou_thr: float, max_missed: int):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.tracks: list[Track] = []

    def update(self, dets: list[Detection]) -> list[Track]:
        """Match detections, age the rest. Returns tracks seen now."""

        updated: list[Track] = []
        unmatched = set(range(len(self.tracks)))

        for det in sorted(dets, key=lambda d: -d.conf):

            best_i = None
            best_iou = self.iou_thr

            for i in unmatched:

                track = self.tracks[i]

                if track.cls != det.cls:
                    continue

                value = iou_xyxy(track.box, det.box)

                if value >= best_iou:
                    best_i = i
                    best_iou = value

            if best_i is None:
                track = Track(cls=det.cls, box=det.box)
                self.tracks.append(track)
            else:
                unmatched.discard(best_i)
                track = self.tracks[best_i]
                track.box = det.box
                track.hits += 1
                track.missed = 0

            updated.append(track)

        for i in unmatched:
            self.tracks[i].missed += 1

        self.tracks = [
            t for t in self.tracks
            if t.missed <= self.max_missed
        ]

        return updated


class PaperPipeline:
    """
    The 5-level cascade of arXiv 2607.03131 (sec. III-G).
    """

    def __init__(self, cfg: dict, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.frame_w = float(frame_w)
        self.frame_h = float(frame_h)
        self.frame_area = self.frame_w * self.frame_h

        self.class_conf = {
            "fire": cfg["fire_conf"],
            "smoke": cfg["smoke_conf"],
        }

        self.tracker = IoUTracker(cfg["iou_thr"], cfg["max_missed"])

        # (cls, cx_norm, cy_norm, time) of past alerts
        self.alerts: list[tuple[str, float, float, float]] = []

    @property
    def predict_conf(self) -> float:
        """Confidence passed to YOLO: the lowest per-class threshold."""
        return min(self.class_conf.values())

    # ---- level 1: per-class confidence ---------------------------
    def extract(self, result) -> list[Detection]:

        dets: list[Detection] = []

        if result.boxes is None or len(result.boxes) == 0:
            return dets

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, conf, cls_id in zip(xyxy, confs, clss):

            cls_name = canonical_class(str(names[int(cls_id)]))

            if cls_name is None:
                continue

            if conf < self.class_conf[cls_name]:
                continue

            dets.append(
                Detection(
                    cls=cls_name,
                    conf=float(conf),
                    box=tuple(float(v) for v in box),
                )
            )

        return dets

    # ---- level 2: size plausibility ------------------------------
    def _size_ok(self, det: Detection) -> bool:

        w = max(0.0, det.box[2] - det.box[0])
        h = max(0.0, det.box[3] - det.box[1])

        fraction = (w * h) / self.frame_area

        return (
            self.cfg["min_area"] <= fraction <= self.cfg["max_area"]
        )

    # ---- level 3: HSV colour check -------------------------------
    def _color_ok(self, frame, det: Detection) -> bool:

        x1 = max(0, int(det.box[0]))
        y1 = max(0, int(det.box[1]))
        x2 = min(frame.shape[1], int(det.box[2]))
        y2 = min(frame.shape[0], int(det.box[3]))

        if x2 - x1 < 2 or y2 - y1 < 2:
            return False

        crop = cv2.resize(
            frame[y1:y2, x1:x2],
            (HSV_CROP_SIZE, HSV_CROP_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]

        p = HSV_PARAMS[det.cls]

        if det.cls == "fire":
            mask = (
                ((h <= p["h_max"]) | (h >= p["h_min2"]))
                & (s >= p["s_min"])
                & (v >= p["v_min"])
            )
        else:
            mask = (
                (s <= p["s_max"])
                & (v >= p["v_min"])
                & (v <= p["v_max"])
            )

        return float(mask.mean()) >= p["min_frac"]

    # ---- level 5: per-location cooldown --------------------------
    def _in_cooldown(self, track: Track, now: float) -> bool:

        cx = (track.box[0] + track.box[2]) / 2 / self.frame_w
        cy = (track.box[1] + track.box[3]) / 2 / self.frame_h

        for cls_name, ax, ay, t in self.alerts:

            if (
                cls_name == track.cls
                and now - t < self.cfg["cooldown_s"]
                and math.hypot(cx - ax, cy - ay) < 0.10
            ):
                return True

        return False

    def _register_alert(self, track: Track, now: float):

        cx = (track.box[0] + track.box[2]) / 2 / self.frame_w
        cy = (track.box[1] + track.box[3]) / 2 / self.frame_h

        self.alerts.append((track.cls, cx, cy, now))

    # ---- full cascade for one inspected frame --------------------
    def process(self, frame, result, now: float) -> bool:
        """
        Returns True when an alarm is raised on this frame.
        """

        dets = self.extract(result)                                   # 1

        if self.cfg["use_size"]:                                      # 2
            dets = [d for d in dets if self._size_ok(d)]

        if self.cfg["use_hsv"]:                                       # 3
            dets = [d for d in dets if self._color_ok(frame, d)]

        tracks = self.tracker.update(dets)                            # 4

        for track in tracks:

            if track.hits < self.cfg["confirm_frames"]:
                continue

            if self._in_cooldown(track, now):                         # 5
                continue

            self._register_alert(track, now)

            return True

        return False


def run_paper_pipeline(
    video_path: Path,
    model: YOLO,
    cfg: dict,
    real_frame: int,
):
    """
    Simulate the paper pipeline on one video.
    """

    fps, frame_count, width, height = get_video_info(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    pipeline = PaperPipeline(cfg, width, height)

    virtual_time = 0.0

    total_inference_time = 0.0
    total_postprocess_time = 0.0
    number_of_inferences = 0

    alarm_frame = None
    alarm_time = None

    while True:

        current_frame = frame_from_time(
            virtual_time,
            fps,
        )

        if current_frame >= frame_count:
            break

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            current_frame,
        )

        ret, frame = cap.read()

        if not ret:
            break

        result, inference_time = run_yolo_raw(
            model=model,
            frame=frame,
            confidence=pipeline.predict_conf,
        )

        number_of_inferences += 1
        total_inference_time += inference_time

        start = time.perf_counter()

        alarm = pipeline.process(
            frame=frame,
            result=result,
            now=virtual_time + inference_time,
        )

        postprocess_time = time.perf_counter() - start

        total_postprocess_time += postprocess_time

        step_time = inference_time + postprocess_time

        if alarm:

            alarm_time = virtual_time + step_time

            alarm_frame = frame_from_time(
                alarm_time,
                fps,
            )

            break

        next_frame_time = (current_frame + 1) / fps + 1e-9

        virtual_time = max(
            virtual_time + step_time,
            next_frame_time,
        )

        _maybe_log_metrics()

    cap.release()

    if alarm_frame is not None:
        latency = (
            alarm_frame - real_frame
        ) / fps
    else:
        latency = None

    return SimulationResult(
        alarm=alarm_frame is not None,
        alarm_frame=alarm_frame,
        alarm_time=alarm_time,
        real_frame=real_frame,
        fps=fps,
        latency_seconds=latency,
        number_of_inferences=number_of_inferences,
        total_inference_time=total_inference_time,
        postprocess_time=total_postprocess_time,
    )


def experiment_name(cfg: dict) -> str:
    # sheet name, the same as in dumitru_positive.py
    return cfg["label"]


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

def run_experiment(
    videos: list[Path],
    experiments: list[dict],
    model: YOLO,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, list[dict]]]:
    """
    Negative videos: stores 1 = false alarm, 0 = no alarm (the desired outcome).
    Runs every (configuration x video) combination in batches and saves
    the intermediate Excel at the end of each batch.
    Returns (all_results, all_metadata).
    """
    all_results: dict[str, dict[str, list[int]]] = {}
    all_metadata: dict[str, list[dict]] = {}

    for exp in experiments:
        all_results[exp["name"]] = {v.stem: [] for v in videos}
        all_metadata[exp["name"]] = []

    runs_per_batch = RUNS_PER_BATCH
    total_batches = (N_RUNS + runs_per_batch - 1) // runs_per_batch

    for batch_idx in range(total_batches):
        batch_start = batch_idx * runs_per_batch + 1
        batch_end   = min((batch_idx + 1) * runs_per_batch, N_RUNS)

        print()
        print("-" * 80)
        print(f"BATCH {batch_idx + 1}/{total_batches} – "
              f"runs {batch_start}…{batch_end}")
        print("-" * 80)

        for video in videos:
            video_name = video.stem
            real_frame = get_real_frame(video)

            print()
            print(f"Video: {video.name}")

            for exp in experiments:
                exp_name = exp["name"]

                for run_number in range(batch_start, batch_end + 1):
                    print(
                        f"  {exp_name} – run {run_number}/{N_RUNS} "
                        f"(batch {batch_idx + 1})...",
                        end=" ",
                        flush=True,
                    )

                    result: SimulationResult = run_paper_pipeline(
                        video_path=video,
                        model=model,
                        cfg=exp["cfg"],
                        real_frame=real_frame,
                    )

                    all_results[exp_name][video_name].append(
                        int(result.alarm)
                    )

                    all_metadata[exp_name].append(
                        {
                            "experiment": exp_name,
                            "video": video.name,
                            "run": run_number,
                            "alarm": result.alarm,
                            "alarm_frame": result.alarm_frame,
                            "fps": result.fps,
                            # real_frame = 0, so this is the second of the video
                            # at which the false alarm happened
                            "false_alarm_at_seconds": result.latency_seconds,
                            "alarm_time_seconds": result.alarm_time,
                            "number_of_inferences": result.number_of_inferences,
                            "total_inference_time_seconds": result.total_inference_time,
                            "postprocess_time_seconds": result.postprocess_time,
                        }
                    )

                    if result.alarm:
                        print(
                            f"FALSE ALARM at frame {result.alarm_frame} "
                            f"({result.latency_seconds:.3f}s), "
                            f"inferences={result.number_of_inferences}"
                        )
                    else:
                        print(
                            f"no alarm (ok), "
                            f"inferences={result.number_of_inferences}"
                        )

        print("\nSaving intermediate Excel for this batch …")

        _write_excel(all_results, all_metadata, PARTIAL_EXCEL_PATH)

        print("  → done.\n")

    return all_results, all_metadata


# ============================================================
# EXCEL
# ============================================================

def results_to_dataframe(results: dict):
    max_runs = max(
        len(values)
        for values in results.values()
    )

    data = {
        "run": list(
            range(
                1,
                max_runs + 1,
            )
        )
    }

    for video_name, values in results.items():

        padded = (
            values
            + [None]
            * (
                max_runs
                - len(values)
            )
        )

        data[video_name] = padded

    return pd.DataFrame(data)


def _write_summary_sheets(
    writer,
    all_results: dict[str, dict[str, list[int]]],
    all_metadata: dict[str, list[dict]],
) -> None:
    """
    Write two extra sheets at the start of the workbook:

        summary: one row per experiment with the false-alarm rate
                 (lower is better; ideal = 0 false alarms)
        details: one row per (experiment, video, run) with the
                 frame/second of the false alarm and inference counts
    """

    rows = []

    for experiment_name_, results in all_results.items():

        values = [
            v
            for video_values in results.values()
            for v in video_values
        ]

        if not values:
            continue

        false_alarms = sum(values)

        rows.append(
            {
                "experiment": experiment_name_,
                "runs": len(values),
                "false_alarms": false_alarms,
                "no_alarm": len(values) - false_alarms,
                "false_alarm_rate": false_alarms / len(values),
            }
        )

    pd.DataFrame(rows).to_excel(
        writer,
        sheet_name="summary",
        index=False,
    )

    pd.DataFrame(
        [
            row
            for experiment_rows in all_metadata.values()
            for row in experiment_rows
        ]
    ).to_excel(
        writer,
        sheet_name="details",
        index=False,
    )


def _write_excel(
    all_results: dict[str, dict[str, list[int]]],
    all_metadata: dict[str, list[dict]],
    path: Path,
) -> None:
    """
    summary + details sheets, then one sheet per experiment
    (1 = false alarm, 0 = no alarm).
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
    ) as writer:

        _write_summary_sheets(writer, all_results, all_metadata)

        used_sheet_names = {"summary", "details"}

        for experiment_name_, results in all_results.items():

            sheet_name = sanitize_filename(
                experiment_name_
            )[:31]

            original_name = sheet_name
            counter = 1

            while sheet_name in used_sheet_names:

                suffix = f"_{counter}"

                sheet_name = (
                    original_name[
                        :31 - len(suffix)
                    ]
                    + suffix
                )

                counter += 1

            used_sheet_names.add(sheet_name)

            df = results_to_dataframe(results)

            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    workbook = load_workbook(path)

    for worksheet in workbook.worksheets:

        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:

            max_length = 0

            column_letter = (
                column_cells[0].column_letter
            )

            for cell in column_cells:

                if cell.value is not None:
                    max_length = max(
                        max_length,
                        len(str(cell.value)),
                    )

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                max_length + 2,
                30,
            )

    workbook.save(path)

    print(
        f"  → Excel saved to: "
        f"{path.resolve()}"
    )


def save_excel(all_results: dict, all_metadata: dict):
    """Final Excel: summary, details and one sheet per experiment."""

    _write_excel(all_results, all_metadata, EXCEL_PATH)

    print()
    print("=" * 80)
    print("Excel saved to:")
    print(EXCEL_PATH.resolve())
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("FIRE DETECTION EXPERIMENT – DUMITRU & SPINU (arXiv 2607.03131), negative videos")
    print("=" * 80)

    # --------------------------------------------------------
    # BUILD EXPERIMENT LIST
    # --------------------------------------------------------
    configs = [{**PAPER_DEFAULTS, **c} for c in PAPER_CONFIGS]

    names = [experiment_name(c) for c in configs]

    if len(set(names)) != len(names):
        raise ValueError(
            "Duplicate experiment names. Give each PAPER_CONFIG a unique "
            f"label: {names}"
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    print()
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    experiments = [
        {
            "name": experiment_name(c),
            "cfg": c,
        }
        for c in configs
    ]

    # --------------------------------------------------------
    # FIND VIDEOS
    # --------------------------------------------------------
    videos = get_video_files()
    print()
    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  - {v.name}")

    print()
    print(f"Experiments:           {len(experiments)}")
    print(f"Runs per configuration: {N_RUNS}")
    print(f"Runs per batch:        {RUNS_PER_BATCH}")

    # --------------------------------------------------------
    # WARM-UP MODEL
    # --------------------------------------------------------
    print()
    print("Warming up model...")
    cap = cv2.VideoCapture(str(videos[0]))
    ret, frame = cap.read()
    cap.release()
    if ret:
        run_yolo_raw(model, frame, 0.25)
    print("Warm-up completed.")

    # --------------------------------------------------------
    # RUN EVERYTHING IN BATCHES
    # --------------------------------------------------------
    all_results, all_metadata = run_experiment(
        videos=videos,
        experiments=experiments,
        model=model,
    )

    # --------------------------------------------------------
    # SAVE FINAL EXCEL
    # --------------------------------------------------------
    save_excel(all_results, all_metadata)

    print()
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
