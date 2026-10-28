from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


import cv2
import pandas as pd
from openpyxl import load_workbook
from ultralytics import YOLO

import numpy as np
import psutil
import csv
import subprocess
import os


# ============================================================
# CONFIGURATION
# ============================================================

# Expected folder structure:
#
# dataset/
# ├── poss(1).mp4
# ├── poss(2).mp4
# ├── poss(3).mp4
# └── firstframes/
#     ├── poss(1).txt
#     ├── poss(2).txt
#     └── poss(3).txt

# Paths (relative to the repository, independent of the folder the script is launched from):
#   models/                          models (Git LFS)
#   data/videos/positive/*.mp4       positive videos  (not versioned; or set SAFERFIRE_VIDEOS)
#   data/videos/negative/*.mp4       negative videos
#   data/onset_frames/<video>.txt    annotated onset frame of each positive video
#   runs/<script name>/              output of the runs (not versioned); add it to results/ with
#                                    analysis/import_runs.py
REPO = Path(__file__).resolve().parents[1]
VIDEOS = Path(os.environ.get("SAFERFIRE_VIDEOS", REPO / "data" / "videos"))
DATASET_DIR = VIDEOS / "positive"

VIDEO_DIR = DATASET_DIR
FIRST_FRAMES_DIR = REPO / "data" / "onset_frames"

# Output
OUTPUT_DIR = REPO / "runs" / "saferfire_positive"
EXCEL_PATH = OUTPUT_DIR / "results.xlsx"


# Models
LIGHT_MODEL_PATH = str(REPO / "models" / "saferfire" / "YOLOv8n_best.onnx")
HEAVY_MODEL_PATH = str(REPO / "models" / "saferfire" / "YOLOv12m_best.onnx")

# Number of repetitions for each video/configuration
N_RUNS = 6
RUNS_PER_BATCH = 2
# YOLO image size
IMG_SIZE = 640

# None = automatic
# Examples:
# DEVICE = "cpu"
# DEVICE = "cuda:0"
DEVICE = None

# IMPORTANT:
# Change these to match the class names in your models.
TARGET_CLASSES = {"fire", "smoke"}



# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

# ------------------------------------------------------------
# APPROACH 1: LIGHT MODEL ONLY
# ------------------------------------------------------------

# Set of confidence thresholds to run:
#   "main"  : 0.25 and 0.50   (Light Only, Best Only, Two-Step with c_H >= c_L)
#   "extra" : 0.125 and 0.75  (the additional thresholds of the revision: every
#             Two-Step combination with c_H >= c_L not covered by "main")
#   "all"   : both
THRESHOLD_SET = "main"
_THRESHOLDS = {"main": [0.25, 0.50], "extra": [0.125, 0.75], "all": [0.125, 0.25, 0.50, 0.75]}[THRESHOLD_SET]
_NM = [(1, 1), (3, 2), (5, 3)]

LIGHT_CONFIGS = [{"conf": c, "N": n, "M": m} for c in _THRESHOLDS for (n, m) in _NM]


# ------------------------------------------------------------
# APPROACH 2: HEAVY MODEL ONLY
# ------------------------------------------------------------

HEAVY_CONFIGS = [{"conf": c} for c in _THRESHOLDS]


# ------------------------------------------------------------
# APPROACH 3: LIGHT -> HEAVY
# ------------------------------------------------------------

# every pair with heavy_conf >= light_conf; "extra" = the pairs that involve 0.125 or 0.75
_ALL_PAIRS = [(cl, ch) for cl in [0.125, 0.25, 0.50, 0.75] for ch in [0.125, 0.25, 0.50, 0.75] if ch >= cl]
_PAIRS = {
    "main": [p for p in _ALL_PAIRS if set(p) <= {0.25, 0.50}],
    "extra": [p for p in _ALL_PAIRS if not set(p) <= {0.25, 0.50}],
    "all": _ALL_PAIRS,
}[THRESHOLD_SET]

TWO_STEP_CONFIGS = [
    {"light_conf": cl, "N": n, "M": m, "heavy_conf": ch}
    for (cl, ch) in _PAIRS for (n, m) in _NM
]



# ------------------------------------------------------------

# ----------------------------------------------------------------------
# SYSTEM LOG (temperature, memory, CPU %, GPU clock)
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
    """Return the Pi’s CPU temperature in °C using vcgencmd."""
    out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
    # Example output:  "temp=48.5'C\n"
    return float(out.split("=")[1].split("'")[0])


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
    Return the total CPU utilisation (0‑100 %).
    Using ``interval=None`` makes the value refer to the time since the
    previous call – exactly what we need for a periodic logger.
    """
    return psutil.cpu_percent(interval=None)


def read_gpu_clock_mhz() -> float:
    """
    Return the VideoCore‑GPU clock in MHz.
    ``vcgencmd measure_clock core`` yields Hz, e.g. ``core=500000000``.
    """
    out = subprocess.check_output(
        ["vcgencmd", "measure_clock", "core"]
    ).decode().strip()
    try:
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
    try:
        temp_c   = read_temperature()
    except Exception as e:
        temp_c = float("nan")
        print(f"[log] Temperature read error: {e}")

    try:
        mem_proc_mib = current_memory_mib()
    except Exception as e:
        mem_proc_mib = float("nan")
        print(f"[log] Process memory read error: {e}")
        
    try:
        mem_sys_mib = current_system_memory_mib()
    except Exception as e:
        mem_sys_mib = float("nan")
        print(f"[log] System memory (MiB) read error: {e}")
        
    try:
        mem_sys_prct = current_system_memory_percent()
    except Exception as e:
        mem_sys_prct = float("nan")
        print(f"[log] System memory (%) read error: {e}")

    try:
        cpu_pct = read_cpu_percent()
    except Exception as e:
        cpu_pct = float("nan")
        print(f"[log] CPU % read error: {e}")

    try:
        gpu_mhz = read_gpu_clock_mhz()
    except Exception as e:
        gpu_mhz = float("nan")
        print(f"[log] GPU clock read error: {e}")

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

    total_inference_time: float
    number_of_inferences: int

    light_inferences: int
    heavy_inferences: int


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
    Read the ground-truth first-fire frame.

    Example:
        firstframes/poss(1).txt

    Contents:
        153
    """

    txt_path = FIRST_FRAMES_DIR / f"{video_path.stem}.txt"

    if not txt_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {txt_path}"
        )

    text = txt_path.read_text().strip()

    match = re.search(r"-?\d+", text)

    if match is None:
        raise ValueError(
            f"Could not find a frame number in {txt_path}"
        )

    return int(match.group())


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


# ============================================================
# YOLO INFERENCE
# ============================================================

def run_yolo(
    model: YOLO,
    frame,
    confidence: float,
):
    """
    Run one YOLO inference.

    Returns:
        detected
        annotated_frame
        inference_time
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

    result = results[0]

    detected = False

    if result.boxes is not None and len(result.boxes) > 0:

        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        for cls_id in classes:

            cls_id = int(cls_id)

            class_name = str(
                names[cls_id]
            ).lower()

            if class_name in TARGET_CLASSES:
                detected = True
                break

    annotated_frame = result.plot()

    return (
        detected,
        annotated_frame,
        inference_time,
    )


# ============================================================
# LIVE STREAM SIMULATION
# ============================================================

def frame_from_time(
    virtual_time: float,
    fps: float,
) -> int:
    """
    Convert simulated live-stream time to frame number.

    Example:

        FPS = 30
        virtual_time = 0.10 seconds

        0.10 * 30 = frame 3
    """

    return max(
        0,
        int(math.floor(virtual_time * fps)),
    )



# ============================================================
# APPROACH 1
# LIGHT MODEL ONLY
# ============================================================

def approach_light_only(
    video_path: Path,
    model: YOLO,
    confidence: float,
    N: int,
    M: int,
    real_frame: int,
):
    """
    Light model only.

    Alarm when at least M of the last N inspected frames
    contain a fire/smoke detection.

    Example:

        N=3, M=2

    Detection history:

        [False, True, True] -> ALARM
        [True, False, True] -> ALARM
        [True, False, False] -> NO ALARM
    """

    fps, frame_count, _, _ = get_video_info(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # This is the simulated clock of the live stream.
    virtual_time = 0.0

    history = deque([False] * N, maxlen=N)

    total_inference_time = 0.0
    number_of_inferences = 0

    trigger_alarm_frame = None
    alarm_frame = None
    alarm_time = None

    annotated_frames = {}
    detection_history = {}
    inference_times = {}
    analysis_models = {}

    while True:

        # Determine which frame has arrived by this point
        # in the simulated live stream.
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

        # ----------------------------------------------------
        # Run light model
        # ----------------------------------------------------

        (
            detected,
            annotated,
            inference_time,
        ) = run_yolo(
            model=model,
            frame=frame,
            confidence=confidence,
        )

        number_of_inferences += 1
        total_inference_time += inference_time

        history.append(detected)



        # ----------------------------------------------------
        # N/M temporal logic
        # ----------------------------------------------------

        if (
            len(history) >= N
            and sum(history) >= M
        ):
            # Frame that was actually analyzed and triggered
            # the alarm.
            trigger_alarm_frame = current_frame

            # Alarm is only known after inference finishes.
            alarm_time = (
                virtual_time
                + inference_time
            )

            # Frame visible when the inference finishes.
            alarm_frame = frame_from_time(
                alarm_time,
                fps,
            )

            break

        # ----------------------------------------------------
        # Advance virtual live-stream clock
        # ----------------------------------------------------

        virtual_time += inference_time

        # ---- periodic logging -------------------------------------------------
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
        total_inference_time=total_inference_time,
        number_of_inferences=number_of_inferences,
        light_inferences=number_of_inferences,
        heavy_inferences=0,
    )


# ============================================================
# APPROACH 2
# HEAVY MODEL ONLY
# ============================================================

def approach_heavy_only(
    video_path: Path,
    model: YOLO,
    confidence: float,
    real_frame: int,
):
    """
    Heavy model only.

    No N/M logic.

    Alarm immediately when the heavy model detects fire/smoke.
    """

    fps, frame_count, _, _ = get_video_info(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    virtual_time = 0.0

    total_inference_time = 0.0
    number_of_inferences = 0

    trigger_alarm_frame = None
    alarm_frame = None
    alarm_time = None

    annotated_frames = {}
    detection_history = {}
    inference_times = {}
    analysis_models = {}

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

        (
            detected,
            annotated,
            inference_time,
        ) = run_yolo(
            model=model,
            frame=frame,
            confidence=confidence,
        )

        number_of_inferences += 1
        total_inference_time += inference_time


        if detected:

            trigger_alarm_frame = current_frame

            alarm_time = (
                virtual_time
                + inference_time
            )

            # Frame visible when the inference finishes.
            alarm_frame = frame_from_time(
                alarm_time,
                fps,
            )

            break

        virtual_time += inference_time

        # ---- periodic logging -------------------------------------------------
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
        total_inference_time=total_inference_time,
        number_of_inferences=number_of_inferences,
        light_inferences=0,
        heavy_inferences=number_of_inferences,
    )


# ============================================================
# APPROACH 3
# LIGHT -> HEAVY
# ============================================================
def approach_two_step(
    video_path: Path,
    light_model: YOLO,
    heavy_model: YOLO,
    light_confidence: float,
    N: int,
    M: int,
    heavy_confidence: float,
    real_frame: int,
):
    """
    Two-step approach.

    1. Run light model.
    2. Apply N/M logic to light detections.
    3. If N/M triggers, run heavy model on the NEXT frame.
    4. Alarm if heavy model detects fire/smoke.

    Both light and heavy inference times are included
    in the simulated live-stream clock.
    """

    fps, frame_count, _, _ = get_video_info(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    virtual_time = 0.0

    history = deque([False] * N, maxlen=N)

    total_inference_time = 0.0

    light_inferences = 0
    heavy_inferences = 0

    trigger_alarm_frame = None
    alarm_frame = None
    alarm_time = None

    annotated_frames = {}
    detection_history = {}
    inference_times = {}
    analysis_models = {}

    # Heavy should run on the next frame
    heavy_pending = False

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

        # ====================================================
        # STEP 1: HEAVY MODEL
        # ====================================================

        if heavy_pending:

            (
                heavy_detected,
                heavy_annotated,
                heavy_time,
            ) = run_yolo(
                model=heavy_model,
                frame=frame,
                confidence=heavy_confidence,
            )

            heavy_inferences += 1
            total_inference_time += heavy_time


            virtual_time += heavy_time

            # ---- periodic logging -------------------------------------------------
            _maybe_log_metrics()

            # Reset trigger
            heavy_pending = False

            # ====================================================
            # ALARM
            # ====================================================

            if heavy_detected:

                trigger_alarm_frame = current_frame

                alarm_time = virtual_time

                alarm_frame = frame_from_time(
                    alarm_time,
                    fps,
                )

                break

            # Heavy did not detect fire.
            # Continue with the next frame.
            continue

        # ====================================================
        # STEP 2: LIGHT MODEL
        # ====================================================

        (
            light_detected,
            light_annotated,
            light_time,
        ) = run_yolo(
            model=light_model,
            frame=frame,
            confidence=light_confidence,
        )

        light_inferences += 1
        total_inference_time += light_time

        history.append(light_detected)


        virtual_time += light_time

        # ---- periodic logging -------------------------------------------------
        _maybe_log_metrics()

        # ====================================================
        # STEP 3: LIGHT N/M LOGIC
        # ====================================================

        light_triggered = (
            sum(history) >= M
        )

        if light_triggered:

            # IMPORTANT:
            # Do NOT run heavy on this frame.
            # Run it on the NEXT frame.
            heavy_pending = True

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
        total_inference_time=total_inference_time,
        number_of_inferences=(
            light_inferences
            + heavy_inferences
        ),
        light_inferences=light_inferences,
        heavy_inferences=heavy_inferences,
    )
# ============================================================
# EXPERIMENT NAMES
# ============================================================

def _c(x):
    # confidence threshold as in the sheet names: 0.125, 0.25, 0.50, 0.75
    return "0.125" if abs(x - 0.125) < 1e-9 else f"{x:.2f}"


# Sheet names (<= 31 characters, as in results/README.md): (W,T) = (N,M)
def light_name(
    confidence: float,
    N: int,
    M: int,
):
    return f"LO_c{_c(confidence)}_W{N}T{M}"


def heavy_name(
    confidence: float,
):
    return f"BO_c{_c(confidence)}"


def two_step_name(
    light_confidence: float,
    N: int,
    M: int,
    heavy_confidence: float,
):
    return f"TS_cL{_c(light_confidence)}_cH{_c(heavy_confidence)}_W{N}T{M}"


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

# -----------------------------------------------------------------
# Função principal – executa tudo em batches
# -----------------------------------------------------------------
def run_experiment(
    videos: list[Path],
    experiments: list[dict],
) -> dict[str, dict[str, list[float | None]]]:
    """
    Executa todas as combinações (configuração × vídeo) em batches de
    ``RUNS_PER_BATCH`` corridas cada um, salvando os resultados
    (metadata + matriz de latências) ao final **de cada batch**.
    O retorno tem o mesmo formato usado por ``save_excel``.
    """
    # ---------- containers ----------
    all_results: dict[str, dict[str, list[float | None]]] = {}
    all_metadata: dict[str, list[dict]] = {}

    for exp in experiments:
        # inicializa dicionários vazios
        all_results[exp["name"]] = {v.stem: [] for v in videos}
        all_metadata[exp["name"]] = []

    runs_per_batch = RUNS_PER_BATCH
    total_batches = (N_RUNS + runs_per_batch - 1) // runs_per_batch   # ceil

    for batch_idx in range(total_batches):
        batch_start = batch_idx * runs_per_batch + 1
        batch_end   = min((batch_idx + 1) * runs_per_batch, N_RUNS)

        print()
        print("-" * 80)
        print(f"BATCH {batch_idx + 1}/{total_batches} – "
              f"runs {batch_start}…{batch_end}")
        print("-" * 80)

        # ----------- percorrer vídeos → experimentos ------------
        for video in videos:
            video_name = video.stem
            real_frame = get_real_frame(video)

            print()
            print(f"Video: {video.name}")
            print(f"Ground‑truth frame: {real_frame}")

            for exp in experiments:
                exp_name    = exp["name"]
                run_fn      = exp["run_fn"]
                make_kwargs = exp["make_kwargs"]

                for run_number in range(batch_start, batch_end + 1):
                    print(
                        f"  {exp_name} – run {run_number}/{N_RUNS} "
                        f"(batch {batch_idx + 1})...",
                        end=" ",
                        flush=True,
                    )

                    # ---------- kwargs ----------
                    kwargs = make_kwargs(
                        video=video,
                        real_frame=real_frame,
                    )

                    # ---------- execução ----------
                    result: SimulationResult = run_fn(**kwargs)

                    # ---------- armazenar ----------
                    all_results[exp_name][video_name].append(
                        result.latency_seconds
                    )

                    all_metadata[exp_name].append(
                        {
                            "experiment": exp_name,
                            "video": video.name,
                            "run": run_number,
                            "real_frame": result.real_frame,
                            "alarm_frame": result.alarm_frame,
                            "fps": result.fps,
                            "latency_seconds": result.latency_seconds,
                            "alarm": result.alarm,
                            "alarm_time_seconds": result.alarm_time,
                            "total_inference_time_seconds": result.total_inference_time,
                            "number_of_inferences": result.number_of_inferences,
                            "light_inferences": result.light_inferences,
                            "heavy_inferences": result.heavy_inferences,
                        }
                    )

                    if result.alarm:
                        print(
                            f"alarm_frame={result.alarm_frame}, "
                            f"latency={result.latency_seconds:.3f}s, "
                            f"inferences={result.number_of_inferences}"
                        )
                    else:
                        print("NO ALARM")

        # ---------- SALVAR INTERMEDIÁRIO ao fim do batch ----------
        print("\nSaving intermediate Excel for this batch …")

        _write_intermediate_excel(
            all_results=all_results,
        )

        print("  → done.\n")


    return all_results

# ============================================================
# EXCEL
# ============================================================

def results_to_dataframe(results: dict):
    """
    Convert:

        {
            "poss(1)": [1.2, 1.3, 1.4],
            "poss(2)": [0.8, 0.9, 1.0]
        }

    into:

        run | poss(1) | poss(2)
        -------------------------
        1   | 1.2     | 0.8
        2   | 1.3     | 0.9
        3   | 1.4     | 1.0
    """

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


PARTIAL_EXCEL_PATH = OUTPUT_DIR / "partial_results.xlsx"


def _write_intermediate_excel(
    all_results: dict[str, dict[str, list[float | None]]],
) -> None:
    """
    Save all currently available results to one Excel file.

    File:
        experimental_results/partial_results.xlsx

    Each experiment gets its own sheet, using the same layout as
    the final results.xlsx.

    The file is overwritten on every call, so it always contains
    everything processed so far.
    """

    PARTIAL_EXCEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        PARTIAL_EXCEL_PATH,
        engine="openpyxl",
    ) as writer:

        used_sheet_names = set()

        for experiment_name, results in all_results.items():

            sheet_name = sanitize_filename(
                experiment_name
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

    # Format Excel
    workbook = load_workbook(PARTIAL_EXCEL_PATH)

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

    workbook.save(PARTIAL_EXCEL_PATH)

    print(
        f"  → Partial results saved to: "
        f"{PARTIAL_EXCEL_PATH.resolve()}"
    )



def save_excel(all_results: dict):
    """
    Create one Excel sheet per experiment.
    """

    EXCEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        EXCEL_PATH,
        engine="openpyxl",
    ) as writer:

        used_sheet_names = set()

        for experiment_name, results in (
            all_results.items()
        ):

            sheet_name = sanitize_filename(
                experiment_name
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

    # Format Excel
    workbook = load_workbook(EXCEL_PATH)

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

    workbook.save(EXCEL_PATH)

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
    print("FIRE DETECTION EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------
    print()
    print("Loading light model...")
    light_model = YOLO(LIGHT_MODEL_PATH)

    print("Loading heavy model...")
    heavy_model = YOLO(HEAVY_MODEL_PATH)

    # --------------------------------------------------------
    # FIND VIDEOS
    # --------------------------------------------------------
    videos = get_video_files()
    print()
    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  - {v.name}")

    print()
    print(f"Runs per configuration: {N_RUNS}")
    print(f"Runs per batch:        {RUNS_PER_BATCH}")

    # --------------------------------------------------------
    # WARM‑UP MODELS
    # --------------------------------------------------------
    print()
    print("Warming up models...")
    cap = cv2.VideoCapture(str(videos[0]))
    ret, frame = cap.read()
    cap.release()
    if ret:
        run_yolo(light_model, frame, 0.25)
        run_yolo(heavy_model, frame, 0.25)
    print("Warm‑up completed.")

    # --------------------------------------------------------
    # PREPARAR LISTA DE EXPERIMENTOS
    # --------------------------------------------------------
    experiments: list[dict] = []

    # ---- APPROACH 1 (light only) ---------------------------------
    for cfg in LIGHT_CONFIGS:
        name = light_name(cfg["conf"], cfg["N"], cfg["M"])

        def make_kwargs(video, real_frame, cfg=cfg):
            return {
                "video_path": video,
                "model": light_model,
                "confidence": cfg["conf"],
                "N": cfg["N"],
                "M": cfg["M"],
                "real_frame": real_frame,
            }

        experiments.append(
            {
                "name": name,
                "run_fn": approach_light_only,
                "make_kwargs": make_kwargs,
            }
        )

    # ---- APPROACH 2 (heavy only) ---------------------------------
    for cfg in HEAVY_CONFIGS:
        name = heavy_name(cfg["conf"])

        def make_kwargs(video, real_frame, cfg=cfg):
            return {
                "video_path": video,
                "model": heavy_model,
                "confidence": cfg["conf"],
                "real_frame": real_frame,
            }

        experiments.append(
            {
                "name": name,
                "run_fn": approach_heavy_only,
                "make_kwargs": make_kwargs,
            }
        )

    # ---- APPROACH 3 (two‑step) -----------------------------------
    for cfg in TWO_STEP_CONFIGS:
        name = two_step_name(
            cfg["light_conf"], cfg["N"], cfg["M"], cfg["heavy_conf"]
        )

        def make_kwargs(video, real_frame, cfg=cfg):
            return {
                "video_path": video,
                "light_model": light_model,
                "heavy_model": heavy_model,
                "light_confidence": cfg["light_conf"],
                "N": cfg["N"],
                "M": cfg["M"],
                "heavy_confidence": cfg["heavy_conf"],
                "real_frame": real_frame,
            }

        experiments.append(
            {
                "name": name,
                "run_fn": approach_two_step,
                "make_kwargs": make_kwargs,
            }
        )

    # --------------------------------------------------------
    # EXECUTAR TODOS OS EXPERIMENTOS EM BATCHES
    # --------------------------------------------------------
    all_results = run_experiment(
        videos=videos,
        experiments=experiments,
    )

    # --------------------------------------------------------
    # SALVAR EXCEL (mesmo código de antes)
    # --------------------------------------------------------
    save_excel(all_results)

    print()
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

