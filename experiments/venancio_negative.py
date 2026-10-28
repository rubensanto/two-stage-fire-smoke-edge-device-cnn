from __future__ import annotations

import os
os.environ["YOLO_AUTOUPDATE"] = "0"

import math
import re
import sys
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


# ============================================================
# CONFIGURATION
# ============================================================

# Expected folder structure (NEGATIVE videos: no fire, no ground truth):
#
# neg_videos/
# ├── old_01_firesense.mp4
# ├── new_pexels_fog_night_avenue_15722036.mp4
# └── ...

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
OUTPUT_DIR = REPO / "runs" / "venancio_negative"
EXCEL_PATH = OUTPUT_DIR / "results.xlsx"


# Venancio et al. model (YOLOv5l fine-tuned on our training data, exported to ONNX)
VENANCIO_MODEL_PATH = str(REPO / "models" / "venancio" / "yolov5_venancio.onnx")

# area variation tracker of the original repository (temporal/tracker.py, MIT license)
VENANCIO_REPO_PATH = str(Path(__file__).resolve().parent / "third_party" / "venancio")
if VENANCIO_REPO_PATH not in sys.path:
    sys.path.append(VENANCIO_REPO_PATH)

from temporal.tracker import ObjectTracker, Log

# Number of repetitions for each video/configuration
N_RUNS = 1
RUNS_PER_BATCH = 1
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
# APPROACH: VENANCIO (YOLO + Temporal Tracker)
# ------------------------------------------------------------

# Same 4 configs as the positive-video run (launch_simulation_fire-detec.py)
VENANCIO_CONFIGS = [
    {"conf": 0.25, "area_thresh": 0.05, "window_size": 20},
    {"conf": 0.25, "area_thresh": 0.10, "window_size": 30},
    {"conf": 0.40, "area_thresh": 0.05, "window_size": 20},
    {"conf": 0.50, "area_thresh": 0.05, "window_size": 30},
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
# NOVA: inferência YOLO que devolve também as bounding boxes
# (necessário para alimentar o tracker do Venâncio)
# ============================================================

def run_yolo_with_boxes(
    model: YOLO,
    frame,
    confidence: float,
):
    """
    Run one YOLO inference and return bounding boxes for the tracker.

    Returns:
        boxes_xyxy  -> lista de [xmin, ymin, xmax, ymax]
        class_ids   -> lista de int
        conf_scores -> lista de float
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

    boxes_xyxy = []
    class_ids = []
    conf_scores = []

    if result.boxes is not None and len(result.boxes) > 0:

        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        names = result.names

        for i in range(len(xyxy)):

            cls_id = int(cls[i])
            class_name = str(names[cls_id]).lower()

            # Só considera classes de interesse (fire / smoke)
            if class_name in TARGET_CLASSES:
                boxes_xyxy.append([
                    int(xyxy[i][0]),
                    int(xyxy[i][1]),
                    int(xyxy[i][2]),
                    int(xyxy[i][3]),
                ])
                class_ids.append(cls_id)
                conf_scores.append(float(conf[i]))

    return (
        boxes_xyxy,
        class_ids,
        conf_scores,
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
# APPROACH: VENANCIO (YOLO + Temporal Tracker)
# ============================================================

def approach_venancio(
    video_path: Path,
    model: YOLO,
    confidence: float,
    area_thresh: float,
    window_size: int,
    real_frame: int,
):
    """
    Venâncio approach: YOLO detector + ObjectTracker
    (Area Variation Technique).

    1. Run YOLO on each frame.
    2. Feed detected boxes to the ObjectTracker.
    3. The tracker suppresses boxes whose area does not grow
       over the sliding window (static false positives).
    4. Alarm when at least one valid (non-suppressed) detection
       remains.
    """

    fps, frame_count, _, _ = get_video_info(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # This is the simulated clock of the live stream.
    virtual_time = 0.0

    total_inference_time = 0.0
    number_of_inferences = 0

    trigger_alarm_frame = None
    alarm_frame = None
    alarm_time = None

    # Initialize Venâncio temporal tracker
    tracker = ObjectTracker(area_thresh, window_size)
    log = Log()

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

        # ----------------------------------------------------
        # Run YOLO and get boxes
        # ----------------------------------------------------

        (
            boxes_xyxy,
            class_ids,
            conf_scores,
            inference_time,
        ) = run_yolo_with_boxes(
            model=model,
            frame=frame,
            confidence=confidence,
        )

        number_of_inferences += 1
        total_inference_time += inference_time

        # ----------------------------------------------------
        # Feed detections into the tracker
        # ----------------------------------------------------

        if len(boxes_xyxy) > 0:

            for bbox, cls_id, conf in zip(
                boxes_xyxy, class_ids, conf_scores
            ):
                log.coord_objs.append(bbox)
                log.classes.append(cls_id)
                log.conf_scores.append(conf)

            # Object tracking (area variation over the window)
            centroids, areas = tracker.tracking(log.coord_objs)
            log.centroids, log.areas = log.update(centroids, areas)

            # Indices of detections that should be suppressed
            idxs = tracker.bbox_suppression(log)

            # Number of valid (non-suppressed) detections
            valid_detections = len(boxes_xyxy) - len(idxs)

            # Clear per-frame lists
            log.coord_objs.clear()
            log.classes.clear()
            log.conf_scores.clear()

        else:
            valid_detections = 0

        # ----------------------------------------------------
        # Alarm decision
        # ----------------------------------------------------

        if valid_detections > 0:

            trigger_alarm_frame = current_frame

            alarm_time = (
                virtual_time
                + inference_time
            )

            alarm_frame = frame_from_time(
                alarm_time,
                fps,
            )

            break

        # ----------------------------------------------------
        # Advance virtual live-stream clock
        # ----------------------------------------------------

        virtual_time += inference_time

        # ---- periodic logging ----
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
# EXPERIMENT NAMES
# ============================================================

def venancio_name(
    confidence: float,
    area_thresh: float,
    window_size: int,
):
    # sheet name (<= 31 characters): AVT_c0.25_area0.05_win20
    return f"AVT_c{confidence:.2f}_area{area_thresh:.2f}_win{window_size}"


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

# -----------------------------------------------------------------
# Função principal – executa tudo em batches
# -----------------------------------------------------------------
def run_experiment(
    videos: list[Path],
    experiments: list[dict],
) -> tuple[dict[str, dict[str, list[int]]], dict[str, list[dict]]]:
    """
    Vídeos negativos: guarda 1 = falso alarme, 0 = sem alarme (o desejado).
    Retorna (all_results, all_metadata).

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
                            "total_inference_time_seconds": result.total_inference_time,
                            "number_of_inferences": result.number_of_inferences,
                            "light_inferences": result.light_inferences,
                            "heavy_inferences": result.heavy_inferences,
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

        # ---------- SALVAR INTERMEDIÁRIO ao fim do batch ----------
        print("\nSaving intermediate Excel for this batch …")

        _write_intermediate_excel(
            all_results=all_results,
            all_metadata=all_metadata,
        )

        print("  → done.\n")


    return all_results, all_metadata

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

    for experiment_name, results in all_results.items():

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
                "experiment": experiment_name,
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


def _write_intermediate_excel(
    all_results: dict[str, dict[str, list[int]]],
    all_metadata: dict[str, list[dict]],
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

        _write_summary_sheets(writer, all_results, all_metadata)

        used_sheet_names = {"summary", "details"}

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



def save_excel(all_results: dict, all_metadata: dict):
    """
    Create summary + details sheets, then one Excel sheet per experiment
    (1 = false alarm, 0 = no alarm).
    """

    EXCEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        EXCEL_PATH,
        engine="openpyxl",
    ) as writer:

        _write_summary_sheets(writer, all_results, all_metadata)

        used_sheet_names = {"summary", "details"}

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
    print("FIRE DETECTION EXPERIMENT – VENANCIO APPROACH (negative videos)")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    print()
    print("Loading Venancio model...")
    venancio_model = YOLO(VENANCIO_MODEL_PATH, task="detect")

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
    # WARM‑UP MODEL
    # --------------------------------------------------------
    print()
    print("Warming up model...")
    cap = cv2.VideoCapture(str(videos[0]))
    ret, frame = cap.read()
    cap.release()
    if ret:
        run_yolo(venancio_model, frame, 0.25)
    print("Warm‑up completed.")

    # --------------------------------------------------------
    # PREPARAR LISTA DE EXPERIMENTOS
    # --------------------------------------------------------
    experiments: list[dict] = []

    # ---- VENANCIO APPROACH ---------------------------------
    for cfg in VENANCIO_CONFIGS:
        name = venancio_name(
            cfg["conf"], cfg["area_thresh"], cfg["window_size"]
        )

        def make_kwargs(video, real_frame, cfg=cfg):
            return {
                "video_path": video,
                "model": venancio_model,
                "confidence": cfg["conf"],
                "area_thresh": cfg["area_thresh"],
                "window_size": cfg["window_size"],
                "real_frame": real_frame,
            }

        experiments.append(
            {
                "name": name,
                "run_fn": approach_venancio,
                "make_kwargs": make_kwargs,
            }
        )

    # --------------------------------------------------------
    # EXECUTAR TODOS OS EXPERIMENTOS EM BATCHES
    # --------------------------------------------------------
    all_results, all_metadata = run_experiment(
        videos=videos,
        experiments=experiments,
    )

    # --------------------------------------------------------
    # SALVAR EXCEL (mesmo código de antes)
    # --------------------------------------------------------
    save_excel(all_results, all_metadata)

    print()
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

