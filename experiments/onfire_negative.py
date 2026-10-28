from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
import torch
from openpyxl import load_workbook

import sys

# OnFire network definitions (third_party/onfire_models, from the original repository)
sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
from onfire_models import nasnet_mobile_onfire, shufflenetv2  # noqa: E402


# ============================================================
# BASELINE: ShuffleNetV2-OnFire / NasNet-A-OnFire
# ============================================================
#
# Thomson, Bhowmik and Breckon, "Efficient and Compact Convolutional
# Neural Network Architectures for Non-temporal Real-time Fire
# Detection", ICMLA 2020.
# Code and model definitions (MIT license):
#   https://github.com/NeelBhowmik/efficient-compact-fire-detection-cnn
#
# The two modes of the paper are run separately, as published, each
# frame being classified on its own (non-temporal):
#
#   onfire_ff       full-frame {fire, no-fire} classifier
#                   (shufflenet_ff.pt / nasnet_ff.pt)
#
#   onfire_sp       superpixel localisation: SLIC + per-superpixel
#                   {fire, no-fire} classifier on every frame
#                   (shufflenet_sp.pt / nasnet_sp.pt)
#
# The alarm is raised on the first inspected frame classified as fire
# (full-frame: P(fire) >= 0.5; superpixel: at least one superpixel with
# P(fire) >= 0.5). Frames are read with the same simulated live stream
# as launch_simulation_batches.py (frames arriving during an inference
# are skipped).
#
# NOTE: the OnFire models detect FIRE only (no smoke class).
#
# NEGATIVE VIDEOS (no fire): every alarm is a FALSE alarm.
# Same output format as launch_simulation_batches_neg.py:
#   summary sheet (false-alarm rate), details sheet (one row per run)
#   and one sheet per experiment with 1 = false alarm, 0 = no alarm.


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
OUTPUT_DIR = REPO / "runs" / "onfire_negative"
EXCEL_PATH = OUTPUT_DIR / "results.xlsx"
PARTIAL_EXCEL_PATH = OUTPUT_DIR / "partial_results.xlsx"

# Models: "shufflenetonfire" (faster) or "nasnetonfire"
ONFIRE_ARCH = "shufflenetonfire"

# Published weights (models/onfire; the NASNet weights are obtained with
# third_party/download_onfire_weights.sh, only ShuffleNetV2 was used in the paper)
WEIGHTS_DIR = REPO / "models" / "onfire"
WEIGHTS = {
    "shufflenetonfire": {
        "ff": WEIGHTS_DIR / "shufflenet_ff.pt",
        "sp": WEIGHTS_DIR / "shufflenet_sp.pt",
    },
    "nasnetonfire": {
        "ff": WEIGHTS_DIR / "nasnet_ff.pt",
        "sp": WEIGHTS_DIR / "nasnet_sp.pt",
    },
}

# Number of repetitions for each video/configuration
N_RUNS = 1
RUNS_PER_BATCH = 1

# CPU threads used by PyTorch (RPi5 has 4 cores)
TORCH_THREADS = 4

# Network input size used by the OnFire models
INPUT_SIZE = 224

# SLIC superpixels (same values as inference_superpixel.py)
SP_REGION_SIZE = 22
SP_ITERATIONS = 10


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

# A frame is "fire" when P(fire) >= threshold.
# The original code uses torch.round(sigmoid), i.e. threshold 0.5.
# N = M = 1: alarm on the first frame classified as fire (no temporal rule).

# ------------------------------------------------------------
# MODE 1: FULL-FRAME (Table III / V of the paper)
# ------------------------------------------------------------

FF_CONFIGS = [
    {"thr": 0.50, "N": 1, "M": 1},
]


# ------------------------------------------------------------
# MODE 2: SUPERPIXEL (Table IV / VI of the paper)
# ------------------------------------------------------------

# min_sp: number of superpixels classified as fire to call the frame fire
SP_CONFIGS = [
    {"thr": 0.50, "min_sp": 1},
]


# ----------------------------------------------------------------------
# NTFY NOTIFICATIONS (end of each batch, end of the experiment, errors)
# ----------------------------------------------------------------------
# Subscribe to the same topic in the ntfy app. Topics are public, so use
# a name that is hard to guess. Empty string = no notifications.
NTFY_SERVER = "https://ntfy.sh"
NTFY_TOPIC = "saferfire-onfire-r7k2"


def notify(title: str, message: str, tags: str = "", priority: str = "default") -> None:
    """Send an ntfy notification. Never stops the experiment if it fails."""

    if not NTFY_TOPIC:
        return

    request = urllib.request.Request(
        f"{NTFY_SERVER}/{NTFY_TOPIC}",
        data=message.encode("utf-8"),
        headers={"Title": title, "Tags": tags, "Priority": priority},
        method="POST",
    )

    try:
        urllib.request.urlopen(request, timeout=15).close()
    except Exception as e:
        print(f"[ntfy] Could not send notification: {e}")


def format_duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


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
    """Return the Pi’s CPU temperature in °C using vcgencmd."""
    out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
    # Example output:  "temp=48.5'C\n"
    return float(out.split("=")[1].split("'")[0])


def current_memory_mib() -> float:
    """Resident-set-size of this Python process in MiB."""
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
    """Return the total CPU utilisation (0-100 %) since the previous call."""
    return psutil.cpu_percent(interval=None)


def read_gpu_clock_mhz() -> float:
    """Return the VideoCore-GPU clock in MHz."""
    out = subprocess.check_output(
        ["vcgencmd", "measure_clock", "core"]
    ).decode().strip()
    try:
        hz = int(out.split("=")[1])
        return hz / 1_000_000   # MHz
    except Exception:
        return float("nan")


def _safe(fn, label):
    try:
        return fn()
    except Exception as e:
        print(f"[log] {label} read error: {e}")
        return float("nan")


def _maybe_log_metrics():
    """Write a CSV line if LOG_INTERVAL seconds have passed."""
    global _last_log_time, _log_measurement_number

    now = time.time()
    if now - _last_log_time < LOG_INTERVAL:
        return                # not time yet

    _log_measurement_number += 1

    temp_c = _safe(read_temperature, "Temperature")
    mem_proc_mib = _safe(current_memory_mib, "Process memory")
    mem_sys_mib = _safe(current_system_memory_mib, "System memory (MiB)")
    mem_sys_prct = _safe(current_system_memory_percent, "System memory (%)")
    cpu_pct = _safe(read_cpu_percent, "CPU %")
    gpu_mhz = _safe(read_gpu_clock_mhz, "GPU clock")

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

    _last_log_time = now


# ============================================================
# RESULT CLASS
# ============================================================

@dataclass
class SimulationResult:
    alarm: bool
    alarm_frame: int | None
    alarm_time: float | None

    real_frame: int | None
    fps: float

    latency_seconds: float | None

    total_inference_time: float
    number_of_inferences: int

    # Kept with the same names as launch_simulation_batches.py:
    # light = full-frame classifier, heavy = superpixel localisation
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
    """Get FPS, number of frames, width and height."""

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


def frame_from_time(virtual_time: float, fps: float) -> int:
    """Convert simulated live-stream time to frame number."""

    return max(
        0,
        int(math.floor(virtual_time * fps)),
    )


def latency_from(alarm_frame, real_frame, fps):
    if alarm_frame is None or real_frame is None:
        return None
    return (alarm_frame - real_frame) / fps


# ============================================================
# ONFIRE MODELS
# ============================================================

DEVICE = torch.device("cpu")

# Normalisation used by inference_ff.py / inference_superpixel.py
NORMALIZATION = {
    "shufflenetonfire": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "nasnetonfire": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def build_model(arch: str):
    if arch == "shufflenetonfire":
        return shufflenetv2.shufflenet_v2_x0_5(
            pretrained=False,
            layers=[4, 8, 4],
            output_channels=[24, 48, 96, 192, 64],
            num_classes=1,
        )
    if arch == "nasnetonfire":
        return nasnet_mobile_onfire.nasnetamobile(
            num_classes=1,
            pretrained=False,
        )
    raise ValueError(f"Unknown OnFire architecture: {arch}")


def load_model(arch: str, weight_path: Path):
    if not weight_path.exists():
        raise FileNotFoundError(
            f"OnFire weights not found: {weight_path}. "
            f"Run download_onfire_weights.sh first."
        )

    model = build_model(arch)
    model.load_state_dict(
        torch.load(weight_path, map_location=DEVICE)
    )
    model.eval()
    model.to(DEVICE)
    return model


def to_tensor(rgb_images: np.ndarray, arch: str) -> torch.Tensor:
    """
    Equivalent to transforms.ToTensor() + transforms.Normalize(...)
    for a batch of RGB uint8 images (B, H, W, 3).
    """

    mean, std = NORMALIZATION[arch]
    x = rgb_images.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    return torch.from_numpy(x).to(DEVICE)


def resize_frame(frame):
    # The original code calls cv2.resize(frame, (224, 224), cv2.INTER_AREA),
    # where the third positional argument is "dst", so the default
    # (bilinear) interpolation is used. Kept identical here.
    return cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))


def fire_probability(logits: torch.Tensor) -> np.ndarray:
    # In the original code, round(sigmoid(output)) == 1 means NO-FIRE,
    # therefore P(fire) = 1 - sigmoid(output).
    return (1.0 - torch.sigmoid(logits)).reshape(-1).cpu().numpy()


def run_full_frame(model, arch: str, frame, threshold: float):
    """
    Full-frame {fire, no-fire} classification.

    Returns:
        detected
        inference_time (pre-processing + forward pass)
    """

    start = time.perf_counter()

    small = resize_frame(frame)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        logits = model(to_tensor(rgb[None], arch))

    p_fire = float(fire_probability(logits)[0])

    inference_time = time.perf_counter() - start

    return p_fire >= threshold, inference_time


def slic_segments(small_bgr):
    if hasattr(cv2, "ximgproc"):
        slic = cv2.ximgproc.createSuperpixelSLIC(
            small_bgr,
            region_size=SP_REGION_SIZE,
        )
        slic.iterate(SP_ITERATIONS)
        return slic.getLabels()

    raise RuntimeError(
        "cv2.ximgproc is not available. Install opencv-contrib-python "
        "(see README.md)."
    )


def run_superpixel(model, arch: str, frame, threshold: float, min_sp: int):
    """
    Superpixel localisation as in inference_superpixel.py:
    SLIC on the 224x224 frame, each superpixel is masked on a black
    background and classified. All superpixels are classified in one
    batch (same result as one by one, faster on CPU).

    Returns:
        detected (at least `min_sp` superpixels classified as fire)
        inference_time (resize + SLIC + forward pass)
        number_of_fire_superpixels
    """

    start = time.perf_counter()

    small = resize_frame(frame)
    segments = slic_segments(small)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    labels = np.unique(segments)
    batch = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    for i, seg in enumerate(labels):
        mask = segments == seg
        batch[i][mask] = rgb[mask]

    with torch.inference_mode():
        logits = model(to_tensor(batch, arch))

    n_fire = int((fire_probability(logits) >= threshold).sum())

    inference_time = time.perf_counter() - start

    return n_fire >= min_sp, inference_time, n_fire


# ============================================================
# APPROACH 1
# FULL-FRAME ONLY
# ============================================================

def approach_ff_only(
    video_path: Path,
    model,
    arch: str,
    threshold: float,
    N: int,
    M: int,
    real_frame: int | None,
):
    """
    Full-frame classifier only.

    Alarm when at least M of the last N inspected frames are
    classified as fire.
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
    number_of_inferences = 0

    alarm_frame = None
    alarm_time = None

    while True:

        current_frame = frame_from_time(virtual_time, fps)

        if current_frame >= frame_count:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        ret, frame = cap.read()

        if not ret:
            break

        detected, inference_time = run_full_frame(
            model=model,
            arch=arch,
            frame=frame,
            threshold=threshold,
        )

        number_of_inferences += 1
        total_inference_time += inference_time

        history.append(detected)

        if sum(history) >= M:

            # Alarm is only known after inference finishes.
            alarm_time = virtual_time + inference_time

            # Frame visible when the inference finishes.
            alarm_frame = frame_from_time(alarm_time, fps)

            break

        virtual_time += inference_time

        _maybe_log_metrics()

    cap.release()

    return SimulationResult(
        alarm=alarm_frame is not None,
        alarm_frame=alarm_frame,
        alarm_time=alarm_time,
        real_frame=real_frame,
        fps=fps,
        latency_seconds=latency_from(alarm_frame, real_frame, fps),
        total_inference_time=total_inference_time,
        number_of_inferences=number_of_inferences,
        light_inferences=number_of_inferences,
        heavy_inferences=0,
    )


# ============================================================
# APPROACH 2
# SUPERPIXEL ONLY
# ============================================================

def approach_sp_only(
    video_path: Path,
    model,
    arch: str,
    threshold: float,
    min_sp: int,
    real_frame: int | None,
):
    """
    Superpixel localisation on every inspected frame.

    Alarm on the first frame with at least `min_sp` superpixels
    classified as fire.
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

    alarm_frame = None
    alarm_time = None

    while True:

        current_frame = frame_from_time(virtual_time, fps)

        if current_frame >= frame_count:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        ret, frame = cap.read()

        if not ret:
            break

        detected, inference_time, _ = run_superpixel(
            model=model,
            arch=arch,
            frame=frame,
            threshold=threshold,
            min_sp=min_sp,
        )

        number_of_inferences += 1
        total_inference_time += inference_time

        if detected:

            # Alarm is only known after inference finishes.
            alarm_time = virtual_time + inference_time

            # Frame visible when the inference finishes.
            alarm_frame = frame_from_time(alarm_time, fps)

            break

        virtual_time += inference_time

        _maybe_log_metrics()

    cap.release()

    return SimulationResult(
        alarm=alarm_frame is not None,
        alarm_frame=alarm_frame,
        alarm_time=alarm_time,
        real_frame=real_frame,
        fps=fps,
        latency_seconds=latency_from(alarm_frame, real_frame, fps),
        total_inference_time=total_inference_time,
        number_of_inferences=number_of_inferences,
        light_inferences=0,
        heavy_inferences=number_of_inferences,
    )


# ============================================================
# EXPERIMENT NAMES
# ============================================================

# Sheet names (<= 31 characters, as in results/README.md)
def ff_name(threshold: float, N: int, M: int):
    rule = "" if (N, M) == (1, 1) else f"_W{N}T{M}"
    return f"FF_thr{threshold:.2f}{rule}"


def sp_name(threshold: float, min_sp: int):
    return f"SP_thr{threshold:.2f}_k{min_sp}"


# ============================================================
# RUN EXPERIMENTS (BATCHES)
# ============================================================

def run_experiment(
    videos: list[Path],
    experiments: list[dict],
) -> tuple[dict[str, dict[str, list[int]]], dict[str, list[dict]]]:
    """
    Negative videos: stores 1 = false alarm, 0 = no alarm (the desired result).
    Returns (all_results, all_metadata).

    Runs all (configuration x video) combinations in batches of
    RUNS_PER_BATCH runs, saving partial results after each batch.
    """

    all_results: dict[str, dict[str, list[int]]] = {}
    all_metadata: dict[str, list[dict]] = {}

    for exp in experiments:
        all_results[exp["name"]] = {v.stem: [] for v in videos}
        all_metadata[exp["name"]] = []

    total_batches = (N_RUNS + RUNS_PER_BATCH - 1) // RUNS_PER_BATCH

    experiment_start = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * RUNS_PER_BATCH + 1
        batch_end = min((batch_idx + 1) * RUNS_PER_BATCH, N_RUNS)

        batch_start_time = time.time()
        batch_alarms = {exp["name"]: 0 for exp in experiments}

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

                    kwargs = exp["make_kwargs"](
                        video=video,
                        real_frame=real_frame,
                    )

                    result: SimulationResult = exp["run_fn"](**kwargs)

                    all_results[exp_name][video_name].append(
                        int(result.alarm)
                    )
                    batch_alarms[exp_name] += result.alarm

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

        print("\nSaving intermediate Excel for this batch …")
        write_excel(all_results, all_metadata, PARTIAL_EXCEL_PATH)
        print("  → done.\n")

        elapsed = time.time() - experiment_start
        remaining = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
        runs_per_exp = len(videos) * (batch_end - batch_start + 1)
        alarm_lines = "\n".join(
            f"{name}: false alarm in {count}/{runs_per_exp} runs"
            for name, count in batch_alarms.items()
        )
        notify(
            f"OnFire NEG batch {batch_idx + 1}/{total_batches} done",
            f"Runs {batch_start}-{batch_end} ({ONFIRE_ARCH}, {len(videos)} negative videos)\n"
            f"Batch time: {format_duration(time.time() - batch_start_time)}\n"
            f"Elapsed: {format_duration(elapsed)}\n"
            f"Estimated remaining: {format_duration(remaining)}\n"
            f"{alarm_lines}",
            tags="hourglass",
        )

    return all_results, all_metadata


# ============================================================
# EXCEL
# ============================================================

def results_to_dataframe(results: dict):
    """
    {video: [1/0 per run]} -> run | video_1 | video_2 | ...
    """

    max_runs = max(len(values) for values in results.values())

    data = {"run": list(range(1, max_runs + 1))}

    for video_name, values in results.items():
        data[video_name] = values + [None] * (max_runs - len(values))

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

    for experiment_name, results in all_results.items():

        values = [v for video_values in results.values() for v in video_values]

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

    pd.DataFrame(rows).to_excel(writer, sheet_name="summary", index=False)

    pd.DataFrame(
        [row for experiment_rows in all_metadata.values() for row in experiment_rows]
    ).to_excel(writer, sheet_name="details", index=False)


def write_excel(all_results: dict, all_metadata: dict, path: Path) -> None:
    """
    summary + details sheets, then one sheet per experiment
    (1 = false alarm, 0 = no alarm). Same layout as
    launch_simulation_batches_neg.py.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        _write_summary_sheets(writer, all_results, all_metadata)

        used_sheet_names = {"summary", "details"}

        for experiment_name, results in all_results.items():

            sheet_name = sanitize_filename(experiment_name)[:31]

            original_name = sheet_name
            counter = 1

            while sheet_name in used_sheet_names:
                suffix = f"_{counter}"
                sheet_name = original_name[:31 - len(suffix)] + suffix
                counter += 1

            used_sheet_names.add(sheet_name)

            results_to_dataframe(results).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    workbook = load_workbook(path)

    for worksheet in workbook.worksheets:

        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:

            max_length = max(
                (len(str(cell.value)) for cell in column_cells if cell.value is not None),
                default=0,
            )

            worksheet.column_dimensions[
                column_cells[0].column_letter
            ].width = min(max_length + 2, 30)

    workbook.save(path)

    print(f"  → Results saved to: {path.resolve()}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("FIRE DETECTION EXPERIMENT – OnFire baseline (negative videos)")
    print("=" * 80)

    torch.set_num_threads(TORCH_THREADS)

    if not hasattr(cv2, "ximgproc") and SP_CONFIGS:
        raise RuntimeError(
            "cv2.ximgproc is not available, needed for the superpixel step. "
            "Install opencv-contrib-python (see README.md) or empty SP_CONFIGS."
        )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------
    print()
    print(f"Loading {ONFIRE_ARCH} full-frame model...")
    ff_model = load_model(ONFIRE_ARCH, WEIGHTS[ONFIRE_ARCH]["ff"])

    sp_model = None
    if SP_CONFIGS:
        print(f"Loading {ONFIRE_ARCH} superpixel model...")
        sp_model = load_model(ONFIRE_ARCH, WEIGHTS[ONFIRE_ARCH]["sp"])

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
    # WARM-UP MODELS
    # --------------------------------------------------------
    print()
    print("Warming up models...")
    cap = cv2.VideoCapture(str(videos[0]))
    ret, frame = cap.read()
    cap.release()
    if ret:
        for _ in range(3):
            _, t_ff = run_full_frame(ff_model, ONFIRE_ARCH, frame, 0.5)
        print(f"  full-frame inference: {t_ff * 1000:.1f} ms")
        if sp_model is not None:
            _, t_sp, _ = run_superpixel(sp_model, ONFIRE_ARCH, frame, 0.5, 1)
            print(f"  superpixel step:      {t_sp * 1000:.1f} ms")
    print("Warm-up completed.")

    # --------------------------------------------------------
    # EXPERIMENTS
    # --------------------------------------------------------
    experiments: list[dict] = []

    for cfg in FF_CONFIGS:

        def make_kwargs(video, real_frame, cfg=cfg):
            return {
                "video_path": video,
                "model": ff_model,
                "arch": ONFIRE_ARCH,
                "threshold": cfg["thr"],
                "N": cfg["N"],
                "M": cfg["M"],
                "real_frame": real_frame,
            }

        experiments.append(
            {
                "name": ff_name(cfg["thr"], cfg["N"], cfg["M"]),
                "run_fn": approach_ff_only,
                "make_kwargs": make_kwargs,
            }
        )

    for cfg in SP_CONFIGS:

        def make_kwargs(video, real_frame, cfg=cfg):
            return {
                "video_path": video,
                "model": sp_model,
                "arch": ONFIRE_ARCH,
                "threshold": cfg["thr"],
                "min_sp": cfg["min_sp"],
                "real_frame": real_frame,
            }

        experiments.append(
            {
                "name": sp_name(cfg["thr"], cfg["min_sp"]),
                "run_fn": approach_sp_only,
                "make_kwargs": make_kwargs,
            }
        )

    start = time.time()

    all_results, all_metadata = run_experiment(
        videos=videos,
        experiments=experiments,
    )

    write_excel(all_results, all_metadata, EXCEL_PATH)

    print()
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)

    false_alarm_lines = "\n".join(
        f"{name}: {sum(sum(v) for v in results.values())} false alarms "
        f"in {sum(len(v) for v in results.values())} runs"
        for name, results in all_results.items()
    )
    notify(
        "OnFire NEG experiment completed",
        f"{ONFIRE_ARCH}: {len(experiments)} experiments x {len(videos)} negative videos "
        f"x {N_RUNS} runs\n"
        f"Total time: {format_duration(time.time() - start)}\n"
        f"{false_alarm_lines}\n"
        f"Results: {EXCEL_PATH}",
        tags="white_check_mark",
        priority="high",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        notify("OnFire NEG experiment stopped", "Stopped with Ctrl+C.", tags="stop_sign")
        raise
    except Exception as e:
        notify(
            "OnFire NEG experiment FAILED",
            f"{type(e).__name__}: {e}",
            tags="rotating_light",
            priority="high",
        )
        raise
