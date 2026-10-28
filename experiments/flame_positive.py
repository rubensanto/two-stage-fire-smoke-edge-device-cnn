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
OUTPUT_DIR = REPO / "runs" / "flame_positive"
EXCEL_PATH = OUTPUT_DIR / "results.xlsx"                  # EDGE (compared with the other methods)
EXCEL_OFFLINE_PATH = OUTPUT_DIR / "results_offline.xlsx"  # OFFLINE (validation of the implementation)

PARTIAL_EDGE_PATH = OUTPUT_DIR / "partial_results_edge.xlsx"
PARTIAL_OFFLINE_PATH = OUTPUT_DIR / "partial_results_offline.xlsx"

GRID_CSV_PATH = OUTPUT_DIR / "tc_grid_search.csv"
METHODOLOGY_TXT_PATH = OUTPUT_DIR / "NOTA_METODOLOGICA_REGIMES.txt"


# Models
MODEL_PATH = str(REPO / "models" / "flame" / "YOLOv8s_best.pt")   # YOLOv8s trained on our data

# Number of repetitions for each video/configuration
N_RUNS = 6          # regime EDGE (não determinístico → média±std)
RUNS_PER_BATCH = 2
# YOLO image size
IMG_SIZE = 640

# None = automatic
DEVICE = None

# IMPORTANT:
# Change these to match the class names in your models.
TARGET_CLASSES = {"fire", "smoke"}


# ------------------------------------------------------------
# FLAGS DE EXECUÇÃO (liga/desliga cada etapa)
# ------------------------------------------------------------
RUN_GRID_SEARCH = True    # Etapa 1: grid search do Tc (Stage I only)
RUN_OFFLINE     = True    # Etapa 2: regime offline (replicação, determinístico)
RUN_EDGE        = True    # Etapa 3: regime edge (matriz comparativa)

GRID_SEARCH_SUBSET = 10   # nº de vídeos usados no grid search
OFFLINE_SUBSET     = 15   # nº de vídeos usados no regime offline


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

# ------------------------------------------------------------
# REGIME EDGE: FLAME (Gragnaniello et al., 2025)
# Após o grid search, ajusta o "conf" se o Tc recomendado diferir de 0.4
# ------------------------------------------------------------

# ------------------------------------------------------------
# CONFIGS (construídas automaticamente à volta do Tc base)
# ------------------------------------------------------------
PAPER_TC = 0.4      # Tc publicado no paper (Gragnaniello et al., 2025)


def build_edge_configs(base_tc):
    """Configs do regime EDGE, centradas no Tc calibrado automaticamente."""
    return [
        {"conf": base_tc, "t_fg": 0.2, "tC": 1.0, "tO": 0.1},   # parâmetros do paper
        {"conf": base_tc, "t_fg": 0.2, "tC": 0.5, "tO": 0.1},   # variação: tC
        {"conf": base_tc, "t_fg": 0.3, "tC": 1.0, "tO": 0.1},   # variação: t_fg
    ]


def build_offline_configs(base_tc):
    """Configs do regime OFFLINE: paper Tc (replicação) + Tc calibrado."""
    cfgs = [
        {"conf": PAPER_TC, "t_fg": 0.2, "tC": 1.0, "tO": 0.1},  # replicação fiel
    ]
    if base_tc != PAPER_TC:
        cfgs.append(
            {"conf": base_tc, "t_fg": 0.2, "tC": 1.0, "tO": 0.1},  # c/ Tc calibrado
        )
    return cfgs


# ------------------------------------------------------------

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
# FLAME TRACKING (Algorithm 1 — Gragnaniello et al., 2025)
# ============================================================

def flame_tracking(tracks, candidates, dt, tC, tO, d_M):
    """
    Algorithm 1 (Motion filtering) do paper FLAME.
    tracks: lista de dicts {centroid, state, time, associated}
    candidates: lista de tuplos (cx, cy)
    dt: tempo entre frames analisados (1/fps no offline; yolo_time no edge)
    """
    confirmed = False
    
    cand_associated = [False] * len(candidates)
    for t in tracks:
        t["associated"] = False
        
    M = []
    for t in tracks:
        row = []
        for c in candidates:
            dist = math.dist(t["centroid"], c)
            sim = max(0.0, 1.0 - (dist / d_M)) if dist < d_M else 0.0
            row.append(sim)
        M.append(row)
        
    while True:
        best_val = 0.0
        best_j = -1
        best_i = -1
        for j, t in enumerate(tracks):
            if t["associated"]: continue
            for i in range(len(candidates)):
                if cand_associated[i]: continue
                if M[j][i] > best_val:
                    best_val = M[j][i]
                    best_j = j
                    best_i = i
                    
        if best_j == -1 or best_i == -1 or best_val <= 0.0:
            break
            
        cand_associated[best_i] = True
        tracks[best_j]["associated"] = True
        tracks[best_j]["centroid"] = candidates[best_i]
        
    for t in tracks:
        if t["associated"]:
            if t["state"] == "Occluded":
                t["state"] = "New"
                t["time"] = 0.0
            else:
                t["time"] += dt
                if t["state"] == "New" and t["time"] > tC:
                    t["state"] = "Confirmed"
                    confirmed = True
        else:
            if t["state"] == "New":
                t["state"] = "Occluded"
                t["time"] = 0.0
            else:
                t["time"] += dt
                if t["state"] == "Occluded" and t["time"] > tO:
                    t["state"] = "Rejected"
                    
    tracks = [t for t in tracks if t["state"] != "Rejected"]
    
    for i, c in enumerate(candidates):
        if not cand_associated[i]:
            tracks.append({
                "centroid": c,
                "state": "New",
                "time": 0.0,
                "associated": False
            })
            
    return tracks, confirmed


# ============================================================
# STAGE I + II comuns (deteção + filtro de background)
# ============================================================

def extract_candidates(frame, results, fg_mask, t_fg):
    """
    Stage I (YOLO) + Stage II (Eq. 1: overlap com foreground).
    Devolve lista de centróides que sobrevivem aos dois estágios.
    """
    candidates = []

    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        for i in range(len(boxes)):
            cls_name = str(names[int(classes[i])]).lower()
            if cls_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, boxes[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = fg_mask[y1:y2, x1:x2]
            overlap = np.count_nonzero(roi) / roi.size if roi.size > 0 else 0.0

            if overlap >= t_fg:
                candidates.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    return candidates


def make_background_subtractor():
    """
    Stage II init: GMM/MOG2 (Zivkovic 2004 = ref [76] do paper).
    h=500, Tm=16, Ng=3, σ̂0=15, máscara binária (sem sombras).
    """
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )
    bg_sub.setNMixtures(3)
    bg_sub.setVarInit(15.0)
    return bg_sub


# ============================================================
# APPROACH: FLAME — REGIME EDGE (simulação live-stream)
# ============================================================

def approach_flame(
    video_path: Path,
    model: YOLO,
    confidence: float,
    t_fg: float,
    tC: float,
    tO: float,
    real_frame: int,
):
    """
    FLAME em regime EDGE: o relógio virtual avança com o tempo de
    inferência; só os frames que "chegam" são analisados.
    dt do FSM = yolo_time (tempo real entre frames analisados).
    """
    fps, frame_count, width, height = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
        
    virtual_time = 0.0
    total_inference_time = 0.0
    number_of_inferences = 0
    alarm_frame = None
    alarm_time = None
    
    bg_sub = make_background_subtractor()
    tracks = []
    d_M = math.sqrt(width**2 + height**2)
    
    warmup_count = min(10, frame_count)
    for _ in range(warmup_count):
        ret, w_frame = cap.read()
        if ret:
            bg_sub.apply(w_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        current_frame = frame_from_time(virtual_time, fps)
        if current_frame >= frame_count:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
            
        start = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=IMG_SIZE,
            verbose=False,
        )
        yolo_time = time.perf_counter() - start
        number_of_inferences += 1
        total_inference_time += yolo_time
        
        fg_mask = bg_sub.apply(frame)
        candidates = extract_candidates(frame, results, fg_mask, t_fg)
        
        tracks, confirmed = flame_tracking(
            tracks=tracks,
            candidates=candidates,
            dt=yolo_time,
            tC=tC,
            tO=tO,
            d_M=d_M
        )
        
        if confirmed:
            alarm_time = virtual_time + yolo_time
            alarm_frame = frame_from_time(alarm_time, fps)
            break
            
        virtual_time += yolo_time
        _maybe_log_metrics()
        
    cap.release()
    latency = (alarm_frame - real_frame) / fps if alarm_frame is not None else None
    
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
# APPROACH: FLAME — REGIME OFFLINE (replicação do paper)
# ============================================================

def approach_flame_offline(
    video_path: Path,
    model: YOLO,
    confidence: float,
    t_fg: float,
    tC: float,
    tO: float,
    real_frame: int,
):
    """
    FLAME em regime OFFLINE: processa TODOS os frames sequencialmente
    (sem seek, sem relógio virtual). dt do FSM = 1/fps (tempo inter-frame
    do vídeo), o regime em que os parâmetros do paper foram calibrados.
    Determinístico: 1 run basta.
    """
    fps, frame_count, width, height = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_inference_time = 0.0
    number_of_inferences = 0
    alarm_frame = None

    bg_sub = make_background_subtractor()
    tracks = []
    d_M = math.sqrt(width**2 + height**2)

    dt = 1.0 / fps          # ← regime do paper
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=IMG_SIZE,
            verbose=False,
        )
        yolo_time = time.perf_counter() - start
        number_of_inferences += 1
        total_inference_time += yolo_time

        fg_mask = bg_sub.apply(frame)
        candidates = extract_candidates(frame, results, fg_mask, t_fg)

        tracks, confirmed = flame_tracking(
            tracks=tracks,
            candidates=candidates,
            dt=dt,
            tC=tC,
            tO=tO,
            d_M=d_M
        )

        if confirmed:
            alarm_frame = current_frame
            break

        current_frame += 1
        _maybe_log_metrics()

    cap.release()

    if alarm_frame is not None:
        alarm_time = alarm_frame / fps
        latency = (alarm_frame - real_frame) / fps
    else:
        alarm_time = None
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
# GRID SEARCH DO Tc (Stage I only — como no paper, Sec. 2.2.1)
# ============================================================

def run_tc_grid_search(videos: list[Path], model: YOLO, subset: int):
    """
    Uma passagem sequencial por vídeo (Stage I apenas), registando a
    confiança máxima de fire/smoke por frame. Depois varre Tc offline
    (0.10–0.90, step 0.05) e calcula métricas video-level (ONFIRE):
      TP = primeira deteção >= ground-truth
      FP = primeira deteção <  ground-truth (alarme prematuro)
      FN = nenhuma deteção
    Grava tc_grid_search.csv e devolve o Tc com melhor F1.
    """
    print()
    print("=" * 80)
    print(f"GRID SEARCH Tc (Stage I) — {min(subset, len(videos))} vídeos")
    print("=" * 80)

    records = {}   # stem -> (fps, [(frame_idx, max_conf), ...])

    for video in videos[:subset]:
        fps, frame_count, _, _ = get_video_info(video)
        cap = cv2.VideoCapture(str(video))
        per_frame = []
        idx = 0

        print(f"  Passing {video.name} ({frame_count} frames)...", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=0.05,          # threshold baixo para registar tudo
                imgsz=IMG_SIZE,
                verbose=False,
            )

            max_conf = 0.0
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                names = result.names
                for i in range(len(confs)):
                    if str(names[int(classes[i])]).lower() in TARGET_CLASSES:
                        max_conf = max(max_conf, float(confs[i]))

            per_frame.append((idx, max_conf))
            idx += 1

        cap.release()
        records[video.stem] = (fps, per_frame)

    # ---- sweep offline dos thresholds ----
    rows = []
    for tc_i in range(17):                      # 0.10 .. 0.90
        tc = round(0.10 + 0.05 * tc_i, 2)
        tp = fp = fn = 0
        delays = []

        for video in videos[:subset]:
            real_frame = get_real_frame(video)
            fps, per_frame = records[video.stem]

            first = None
            for fidx, conf in per_frame:
                if conf >= tc:
                    first = fidx
                    break

            if first is None:
                fn += 1
            elif first < real_frame:
                fp += 1
            else:
                tp += 1
                delays.append((first - real_frame) / fps)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        mean_delay = sum(delays) / len(delays) if delays else float("nan")

        rows.append({
            "tc": tc, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": f1, "mean_delay_s": mean_delay,
        })

    # ---- gravar csv ----
    GRID_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRID_CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["tc", "tp", "fp", "fn", "precision", "recall", "f1", "mean_delay_s"]
        )
        for r in rows:
            writer.writerow([
                f"{r['tc']:.2f}", r["tp"], r["fp"], r["fn"],
                f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                f"{r['f1']:.3f}", f"{r['mean_delay_s']:.3f}",
            ])

    print()
    print(f"{'Tc':>5} | {'TP':>3} {'FP':>3} {'FN':>3} | "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} | {'Delay(s)':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['tc']:>5.2f} | {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} | "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} | "
              f"{r['mean_delay_s']:>8.3f}")

    best = max(rows, key=lambda r: r["f1"])
    print()
    print("=" * 80)
    print(f"Tc RECOMENDADO para o TEU detetor: {best['tc']:.2f} "
          f"(F1={best['f1']:.3f})")
    print(f"CSV gravado em: {GRID_CSV_PATH.resolve()}")
    print("→ Se diferir de 0.4, atualiza o 'conf' nas FLAME_CONFIGS / "
          "FLAME_OFFLINE_CONFIGS antes do lançamento final.")
    print("=" * 80)

    return best["tc"]


# ============================================================
# EXPERIMENT NAMES
# ============================================================

def flame_name(prefix, conf, t_fg, tC, tO):
    # sheet name (<= 31 characters): edge_c0.25_fg0.20_tC1.0 / offline_c0.25_fg0.20_tC1.0;
    # tO is only written when it differs from the 0.1 s of the paper
    regime = "offline" if prefix == "off" else prefix
    t_o = "" if abs(tO - 0.1) < 1e-9 else f"_tO{tO:.1f}"
    return f"{regime}_c{conf:.2f}_fg{t_fg:.2f}_tC{tC:.1f}{t_o}"


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    videos: list[Path],
    experiments: list[dict],
    partial_path: Path,
    n_runs: int,
) -> dict[str, dict[str, list[float | None]]]:
    """
    Executa todas as combinações (configuração × vídeo) em batches de
    ``RUNS_PER_BATCH`` corridas cada um, salvando os resultados
    (metadata + matriz de latências) ao final **de cada batch**.
    ``n_runs`` permite 1 run no regime offline (determinístico)
    e N_RUNS no regime edge (não determinístico).
    """
    # ---------- containers ----------
    all_results: dict[str, dict[str, list[float | None]]] = {}
    all_metadata: dict[str, list[dict]] = {}

    for exp in experiments:
        all_results[exp["name"]] = {v.stem: [] for v in videos}
        all_metadata[exp["name"]] = []

    runs_per_batch = RUNS_PER_BATCH
    total_batches = (n_runs + runs_per_batch - 1) // runs_per_batch   # ceil

    for batch_idx in range(total_batches):
        batch_start = batch_idx * runs_per_batch + 1
        batch_end   = min((batch_idx + 1) * runs_per_batch, n_runs)

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
                        f"  {exp_name} – run {run_number}/{n_runs} "
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
            partial_path=partial_path,
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


def _write_intermediate_excel(
    all_results: dict[str, dict[str, list[float | None]]],
    partial_path: Path,
) -> None:
    """
    Save all currently available results to one Excel file.
    Each experiment gets its own sheet. Overwritten on every call.
    """

    partial_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        partial_path,
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
    workbook = load_workbook(partial_path)

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

    workbook.save(partial_path)

    print(
        f"  → Partial results saved to: "
        f"{partial_path.resolve()}"
    )



def save_excel(all_results: dict, excel_path: Path):
    """
    Create one Excel sheet per experiment.
    """

    excel_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        excel_path,
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
    workbook = load_workbook(excel_path)

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

    workbook.save(excel_path)

    print()
    print("=" * 80)
    print("Excel saved to:")
    print(excel_path.resolve())
    print("=" * 80)


# ============================================================
# NOTA METODOLÓGICA (txt na pasta de output)
# ============================================================

METHODOLOGY_NOTE = """\
NOTA METODOLOGICA - PORQUE DOIS REGIMES DE TESTE (OFFLINE + EDGE)
=================================================================
Pipeline avaliado: FLAME (Gragnaniello, Greco, Sansone, Vento,
Neural Computing and Applications, 2025) - Familia 4
(Heuristic-to-Learned: detetor + analise de movimento baseada em modelo).

Estagios: (I) YOLOv8s (flame+smoke, Tc); (II) subtracao de fundo GMM
(h=500, Tm=16, Ng=3, sigma0=15, T_fg=0.2); (III) tracking por centroides
com FSM New/Occluded/Confirmed/Rejected (tC=1s, tO=0.1s).

-----------------------------------------------------------------
1. REGIME OFFLINE (results_offline.xlsx) - REPLICACAO
-----------------------------------------------------------------
- Processa TODOS os frames do video, sequencialmente (sem seek).
- dt do FSM = 1/fps (tempo inter-frame do video).
- E o regime em que os autores calibraram tC, tO e Tc (sistema a
  acompanhar o frame-rate da camara, >=10-25 fps).
- Deterministico: a sequencia de frames e fixa, portanto 1 run basta.
- OBJETIVO: validar a implementacao. Se as latencias ficarem na casa
  dos segundos e forem estaveis, o codigo reproduz o comportamento
  publicado. Se o offline falhar, o defeito e da implementacao.

-----------------------------------------------------------------
2. REGIME EDGE (results.xlsx) - DEPLOY REAL NO RPi5
-----------------------------------------------------------------
- Relogio virtual: o tempo avanca com o tempo de inferencia; so os
  frames que "chegam" dado o throughput do dispositivo sao analisados
  (~2-5 fps com YOLOv8s@640 no RPi5).
- dt do FSM = tempo real entre frames analisados (yolo_time).
- E o regime comparavel com as outras 3 familias da matriz
  (YOLOv8n->YOLOv8L, Venancio, Bahhar), que usam o mesmo protocolo.
- OBJETIVO: medir o custo real de deploy e comparar familias em
  condicoes identicas.

-----------------------------------------------------------------
3. PORQUE NAO APENAS O REGIME EDGE?
-----------------------------------------------------------------
a) No edge confundem-se duas causas distintas:
   (i) erros de implementacao;
   (ii) desajuste de regime (parametros calibrados para outro throughput).
   Sem o offline nao e possivel separa-las: um NO ALARM no edge pode
   ser um bug OU ser o tO=0.1s demasiado curto para o FPS efetivo
   (com dt~0.5s, UMA unica amostra falhada ja excede tO e rejeita o
   track, forcando o recomeco da confirmacao).
b) O Tc=0.4 foi calibrado pelos autores no validation set deles, com o
   YOLOv8s treinado no LFDN (>4000 negativos dificeis). O nosso detetor
   tem outra curva de confianca; o grid search de Tc (tc_grid_search.csv)
   e o regime offline isolam esse efeito antes de interpretar o edge.
c) O edge e nao-deterministico (tempo de inferencia varia com carga e
   temperatura; seek H.264 nao e frame-exact). O offline da uma
   referencia estavel e auditavel para revisores.

-----------------------------------------------------------------
4. PORQUE NAO APENAS O REGIME OFFLINE?
-----------------------------------------------------------------
- Nao reflete o deploy: num dispositivo real o sistema nao acompanha
  todos os frames; a latencia percebida depende do throughput.
- O proprio paper reconhece isto: a metrica FDS penaliza metodos
  abaixo de 10 fps. Avaliar so offline esconderia a limitacao de
  deploy que nos queremos quantificar.

-----------------------------------------------------------------
5. COMO LER OS RESULTADOS
-----------------------------------------------------------------
- results_offline.xlsx: latencias em tempo de video, estaveis.
  Valores de poucos segundos = implementacao fiel ao paper.
- results.xlsx: latencias maiores e com variancia (reportar media±std).
- A DIFERENCA entre os dois regimes e o finding cientifico: quantifica
  a degradacao de metodos de confirmacao temporal quando o throughput
  do dispositivo fica muito abaixo do frame-rate de calibracao.
- tc_grid_search.csv: P/R/F1/delay por Tc (0.10-0.90) medidos no Stage I
  com o NOSSO detetor; usar o Tc recomendado para interpretar configs.

Ficheiros gerados:
  - tc_grid_search.csv            (grid search do Tc, Stage I)
  - results_offline.xlsx          (regime offline, subconjunto, 1 run)
  - results.xlsx                  (regime edge, todos os videos, 6 runs)
  - partial_results_offline.xlsx  (intermedio offline)
  - partial_results_edge.xlsx     (intermedio edge)
  - system_log.csv                (temperatura/memoria/CPU/GPU do RPi5)
"""


def write_methodology_note():
    METHODOLOGY_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METHODOLOGY_TXT_PATH.write_text(METHODOLOGY_NOTE, encoding="utf-8")
    print(f"Methodology note written to: {METHODOLOGY_TXT_PATH.resolve()}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("FIRE DETECTION EXPERIMENT - FLAME (Família 4)")
    print("Execução única automática: grid search Tc → OFFLINE → EDGE")
    print("=" * 80)

    # --------------------------------------------------------
    # NOTA METODOLÓGICA (txt na pasta de output)
    # --------------------------------------------------------
    write_methodology_note()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    print()
    print("Loading FLAME model...")
    flame_model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # FIND VIDEOS
    # --------------------------------------------------------
    videos = get_video_files()
    print()
    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  - {v.name}")

    print()
    print(f"Runs per configuration (edge): {N_RUNS}")
    print(f"Runs per batch:                {RUNS_PER_BATCH}")
    print(f"Offline subset:                {min(OFFLINE_SUBSET, len(videos))} vídeos")
    print(f"Grid-search subset:            {min(GRID_SEARCH_SUBSET, len(videos))} vídeos")

    # --------------------------------------------------------
    # WARM‑UP MODEL
    # --------------------------------------------------------
    print()
    print("Warming up model...")
    cap = cv2.VideoCapture(str(videos[0]))
    ret, frame = cap.read()
    cap.release()
    if ret:
        flame_model.predict(source=frame, conf=0.4, imgsz=IMG_SIZE, verbose=False)
    print("Warm‑up completed.")

    # --------------------------------------------------------
    # ETAPA 1: GRID SEARCH DO Tc (automático)
    # --------------------------------------------------------
    base_tc = PAPER_TC
    if RUN_GRID_SEARCH:
        best = run_tc_grid_search(videos, flame_model, GRID_SEARCH_SUBSET)
        if best is not None:
            base_tc = best
        print()
        print(f"→ Tc base aplicado automaticamente nas etapas seguintes: {base_tc:.2f}")

    offline_configs = build_offline_configs(base_tc)
    edge_configs = build_edge_configs(base_tc)

    # --------------------------------------------------------
    # ETAPA 2: REGIME OFFLINE (automático, 1 run, subconjunto)
    # --------------------------------------------------------
    if RUN_OFFLINE:
        print()
        print("=" * 80)
        print("REGIME OFFLINE (replicação: todos os frames, dt=1/fps)")
        print("=" * 80)

        videos_offline = videos[:OFFLINE_SUBSET]

        experiments_offline: list[dict] = []
        for cfg in offline_configs:
            name = flame_name("off", cfg["conf"], cfg["t_fg"], cfg["tC"], cfg["tO"])

            def make_kwargs_off(video, real_frame, cfg=cfg):
                return {
                    "video_path": video,
                    "model": flame_model,
                    "confidence": cfg["conf"],
                    "t_fg": cfg["t_fg"],
                    "tC": cfg["tC"],
                    "tO": cfg["tO"],
                    "real_frame": real_frame,
                }

            experiments_offline.append(
                {
                    "name": name,
                    "run_fn": approach_flame_offline,
                    "make_kwargs": make_kwargs_off,
                }
            )

        all_offline = run_experiment(
            videos=videos_offline,
            experiments=experiments_offline,
            partial_path=PARTIAL_OFFLINE_PATH,
            n_runs=1,
        )

        save_excel(all_offline, EXCEL_OFFLINE_PATH)

    # --------------------------------------------------------
    # ETAPA 3: REGIME EDGE (automático, 6 runs, todos os vídeos)
    # --------------------------------------------------------
    if RUN_EDGE:
        print()
        print("=" * 80)
        print("REGIME EDGE (live-stream simulado, dt=yolo_time)")
        print("=" * 80)

        experiments_edge: list[dict] = []
        for cfg in edge_configs:
            name = flame_name("edge", cfg["conf"], cfg["t_fg"], cfg["tC"], cfg["tO"])

            def make_kwargs_edge(video, real_frame, cfg=cfg):
                return {
                    "video_path": video,
                    "model": flame_model,
                    "confidence": cfg["conf"],
                    "t_fg": cfg["t_fg"],
                    "tC": cfg["tC"],
                    "tO": cfg["tO"],
                    "real_frame": real_frame,
                }

            experiments_edge.append(
                {
                    "name": name,
                    "run_fn": approach_flame,
                    "make_kwargs": make_kwargs_edge,
                }
            )

        all_edge = run_experiment(
            videos=videos,
            experiments=experiments_edge,
            partial_path=PARTIAL_EDGE_PATH,
            n_runs=N_RUNS,
        )

        save_excel(all_edge, EXCEL_PATH)

    print()
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
