# A Two-Stage Vision-Based Fire and Smoke Detection System for Real-Time Indoor Monitoring on Edge Devices

Data, models, experiment scripts and analysis code of the paper *A Two-Stage Vision-Based Fire and Smoke Detection
System for Real-Time Indoor Monitoring on Edge Devices* (R. Santo, C. Couto, A. Sousa, H. Dias, T. Silva,
J. Ferraz de Abreu; submitted to Neural Computing and Applications).

A lightweight detector (YOLOv8n) monitors every frame and, when a temporal rule (W,T) is satisfied, a high-capacity
detector (YOLOv12m) verifies the event before the alarm is raised (Two-Step). It is compared with the two
single-model baselines (Light Only, Best Only) and with four video-based methods from the literature
(Venâncio et al., OnFire, FLAME, Dumitru and Spînu), all run on a Raspberry Pi 5 on the same 69 positive and
69 negative videos.

## Data

- **Training dataset (FireSmokeDS)**: [Google Drive](https://drive.google.com/drive/folders/1u421L9LPUVPZ2AgxSgd_HPG0uSyjq0lG?usp=sharing).
  The source videos of its Selfmade subset (frames extracted and annotated by the authors) are listed, with the
  number of frames extracted from each one, in `data/selfmade_training_sources.csv`.
- **Test videos (69 positive, 69 negative)**: the videos are third-party footage and are not redistributed; they are
  given by reference in `data/video_sources_positive.csv` and `data/video_sources_negative.csv`:
  - positive videos P13–P69: YouTube link, start and end time in the original video, crop box and resize to 640×640;
  - positive videos P1–P12: link of the post or page where the footage was published (YouTube, LinkedIn, Reddit,
    news website, NIST), with a description; the cut of these segments was not recorded;
  - negative videos: FIRESENSE (Zenodo record 836749, CC BY 4.0, exact file in the archive), KMU Fire & Smoke
    Database (exact file), Pexels (video id) and YouTube.

  `python data/prepare_videos.py` downloads and rebuilds the YouTube, FIRESENSE and KMU videos (needs `ffmpeg` and
  `yt-dlp`) and lists the others for manual download. Place the videos in `data/videos/positive` and
  `data/videos/negative`, or set `SAFERFIRE_VIDEOS`. Re-encoded videos differ slightly from the originals, so the
  results are reproduced only approximately. Third-party videos remain subject to the terms of their sources.
- **Annotations**: `data/onset_frames/` gives the frame at which the fire or smoke becomes visible in each positive
  video (reference for the latency); see `data/README.md`.

## Contents

| Folder | Contents |
|---|---|
| `models/` | Trained models (Git LFS): YOLOv8n and YOLOv12m in ONNX format as run on the Raspberry Pi 5, and the detectors trained for the methods from the literature — see `models/README.md` |
| `experiments/` | Scripts run on the Raspberry Pi 5 for each method, on the positive and negative videos — see `experiments/README.md` |
| `results/` | Raw results of every run: latency per positive video and false alarm per negative video, one workbook per method — see `results/README.md` |
| `analysis/` | Scripts that compute the figures and tables of the paper from `results/` (`make_all.py`) — see `analysis/README.md`; the regenerated figures and tables are written to `paper/figs` and `paper/tables` |
| `data/` | Onset annotations and list of the negative videos (the videos are on Google Drive) |

## Usage

```bash
git lfs install            # once, for the models
git clone https://github.com/rubensanto/two-stage-fire-smoke-edge-device-cnn.git
cd two-stage-fire-smoke-edge-device-cnn/analysis
uv run python make_all.py  # figures and tables of the paper from results/
```

The experiments (`experiments/*.py`) need the videos and the requirements in `experiments/requirements.txt`; they
write their output to `runs/<script name>/`, which `analysis/import_runs.py` adds to `results/`. The pre-trained
OnFire weights are downloaded from their original source with `experiments/third_party/download_onfire_weights.sh`.

## Third-party code and models

- `experiments/third_party/onfire_models/`: OnFire model definitions (MIT License, Durham University; see the
  `LICENSE` file in that folder).
- `experiments/third_party/venancio/`: tracker of Venâncio et al., based on the simple object tracker of
  PyImageSearch.
- The YOLO models were trained with Ultralytics and are subject to the Ultralytics license (AGPL-3.0).
