# Experiments (Raspberry Pi 5)

Each script streams the videos as a live feed (virtual clock driven by the inference time on the device), runs one
method in batches and writes its workbook to `runs/<script name>/results.xlsx` (plus `partial_results.xlsx`, saved
after every batch, and `system_log.csv` with temperature, memory and CPU). The sheet names follow `results/README.md`.

| Script | Method | Videos | Configurations |
|---|---|---|---|
| `saferfire_positive.py` | Light Only, Best Only, Two-Step | positive | `THRESHOLD_SET = "main"` (0.25/0.50, 17 configurations) or `"extra"` (0.125/0.75, 29) |
| `saferfire_negative.py` | Light Only, Best Only, Two-Step | negative | all 46 configurations, 1 run |
| `venancio_positive.py` | Venâncio et al. (YOLOv5l + AVT) | positive | 4 configurations |
| `venancio_negative.py` | Venâncio et al. | negative | same 4 configurations, 1 run |
| `onfire_positive.py` | OnFire (ShuffleNetV2), full frame and superpixels | positive | thr = 0.50 |
| `onfire_negative.py` | OnFire | negative | thr = 0.50, 1 run |
| `flame_positive.py` | FLAME (YOLOv8s + motion analysis) | positive | grid search of the detector threshold, offline regime (validation) and edge regime (3 configurations) |
| `flame_negative.py` | FLAME | negative | edge regime, same 3 configurations, 1 run |
| `dumitru_positive.py` | Dumitru & Spînu (arXiv:2607.03131, YOLOv10n + 5-level validation cascade) | positive | ablation of the cascade, 6 configurations |
| `dumitru_negative.py` | Dumitru & Spînu | negative | same 6 configurations, 1 run |
| `ideal_latency.py` | Light Only, Best Only, Two-Step with ideal models (no model is run) | positive | the 3 temporal rules; writes `results/positive_videos/ideal_latency.csv` directly |

The number of runs (`N_RUNS`, `RUNS_PER_BATCH`) and the configurations are set at the top of each script.

## Requirements

- Python 3.10+ with `pip install -r requirements.txt` (OnFire needs `torch`; Venâncio et al. needs `scipy`).
- Models in `models/` (Git LFS) and videos in `data/videos/positive|negative` (or `SAFERFIRE_VIDEOS`).
- Third-party code in `third_party/`: the area variation tracker of Venâncio et al. (`venancio/temporal/tracker.py`,
  MIT license) and the OnFire network definitions (`onfire_models/`); `download_onfire_weights.sh` downloads the
  NASNet weights, which were not used in the paper.

## Running and collecting the results

```bash
python experiments/saferfire_positive.py          # from any folder; output in runs/saferfire_positive/
cd analysis
python import_runs.py ../runs/saferfire_positive/results.xlsx ../results/positive_videos/saferfire_latency.xlsx
```

`ideal_latency.py` replays the same simulated stream with ideal models: a frame is detected if and only if it is at
or after the annotated onset, and every inference takes the mean time of Table 6 (t_L = 0.184 s, t_H = 1.371 s).
It only reads the FPS and frame count of the videos and takes a few seconds (`python experiments/ideal_latency.py`).

Notes on the FLAME implementation and its two regimes: `docs/flame_readme.md` and `docs/flame_regimes_note.txt`.
Notes on the Dumitru & Spînu implementation (parameters taken from the paper and those chosen by us):
`docs/dumitru_readme.md`.
