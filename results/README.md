# Results of the system-level evaluation (Raspberry Pi 5)

One workbook per method and video set. Every sheet is one configuration, with one row per run (`run`)
and one column per video.

| File | Contents | Runs per video |
|---|---|---|
| `positive_videos/saferfire_latency.xlsx` | Light Only, Best Only and Two-Step, 46 configurations | 14 with thresholds 0.25/0.50, 6 with 0.125/0.75 |
| `positive_videos/venancio_latency.xlsx` | Venâncio et al. (YOLOv5l + area variation technique) | 6 |
| `positive_videos/onfire_latency.xlsx` | OnFire (ShuffleNetV2), full frame and superpixels | 6 |
| `positive_videos/flame_latency.xlsx` | FLAME, edge regime (`edge_*`, used in the paper) and offline validation on 15 videos (`offline_*`) | 6 (edge), 1 (offline) |
| `positive_videos/ideal_latency.csv` | Light Only, Best Only and Two-Step with ideal models (`experiments/ideal_latency.py`): FPS, frame count, onset and latency per video | 1 (deterministic) |
| `positive_videos/flame_tc_grid_search.csv` | FLAME: precision/recall/F1 of the detector threshold on 10 videos (choice of c = 0.25) | — |
| `positive_videos/dumitru_latency.xlsx` | Dumitru & Spînu (arXiv:2607.03131, YOLOv10n + 5-level cascade), 6 configurations (Table 11) | 6 |
| `positive_videos/dumitru_latency_details.csv` | Same runs: alarm frame and time, premature alarm, inference count and inference/post-processing time per video and run | 6 |
| `negative_videos/saferfire_false_alarms.xlsx` | Light Only, Best Only and Two-Step, 46 configurations | 1 |
| `negative_videos/onfire_false_alarms.xlsx` | OnFire, full frame and superpixels | 1 |
| `negative_videos/venancio_false_alarms.xlsx` | Venâncio et al., 4 configurations | 1 |
| `negative_videos/flame_false_alarms.xlsx` | FLAME, edge regime, 3 configurations | 1 |
| `negative_videos/dumitru_false_alarms.xlsx` | Dumitru & Spînu, 6 configurations | 1 |

## Values

- Positive videos: latency in seconds after the annotated onset (`data/onset_frames`); negative = alarm before the
  onset (premature alarm); empty = no alarm.
- Negative videos: 1 = false alarm, 0 = no alarm. These workbooks also have a `summary` sheet (false alarms per
  configuration) and a `details` sheet (alarm frame and time, inference counts, ... per video and run).

## Sheet names

| Method | Sheet | Meaning |
|---|---|---|
| SAFERFIRE | `LO_c0.25_W1T1` | Light Only, confidence threshold 0.25, temporal rule (W,T) = (1,1) |
| | `BO_c0.50` | Best Only, confidence threshold 0.50 |
| | `TS_cL0.125_cH0.75_W5T3` | Two-Step, c_L = 0.125, c_H = 0.75, (W,T) = (5,3) |
| Venâncio et al. | `AVT_c0.25_area0.05_win20` | confidence 0.25, area variation threshold 5 %, window of 20 frames |
| OnFire | `FF_thr0.50` | full-frame classifier, alarm when P(fire) ≥ 0.50 |
| | `SP_thr0.50_k1` | superpixels (SLIC), alarm when at least k = 1 superpixel has P(fire) ≥ 0.50 |
| FLAME | `edge_c0.25_fg0.20_tC1.0` | edge regime, detector threshold 0.25, foreground overlap T_fg = 0.20, confirmation time t_C = 1.0 s (t_O = 0.1 s) |
| | `offline_c0.25_fg0.20_tC1.0` | same, every frame processed (validation of the implementation) |
| Dumitru & Spînu | `raw_K1` | level 1 only (per-class confidence: fire ≥ 0.40, smoke ≥ 0.55), alarm on the first detection |
| | `size_K1`, `size_hsv_K1` | + size plausibility (0.3–85 % of the frame), + HSV colour check |
| | `size_K3` | size + temporal confirmation over K = 3 consecutive inspected frames (IoU tracking), no HSV |
| | `full_K3`, `full_K5` | all 5 levels (confidence, size, HSV, temporal confirmation with K = 3 or 5, cooldown) |

The experiment scripts write these names directly. Workbooks of the older scripts (truncated names such as
`two_step_Lconf0.25_N1_M1_Hcon_1`) are translated by `analysis/import_runs.py --legacy main|extra`.

## Adding runs

```bash
cd analysis
python import_runs.py ../runs/saferfire_positive/results.xlsx ../results/positive_videos/saferfire_latency.xlsx --dry-run
python import_runs.py ../runs/saferfire_positive/results.xlsx ../results/positive_videos/saferfire_latency.xlsx
uv run python make_all.py
```
