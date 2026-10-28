# Data

| Path | Contents |
|---|---|
| `onset_frames/<video>.txt` | Annotated frame at which the fire or smoke becomes visible in each of the 69 positive videos (reference for the latency) |
| `negative_videos.csv` | The 69 negative videos: identifier N1–N69 (as in the paper), file, set, original file and origin |
| `negative_videos_confusing_frames.csv` | Frame of each negative video scored as most fire- or smoke-like by YOLOv12m (thumbnails of Table 7) |
| `videos/positive/`, `videos/negative/` | The videos (1.5 GB, **not versioned**); alternatively set `SAFERFIRE_VIDEOS` to a folder with these two subfolders |

The positive videos are numbered P1–P69 in the natural order of their file names (`personal2` before `personal10`),
as in Tables A.1–A.9 (`analysis/latency_analysis.py`, `create_video_mapping`).
All videos are publicly available at https://github.com/rubensanto/two-stage-fire-smoke-edge-device-cnn.
