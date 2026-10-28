# Data

| Path | Contents |
|---|---|
| `onset_frames/<video>.txt` | Annotated frame at which the fire or smoke becomes visible in each of the 69 positive videos (reference for the latency) |
| `negative_videos.csv` | The 69 negative videos: identifier N1–N69 (as in the paper), file, set, original file and origin |
| `negative_videos_confusing_frames.csv` | Frame of each negative video scored as most fire- or smoke-like by YOLOv12m (thumbnails of Table 7) |
| `video_sources_positive.csv` | Source of each positive video P1–P69: link, cut (start and end time in the original), crop box, output size, frame rate, number of frames, onset frame and notes |
| `video_sources_negative.csv` | Source of each negative video N1–N69: link, file in the original archive (FIRESENSE, KMU) and processing |
| `selfmade_training_sources.csv` | Source videos of the Selfmade subset of the training dataset, with the number of frames extracted from each one |
| `prepare_videos.py` | Downloads and rebuilds the test videos from the two manifests above (YouTube, FIRESENSE, KMU) and lists the others for manual download |
| `videos/positive/`, `videos/negative/` | The test videos (1.5 GB, **not versioned**, third-party footage); alternatively set `SAFERFIRE_VIDEOS` to a folder with these two subfolders |

The positive videos are numbered P1–P69 in the natural order of their file names (`personal2` before `personal10`),
as in Tables A.1–A.9 (`analysis/latency_analysis.py`, `create_video_mapping`).
The test videos are not redistributed: they are given by reference in the two `video_sources_*.csv` files, and
third-party videos remain subject to the terms of their original sources.
