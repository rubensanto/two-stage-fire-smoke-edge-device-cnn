# Models (Git LFS)

| File | Used by | Notes |
|---|---|---|
| `saferfire/YOLOv8n_best.onnx` | `saferfire_*.py`, `analysis/alarm_frames` | Lightweight model (first stage, Light Only); exact file used on the RPi5 |
| `saferfire/YOLOv12m_best.onnx` | `saferfire_*.py`, `analysis/alarm_frames` | High-capacity model (verification, Best Only); exact file used on the RPi5 |
| `venancio/yolov5_venancio.onnx` | `venancio_*.py` | YOLOv5l fine-tuned on our training data (weights embedded in the .onnx) |
| `flame/YOLOv8s_best.pt` | `flame_*.py` | YOLOv8s trained on our data (Tables 4 and 5), detector of FLAME |
| `onfire/shufflenet_ff.pt`, `onfire/shufflenet_sp.pt` | `onfire_*.py` | Published ShuffleNetV2-OnFire weights (full frame and superpixel); not included here, download them from the original source with `experiments/third_party/download_onfire_weights.sh` |
| `dumitru/YOLOv10n_best.pt` | `dumitru_*.py` | YOLOv10n trained on our data (fire/smoke), detector of Dumitru & Spînu (the paper does not give the variant) |

The files are stored with Git LFS: run `git lfs install` once before cloning (or `git lfs pull` after cloning).
