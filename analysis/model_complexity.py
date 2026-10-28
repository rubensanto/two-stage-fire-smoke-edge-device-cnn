"""
Parameters, MACs and FLOPs of the models in Table 6 (ONNX format, 640x640 input, Section 4.4).

Two steps, run in separate environments because Ultralytics 8.3.99 needs numpy <= 2.1.1 and onnx-opcounter 0.1.0
needs numpy >= 2.5:

1. export: every .pt checkpoint in --pt-dir is exported to ONNX (output/onnx_export/) with the same settings as
   the models deployed on the RPi5 (Ultralytics 8.3.99, batch 1, imgsz 640, simplify=True, no NMS):

    uv run --no-project --python 3.12 --with ultralytics==8.3.99 --with torch==2.4.1 --with torchvision==0.19.1 \
        --with onnx --with onnxslim python model_complexity.py export --pt-dir <folder with *_best.pt>

2. count: parameters and MACs of the deployed ONNX models (models/saferfire/) and of the exported ones, with
   onnx-opcounter 0.1.0. A model present in both is checked to give the same counts, which confirms that the
   export reproduces the deployed graphs. Writes output/model_complexity.csv.

    uv run --no-project --python 3.12 --with onnx-opcounter==0.1.0 python model_complexity.py count

FLOPs are reported as 2 x MACs (one MAC = one multiplication and one addition). onnx-opcounter 0.0.4 gives slightly
different counts (e.g. 8.157 instead of 8.149 GFLOPs for YOLOv8n), and with numpy 1.x on Windows its MACs overflow.
"""

import argparse
import csv
import shutil
from pathlib import Path

from configurations import MODELS

IMGSZ = 640
OUTPUT = Path(__file__).resolve().parent / "output"
EXPORT_DIR = OUTPUT / "onnx_export"


def export(pt_dir):
    from ultralytics import RTDETR, YOLO

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for pt_path in sorted(pt_dir.glob("*.pt")):
        # Ultralytics writes the .onnx next to the checkpoint, so export from a copy
        local_pt = EXPORT_DIR / pt_path.name
        shutil.copy2(pt_path, local_pt)
        model_class = RTDETR if pt_path.name.upper().startswith("RT-DETR") else YOLO
        model_class(str(local_pt)).export(format="onnx", imgsz=IMGSZ, batch=1, simplify=True, dynamic=False,
                                          half=False, nms=False)
        local_pt.unlink()


def complexity(onnx_path):
    import onnx
    from onnx_opcounter import calculate_macs, calculate_params

    model = onnx.load(str(onnx_path))
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    macs = int(calculate_macs(model))
    if macs <= 0:
        raise RuntimeError(f"{onnx_path.name}: invalid MAC count {macs} (integer overflow, use numpy >= 2)")
    return {"input": "x".join(map(str, shape)), "params": int(calculate_params(model)), "macs": macs,
            "gflops": 2 * macs / 1e9}


def count():
    rows = []
    for source, folder in (("deployed onnx", MODELS / "saferfire"), ("exported from pt", EXPORT_DIR)):
        for onnx_path in sorted(folder.glob("*.onnx")):
            rows.append({"model": onnx_path.stem, "source": source, **complexity(onnx_path)})

    print(f"{'Model':<24} {'Source':<18} {'Input':<14} {'Parameters':>12} {'MACs':>16} {'GFLOPs':>9}")
    for row in rows:
        print(f"{row['model']:<24} {row['source']:<18} {row['input']:<14} {row['params']:>12,} "
              f"{row['macs']:>16,} {row['gflops']:>9.3f}")

    exported = {row["model"]: row for row in rows if row["source"] == "exported from pt"}
    for ref in (row for row in rows if row["source"] == "deployed onnx"):
        if ref["model"] in exported:
            same = all(exported[ref["model"]][key] == ref[key] for key in ("input", "params", "macs"))
            print(f"Check {ref['model']}: export {'matches' if same else 'DIFFERS FROM'} the deployed ONNX")

    OUTPUT.mkdir(exist_ok=True)
    with open(OUTPUT / "model_complexity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "source", "input", "params", "macs", "gflops"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT / 'model_complexity.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="step", required=True)
    export_parser = subparsers.add_parser("export", help="export the .pt checkpoints to ONNX")
    export_parser.add_argument("--pt-dir", type=Path, required=True, help="folder with the trained .pt checkpoints")
    subparsers.add_parser("count", help="count parameters, MACs and FLOPs of the ONNX models")
    args = parser.parse_args()

    if args.step == "export":
        export(args.pt_dir)
    else:
        count()


if __name__ == "__main__":
    main()
