"""Step 3: train YOLOv8 on dataset/data.yaml, on the GPU if there is one.

    python train_yolo.py                       # yolov8n, 60 epochs
    python train_yolo.py --model yolov8s.pt --epochs 80

Ends by printing where best.pt is and how to deploy it to the robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n.pt", help="starting weights (n = fastest, s = a bit more accurate)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="cup")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found; run dataset_label.py first")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(f"missing dependency ({e}); install torch + ultralytics first")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU (slow: expect 10x longer)'}")

    model = YOLO(args.model)
    results = model.train(data=str(DATA_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          device=device, project=str(ROOT / "runs"), name=args.name, exist_ok=True,
                          patience=20, plots=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[train] done. best weights: {best}")
    print("[train] deploy: copy best.pt to the robot's computer, then in the Viam app set the yolo-detector "
          "service's model_location to that path and save. Detections will use your class names.")
    print("[train] the yolo-detector's labels become the names in dataset/data.yaml (e.g. 'cup'), so main.py "
          "and objects-3d keep working unchanged.")


if __name__ == "__main__":
    main()
