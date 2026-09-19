"""Step 2: label the captured images, then split them into a YOLO dataset.

Auto-labels with the stock COCO YOLO (which already knows 'cup', 'bottle',
'cell phone', ...), then walks you through each image to review:

    python dataset_label.py --classes cup
    python dataset_label.py --classes cup bottle --conf 0.25

Review keys:  ENTER/Y keep    N discard this image    D clear the boxes
              drag with the mouse to draw a box (class = first --classes entry,
              press 1-9 first to pick another)    U undo last box    Q stop

Writes dataset/images/{train,val}, dataset/labels/{train,val}, dataset/data.yaml.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent / "dataset"
RAW_DIR = ROOT / "raw"


def load_yolo(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed: pip install ultralytics")
    return YOLO(weights)


def autolabel(model, image, class_names: list[str], conf: float) -> list[tuple[int, float, float, float, float]]:
    """Boxes as (class_index, cx, cy, w, h) normalized, YOLO label format."""
    result = model.predict(image, conf=conf, verbose=False)[0]
    h, w = image.shape[:2]
    out = []
    for xyxy, k in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        name = result.names[int(k)]
        if name not in class_names:
            continue
        x0, y0, x1, y1 = xyxy
        out.append((class_names.index(name), (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
    return out


class Reviewer:
    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        self.current_class = 0
        self.boxes: list[tuple[int, float, float, float, float]] = []
        self.drag_start = None
        self.drag_now = None
        self.image = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            h, w = self.image.shape[:2]
            (x0, y0), (x1, y1) = self.drag_start, (x, y)
            x0, x1 = sorted((max(0, x0), min(w, x1)))
            y0, y1 = sorted((max(0, y0), min(h, y1)))
            if x1 - x0 > 8 and y1 - y0 > 8:
                self.boxes.append((self.current_class, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))
            self.drag_start = self.drag_now = None

    def render(self, title: str):
        view = self.image.copy()
        h, w = view.shape[:2]
        for k, cx, cy, bw, bh in self.boxes:
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 220, 0), 2)
            cv2.putText(view, self.class_names[k], (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        if self.drag_start and self.drag_now:
            cv2.rectangle(view, self.drag_start, self.drag_now, (0, 200, 255), 2)
        cv2.rectangle(view, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(view, f"{title}  |  class: {self.class_names[self.current_class]}  |  "
                          "ENTER keep  N discard  D clear  drag=draw  U undo  Q stop",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", nargs="+", required=True, help="COCO class names to keep, e.g. cup bottle")
    ap.add_argument("--weights", default="yolov8m.pt", help="stock model used for auto-labeling")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--val", type=float, default=0.2, help="fraction of images for validation")
    ap.add_argument("--no-review", action="store_true", help="trust the auto-labels, skip the viewer")
    args = ap.parse_args()

    images = sorted(RAW_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {RAW_DIR}; run dataset_capture.py first")
    model = load_yolo(args.weights)
    print(f"[label] {len(images)} images, auto-labeling classes {args.classes} with {args.weights}")

    labeled: list[tuple[Path, list]] = []
    reviewer = Reviewer(args.classes)
    window = "Label review"
    if not args.no_review:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, reviewer.on_mouse)

    for i, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            continue
        boxes = autolabel(model, image, args.classes, args.conf)
        if args.no_review:
            if boxes:
                labeled.append((path, boxes))
            continue
        reviewer.image, reviewer.boxes = image, boxes
        keep = None
        while keep is None:
            cv2.imshow(window, reviewer.render(f"{i + 1}/{len(images)} {path.name}"))
            key = cv2.waitKey(20) & 0xFF
            if key in (13, ord("y")):
                keep = True
            elif key == ord("n"):
                keep = False
            elif key == ord("d"):
                reviewer.boxes = []
            elif key == ord("u") and reviewer.boxes:
                reviewer.boxes.pop()
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(args.classes):
                reviewer.current_class = key - ord("1")
            elif key == ord("q"):
                keep = False
                images = images[:i]
                break
        if keep and reviewer.boxes:
            labeled.append((path, list(reviewer.boxes)))
        elif keep:
            print(f"[label] {path.name}: no boxes, skipped")
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

    if len(labeled) < 10:
        raise SystemExit(f"only {len(labeled)} labeled images; need more before training")

    random.seed(0)
    random.shuffle(labeled)
    n_val = max(1, int(len(labeled) * args.val))
    splits = {"val": labeled[:n_val], "train": labeled[n_val:]}
    for split, items in splits.items():
        img_dir, lbl_dir = ROOT / "images" / split, ROOT / "labels" / split
        shutil.rmtree(img_dir, ignore_errors=True)
        shutil.rmtree(lbl_dir, ignore_errors=True)
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for path, boxes in items:
            shutil.copy(path, img_dir / path.name)
            (lbl_dir / f"{path.stem}.txt").write_text(
                "".join(f"{k} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for k, cx, cy, w, h in boxes))
    data_yaml = ROOT / "data.yaml"
    data_yaml.write_text(
        f"path: {ROOT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(args.classes)))
    print(f"[label] wrote {len(splits['train'])} train / {len(splits['val'])} val images and {data_yaml}")


if __name__ == "__main__":
    main()
