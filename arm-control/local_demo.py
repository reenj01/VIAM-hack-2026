"""No-robot test harness for the gaze -> YOLO lock-on.

Runs YOLO locally (ultralytics) on a "scene" -- a still image, a video file,
or a second camera -- while the laptop webcam tracks your gaze. Look at a
detected object, hold, and it locks with a frozen snapshot. It uses the same
GazeLockController as main.py, so what you validate here is exactly what the
robot pipeline runs.

    python local_demo.py --scene path/to/photo.jpg
    python local_demo.py --scene path/to/clip.mp4
    python local_demo.py --scene 1            # a second camera, by index

Keys: Q quit, R release the lock, C recalibrate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from gaze_lock import Box, GazeLockController, draw_live, draw_locked, filter_background_boxes
from webcam_gaze import (
    BLINK_EAR_THRESHOLD,
    CALIBRATION_PATH,
    GazeCalibration,
    GazeSmoother,
    WebcamGazeTracker,
    estimate_gaze,
    run_calibration,
)

WINDOW = "Gaze lock demo (Q quit, R release, C recalibrate)"
MAX_SCENE_WIDTH = 1280


class Scene:
    """A still image, a video file (looped), or a camera index."""

    def __init__(self, source: str):
        self.image = None
        self.cap = None
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
        elif Path(source).is_file() and (img := cv2.imread(source)) is not None:
            self.image = img
        else:
            self.cap = cv2.VideoCapture(source)
        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f"Could not open scene source: {source}")

    @property
    def static(self) -> bool:
        return self.image is not None

    def read(self):
        if self.image is not None:
            return self.image.copy()
        ok, frame = self.cap.read()
        if not ok and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cap is not None:
            self.cap.release()


def fit_width(frame, max_w: int):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    return cv2.resize(frame, (max_w, int(h * max_w / w)))


def load_yolo(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed. Install it with: pip install ultralytics")
    return YOLO(weights)


def detect(model, frame, conf: float) -> list[Box]:
    result = model.predict(frame, conf=conf, verbose=False)[0]
    boxes: list[Box] = []
    rows = zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
    for i, (xyxy, c, k) in enumerate(rows):
        x0, y0, x1, y1 = (int(v) for v in xyxy)
        boxes.append(Box(x0, y0, x1, y1, result.names[int(k)], float(c), i))
    return filter_background_boxes(boxes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="image path, video path, or camera index")
    ap.add_argument("--webcam-index", type=int, default=0, help="camera that watches your face")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--detect-every", type=int, default=2, help="run YOLO every N frames (video/camera)")
    ap.add_argument("--skip-calibration", action="store_true", help="reuse the last saved calibration")
    args = ap.parse_args()

    scene = Scene(args.scene)
    first = scene.read()
    if first is None:
        raise SystemExit("Scene produced no frames")
    frame_h, frame_w = fit_width(first, MAX_SCENE_WIDTH).shape[:2]

    model = load_yolo(args.weights)
    gaze = WebcamGazeTracker(camera_index=args.webcam_index)
    smoother = GazeSmoother()
    lock = GazeLockController()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def calibrate() -> GazeCalibration:
        return run_calibration(gaze, frame_w, frame_h, window_name=WINDOW, keep_window=True)

    calib = None
    if args.skip_calibration and CALIBRATION_PATH.exists():
        calib = GazeCalibration.load()
        if (calib.frame_w, calib.frame_h) != (frame_w, frame_h):
            print("[demo] saved calibration is for a different frame size, recalibrating")
            calib = None
    if calib is None:
        calib = calibrate()

    boxes: list[Box] = []
    frame_idx = 0
    try:
        while True:
            if lock.is_locked:
                cv2.imshow(WINDOW, draw_locked(lock.locked, ["Press R to release, Q to quit"]))
                key = cv2.waitKey(30) & 0xFF
                if key == ord("r"):
                    lock.release()
                    smoother.reset()
                elif key == ord("q"):
                    break
                continue

            frame = scene.read()
            if frame is None:
                break
            frame = fit_width(frame, MAX_SCENE_WIDTH)
            if scene.static:
                if frame_idx == 0:
                    boxes = detect(model, frame, args.conf)
            elif frame_idx % args.detect_every == 0:
                boxes = detect(model, frame, args.conf)
            frame_idx += 1

            gaze_pt, ear = estimate_gaze(gaze, calib, smoother)
            hovered, progress, locked = lock.update(frame, boxes, gaze_pt)
            if locked is not None:
                b = locked.box
                print(f"[lock] locked onto '{b.label}' ({b.confidence:.2f}) "
                      f"box=({b.x0},{b.y0})-({b.x1},{b.y1})")
                continue

            view = draw_live(frame, boxes, hovered, progress, gaze_pt)
            if gaze_pt is None:
                cv2.putText(view, "No face detected", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 80, 255), 2, cv2.LINE_AA)
            elif ear is not None and ear < BLINK_EAR_THRESHOLD:
                cv2.putText(view, "BLINK", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calib = calibrate()
                smoother.reset()
    finally:
        gaze.close()
        scene.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
