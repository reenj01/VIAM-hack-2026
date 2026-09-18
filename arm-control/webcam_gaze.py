"""Reusable webcam gaze tracker.

The feature extraction, calibration math, and the face-framing gate are ported
directly from gaze_dot.py at the repo root, which the team tested and found
accurate. This module wraps that same algorithm in a class so it can be
driven frame-by-frame from another program's loop, and adds gaze smoothing,
blink detection, and dwell timing on top (gaze_dot.py only draws a cursor
dot; it has no selection logic).

Unlike gaze_dot.py's standalone script, calibration here targets the pixel
space of whatever window you tell it to calibrate against (e.g. the window
showing the RealSense feed), so gaze coordinates land directly in the same
pixel space as the video you're hit-testing against. Calibrate and run in the
SAME window, at the same screen position: the mapping is to pixels on your
physical screen, so moving the window afterwards shifts everything.

No training on an eye dataset happens here or in gaze_dot.py: the MediaPipe
face/iris model is pretrained, and calibration is a small per-user ridge
regression fit at startup.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
CALIBRATION_PATH = Path(__file__).parent / "webcam_gaze_calibration.json"

# MediaPipe FaceLandmarker's 478-point mesh: 468-472 = right iris, 473-477 = left iris.
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
# Vertical lid points, used only for blink detection (not part of gaze_dot.py's
# original feature set).
RIGHT_LID = (159, 145)
LEFT_LID = (386, 374)

BLINK_EAR_THRESHOLD = 0.17
DWELL_SECONDS = 0.9
DWELL_GRACE_SECONDS = 0.25   # a brief gaze wobble outside the box doesn't reset the dwell
SMOOTHING = 0.80             # EMA weight on the previous gaze point (same as gaze_dot.py)

CALIBRATION_TARGETS = [
    (0.15, 0.15), (0.50, 0.15), (0.85, 0.15),
    (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),
    (0.15, 0.85), (0.50, 0.85), (0.85, 0.85),
]

# ID-photo-style framing gate, run once before the 9-point calibration starts
# (from gaze_dot.py). Keeps the face centered and at a consistent distance so
# the calibration is taken from the same head position it will be used from.
FRAME_TARGET_CX = 0.50
FRAME_TARGET_CY = 0.45
FRAME_TARGET_W = 0.34
FRAME_TARGET_H = 0.62
FRAME_POSITION_TOLERANCE = 0.05
FRAME_SIZE_RATIO_LOW = 0.85
FRAME_SIZE_RATIO_HIGH = 1.20
FRAME_HOLD_SECONDS = 1.0
FRAME_MESSAGES = {
    "move_left": "Move left to center your face in the frame",
    "move_right": "Move right to center your face in the frame",
    "move_up": "Move up to center your face in the frame",
    "move_down": "Move down to center your face in the frame",
    "move_closer": "Move closer to the camera",
    "move_back": "Move back from the camera",
}


def _landmark_xy(landmarks, index: int) -> np.ndarray:
    point = landmarks[index]
    return np.array([point.x, point.y], dtype=np.float64)


def gaze_features(landmarks) -> Optional[np.ndarray]:
    """Same feature vector as gaze_dot.py's gaze_features()."""
    try:
        left_iris = np.mean([_landmark_xy(landmarks, i) for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([_landmark_xy(landmarks, i) for i in RIGHT_IRIS], axis=0)
        left_inner, left_outer = (_landmark_xy(landmarks, i) for i in LEFT_EYE_CORNERS)
        right_inner, right_outer = (_landmark_xy(landmarks, i) for i in RIGHT_EYE_CORNERS)
    except IndexError:
        return None

    left_width = max(np.linalg.norm(left_outer - left_inner), 1e-5)
    right_width = max(np.linalg.norm(right_outer - right_inner), 1e-5)

    left_center = (left_inner + left_outer) / 2
    right_center = (right_inner + right_outer) / 2

    left_relative = (left_iris - left_center) / left_width
    right_relative = (right_iris - right_center) / right_width

    face_center = (left_center + right_center) / 2
    eye_distance = np.linalg.norm(right_center - left_center)

    mean_x = (left_relative[0] + right_relative[0]) / 2
    mean_y = (left_relative[1] + right_relative[1]) / 2

    return np.array([
        left_relative[0], left_relative[1],
        right_relative[0], right_relative[1],
        mean_x, mean_y,
        mean_x * mean_x, mean_y * mean_y, mean_x * mean_y,
        face_center[0], face_center[1], eye_distance,
        1.0,
    ], dtype=np.float64)


def eye_aspect_ratio(landmarks) -> float:
    """Mean eye-aspect-ratio across both eyes; drops sharply on a blink."""

    def ear(top_i, bottom_i, outer_i, inner_i):
        top, bottom = _landmark_xy(landmarks, top_i), _landmark_xy(landmarks, bottom_i)
        outer, inner = _landmark_xy(landmarks, outer_i), _landmark_xy(landmarks, inner_i)
        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(outer - inner)
        return vertical / (horizontal + 1e-6)

    r = ear(*RIGHT_LID, *RIGHT_EYE_CORNERS)
    l = ear(*LEFT_LID, *LEFT_EYE_CORNERS)
    return (r + l) / 2


def fit_calibration(samples: list[np.ndarray], targets_px: list[np.ndarray]) -> np.ndarray:
    """Same ridge-regularized fit as gaze_dot.py's fit_calibration()."""
    x = np.vstack(samples)
    y = np.vstack(targets_px)
    regularization = 1e-3
    return np.linalg.solve(x.T @ x + regularization * np.eye(x.shape[1]), x.T @ y)


def calibration_error(samples, targets_px, mapping) -> float:
    """Mean pixel error on the calibration points themselves."""
    pred = np.vstack(samples) @ mapping
    return float(np.mean(np.linalg.norm(pred - np.vstack(targets_px), axis=1)))


def face_bbox_normalized(landmarks) -> tuple[float, float, float, float]:
    """Bounding box (left, top, right, bottom) of all face landmarks, normalized [0, 1]."""
    xs = np.array([p.x for p in landmarks])
    ys = np.array([p.y for p in landmarks])
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def evaluate_framing(bbox) -> tuple[str, bool]:
    """Compare the face bbox to the target oval and return (status_code, aligned)."""
    left, top, right, bottom = bbox
    face_cx = (left + right) / 2
    face_cy = (top + bottom) / 2
    face_h = bottom - top

    dx = face_cx - FRAME_TARGET_CX
    dy = face_cy - FRAME_TARGET_CY
    size_ratio = face_h / FRAME_TARGET_H

    # The frame is mirrored (selfie view), so on-screen directions read like a
    # real mirror: face right-of-center on screen -> tell the user to move left.
    if abs(dx) > FRAME_POSITION_TOLERANCE:
        return ("move_left" if dx > 0 else "move_right"), False
    if abs(dy) > FRAME_POSITION_TOLERANCE:
        return ("move_up" if dy > 0 else "move_down"), False
    if size_ratio < FRAME_SIZE_RATIO_LOW:
        return "move_closer", False
    if size_ratio > FRAME_SIZE_RATIO_HIGH:
        return "move_back", False
    return "aligned", True


def draw_id_frame(canvas, aligned: bool) -> None:
    height, width = canvas.shape[:2]
    center = (int(FRAME_TARGET_CX * width), int(FRAME_TARGET_CY * height))
    axes = (int(FRAME_TARGET_W * width / 2), int(FRAME_TARGET_H * height / 2))
    color = (0, 200, 0) if aligned else (0, 165, 255)
    cv2.ellipse(canvas, center, axes, 0, 0, 360, color, 3)


def letterbox(frame, width: int, height: int) -> np.ndarray:
    """Fit frame inside width x height keeping its aspect ratio, black bars around."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    fh, fw = frame.shape[:2]
    scale = min(width / fw, height / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh))
    return canvas


def _draw_status(canvas, message: str) -> None:
    cv2.rectangle(canvas, (10, 10), (min(canvas.shape[1] - 10, 900), 75), (0, 0, 0), -1)
    cv2.putText(canvas, message, (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def _draw_target(canvas, point_px, number: int, progress: float, settling: bool) -> None:
    x, y = point_px
    color = (0, 165, 255) if settling else (0, 255, 255)
    cv2.circle(canvas, (x, y), 24, color, 3)
    cv2.circle(canvas, (x, y), max(1, int(20 * progress)), color, -1)
    cv2.putText(canvas, str(number), (x - 8, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


@dataclass
class GazeCalibration:
    mapping: np.ndarray
    frame_w: int
    frame_h: int

    def predict(self, feat: np.ndarray) -> tuple[float, float]:
        pred = feat @ self.mapping
        x = float(np.clip(pred[0], 0, self.frame_w - 1))
        y = float(np.clip(pred[1], 0, self.frame_h - 1))
        return x, y

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.write_text(json.dumps({
            "mapping": self.mapping.tolist(),
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
        }))

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "GazeCalibration":
        data = json.loads(path.read_text())
        return cls(
            mapping=np.array(data["mapping"]),
            frame_w=data["frame_w"],
            frame_h=data["frame_h"],
        )


class WebcamGazeTracker:
    """Owns the laptop webcam + MediaPipe FaceLandmarker for live gaze tracking."""

    def __init__(self, camera_index: int = 0, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Missing MediaPipe model at {model_path}")

        base_options = BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}")
        self._start = time.monotonic()
        self._last_timestamp_ms = -1
        # Camera warmup: auto-exposure can take several frames to settle.
        for _ in range(15):
            self._cap.read()

    def read(self):
        """Returns (frame_bgr, landmarks|None, features|None, ear|None)."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, None, None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._start) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return frame, None, None, None
        landmarks = result.face_landmarks[0]
        feats = gaze_features(landmarks)
        ear = eye_aspect_ratio(landmarks)
        return frame, landmarks, feats, ear

    def close(self):
        self._cap.release()
        self._landmarker.close()


class GazeSmoother:
    """Exponential moving average over the predicted gaze point."""

    def __init__(self, smoothing: float = SMOOTHING):
        self.smoothing = smoothing
        self._pt: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._pt = None

    def update(self, pt: tuple[float, float]) -> tuple[float, float]:
        p = np.array(pt, dtype=np.float64)
        self._pt = p if self._pt is None else self.smoothing * self._pt + (1 - self.smoothing) * p
        return float(self._pt[0]), float(self._pt[1])


def estimate_gaze(tracker: WebcamGazeTracker, calib: GazeCalibration,
                  smoother: GazeSmoother) -> tuple[Optional[tuple[float, float]], Optional[float]]:
    """One webcam frame -> (smoothed gaze point in window pixels or None, eye aspect ratio)."""
    _, _, feats, ear = tracker.read()
    if feats is None:
        smoother.reset()
        return None, ear
    return smoother.update(calib.predict(feats)), ear


class DwellSelector:
    """Tracks how long gaze has continuously hovered the same label."""

    def __init__(self, dwell_seconds: float = DWELL_SECONDS,
                 grace_seconds: float = DWELL_GRACE_SECONDS):
        self.dwell_seconds = dwell_seconds
        self.grace_seconds = grace_seconds
        self._target: Optional[str] = None
        self._started_at: Optional[float] = None
        self._last_seen: Optional[float] = None

    def reset(self) -> None:
        self._target, self._started_at, self._last_seen = None, None, None

    def update(self, hovered: Optional[str]) -> tuple[Optional[str], float]:
        """Returns (selected_label_or_None, progress_0_to_1)."""
        now = time.monotonic()
        if hovered is None:
            if self._target is not None and now - self._last_seen <= self.grace_seconds:
                return None, min(1.0, (now - self._started_at) / self.dwell_seconds)
            self.reset()
            return None, 0.0
        if hovered != self._target:
            self._target, self._started_at, self._last_seen = hovered, now, now
            return None, 0.0
        self._last_seen = now
        elapsed = now - self._started_at
        if elapsed >= self.dwell_seconds:
            return hovered, 1.0
        return None, elapsed / self.dwell_seconds


def run_calibration(tracker: WebcamGazeTracker, frame_w: int, frame_h: int,
                     window_name: str = "Gaze Calibration",
                     settle_seconds: float = 0.5,
                     hold_seconds: float = 1.5,
                     min_samples_per_point: int = 10,
                     keep_window: bool = False) -> GazeCalibration:
    """Framing gate, then 9-point calibration, against a window of size (frame_w, frame_h).

    Same flow and timing as gaze_dot.py: center your face in the oval, hold
    for a second, then each target is shown for settle + hold seconds and the
    median of the samples collected after the settle window is kept.
    Pass keep_window=True to leave the window open for the caller to reuse
    (so the run happens at the exact screen position that was calibrated).
    """
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def finish_window():
        if not keep_window:
            cv2.destroyWindow(window_name)

    def show(canvas):
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            finish_window()
            raise SystemExit("Calibration cancelled")

    def webcam_canvas(frame):
        if frame is None:
            return np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        return letterbox(frame, frame_w, frame_h)

    # Phase 1: framing gate.
    hold_started = 0.0
    while True:
        frame, landmarks, _, _ = tracker.read()
        now = time.monotonic()
        aligned, remaining = False, 1.0
        if landmarks is None:
            hold_started = 0.0
            message = "Face not found. Center your face in the frame."
        else:
            status, aligned = evaluate_framing(face_bbox_normalized(landmarks))
            if aligned:
                if hold_started == 0.0:
                    hold_started = now
                remaining = max(0.0, FRAME_HOLD_SECONDS - (now - hold_started))
                message = "Hold still..." if remaining > 0 else "Starting calibration..."
            else:
                hold_started = 0.0
                message = FRAME_MESSAGES[status]

        # The oval goes on the raw webcam frame, so it sits in the same
        # normalized space the face bbox is judged in, and is then letterboxed
        # with the frame rather than stretched.
        source = frame if frame is not None else np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        draw_id_frame(source, aligned)
        canvas = webcam_canvas(source)
        _draw_status(canvas, message)
        show(canvas)
        if aligned and remaining <= 0.0:
            break

    # Phase 2: the nine targets.
    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for idx, (tx, ty) in enumerate(CALIBRATION_TARGETS):
        px, py = int(tx * frame_w), int(ty * frame_h)
        samples: list[np.ndarray] = []
        started = time.monotonic()
        while True:
            frame, _, feats, _ = tracker.read()
            canvas = webcam_canvas(frame)
            elapsed = time.monotonic() - started
            settling = elapsed < settle_seconds
            progress = min(elapsed / (hold_seconds + settle_seconds), 1.0)
            _draw_target(canvas, (px, py), idx + 1, progress, settling)
            _draw_status(canvas, f"Look directly at target {idx + 1} of {len(CALIBRATION_TARGETS)}")
            if feats is not None and not settling:
                samples.append(feats)
            show(canvas)
            if elapsed >= hold_seconds + settle_seconds:
                break

        if len(samples) >= min_samples_per_point:
            all_features.append(np.median(samples, axis=0))
            all_targets.append(np.array([px, py], dtype=np.float64))
        else:
            print(f"[calibration] target {idx + 1} skipped: face not tracked reliably")

    finish_window()

    if len(all_features) < 6:
        raise RuntimeError("Too few good calibration points captured; try again with better lighting.")

    mapping = fit_calibration(all_features, all_targets)
    calib = GazeCalibration(mapping=mapping, frame_w=frame_w, frame_h=frame_h)
    calib.save()
    err = calibration_error(all_features, all_targets, mapping)
    print(f"[calibration] done on {len(all_features)}/{len(CALIBRATION_TARGETS)} points, "
          f"mean fit error {err:.1f}px, saved to {CALIBRATION_PATH}")
    return calib
