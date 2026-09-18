"""Reusable webcam gaze tracker.

The feature extraction and calibration math here are ported directly from
github/gaze_dot.py, which the team tested and found accurate. This module
wraps that same algorithm in a class so it can be driven frame-by-frame from
main.py's loop, and adds dwell-based selection + blink detection on top
(gaze_dot.py itself only draws a cursor dot; it has no selection logic).

Unlike gaze_dot.py's standalone script, calibration here targets the pixel
space of whatever window you tell it to calibrate against (e.g. the window
showing the RealSense feed), not the full OS screen -- so gaze coordinates
land directly in the same pixel space as the video you're hit-testing
against, with no extra screen-to-window remapping step.

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


def _landmark_xy(landmarks, index: int) -> np.ndarray:
    point = landmarks[index]
    return np.array([point.x, point.y], dtype=np.float64)


def gaze_features(landmarks) -> Optional[np.ndarray]:
    """Same feature vector as github/gaze_dot.py's gaze_features()."""
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
    """Same ridge-regularized fit as github/gaze_dot.py's fit_calibration()."""
    x = np.vstack(samples)
    y = np.vstack(targets_px)
    regularization = 1e-3
    return np.linalg.solve(x.T @ x + regularization * np.eye(x.shape[1]), x.T @ y)


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


def run_calibration(tracker: WebcamGazeTracker, frame_w: int, frame_h: int,
                     window_name: str = "Gaze Calibration",
                     samples_per_point: int = 20,
                     settle_seconds: float = 0.5,
                     hold_seconds: float = 1.5) -> GazeCalibration:
    """9-point calibration against a window of size (frame_w, frame_h).

    Same target layout and per-point timing as github/gaze_dot.py, but driven
    automatically by a hold duration rather than a SPACE keypress, and scaled
    to an arbitrary window size rather than the full screen.
    """
    targets = [
        (0.15, 0.15), (0.50, 0.15), (0.85, 0.15),
        (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),
        (0.15, 0.85), (0.50, 0.85), (0.85, 0.85),
    ]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_w, frame_h)

    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for idx, (tx, ty) in enumerate(targets):
        px, py = int(tx * frame_w), int(ty * frame_h)
        target_samples: list[np.ndarray] = []
        phase_start = time.monotonic()

        while True:
            frame, landmarks, feats, _ = tracker.read()
            canvas = frame.copy() if frame is not None else np.zeros((frame_h, frame_w, 3), np.uint8)
            canvas = cv2.resize(canvas, (frame_w, frame_h))

            elapsed = time.monotonic() - phase_start
            settling = elapsed < settle_seconds
            progress = min(elapsed / (hold_seconds + settle_seconds), 1.0)
            color = (0, 165, 255) if settling else (0, 255, 255)
            cv2.circle(canvas, (px, py), 24, color, 3)
            cv2.circle(canvas, (px, py), max(1, int(20 * progress)), color, -1)
            cv2.putText(canvas, f"Target {idx + 1}/{len(targets)} - look at the dot",
                        (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if feats is not None and not settling:
                target_samples.append(feats)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyWindow(window_name)
                raise SystemExit("Calibration cancelled")

            if elapsed >= hold_seconds + settle_seconds:
                break

        if len(target_samples) >= min(10, samples_per_point // 2):
            all_features.append(np.median(target_samples, axis=0))
            all_targets.append(np.array([px, py], dtype=np.float64))
        else:
            print(f"[calibration] target {idx + 1} skipped: face not tracked reliably")

    cv2.destroyWindow(window_name)

    if len(all_features) < 6:
        raise RuntimeError("Too few good calibration points captured; try again with better lighting.")

    mapping = fit_calibration(all_features, all_targets)
    calib = GazeCalibration(mapping=mapping, frame_w=frame_w, frame_h=frame_h)
    calib.save()
    print(f"[calibration] done ({len(all_features)}/{len(targets)} points), saved to {CALIBRATION_PATH}")
    return calib


class DwellSelector:
    """Tracks how long gaze has continuously hovered the same label."""

    def __init__(self, dwell_seconds: float = DWELL_SECONDS):
        self.dwell_seconds = dwell_seconds
        self._target: Optional[str] = None
        self._started_at: Optional[float] = None

    def update(self, hovered: Optional[str]) -> tuple[Optional[str], float]:
        """Returns (selected_label_or_None, progress_0_to_1)."""
        if hovered is None:
            self._target, self._started_at = None, None
            return None, 0.0
        if hovered != self._target:
            self._target, self._started_at = hovered, time.monotonic()
            return None, 0.0
        elapsed = time.monotonic() - self._started_at
        progress = min(1.0, elapsed / self.dwell_seconds)
        if elapsed >= self.dwell_seconds:
            return hovered, 1.0
        return None, progress
