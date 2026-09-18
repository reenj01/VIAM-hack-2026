"""Standalone webcam gaze-dot prototype.

Press C to calibrate the nine on-screen targets. Press R to recalibrate,
and Q or Escape to quit. This program never connects to Viam or any robot.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# Must be set before mediapipe is imported — prevents Metal/GPU init crash on macOS.
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
import numpy as np


WINDOW_NAME = "Gaze dot prototype"
MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
CAMERA_INDEX = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CALIBRATION_SECONDS = 1.5
SETTLE_SECONDS = 0.5          # NEW: ignore samples while the eyes are still moving
MIN_SAMPLES_PER_POINT = 10    # lowered: settle window eats part of the capture
DOT_RADIUS = 14
SMOOTHING = 0.80

TARGETS = [
    (0.15, 0.15), (0.50, 0.15), (0.85, 0.15),
    (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),
    (0.15, 0.85), (0.50, 0.85), (0.85, 0.85),
]

# FIX: MediaPipe iris indices are 468-472 = RIGHT eye, 473-477 = LEFT eye.
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)


def landmark_xy(landmarks, index: int) -> np.ndarray:
    point = landmarks[index]
    return np.array([point.x, point.y], dtype=np.float64)


def gaze_features(landmarks) -> np.ndarray | None:
    """Create scale-normalized iris features from a detected face."""
    try:
        left_iris = np.mean([landmark_xy(landmarks, i) for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([landmark_xy(landmarks, i) for i in RIGHT_IRIS], axis=0)
        left_inner, left_outer = (landmark_xy(landmarks, i) for i in LEFT_EYE_CORNERS)
        right_inner, right_outer = (landmark_xy(landmarks, i) for i in RIGHT_EYE_CORNERS)
    except IndexError:
        return None

    left_width = max(np.linalg.norm(left_outer - left_inner), 1e-5)
    right_width = max(np.linalg.norm(right_outer - right_inner), 1e-5)

    left_center = (left_inner + left_outer) / 2
    right_center = (right_inner + right_outer) / 2

    # FIX: normalize BOTH axes by eye width. The corners share almost the same
    # y, so the old vertical divisor was ~0 and amplified noise enormously.
    left_relative = (left_iris - left_center) / left_width
    right_relative = (right_iris - right_center) / right_width

    face_center = (left_center + right_center) / 2
    eye_distance = np.linalg.norm(right_center - left_center)

    # Averaged eye signal is steadier than either eye alone.
    mean_x = (left_relative[0] + right_relative[0]) / 2
    mean_y = (left_relative[1] + right_relative[1]) / 2

    # FIX: quadratic terms — a purely linear fit maps gaze poorly, especially
    # vertically. Still well-conditioned against 9 averaged calibration points.
    return np.array([
        left_relative[0], left_relative[1],
        right_relative[0], right_relative[1],
        mean_x, mean_y,
        mean_x * mean_x, mean_y * mean_y, mean_x * mean_y,
        face_center[0], face_center[1], eye_distance,
        1.0,
    ], dtype=np.float64)


def fit_calibration(samples: list[np.ndarray], targets_px: list[np.ndarray]) -> np.ndarray:
    """Fit ridge-regularized regression from eye features to screen x/y."""
    x = np.vstack(samples)
    y = np.vstack(targets_px)
    regularization = 1e-3
    return np.linalg.solve(x.T @ x + regularization * np.eye(x.shape[1]), x.T @ y)


def calibration_error(samples, targets_px, mapping) -> float:
    """NEW: mean pixel error on the calibration points themselves."""
    pred = np.vstack(samples) @ mapping
    return float(np.mean(np.linalg.norm(pred - np.vstack(targets_px), axis=1)))


def draw_target(frame, point, number, progress, settling) -> None:
    height, width = frame.shape[:2]
    x, y = int(point[0] * width), int(point[1] * height)
    color = (0, 165, 255) if settling else (0, 255, 255)
    cv2.circle(frame, (x, y), 24, color, 3)
    cv2.circle(frame, (x, y), max(1, int(20 * progress)), color, -1)
    cv2.putText(frame, str(number), (x - 8, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def draw_status(frame, message: str) -> None:
    cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 900), 75), (0, 0, 0), -1)
    cv2.putText(frame, message, (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Download it using the command in README.md."
        )

    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Try CAMERA_INDEX = 1 in gaze_dot.py.")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_face_blendshapes=False,
    )

    calibration_samples: list[np.ndarray] = []
    calibration_targets: list[np.ndarray] = []
    calibrated = False
    calibrating = False
    target_index = 0
    target_started_at = 0.0
    target_samples: list[np.ndarray] = []
    mapping: np.ndarray | None = None
    smoothed_dot: np.ndarray | None = None
    last_timestamp_ms = -1   # NEW: Tasks VIDEO mode requires strictly increasing stamps

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Press C to start calibration. Press Q or Escape to quit.")

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms)
            features = gaze_features(result.face_landmarks[0]) if result.face_landmarks else None

            now = time.monotonic()
            height, width = frame.shape[:2]

            if calibrating:
                if target_started_at == 0.0:
                    target_started_at = now
                elapsed = now - target_started_at
                settling = elapsed < SETTLE_SECONDS
                progress = min(elapsed / (CALIBRATION_SECONDS + SETTLE_SECONDS), 1.0)
                draw_target(frame, TARGETS[target_index], target_index + 1, progress, settling)
                draw_status(frame, f"Look directly at target {target_index + 1} of {len(TARGETS)}")

                # NEW: only collect after the settle window
                if features is not None and not settling:
                    target_samples.append(features)

                if elapsed >= CALIBRATION_SECONDS + SETTLE_SECONDS:
                    if len(target_samples) >= MIN_SAMPLES_PER_POINT:
                        # NEW: median is robust to blinks mid-capture
                        calibration_samples.append(np.median(target_samples, axis=0))
                        calibration_targets.append(np.array([
                            TARGETS[target_index][0] * width,
                            TARGETS[target_index][1] * height,
                        ]))
                    else:
                        print(f"Target {target_index + 1} skipped: face not tracked reliably.")
                    target_index += 1        # FIX: always advance, even on skip
                    target_samples = []
                    target_started_at = 0.0

                    if target_index == len(TARGETS):
                        # NEW: refuse to fit an underdetermined mapping
                        if len(calibration_samples) < 6:
                            print("Too few good targets; press C to try again.")
                            calibrating = False
                        else:
                            mapping = fit_calibration(calibration_samples, calibration_targets)
                            err = calibration_error(calibration_samples, calibration_targets, mapping)
                            print(f"Calibration complete on {len(calibration_samples)} points. "
                                  f"Mean fit error: {err:.1f}px")
                            calibrated = True
                            calibrating = False
                            smoothed_dot = None

            elif calibrated and features is not None and mapping is not None:
                predicted = features @ mapping
                predicted[0] = np.clip(predicted[0], 0, width - 1)
                predicted[1] = np.clip(predicted[1], 0, height - 1)
                smoothed_dot = (predicted if smoothed_dot is None
                                else SMOOTHING * smoothed_dot + (1 - SMOOTHING) * predicted)
                center = tuple(np.round(smoothed_dot).astype(int))
                cv2.circle(frame, center, DOT_RADIUS, (0, 0, 255), -1)
                cv2.circle(frame, center, DOT_RADIUS + 3, (255, 255, 255), 2)
                draw_status(frame, "Calibrated: red dot follows gaze. R recalibrates; Q quits.")

            elif calibrated:
                draw_status(frame, "Face not found. Face the camera, then look at the preview.")
            else:
                draw_status(frame, "Press C to calibrate. Sit still and keep your face in view.")

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("c"), ord("r")):
                calibration_samples, calibration_targets, target_samples = [], [], []
                calibrated = calibrating = False
                calibrating = True
                target_index = 0
                target_started_at = 0.0
                mapping = None
                smoothed_dot = None
                print("Calibration started. Look at each target until it fills.")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
