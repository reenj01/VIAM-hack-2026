"""Reusable webcam gaze tracker.

Built on MediaPipe's pretrained FaceLandmarker (no training here). Per-user
calibration is a small ridge regression from eye + head-pose features to
window pixels, fit at startup.

Design notes:
- Features: each iris's position inside its own eye (along the eye's own
  corner-to-corner axis, in units of eye width), plus head yaw/pitch proxies
  (nose tip relative to the eye midpoint, in units of inter-eye distance),
  plus their products. When the head turns left while the eyes stay on the
  same screen point, the irises rotate right inside the head; only a model
  that sees BOTH can tell that apart from actually looking right.
- Calibration therefore runs in stages: head still (a 5x5 grid, eyes only),
  then eight directions where the head follows the eyes slightly, two targets
  each. One stage would leave the head features constant and their weights
  arbitrary (the "move your head and the dot jumps" failure).
  --quick-calibration = straight only.
- Blinks: per-user threshold from an open-eye baseline; blink frames are
  skipped while calibrating, and live the cursor holds until the eyes are
  fully open again for a moment (the reopening lids corrupt the iris fit).
- Out-of-range predictions (head far outside the calibrated range) return
  None rather than a point pinned to the screen edge.
- Calibrate and run in the SAME window at the same screen position.
"""

from __future__ import annotations

import json
import math
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
CALIBRATION_VERSION = 3   # bump when the feature vector changes

# MediaPipe FaceLandmarker's 478-point mesh: 468-472 = right iris, 473-477 = left iris.
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
RIGHT_LID = (159, 145)
LEFT_LID = (386, 374)
NOSE_TIP = 1
LEFT_EYE_LANDMARKS = [*LEFT_IRIS, *LEFT_EYE_CORNERS, *LEFT_LID]
RIGHT_EYE_LANDMARKS = [*RIGHT_IRIS, *RIGHT_EYE_CORNERS, *RIGHT_LID]

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

BLINK_EAR_THRESHOLD = 0.17        # fallback if no per-user baseline was measured
BLINK_BASELINE_FRACTION = 0.65    # blink: EAR below this fraction of the open-eye baseline
OPEN_BASELINE_FRACTION = 0.85     # fully open again: EAR back above this fraction...
REOPEN_HOLD_S = 0.08              # ...and held there this long before predictions resume
DWELL_SECONDS = 0.9
DWELL_GRACE_SECONDS = 0.35
SMOOTHING = 0.80
RIDGE_LAMBDA = 0.05               # on standardized features
OUT_OF_RANGE_MARGIN = 0.15        # prediction beyond the window by more than this fraction -> None

# Calibration targets: an n x n grid inside CAL_MARGIN, ordered center first,
# then ring by ring outward, each ring walked around its perimeter (short hops).
CAL_MARGIN = 0.15
STRAIGHT_GRID_N = 5   # the head pose used most: dense, so mid-diagonals and interior points are measured


def grid_targets(n: int) -> list[tuple[float, float]]:
    levels = np.linspace(CAL_MARGIN, 1 - CAL_MARGIN, n)
    c = (n - 1) / 2
    pts = []
    for iy in range(n):
        for ix in range(n):
            ring = max(abs(ix - c), abs(iy - c))
            angle = math.atan2(ix - c, -(iy - c)) % (2 * math.pi)   # 0 = top of the ring, clockwise
            pts.append((ring, angle, (float(levels[ix]), float(levels[iy]))))
    pts.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in pts]


# Head-pose stages: (name, where, targets). The straight stage is the 5x5 grid
# with the head still, eyes only. Each directional stage starts with a
# "turn your head" step (webcam view, an arrow, the instruction): a SLIGHT
# turn toward that side, like an attention shift, not a full turn. Then two
# dots: the midpoint toward that side and the edge/corner, e.g. up-left =
# upper-middle-left and upper-left. Those same points are also in the straight
# grid, so the model sees each one with and without the head's help, which is
# what teaches it the head/eye coupling.
# The turn step is timed, not gated: an earlier version waited for a measured
# turn and learned left/up from the first stage, which subtle movements never
# reached, so its gauge accepted any direction. Space starts the dots early.
LO, NEAR, MID, FAR, HI = (float(v) for v in np.linspace(CAL_MARGIN, 1 - CAL_MARGIN, STRAIGHT_GRID_N))
HEAD_STAGES = [
    ("straight", "", None),
    ("up-left", "UPPER LEFT", [(NEAR, NEAR), (LO, LO)]),
    ("up", "TOP", [(MID, NEAR), (MID, LO)]),
    ("up-right", "UPPER RIGHT", [(FAR, NEAR), (HI, LO)]),
    ("right", "RIGHT", [(FAR, MID), (HI, MID)]),
    ("down-right", "LOWER RIGHT", [(FAR, FAR), (HI, HI)]),
    ("down", "BOTTOM", [(MID, FAR), (MID, HI)]),
    ("down-left", "LOWER LEFT", [(NEAR, FAR), (LO, HI)]),
    ("left", "LEFT", [(NEAR, MID), (LO, MID)]),
]
CAL_PREROLL_S = 0.6        # the first dot shows this long before capture
CAL_SETTLE_S = 0.45        # eyes only: a saccade settles fast
HEAD_TURN_S = 3.0          # "turn your head slightly" step before a directional stage (Space skips ahead)
HEAD_SETTLE_S = 0.6        # directional stages: a little longer per dot
CAL_MIN_SAMPLES = 8
CAL_TARGET_SAMPLES = 12
CAL_MAX_CAPTURE_S = 2.0

# Face-framing gate (from gaze_dot.py).
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


def _eye_local(iris: np.ndarray, corner_a: np.ndarray, corner_b: np.ndarray) -> tuple[float, float]:
    """Iris offset from the eye's center along the eye's own axis / perpendicular, in eye widths."""
    axis = corner_b - corner_a
    width = max(float(np.linalg.norm(axis)), 1e-5)
    u = axis / width
    v = np.array([-u[1], u[0]])
    rel = iris - (corner_a + corner_b) / 2
    return float(np.dot(rel, u) / width), float(np.dot(rel, v) / width)


def head_pose_proxies(landmarks) -> tuple[float, float]:
    """Yaw/pitch proxies: nose tip relative to the eye midpoint, in inter-eye
    distances. Turning left/right moves the nose sideways; tilting up/down
    moves it toward/away from the eyes. Position- and scale-invariant."""
    la, lb = (_landmark_xy(landmarks, i) for i in LEFT_EYE_CORNERS)
    ra, rb = (_landmark_xy(landmarks, i) for i in RIGHT_EYE_CORNERS)
    left_c, right_c = (la + lb) / 2, (ra + rb) / 2
    mid = (left_c + right_c) / 2
    axis = right_c - left_c
    eye_dist = max(float(np.linalg.norm(axis)), 1e-5)
    u = axis / eye_dist                      # along the eye line...
    v = np.array([-u[1], u[0]])              # ...and perpendicular: roll-invariant like the iris features
    rel = _landmark_xy(landmarks, NOSE_TIP) - mid
    return float(np.dot(rel, u) / eye_dist), float(np.dot(rel, v) / eye_dist)


def gaze_features(landmarks) -> Optional[np.ndarray]:
    try:
        left_iris = np.mean([_landmark_xy(landmarks, i) for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([_landmark_xy(landmarks, i) for i in RIGHT_IRIS], axis=0)
        la, lb = (_landmark_xy(landmarks, i) for i in LEFT_EYE_CORNERS)
        ra, rb = (_landmark_xy(landmarks, i) for i in RIGHT_EYE_CORNERS)
        yaw, pitch = head_pose_proxies(landmarks)
    except IndexError:
        return None
    lx, ly = _eye_local(left_iris, la, lb)
    rx, ry = _eye_local(right_iris, ra, rb)
    mx, my = (lx + rx) / 2, (ly + ry) / 2
    return np.array([
        lx, ly, rx, ry, mx, my, mx * mx, my * my, mx * my,
        yaw, pitch, yaw * mx, yaw * my, pitch * mx, pitch * my, yaw * yaw, pitch * pitch,
        1.0,
    ], dtype=np.float64)


N_FEATURES = 18
HEAD_YAW, HEAD_PITCH = 9, 10    # where gaze_features puts the head-pose proxies


def eye_aspect_ratio(landmarks) -> float:
    def ear(top_i, bottom_i, outer_i, inner_i):
        top, bottom = _landmark_xy(landmarks, top_i), _landmark_xy(landmarks, bottom_i)
        outer, inner = _landmark_xy(landmarks, outer_i), _landmark_xy(landmarks, inner_i)
        return np.linalg.norm(top - bottom) / (np.linalg.norm(outer - inner) + 1e-6)

    return (ear(*RIGHT_LID, *RIGHT_EYE_CORNERS) + ear(*LEFT_LID, *LEFT_EYE_CORNERS)) / 2


def fit_calibration(samples: list[np.ndarray], targets_px: list[np.ndarray],
                    ridge_lambda: float = RIDGE_LAMBDA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge regression on standardized features (bias column untouched and
    unpenalized). Returns (mapping, feature_mean, feature_std). A feature that
    never varied during calibration gets std 1 and mean = its value, so it
    contributes nothing instead of an arbitrary weight."""
    x = np.vstack(samples)
    y = np.vstack(targets_px)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mean[-1], std[-1] = 0.0, 1.0
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    reg = np.eye(z.shape[1]) * ridge_lambda
    reg[-1, -1] = 0.0
    mapping = np.linalg.solve(z.T @ z + reg, z.T @ y)
    return mapping, mean, std


def face_bbox_normalized(landmarks) -> tuple[float, float, float, float]:
    xs = np.array([p.x for p in landmarks])
    ys = np.array([p.y for p in landmarks])
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def evaluate_framing(bbox) -> tuple[str, bool]:
    left, top, right, bottom = bbox
    face_cx, face_cy, face_h = (left + right) / 2, (top + bottom) / 2, bottom - top
    dx, dy = face_cx - FRAME_TARGET_CX, face_cy - FRAME_TARGET_CY
    size_ratio = face_h / FRAME_TARGET_H
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
    cv2.ellipse(canvas, center, axes, 0, 0, 360, (0, 200, 0) if aligned else (0, 165, 255), 3)


def letterbox(frame, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    fh, fw = frame.shape[:2]
    scale = min(width / fw, height / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh))
    return canvas


def _crop_around(frame, landmarks, indices, margin: float) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bw = max(x1 - x0, 1.0)
    x0, x1 = int(max(0, x0 - margin * bw)), int(min(w, x1 + margin * bw))
    y0, y1 = int(max(0, y0 - margin * bw)), int(min(h, y1 + margin * bw))
    crop = frame[y0:y1, x0:x1]
    return crop if crop.size else None


EYE_CROP_MARGIN = 0.08   # fraction of the eye's width added around it; ~0 = just the eyeball


def eyes_strip(frame, landmarks, height: int = 240) -> np.ndarray:
    """Just the two eyeballs, each cropped tightly and shown side by side.
    (Display only: the model uses landmark coordinates, never pixels, so the
    background and the head outline don't reach it at all.)"""
    crops = []
    for indices in (RIGHT_EYE_LANDMARKS, LEFT_EYE_LANDMARKS):  # mirrored frame: right eye is on the left
        c = _crop_around(frame, landmarks, indices, margin=EYE_CROP_MARGIN)
        if c is not None:
            scale = height / c.shape[0]
            crops.append(cv2.resize(c, (max(1, int(c.shape[1] * scale)), height)))
    if not crops:
        return frame
    gap = np.zeros((height, 24, 3), dtype=np.uint8)
    return np.hstack([crops[0], gap, crops[1]]) if len(crops) == 2 else crops[0]


def _draw_status(canvas, message: str, y_from_bottom: int = 24) -> None:
    h = canvas.shape[0]
    cv2.putText(canvas, message, (16, h - y_from_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    # Same weight as the outline: OpenCV 5 draws 1-px text in a narrower face, so it wouldn't line up.
    cv2.putText(canvas, message, (16, h - y_from_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)


def _draw_target(canvas, point_px, progress: float, settling: bool) -> None:
    x, y = point_px
    color = (0, 165, 255) if settling else (0, 255, 255)
    cv2.circle(canvas, (x, y), 26, color, 3, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), max(2, int(20 * progress)), color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 3, (0, 0, 0), -1, cv2.LINE_AA)


def _draw_turn_prompt(canvas, where: str, dx: int, dy: int, seconds_left: float) -> None:
    """Big instruction + an arrow from the center toward that side (the view is
    mirrored, so screen-left is the user's left) + the countdown to the dots."""
    h, w = canvas.shape[:2]
    lines = [(f"Turn your head SLIGHTLY toward the {where}", 1.1), ("a small turn, not a full one", 0.8),
             (f"dots start in {math.ceil(seconds_left)}s", 0.8)]
    for i, (text, scale) in enumerate(lines):
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
        org = ((w - size[0]) // 2, 60 + 48 * i)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255) if i == 0 else (240, 240, 240),
                    3, cv2.LINE_AA)
    cx, cy = w // 2, h // 2 + 40
    length = 170 / math.hypot(dx, dy)
    tip = (int(cx + dx * length), int(cy + dy * length))
    cv2.arrowedLine(canvas, (cx, cy), tip, (0, 0, 0), 16, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(canvas, (cx, cy), tip, (0, 255, 255), 9, cv2.LINE_AA, tipLength=0.3)


@dataclass
class GazeCalibration:
    mapping: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray
    frame_w: int
    frame_h: int
    blink_ear: float = BLINK_EAR_THRESHOLD
    open_ear: float = BLINK_EAR_THRESHOLD * OPEN_BASELINE_FRACTION / BLINK_BASELINE_FRACTION
    version: int = CALIBRATION_VERSION

    def predict(self, feat: np.ndarray) -> Optional[tuple[float, float]]:
        """Window pixel, or None when the prediction is far outside the window
        (head outside the calibrated range) rather than a point pinned to an edge."""
        pred = ((feat - self.feat_mean) / self.feat_std) @ self.mapping
        mx, my = OUT_OF_RANGE_MARGIN * self.frame_w, OUT_OF_RANGE_MARGIN * self.frame_h
        if not (-mx <= pred[0] <= self.frame_w + mx and -my <= pred[1] <= self.frame_h + my):
            return None
        return (float(np.clip(pred[0], 0, self.frame_w - 1)),
                float(np.clip(pred[1], 0, self.frame_h - 1)))

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.write_text(json.dumps({
            "version": self.version, "mapping": self.mapping.tolist(),
            "feat_mean": self.feat_mean.tolist(), "feat_std": self.feat_std.tolist(),
            "frame_w": self.frame_w, "frame_h": self.frame_h,
            "blink_ear": self.blink_ear, "open_ear": self.open_ear,
        }))

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> Optional["GazeCalibration"]:
        """None if there is no saved calibration or it's from an older feature set."""
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        mapping = np.array(data.get("mapping", []))
        if data.get("version") != CALIBRATION_VERSION or mapping.shape[:1] != (N_FEATURES,):
            return None
        return cls(mapping=mapping, feat_mean=np.array(data["feat_mean"]), feat_std=np.array(data["feat_std"]),
                   frame_w=data["frame_w"], frame_h=data["frame_h"],
                   blink_ear=float(data["blink_ear"]), open_ear=float(data["open_ear"]))


class WebcamGazeTracker:
    """Owns the laptop webcam + MediaPipe FaceLandmarker for live gaze tracking."""

    def __init__(self, camera_index: int = 0, model_path: Path = MODEL_PATH,
                 width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        if not model_path.exists():
            raise FileNotFoundError(f"Missing MediaPipe model at {model_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
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
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._start = time.monotonic()
        self._last_timestamp_ms = -1
        frame = None
        for _ in range(15):  # auto-exposure warmup
            _, frame = self._cap.read()
        got = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "unknown"
        print(f"[gaze] webcam {camera_index} capturing at {got} (asked for {width}x{height})")

    def read(self):
        """Returns (frame_bgr, landmarks|None, features|None, ear|None)."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, None, None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.monotonic() - self._start) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not result.face_landmarks:
            return frame, None, None, None
        landmarks = result.face_landmarks[0]
        return frame, landmarks, gaze_features(landmarks), eye_aspect_ratio(landmarks)

    def close(self):
        self._cap.release()
        self._landmarker.close()


class GazeSmoother:
    def __init__(self, smoothing: float = SMOOTHING):
        self.smoothing = smoothing
        self._pt: Optional[np.ndarray] = None

    @property
    def last(self) -> Optional[tuple[float, float]]:
        return None if self._pt is None else (float(self._pt[0]), float(self._pt[1]))

    def reset(self) -> None:
        self._pt = None

    def update(self, pt: tuple[float, float]) -> tuple[float, float]:
        p = np.array(pt, dtype=np.float64)
        self._pt = p if self._pt is None else self.smoothing * self._pt + (1 - self.smoothing) * p
        return float(self._pt[0]), float(self._pt[1])


class BlinkGate:
    """Eyes count as open only once the aspect ratio is back above the open
    level AND has stayed there for reopen_hold_s: the reopening lids corrupt
    the iris fit for a few frames after a blink."""

    def __init__(self, blink_ear: float, open_ear: float, reopen_hold_s: float = REOPEN_HOLD_S):
        self.blink_ear, self.open_ear, self.reopen_hold_s = blink_ear, open_ear, reopen_hold_s
        self._open_since: Optional[float] = None

    def update(self, ear: Optional[float], now: Optional[float] = None) -> tuple[bool, bool]:
        """Returns (eyes_fully_open, blinking_or_reopening)."""
        now = time.monotonic() if now is None else now
        if ear is None:
            return False, False
        if ear < self.open_ear:            # closed, or only part-way open
            self._open_since = None
            return False, True
        if self._open_since is None:
            self._open_since = now
        if now - self._open_since < self.reopen_hold_s:
            return False, True
        return True, False


class GazeEstimator:
    """Live gaze: webcam frame -> window pixel, with blink holding and smoothing."""

    def __init__(self, tracker: WebcamGazeTracker, calib: GazeCalibration):
        self.tracker, self.calib = tracker, calib
        self.smoother = GazeSmoother()
        self.gate = BlinkGate(calib.blink_ear, calib.open_ear)
        self.last_head: Optional[tuple[float, float]] = None   # head-pose proxies of the last frame read

    def reset(self) -> None:
        self.smoother.reset()

    def read(self) -> tuple[Optional[tuple[float, float]], Optional[float], bool]:
        """Returns (gaze point or None, eye aspect ratio, blinking/reopening)."""
        _, landmarks, feats, ear = self.tracker.read()
        self.last_head = head_pose_proxies(landmarks) if landmarks is not None else None
        if feats is None:
            self.smoother.reset()
            return None, ear, False
        fully_open, blinking = self.gate.update(ear)
        if not fully_open:
            return self.smoother.last, ear, True
        pred = self.calib.predict(feats)
        if pred is None:                   # out of the calibrated range: no point, no evidence
            return None, ear, False
        return self.smoother.update(pred), ear, False


class DwellSelector:
    """Simple hover timer (gaze_lock.py's evidence model is what main.py uses)."""

    def __init__(self, dwell_seconds: float = DWELL_SECONDS, grace_seconds: float = DWELL_GRACE_SECONDS):
        self.dwell_seconds, self.grace_seconds = dwell_seconds, grace_seconds
        self._target: Optional[str] = None
        self._started_at: Optional[float] = None
        self._last_seen: Optional[float] = None

    def reset(self) -> None:
        self._target, self._started_at, self._last_seen = None, None, None

    def update(self, hovered: Optional[str]) -> tuple[Optional[str], float]:
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
        return (hovered, 1.0) if elapsed >= self.dwell_seconds else (None, elapsed / self.dwell_seconds)


def run_calibration(tracker: WebcamGazeTracker, frame_w: int, frame_h: int,
                     window_name: str = "Gaze Calibration",
                     keep_window: bool = False,
                     quick: bool = False) -> GazeCalibration:
    """Framing gate -> the straight stage (5x5 grid, head still) -> eight
    directional stages (two dots each, the head follows the eyes slightly).
    quick=True does the straight stage only (fine if the head will stay still).

    Keys during calibration: S skips the current stage, Q/ESC cancels.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    stages = HEAD_STAGES[:1] if quick else HEAD_STAGES

    def finish_window():
        if not keep_window:
            cv2.destroyWindow(window_name)

    def show(canvas) -> int:
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            finish_window()
            raise SystemExit("Calibration cancelled")
        return key

    def face_canvas(frame):
        return letterbox(frame if frame is not None else np.zeros((frame_h, frame_w, 3), np.uint8), frame_w, frame_h)

    def eyes_canvas(frame, landmarks):
        if frame is None:
            return np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        src = eyes_strip(frame, landmarks) if landmarks is not None else frame
        return (letterbox(src, frame_w, frame_h) * 0.55).astype(np.uint8)

    # Phase 1: framing gate, measuring the open-eye EAR baseline and the straight head pose.
    hold_started = 0.0
    open_ears: list[float] = []
    straight_poses: list[tuple[float, float]] = []
    while True:
        frame, landmarks, _, ear = tracker.read()
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
                if ear is not None:
                    open_ears.append(ear)
                straight_poses.append(head_pose_proxies(landmarks))
                remaining = max(0.0, FRAME_HOLD_SECONDS - (now - hold_started))
                message = "Hold still, eyes open, looking at the screen..." if remaining > 0 else "Starting..."
            else:
                hold_started = 0.0
                message = FRAME_MESSAGES[status]
        source = frame if frame is not None else np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        draw_id_frame(source, aligned)
        canvas = face_canvas(source)
        _draw_status(canvas, message)
        show(canvas)
        if aligned and remaining <= 0.0:
            break

    baseline = float(np.median(open_ears)) if open_ears else None
    blink_ear = min(0.24, max(0.10, BLINK_BASELINE_FRACTION * baseline)) if baseline else BLINK_EAR_THRESHOLD
    open_ear = min(0.30, max(blink_ear + 0.02, OPEN_BASELINE_FRACTION * baseline)) if baseline else blink_ear + 0.03
    yaw0, pitch0 = (np.median(np.array(straight_poses), axis=0) if straight_poses else (0.0, 0.0))
    print(f"[calibration] EAR baseline {baseline if baseline else float('nan'):.3f} -> blink < {blink_ear:.3f}, "
          f"open > {open_ear:.3f}; straight head yaw={yaw0:+.3f} pitch={pitch0:+.3f}")
    gate = BlinkGate(blink_ear, open_ear)

    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for stage_idx, (name, where, stage_targets) in enumerate(stages):
        stage_label = f"Stage {stage_idx + 1}/{len(stages)}: {name}   |   S = skip this stage"
        head_follows = stage_targets is not None
        targets = stage_targets if head_follows else grid_targets(STRAIGHT_GRID_N)
        settle_s = HEAD_SETTLE_S if head_follows else CAL_SETTLE_S
        instruction = (f"Keep your head turned slightly toward the {where}; follow the dots with your eyes"
                       if head_follows else "Keep your head still and follow the dot with your eyes")
        skipped = False

        # Turn-your-head step: timed, never waits on a measured head angle.
        if head_follows:
            dx = -1 if "left" in name else (1 if "right" in name else 0)
            dy = -1 if "up" in name else (1 if "down" in name else 0)
            t0 = time.monotonic()
            while (left := HEAD_TURN_S - (time.monotonic() - t0)) > 0:
                frame, _, _, _ = tracker.read()
                canvas = face_canvas(frame)
                _draw_turn_prompt(canvas, where, dx, dy, left)
                _draw_status(canvas, stage_label, 52)
                _draw_status(canvas, "Space = ready now")
                key = show(canvas)
                if key == ord("s"):
                    skipped = True
                    break
                if key == ord(" "):
                    break

        # Pre-roll: the first dot (not captured yet) and the instruction.
        fx, fy = int(targets[0][0] * frame_w), int(targets[0][1] * frame_h)
        t0 = time.monotonic()
        while not skipped and time.monotonic() - t0 < CAL_PREROLL_S:
            frame, landmarks, _, _ = tracker.read()
            canvas = eyes_canvas(frame, landmarks)
            _draw_target(canvas, (fx, fy), 0.0, settling=True)
            _draw_status(canvas, stage_label, 52)
            _draw_status(canvas, instruction)
            if show(canvas) == ord("s"):
                skipped = True
                break

        # The targets.
        stage_ok, stage_heads = 0, []
        for idx, (tx, ty) in enumerate(targets):
            if skipped:
                break
            px, py = int(tx * frame_w), int(ty * frame_h)
            samples: list[np.ndarray] = []
            started = time.monotonic()
            capture_started: Optional[float] = None
            while True:
                frame, landmarks, feats, ear = tracker.read()
                now = time.monotonic()
                settling = now - started < settle_s
                fully_open, blinking = gate.update(ear, now)
                if not settling:
                    capture_started = capture_started or now
                    if feats is not None and fully_open:
                        samples.append(feats)
                canvas = eyes_canvas(frame, landmarks)
                _draw_target(canvas, (px, py), min(1.0, len(samples) / CAL_TARGET_SAMPLES), settling)
                if blinking:
                    cv2.circle(canvas, (px, py), 34, (0, 120, 255), 2, cv2.LINE_AA)
                _draw_status(canvas, f"{stage_label}   |   dot {idx + 1}/{len(targets)}", 52)
                _draw_status(canvas, instruction)
                if show(canvas) == ord("s"):
                    skipped = True
                    break
                if len(samples) >= CAL_TARGET_SAMPLES:
                    break
                if capture_started is not None and now - capture_started > CAL_MAX_CAPTURE_S:
                    break
            if skipped:
                break
            if len(samples) >= CAL_MIN_SAMPLES:
                feat = np.median(samples, axis=0)
                all_features.append(feat)
                all_targets.append(np.array([px, py], dtype=np.float64))
                stage_heads.append(feat[[HEAD_YAW, HEAD_PITCH]])
                stage_ok += 1
            else:
                print(f"[calibration] {name} dot {idx + 1} skipped: only {len(samples)} clean samples")
        if skipped:
            print(f"[calibration] stage '{name}' skipped")
            continue
        # For the record only (nothing is gated on it): how far the head moved.
        moved = (f"; head moved yaw {np.mean(stage_heads, axis=0)[0] - yaw0:+.3f}, "
                 f"pitch {np.mean(stage_heads, axis=0)[1] - pitch0:+.3f} from straight" if stage_heads else "")
        print(f"[calibration] stage '{name}': {stage_ok}/{len(targets)} dots{moved}")

    finish_window()
    if len(all_features) < 6:
        raise RuntimeError("Too few good calibration points; try again with better lighting, eyes open.")

    mapping, mean, std = fit_calibration(all_features, all_targets)
    calib = GazeCalibration(mapping=mapping, feat_mean=mean, feat_std=std, frame_w=frame_w, frame_h=frame_h,
                            blink_ear=blink_ear, open_ear=open_ear)
    calib.save()
    pred = ((np.vstack(all_features) - mean) / std) @ mapping
    err = float(np.mean(np.linalg.norm(pred - np.vstack(all_targets), axis=1)))
    print(f"[calibration] done: {len(all_features)} dots over {len(stages)} stage(s), "
          f"mean fit error {err:.1f}px, saved to {CALIBRATION_PATH}")
    return calib
