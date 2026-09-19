"""Head-motion steering for users with some head mobility (main.py --user head).

Follows AMiCUS (Rudigkeit et al., Sensors 2019; AMiCUS 2.0, Sensors 2020), a
head-motion robot interface all six tetraplegic participants (C0-C4, severely
restricted neck motion) could operate:
- the neutral head position and how far the user can comfortably turn in each
  direction are calibrated per user, and commands are normalized to that
  personal range, so a small comfortable motion is full scale;
- a dead zone around neutral, then a smooth ramp (they used a Gompertz curve);
- two degrees of freedom at a time, grouped in planes and switched with an
  on-screen dwell control (their nod gesture worked for 41% of attempts in
  users with restricted motion, their dwell control for 83%);
- control pauses on a sudden head movement or a lost face and resumes only
  after the head is back at neutral.
AMiCUS streams velocities to its robot; here each command is a short
motion-planned step, so every move is collision-checked and bounded.

The head-pose values are webcam_gaze.head_pose_proxies (nose tip relative to
the eye midpoint in units of inter-eye distance), so they don't change when
the head merely shifts or moves closer to the camera.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from webcam_gaze import head_pose_proxies, letterbox

HEAD_RANGE_PATH = Path(__file__).parent / "head_range.json"

DIRECTIONS = ("left", "right", "up", "down")
AXIS_OF = {"left": "yaw", "right": "yaw", "up": "pitch", "down": "pitch"}
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}

DEAD_ZONE = 0.35            # fraction of the user's own range that moves nothing
FULL_SCALE = 0.85           # fraction of the range that gives the largest step
STEP_MIN_MM = 10.0          # step size just past the dead zone
STEP_MAX_MM = 35.0          # step size at full scale
HOLD_BEFORE_MOVE_S = 0.35   # head held past the dead zone this long before the first step
NEUTRAL_HOLD_S = 1.0        # holding still this long captures the neutral position
NEUTRAL_TOLERANCE = 0.012   # proxy units: "still" means within this spread
RESUME_NEUTRAL_S = 0.5      # after a pause, back in the dead zone this long to resume
JUMP_LIMIT = 1.0            # normalized change within JUMP_WINDOW_S that counts as a jerk
JUMP_WINDOW_S = 0.15
MIN_RANGE = 0.02            # a direction the user can't reach this far is not used
CAL_STABLE_S = 0.8          # calibration: hold the turned position this long
CAL_TIMEOUT_S = 12.0        # calibration: a direction not reached by then is marked unusable


# --------------------------------------------------------------------------- per-user range

@dataclass
class HeadRange:
    ranges: dict              # direction -> comfortable magnitude (proxy units), or None if unusable
    signs: dict               # "left": sign of the yaw change when turning left; "up": pitch sign for up

    def usable(self, direction: str) -> bool:
        return bool(self.ranges.get(direction))

    def normalized(self, d_yaw: float, d_pitch: float) -> tuple[float, float]:
        """(lateral, vertical) in units of the user's own range: lateral + is
        the user's left, vertical + is up. 0 where a direction is unusable."""
        def axis(d: float, pos: str, neg: str) -> float:
            sign = self.signs.get(pos)
            if sign is None:
                return 0.0
            toward_pos = d * sign
            rng = self.ranges.get(pos) if toward_pos > 0 else self.ranges.get(neg)
            if not rng:
                return 0.0
            return float(max(-1.5, min(1.5, toward_pos / rng)))
        return axis(d_yaw, "left", "right"), axis(d_pitch, "up", "down")

    def save(self, path: Path = HEAD_RANGE_PATH) -> None:
        path.write_text(json.dumps({"ranges": self.ranges, "signs": self.signs}, indent=2))

    @classmethod
    def load(cls, path: Path = HEAD_RANGE_PATH) -> Optional["HeadRange"]:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(ranges=data.get("ranges", {}), signs=data.get("signs", {}))


def shape(n: float) -> float:
    """Dead zone, then a smooth ramp to 1 (smoothstep), sign kept."""
    a = abs(n)
    if a <= DEAD_ZONE:
        return 0.0
    x = min(1.0, (a - DEAD_ZONE) / (FULL_SCALE - DEAD_ZONE))
    return math.copysign(x * x * (3.0 - 2.0 * x), n)


def step_mm(m: float) -> float:
    if m == 0.0:
        return 0.0
    return math.copysign(STEP_MIN_MM + abs(m) * (STEP_MAX_MM - STEP_MIN_MM), m)


# --------------------------------------------------------------------------- steering

@dataclass
class SteerCommand:
    state: str                    # arming | neutral | pending | move | paused
    n: tuple[float, float]        # normalized (lateral, vertical)
    m: tuple[float, float]        # shaped (lateral, vertical), 0 inside the dead zone
    reason: str = ""


class HeadSteer:
    """Head pose -> step commands. Pure logic: feed it head poses and times."""

    def __init__(self, head_range: HeadRange):
        self.rng = head_range
        self.neutral: Optional[tuple[float, float]] = None
        self.state = "arming"
        self.reason = ""
        self._samples: deque = deque()
        self._last: Optional[tuple[float, tuple[float, float]]] = None
        self._engaged_since: Optional[float] = None
        self._neutral_since: Optional[float] = None

    def _pause(self, reason: str) -> None:
        self.state, self.reason = "paused", reason
        self._engaged_since = None
        self._neutral_since = None

    def update(self, head: Optional[tuple[float, float]], now: Optional[float] = None) -> SteerCommand:
        now = time.monotonic() if now is None else now
        if head is None:
            self._samples.clear()
            self._last = None
            if self.state != "arming":
                self._pause("face lost")
            return SteerCommand(self.state, (0.0, 0.0), (0.0, 0.0), self.reason or "face lost")

        if self.state == "arming":
            self._samples.append((now, head))
            while self._samples and now - self._samples[0][0] > NEUTRAL_HOLD_S:
                self._samples.popleft()
            pts = np.array([h for _, h in self._samples])
            span = now - self._samples[0][0]
            if span >= NEUTRAL_HOLD_S * 0.95 and float(np.ptp(pts, axis=0).max()) <= NEUTRAL_TOLERANCE:
                self.neutral = tuple(np.median(pts, axis=0))
                self.state, self.reason = "ready", ""
            return SteerCommand("arming", (0.0, 0.0), (0.0, 0.0), "hold your head still in the center")

        n = self.rng.normalized(head[0] - self.neutral[0], head[1] - self.neutral[1])
        if self._last is not None and now - self._last[0] <= JUMP_WINDOW_S:
            if max(abs(n[0] - self._last[1][0]), abs(n[1] - self._last[1][1])) > JUMP_LIMIT:
                self._pause("sudden movement")
        self._last = (now, n)
        m = (shape(n[0]), shape(n[1]))

        if self.state == "paused":
            if m == (0.0, 0.0):
                self._neutral_since = self._neutral_since or now
                if now - self._neutral_since >= RESUME_NEUTRAL_S:
                    self.state, self.reason = "ready", ""
            else:
                self._neutral_since = None
            if self.state == "paused":
                return SteerCommand("paused", n, m, self.reason)

        if m == (0.0, 0.0):
            self._engaged_since = None
            return SteerCommand("neutral", n, m)
        self._engaged_since = self._engaged_since or now
        if now - self._engaged_since < HOLD_BEFORE_MOVE_S:
            return SteerCommand("pending", n, m)
        return SteerCommand("move", n, m)


def steer_vector(cmd: SteerCommand, plane: str, left: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """World-frame step (mm). plane 'vertical': head left/right -> user's
    left/right, head up/down -> up/down. plane 'horizontal': head left/right ->
    left/right, head up -> away from the user, head down -> closer (like
    looking further across a table)."""
    lat, vert = step_mm(cmd.m[0]), step_mm(cmd.m[1])
    v = lat * left
    if plane == "vertical":
        v = v + np.array([0.0, 0.0, vert])
    else:
        v = v - vert * toward
    return v


# --------------------------------------------------------------------------- calibration UI

def _text(canvas, msg: str, y: int, scale: float = 0.8, color=(255, 255, 255)) -> None:
    cv2.putText(canvas, msg, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, msg, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def calibrate_head_range(tracker, frame_w: int, frame_h: int, window_name: str) -> HeadRange:
    """Neutral, then as far as comfortable left / right / up / down.
    S skips a direction (marked unusable), Q/ESC cancels."""
    def frame_and_head():
        frame, landmarks, _, _ = tracker.read()
        head = head_pose_proxies(landmarks) if landmarks is not None else None
        canvas = letterbox(frame if frame is not None else np.zeros((frame_h, frame_w, 3), np.uint8),
                           frame_w, frame_h)
        return canvas, head

    def show(canvas) -> int:
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise SystemExit("Head calibration cancelled")
        return key

    # Neutral.
    samples: deque = deque()
    while True:
        canvas, head = frame_and_head()
        now = time.monotonic()
        _text(canvas, "Head control setup: look at the screen, head relaxed and still", 44)
        if head is None:
            samples.clear()
            _text(canvas, "Face not found", 84, color=(0, 80, 255))
        else:
            samples.append((now, head))
            while samples and now - samples[0][0] > NEUTRAL_HOLD_S * 1.5:
                samples.popleft()
            pts = np.array([h for _, h in samples])
            if now - samples[0][0] >= NEUTRAL_HOLD_S * 1.4 and float(np.ptp(pts, axis=0).max()) <= NEUTRAL_TOLERANCE:
                neutral = tuple(np.median(pts, axis=0))
                break
        show(canvas)

    ranges: dict = {d: None for d in DIRECTIONS}
    signs: dict = {}

    def expected(direction: str) -> Optional[float]:
        axis_key = "left" if AXIS_OF[direction] == "yaw" else "up"
        base = signs.get(axis_key)
        if base is None:
            return None
        return base if direction == axis_key else -base

    for direction in DIRECTIONS:
        axis = 0 if AXIS_OF[direction] == "yaw" else 1
        want = expected(direction)
        window: deque = deque()
        started = time.monotonic()
        captured = None
        wrong_way = False
        while True:
            canvas, head = frame_and_head()
            now = time.monotonic()
            verb = "Turn" if axis == 0 else "Tilt"
            _text(canvas, f"{verb} your head {direction.upper()} as far as is comfortable, and hold", 44)
            _text(canvas, "S = I can't move that way", frame_h - 24, 0.6, (200, 200, 200))
            d = None
            if head is not None:
                d = head[axis] - neutral[axis]
                wrong_way = want is not None and d * want < 0 and abs(d) >= MIN_RANGE
                window.append((now, d))
                while window and now - window[0][0] > CAL_STABLE_S:
                    window.popleft()
                vals = np.array([v for _, v in window])
                med = float(np.median(vals))
                steady = (now - window[0][0] >= CAL_STABLE_S * 0.9
                          and float(np.ptp(vals)) <= max(0.004, 0.2 * abs(med)))
                if abs(med) >= MIN_RANGE and steady and not wrong_way:
                    captured = med
            if wrong_way:
                _text(canvas, "That's the other way", 84, color=(0, 0, 255))
            bar = int(min(1.0, abs(d or 0.0) / 0.15) * 320)
            cv2.rectangle(canvas, (24, 104), (344, 128), (60, 60, 60), -1)
            cv2.rectangle(canvas, (24, 104), (24 + bar, 128), (0, 200, 0) if captured else (0, 165, 255), -1)
            key = show(canvas)
            if captured is not None:
                ranges[direction] = abs(captured)
                axis_key = "left" if axis == 0 else "up"
                sign = math.copysign(1.0, captured)
                signs.setdefault(axis_key, sign if direction == axis_key else -sign)
                print(f"[head] {direction}: range {abs(captured):.3f}")
                break
            if key == ord("s") or now - started > CAL_TIMEOUT_S:
                print(f"[head] {direction}: not usable (skipped or not reached)")
                break

        # Back to the middle before the next direction.
        back_since = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0:
            canvas, head = frame_and_head()
            _text(canvas, "Back to the center", 44)
            show(canvas)
            if head is not None and max(abs(head[0] - neutral[0]), abs(head[1] - neutral[1])) < MIN_RANGE:
                back_since = back_since or time.monotonic()
                if time.monotonic() - back_since > 0.4:
                    break
            else:
                back_since = None

    rng = HeadRange(ranges=ranges, signs=signs)
    rng.save()
    usable = [d for d in DIRECTIONS if rng.usable(d)]
    print(f"[head] usable directions: {usable or 'none'}; saved to {HEAD_RANGE_PATH}")
    return rng


# --------------------------------------------------------------------------- drawing

def draw_steer_panel(canvas, cmd: SteerCommand, plane: str, center: tuple[int, int], radius: int) -> None:
    """A joystick view of the head: dead zone, full-scale ring, and the head's
    current position (the user's left is the screen's left)."""
    cx, cy = center
    moving = cmd.state in ("pending", "move")
    cv2.circle(canvas, (cx, cy), radius, (90, 90, 90), 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(radius * FULL_SCALE), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(radius * DEAD_ZONE), (0, 120, 0) if not moving else (60, 60, 60), -1, cv2.LINE_AA)
    up, down = ("UP", "DOWN") if plane == "vertical" else ("AWAY", "CLOSER")
    for label, (x, y) in ((up, (cx - 30, cy - radius - 14)), (down, (cx - 45, cy + radius + 32)),
                          ("LEFT", (cx - radius - 90, cy + 8)), ("RIGHT", (cx + radius + 14, cy + 8))):
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    nx, ny = max(-1.2, min(1.2, cmd.n[0])), max(-1.2, min(1.2, cmd.n[1]))
    dot = (int(cx - nx * radius), int(cy - ny * radius))
    color = {"move": (0, 220, 0), "pending": (0, 220, 220), "paused": (0, 0, 255)}.get(cmd.state, (255, 255, 255))
    cv2.circle(canvas, dot, 14, color, -1, cv2.LINE_AA)
