"""Gaze-to-detection lock-on.

Feed it, once per frame, the raw camera frame, the YOLO boxes found in it,
and the current gaze point. It reports which box the gaze is hovering and how
far along the dwell timer is; once the dwell completes it LOCKS: it keeps a
snapshot of that exact frame plus the chosen box, and stops listening to gaze
until release() is called.

The snapshot matters because the RealSense rides on the arm: the moment the
arm starts moving, the live view no longer shows what was selected. Everything
downstream (3D segmentation, motion) is keyed off the locked target, not the
live feed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from webcam_gaze import DWELL_GRACE_SECONDS, DWELL_SECONDS, DwellSelector

HOVER_COLOR = (0, 220, 0)
IDLE_COLOR = (110, 110, 110)
LOCK_COLOR = (255, 200, 0)
GAZE_COLOR = (0, 0, 255)
RING_COLOR = (0, 255, 255)


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    label: str
    confidence: float
    index: int

    @property
    def key(self) -> str:
        return f"{self.index}:{self.label}"

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


# Surfaces YOLO reports as objects. They're never pick targets, and their boxes
# swallow everything on them.
IGNORE_LABELS = {"dining table", "bed", "couch", "bench"}
ENGULF_FRACTION = 0.7        # this much of the smaller box lies inside the bigger one...
ENGULF_MIN_AREA_RATIO = 3.0  # ...and the bigger box is at least this many times larger


def overlap_fraction(outer: Box, inner: Box) -> float:
    """Fraction of inner's area that lies inside outer."""
    ix0, iy0 = max(outer.x0, inner.x0), max(outer.y0, inner.y0)
    ix1, iy1 = min(outer.x1, inner.x1), min(outer.y1, inner.y1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return inter / inner.area if inner.area else 0.0


def engulfs(outer: Box, inner: Box) -> bool:
    return (outer.area >= ENGULF_MIN_AREA_RATIO * inner.area
            and overlap_fraction(outer, inner) >= ENGULF_FRACTION)


def filter_background_boxes(boxes: Sequence[Box]) -> list[Box]:
    """Drop denylisted labels and any box that engulfs a smaller box.

    A much bigger box with another detection sitting inside it is the scene
    (table, background), not a thing on it; ignoring it means gaze always
    resolves to what's on top. Original indices are kept so 3D-object
    matching still lines up.
    """
    kept = [b for b in boxes if b.label.lower() not in IGNORE_LABELS]
    return [b for b in kept if not any(engulfs(b, o) for o in kept if o is not b)]


@dataclass
class LockedTarget:
    box: Box
    snapshot: np.ndarray
    locked_at: float


class GazeLockController:
    def __init__(self, dwell_seconds: float = DWELL_SECONDS,
                 grace_seconds: float = DWELL_GRACE_SECONDS):
        self._dwell = DwellSelector(dwell_seconds, grace_seconds)
        self.locked: Optional[LockedTarget] = None

    @property
    def is_locked(self) -> bool:
        return self.locked is not None

    def update(self, frame: np.ndarray, boxes: Sequence[Box],
               gaze_pt: Optional[tuple[float, float]],
               ) -> tuple[Optional[Box], float, Optional[LockedTarget]]:
        """Returns (hovered_box, dwell_progress, newly_locked_target)."""
        if self.locked is not None:
            return None, 1.0, None

        hovered: Optional[Box] = None
        if gaze_pt is not None:
            hits = [b for b in boxes if b.contains(*gaze_pt)]
            if hits:
                # Overlapping boxes: the innermost one is the one being looked at.
                hovered = min(hits, key=lambda b: b.area)

        selected_key, progress = self._dwell.update(hovered.key if hovered else None)
        if selected_key is not None and hovered is not None:
            self.locked = LockedTarget(box=hovered, snapshot=frame.copy(),
                                       locked_at=time.monotonic())
            self._dwell.reset()
            return hovered, 1.0, self.locked
        return hovered, progress, None

    def release(self) -> None:
        self.locked = None
        self._dwell.reset()


def draw_live(frame: np.ndarray, boxes: Sequence[Box], hovered: Optional[Box],
              progress: float, gaze_pt: Optional[tuple[float, float]]) -> np.ndarray:
    out = frame.copy()
    for b in boxes:
        is_hover = hovered is not None and b.key == hovered.key
        color = HOVER_COLOR if is_hover else IDLE_COLOR
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), color, 3 if is_hover else 2)
        cv2.putText(out, f"{b.label} {b.confidence:.2f}", (b.x0, max(14, b.y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if gaze_pt is not None:
        gx, gy = int(gaze_pt[0]), int(gaze_pt[1])
        cv2.circle(out, (gx, gy), 8, GAZE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, (gx, gy), 14, (255, 255, 255), 2, cv2.LINE_AA)
        if hovered is not None:
            cv2.ellipse(out, (gx, gy), (24, 24), -90, 0, 360 * progress,
                        RING_COLOR, 3, cv2.LINE_AA)
    return out


def draw_locked(locked: LockedTarget, status_lines: Sequence[str] = ()) -> np.ndarray:
    """The frozen snapshot with everything but the locked box dimmed."""
    snap = locked.snapshot
    h, w = snap.shape[:2]
    b = locked.box
    x0, y0 = max(0, b.x0), max(0, b.y0)
    x1, y1 = min(w, b.x1), min(h, b.y1)

    out = (snap * 0.35).astype(np.uint8)
    out[y0:y1, x0:x1] = snap[y0:y1, x0:x1]
    cv2.rectangle(out, (x0, y0), (x1, y1), LOCK_COLOR, 4)

    cv2.rectangle(out, (0, 0), (w, 44 + 26 * len(status_lines)), (0, 0, 0), -1)
    cv2.putText(out, f"LOCKED: {b.label} {b.confidence:.2f}", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, LOCK_COLOR, 2, cv2.LINE_AA)
    for i, line in enumerate(status_lines):
        cv2.putText(out, line, (16, 60 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (230, 230, 230), 2, cv2.LINE_AA)
    return out
