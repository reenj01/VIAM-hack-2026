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

import math

from webcam_gaze import DWELL_SECONDS

# Soft targeting: the gaze is noisy evidence of intent, not a precise pointer.
# Each frame every box earns a score from how close the gaze is (1 inside,
# falling off outside), the scores accumulate over time as "effective seconds
# of attention" with a slow decay, and a box is selected once it has enough
# and clearly more than any other. Gaze that wanders around and just outside
# an object still piles evidence onto it; jitter across an edge resets nothing.
SELECT_SECONDS = DWELL_SECONDS   # effective seconds of attention needed to select
SOFT_SIGMA_PX = 90.0             # score = exp(-(distance_outside_box / sigma)^2)
EVIDENCE_TAU_S = 1.5             # evidence decays by 1/e over this long when looking elsewhere
DOMINANCE = 2.0                  # winner needs this many times the runner-up's evidence
NESTED_OUTER_SCORE = 0.5         # gaze inside several boxes: innermost gets 1, the ones around it this
LEADER_MIN_EVIDENCE_S = 0.12     # show the leading box as "hovered" once it has this much
MAX_DT_S = 0.2                   # a stalled frame can't count as a long fixation
EVIDENCE_CAP_FACTOR = 1.5        # evidence never exceeds this x SELECT_SECONDS
TRACK_MIN_IOU = 0.3              # a new box continues an existing track above this overlap
TRACK_MISSING_GRACE_S = 0.6      # a track survives a detection flicker this long

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


def distance_outside(box: Box, x: float, y: float) -> float:
    dx = max(box.x0 - x, 0.0, x - box.x1)
    dy = max(box.y0 - y, 0.0, y - box.y1)
    return math.hypot(dx, dy)


def gaze_scores(boxes: Sequence[Box], gaze_pt: tuple[float, float],
                sigma_px: float = SOFT_SIGMA_PX) -> dict[str, float]:
    """Per-box attention score for one gaze sample, in [0, 1]."""
    gx, gy = gaze_pt
    inside = [b for b in boxes if b.contains(gx, gy)]
    innermost = min(inside, key=lambda b: b.area).key if inside else None
    scores = {}
    for b in boxes:
        if b.key == innermost:
            scores[b.key] = 1.0
        elif b.contains(gx, gy):
            scores[b.key] = NESTED_OUTER_SCORE
        else:
            d = distance_outside(b, gx, gy)
            scores[b.key] = math.exp(-(d / sigma_px) ** 2)
    return scores


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    box: Box            # latest box seen for this object (carries the current detection index)
    evidence: float
    last_seen: float
    present: bool = True


class EvidenceSelector:
    """Accumulates attention evidence per tracked object and picks a clear winner.

    Objects are tracked by overlap (IoU) + label rather than by the detector's
    list index, so a detection that flickers out for a frame, or a new one
    appearing earlier in the list, doesn't reset or reshuffle the evidence.
    """

    def __init__(self, select_seconds: float = SELECT_SECONDS, sigma_px: float = SOFT_SIGMA_PX,
                 tau_s: float = EVIDENCE_TAU_S, dominance: float = DOMINANCE):
        self.select_seconds, self.sigma_px, self.tau_s, self.dominance = select_seconds, sigma_px, tau_s, dominance
        self.tracks: dict[int, Track] = {}
        self._next_id = 0
        self._last_t: Optional[float] = None

    @property
    def evidence(self) -> dict[str, float]:
        return {t.box.key: t.evidence for t in self.tracks.values()}

    def reset(self) -> None:
        self.tracks.clear()
        self._last_t = None

    def hold(self, now: Optional[float] = None) -> None:
        """Freeze time (e.g. during a blink): the gap counts neither for nor
        against any object when updates resume."""
        self._last_t = time.monotonic() if now is None else now

    def _associate(self, boxes: Sequence[Box], now: float) -> None:
        """Match this frame's boxes to existing tracks (best IoU, same label)."""
        for t in self.tracks.values():
            t.present = False
        unclaimed = set(self.tracks)
        for b in sorted(boxes, key=lambda b: -b.area):
            best_id, best_iou = None, TRACK_MIN_IOU
            for tid in unclaimed:
                t = self.tracks[tid]
                if t.box.label == b.label and (score := iou(t.box, b)) > best_iou:
                    best_id, best_iou = tid, score
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.tracks[best_id] = Track(box=b, evidence=0.0, last_seen=now)
            else:
                unclaimed.discard(best_id)
            t = self.tracks[best_id]
            t.box, t.last_seen, t.present = b, now, True
        # A track that vanished for longer than a flicker is gone.
        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if t.present or now - t.last_seen <= TRACK_MISSING_GRACE_S}

    def update(self, boxes: Sequence[Box], gaze_pt: Optional[tuple[float, float]],
               now: Optional[float] = None) -> tuple[Optional[Box], float, Optional[Box]]:
        """Returns (leading_box, progress_0_to_1, selected_box_or_None)."""
        now = time.monotonic() if now is None else now
        dt = 0.0 if self._last_t is None else min(MAX_DT_S, max(0.0, now - self._last_t))
        self._last_t = now

        self._associate(boxes, now)
        # Score against each track's last-known box, so a box that flickered
        # out this frame still collects evidence if the gaze is on it.
        tracked_boxes = [t.box for t in self.tracks.values()]
        scores = gaze_scores(tracked_boxes, gaze_pt, self.sigma_px) if (gaze_pt is not None and tracked_boxes) else {}

        # Evidence leaks from objects the gaze is NOT on (decay scaled by
        # 1 - score): a fixated object fills in select_seconds, one looked away
        # from is forgotten over tau_s. Capped so evidence can't run away while
        # the gaze lingers near something (at a partial score the growth/leak
        # balance can otherwise creep past the threshold), and so two objects
        # both stared at stay comparable.
        cap = self.select_seconds * EVIDENCE_CAP_FACTOR
        for t in self.tracks.values():
            s = scores.get(t.box.key, 0.0)
            t.evidence = min(cap, t.evidence * math.exp(-dt * (1.0 - s) / self.tau_s) + s * dt)

        if not self.tracks:
            return None, 0.0, None
        ranked = sorted(self.tracks.values(), key=lambda t: t.evidence, reverse=True)
        best = ranked[0]
        runner_up = ranked[1].evidence if len(ranked) > 1 else 0.0
        leader = best.box if best.evidence >= LEADER_MIN_EVIDENCE_S else None
        progress = min(1.0, best.evidence / self.select_seconds)
        # Only a currently-visible object can be selected: the segmenter needs
        # a live detection to match against.
        if (best.present and best.evidence >= self.select_seconds
                and best.evidence >= self.dominance * runner_up):
            return leader, 1.0, best.box
        return leader, progress, None


class GazeLockController:
    def __init__(self, select_seconds: float = SELECT_SECONDS):
        self._selector = EvidenceSelector(select_seconds)
        self.locked: Optional[LockedTarget] = None

    @property
    def is_locked(self) -> bool:
        return self.locked is not None

    @property
    def evidence(self) -> dict[str, float]:
        return self._selector.evidence

    def update(self, frame: np.ndarray, boxes: Sequence[Box],
               gaze_pt: Optional[tuple[float, float]], now: Optional[float] = None,
               ) -> tuple[Optional[Box], float, Optional[LockedTarget]]:
        """Returns (leading_box, progress, newly_locked_target)."""
        if self.locked is not None:
            return None, 1.0, None
        leader, progress, selected = self._selector.update(boxes, gaze_pt, now)
        if selected is not None:
            self.locked = LockedTarget(box=selected, snapshot=frame.copy(),
                                       locked_at=time.monotonic())
            self._selector.reset()
            return selected, 1.0, self.locked
        return leader, progress, None

    def hold(self, now: Optional[float] = None) -> None:
        if self.locked is None:
            self._selector.hold(now)

    def release(self) -> None:
        self.locked = None
        self._selector.reset()


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
