"""What happens after the arm has picked something up: the HOLDING phase.

While the arm holds an object the live object selection is off (a new
selection would open the gripper and drop it). Instead the screen shows a
large gaze menu; the user looks at a tile and holds (dwell), exactly like
picking an object:

    Bring to me   Raise          Closer   Put it back
    Place down    [ rest zone / status ]  Place where I look
    Let go        Lower          Away     Steer with head (head profile)

- Bring to me: carry it level to the taught serve pose (main.py --set-serve).
- Raise / Lower / Closer / Away: 50 mm nudges; Closer/Away are relative to
  the user (the direction from the arm base to the serve pose).
- Put it back: where it was picked up, set down, released.
- Place down: straight down where it is now, released.
- Place where I look: back to the observe pose, the live camera view comes
  up, the user looks at an empty spot on the table and holds; the spot is
  found by intersecting that pixel's camera ray with the table plane
  (as in GazeGrasp, where users place by fixating an empty spot).
- Let go: opens the gripper where it is, after a Yes/No confirmation.
- Steer with head: head-motion steering (head_control.py).

The center of the screen is a rest zone: looking there selects nothing.
While the arm moves, the bottom of the screen is one big STOP (short dwell);
stopping is the safe failure, so it is deliberately easy to trigger.

Every target is clamped to a workspace: never below the height that rests
the object on the table, never past the serve pose toward the user, within
reach, and not into the arm's own base.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from viam.proto.common import Pose, PoseInFrame, WorldState

from gaze_lock import Box, EvidenceSelector
from head_control import HeadRange, HeadSteer, SteerCommand, draw_steer_panel, steer_vector

NUDGE_MM = 50.0
PLACE_CLEARANCE_MM = 5.0        # set the object down this far above where it was resting
APPROACH_ABOVE_PLACE_MM = 100.0
LET_GO_LIFT_MM = 40.0
MAX_TCP_Z_MM = 500.0
MIN_RADIUS_MM = 200.0           # don't pull the object into the arm's base
PLACE_OBSTACLE_MARGIN_MM = 25.0
MIN_MOVE_MM = 2.0

MENU_DWELL_S = 1.0
CONFIRM_DWELL_S = 1.2
STOP_DWELL_S = 0.6
TILE_SIGMA_PX = 35.0            # tiles are big and adjacent: little evidence spills to neighbours
POINT_DWELL_S = 1.5
POINT_RADIUS_PX = 45.0


# --------------------------------------------------------------------------- state

@dataclass
class Held:
    label: str
    pick_grasp: Pose          # TCP pose where it was grasped (world)
    pick_approach: Pose       # above that
    orientation: dict         # wrist orientation kept the whole time
    obstacles: Optional[WorldState]
    place_z: float            # TCP z that rests the object on the table (+ clearance)
    width_mm: float
    pose: Pose                # where the TCP is now (last commanded)
    home: Optional[Pose]      # observe pose to return to
    footprint: Optional[WorldState] = None   # the object itself where it was picked (world), for re-avoiding it


class Status:
    def __init__(self):
        self.status = ""


@dataclass
class UserAxes:
    toward: np.ndarray        # horizontal unit vector from the arm toward the user (world)
    left: np.ndarray          # the user's left, as the user faces the arm
    serve: Pose


def user_axes(serve: Optional[Pose]) -> Optional[UserAxes]:
    if serve is None:
        return None
    r = math.hypot(serve.x, serve.y)
    if r < 1e-6:
        return None
    t = np.array([serve.x / r, serve.y / r, 0.0])
    left = np.array([t[1], -t[0], 0.0])     # user faces -t; rotate that +90 deg about z
    return UserAxes(toward=t, left=left, serve=serve)


def pose_at(p, orientation: dict) -> Pose:
    return Pose(x=float(p[0]), y=float(p[1]), z=float(p[2]), **orientation)


def xyz(p: Pose) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=float)


def clamp_to_workspace(p: np.ndarray, held: Held, axes: Optional[UserAxes], reach_mm: float) -> np.ndarray:
    x, y, z = (float(v) for v in p)
    z = min(max(z, held.place_z), MAX_TCP_Z_MM)
    r = math.hypot(x, y)
    if r > reach_mm:
        x, y = x * reach_mm / r, y * reach_mm / r
    elif 1e-6 < r < MIN_RADIUS_MM:
        x, y = x * MIN_RADIUS_MM / r, y * MIN_RADIUS_MM / r
    if axes is not None:
        limit = float(axes.serve.x * axes.toward[0] + axes.serve.y * axes.toward[1])
        d = x * axes.toward[0] + y * axes.toward[1]
        if d > limit:
            x, y = x - (d - limit) * axes.toward[0], y - (d - limit) * axes.toward[1]
    return np.array([x, y, z])


def nudge_target(held: Held, axes: Optional[UserAxes], action: str, reach_mm: float) -> Optional[Pose]:
    here = xyz(held.pose)
    if action == "raise":
        step = np.array([0.0, 0.0, NUDGE_MM])
    elif action == "lower":
        step = np.array([0.0, 0.0, -NUDGE_MM])
    elif axes is None:
        return None
    elif action == "closer":
        step = NUDGE_MM * axes.toward
    else:
        step = -NUDGE_MM * axes.toward
    target = clamp_to_workspace(here + step, held, axes, reach_mm)
    if np.linalg.norm(target - here) < MIN_MOVE_MM:
        return None
    return pose_at(target, held.orientation)


# --------------------------------------------------------------------------- tiles

@dataclass
class Tile:
    action: str
    label: str
    box: Box
    enabled: bool = True
    hint: str = ""


def _cell(col: int, row: int, cols: int, rows: int, w: int, h: int, span: int = 1, pad: int = 10) -> tuple:
    cw, ch = w / cols, h / rows
    return (int(col * cw + pad), int(row * ch + pad), int((col + span) * cw - pad), int((row + 1) * ch - pad))


def menu_tiles(w: int, h: int, head_mode: bool, have_axes: bool, have_intrinsics: bool) -> list[Tile]:
    layout = [
        ("bring", "Bring to me", 0, 0, have_axes, "needs --set-serve"),
        ("raise", "Raise", 1, 0, True, ""),
        ("closer", "Closer", 2, 0, have_axes, "needs --set-serve"),
        ("put_back", "Put it back", 3, 0, True, ""),
        ("place_down", "Place down", 0, 1, True, ""),
        ("place_look", "Place where I look", 3, 1, have_intrinsics, "no camera intrinsics"),
        ("let_go", "Let go", 0, 2, True, ""),
        ("lower", "Lower", 1, 2, True, ""),
        ("away", "Away", 2, 2, have_axes, "needs --set-serve"),
    ]
    if head_mode:
        layout.append(("steer", "Steer with head", 3, 2, have_axes, "needs --set-serve"))
    tiles = []
    for i, (action, label, col, row, enabled, hint) in enumerate(layout):
        x0, y0, x1, y1 = _cell(col, row, 4, 3, w, h)
        tiles.append(Tile(action, label, Box(x0, y0, x1, y1, action, 1.0, i), enabled, "" if enabled else hint))
    return tiles


def rest_zone(w: int, h: int) -> tuple:
    return _cell(1, 1, 4, 3, w, h, span=2)


def draw_tiles(canvas, tiles: list[Tile], leader: Optional[Box], progress: float) -> None:
    for t in tiles:
        b = t.box
        lead = leader is not None and leader.label == t.action
        fill = (40, 40, 40) if not t.enabled else ((60, 90, 60) if lead else (55, 55, 55))
        cv2.rectangle(canvas, (b.x0, b.y0), (b.x1, b.y1), fill, -1)
        if lead:
            px = int(b.x0 + (b.x1 - b.x0) * progress)
            cv2.rectangle(canvas, (b.x0, b.y1 - 14), (px, b.y1), (0, 220, 255), -1)
        edge = (0, 220, 0) if lead else ((90, 90, 90) if t.enabled else (60, 60, 60))
        cv2.rectangle(canvas, (b.x0, b.y0), (b.x1, b.y1), edge, 3 if lead else 2)
        color = (240, 240, 240) if t.enabled else (110, 110, 110)
        size = cv2.getTextSize(t.label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = b.x0 + max(8, (b.x1 - b.x0 - size[0]) // 2)
        ty = (b.y0 + b.y1) // 2 + size[1] // 2
        cv2.putText(canvas, t.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        if t.action in ("raise", "lower"):
            cx = (b.x0 + b.x1) // 2
            top, bottom = b.y0 + 25, ty - size[1] - 18
            start, end = ((cx, bottom), (cx, top)) if t.action == "raise" else ((cx, top), (cx, bottom))
            cv2.arrowedLine(canvas, start, end, color, 4, cv2.LINE_AA, tipLength=0.3)
        if t.hint:
            cv2.putText(canvas, t.hint, (b.x0 + 10, b.y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)


def put_lines(canvas, lines: list[str], x: int, y: int, scale: float = 0.7, color=(230, 230, 230)) -> None:
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (x, y + i * int(34 * scale / 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)


def draw_gaze(canvas, gaze_pt) -> None:
    if gaze_pt is not None:
        gx, gy = int(gaze_pt[0]), int(gaze_pt[1])
        cv2.circle(canvas, (gx, gy), 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 15, (255, 255, 255), 2, cv2.LINE_AA)


class PointDwell:
    """Selects a point where the gaze rests within a radius for long enough."""

    def __init__(self, seconds: float = POINT_DWELL_S, radius: float = POINT_RADIUS_PX):
        self.seconds, self.radius = seconds, radius
        self.pts: list = []
        self.start: Optional[float] = None

    def reset(self) -> None:
        self.pts, self.start = [], None

    def update(self, pt, now: float):
        """(anchor or None, progress, selected point or None)."""
        if pt is None:
            self.reset()
            return None, 0.0, None
        p = np.array(pt, dtype=float)
        if self.pts:
            anchor = np.median(np.array(self.pts), axis=0)
            if np.linalg.norm(p - anchor) > self.radius:
                self.reset()
        if not self.pts:
            self.start = now
        self.pts.append(p)
        anchor = np.median(np.array(self.pts), axis=0)
        progress = min(1.0, (now - self.start) / self.seconds)
        if progress >= 1.0:
            self.reset()
            return anchor, 1.0, (float(anchor[0]), float(anchor[1]))
        return anchor, progress, None


# --------------------------------------------------------------------------- robot access

@dataclass
class DeliveryIO:
    """Everything the HOLDING phase needs from main.py, injected (no import cycle)."""
    move: Callable            # async (pose, label, status, world_state=None, constraints=None) -> bool
    upright: Callable         # () -> Constraints
    straight_line: Callable   # () -> Constraints
    open_gripper: Callable    # async (status) -> None
    stop_arm: Callable        # async () -> None
    read_pose: Callable       # async () -> Optional[Pose]
    transform: Callable       # async (pose, from_frame, to_frame) -> Pose
    dry_run: bool
    reach_mm: float
    table_top_z: float
    camera_name: str
    world_frame: str
    intrinsics: object        # main.Intrinsics or None
    frame_w: int
    frame_h: int


async def table_point_from_pixel(io: DeliveryIO, u: float, v: float) -> Optional[np.ndarray]:
    """Where the camera ray through display pixel (u, v) meets the table plane.
    Camera frame: x right, y down, z forward (mm)."""
    intr = io.intrinsics
    if intr is None:
        return None
    su, sv = intr.width / io.frame_w, intr.height / io.frame_h
    rx, ry = (u * su - intr.cx) / intr.fx, (v * sv - intr.cy) / intr.fy
    o = await io.transform(Pose(x=0.0, y=0.0, z=0.0, o_z=1.0), io.camera_name, io.world_frame)
    p = await io.transform(Pose(x=rx * 1000.0, y=ry * 1000.0, z=1000.0, o_z=1.0), io.camera_name, io.world_frame)
    dz = p.z - o.z
    if dz >= -1e-6:
        return None                   # ray doesn't point down toward the table
    t = (io.table_top_z - o.z) / dz
    if t <= 0:
        return None
    return np.array([o.x + t * (p.x - o.x), o.y + t * (p.y - o.y), io.table_top_z])


# --------------------------------------------------------------------------- the session

class DeliverySession:
    def __init__(self, io: DeliveryIO, held: Held, serve: Optional[Pose],
                 head_mode: bool = False, head_range: Optional[HeadRange] = None):
        self.io, self.held = io, held
        self.axes = user_axes(serve)
        self.head_mode = head_mode and head_range is not None
        self.head_range = head_range
        self.screen = "menu"          # menu | confirm | place_look | steer
        self.status = Status()
        self.message = f"Holding: {held.label}"
        self.finished = False         # object released: main goes back to live selection
        self.task: Optional[asyncio.Task] = None
        w, h = io.frame_w, io.frame_h
        self.tiles = menu_tiles(w, h, self.head_mode, self.axes is not None, io.intrinsics is not None)
        self.confirm_tiles = [
            Tile("yes", "Yes, let go", Box(40, h // 3, w // 2 - 80, h - 80, "yes", 1.0, 0)),
            Tile("no", "No, keep holding", Box(w // 2 + 80, h // 3, w - 40, h - 80, "no", 1.0, 1)),
        ]
        self.cancel_tile = Tile("cancel", "Cancel", Box(20, 20, 300, 130, "cancel", 1.0, 0))
        self.steer_tiles = [
            Tile("plane_vertical", "Up/down + left/right", Box(20, 20, 420, 130, "plane_vertical", 1.0, 0)),
            Tile("plane_horizontal", "Closer/away + left/right", Box(440, 20, 880, 130, "plane_horizontal", 1.0, 1)),
            Tile("steer_done", "Done", Box(w - 300, 20, w - 20, 130, "steer_done", 1.0, 2)),
        ]
        self.stop_box = Box(20, int(h * 0.6), w - 20, h - 20, "stop", 1.0, 0)
        self.selector = EvidenceSelector(select_seconds=MENU_DWELL_S, sigma_px=TILE_SIGMA_PX)
        self.stop_selector = EvidenceSelector(select_seconds=STOP_DWELL_S, sigma_px=60.0)
        self.point = PointDwell()
        self.steer: Optional[HeadSteer] = None
        self.plane = "vertical"
        self._leader, self._progress = None, 0.0

    # ---- helpers

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def wants_feed(self) -> bool:
        return self.screen == "place_look"

    def _run(self, coro) -> None:
        self.selector.reset()
        self.stop_selector.reset()
        self.task = asyncio.create_task(self._guard(coro))

    async def _guard(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.message = f"Error: {e}"
            print(f"[deliver] {self.message}")

    async def _move(self, target: Pose, label: str, constraints) -> bool:
        ok = await self.io.move(target, label, self.status, self.held.obstacles, constraints)
        if ok:
            self.held.pose = target
        else:
            self.message = f"Couldn't plan the move to {label}; try something else"
        return ok

    async def stop(self, reason: str = "Stopped") -> None:
        if self.busy:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if not self.io.dry_run:
            await self.io.stop_arm()
            actual = await self.io.read_pose()
            if actual is not None:
                self.held.pose = pose_at(xyz(actual), self.held.orientation)
        self.message = reason
        print(f"[deliver] {reason}")

    # ---- actions

    def _start(self, action: str) -> None:
        if action == "let_go":
            self.screen = "confirm"
            self.selector.reset()
            return
        if action == "steer":
            self.screen = "steer"
            self.steer = HeadSteer(self.head_range)
            self.selector.reset()
            self.message = "Hold your head still in the center"
            return
        if action in ("raise", "lower", "closer", "away"):
            target = nudge_target(self.held, self.axes, action, self.io.reach_mm - 30.0)
            if target is None:
                self.message = "Can't go further that way"
                return
            self._run(self._move(target, action, self.io.straight_line()))
            return
        runs = {"bring": self._bring, "put_back": self._put_back, "place_down": self._place_down,
                "place_look": self._go_look}
        self._run(runs[action]())

    async def _bring(self) -> None:
        s = self.axes.serve
        target = clamp_to_workspace(np.array([s.x, s.y, s.z]), self.held, self.axes, self.io.reach_mm - 30.0)
        if await self._move(pose_at(target, self.held.orientation), "you (serve pose)", self.io.upright()):
            self.message = "Here you go"

    async def _release_at(self, above: Pose, place: Pose, where: str) -> None:
        """Above -> straight down -> open -> straight up -> home. Opens only if
        the descent succeeded, so a failed plan never drops the object."""
        if np.linalg.norm(xyz(above) - xyz(self.held.pose)) >= MIN_MOVE_MM:
            if not await self._move(above, f"above {where}", self.io.upright()):
                return
        if not await self._move(place, f"down to {where}", self.io.straight_line()):
            await self._move(above, "back up", self.io.straight_line())
            return
        await self.io.open_gripper(self.status)
        await self._move(above, "lift (straight up)", self.io.straight_line())
        await self._go_home_and_finish(f"Placed the {self.held.label}")

    async def _put_back(self) -> None:
        g = self.held.pick_grasp
        above = pose_at(xyz(self.held.pick_approach), self.held.orientation)
        place = pose_at([g.x, g.y, self.held.place_z], self.held.orientation)
        await self._release_at(above, place, "where it was")

    async def _place_down(self) -> None:
        here = xyz(self.held.pose)
        place = pose_at([here[0], here[1], self.held.place_z], self.held.orientation)
        await self._release_at(self.held.pose, place, "the table here")

    async def _go_look(self) -> None:
        home = self.held.home
        if home is not None and not await self._move(pose_at(xyz(home), self.held.orientation),
                                                     "the observe pose", self.io.upright()):
            return
        self.screen = "place_look"
        self.point.reset()
        self.message = "Look at an empty spot on the table and hold"

    def place_problem(self, spot: Optional[np.ndarray], u: float, v: float, boxes: list[Box]) -> Optional[str]:
        if spot is None:
            return "that isn't on the table"
        r = math.hypot(spot[0], spot[1])
        if r > self.io.reach_mm - 30.0 or r < MIN_RADIUS_MM:
            return "out of the arm's reach"
        if self.axes is not None:
            limit = self.axes.serve.x * self.axes.toward[0] + self.axes.serve.y * self.axes.toward[1]
            if spot[0] * self.axes.toward[0] + spot[1] * self.axes.toward[1] > limit:
                return "too close to you"
        for b in boxes:
            if b.contains(u, v):
                return f"there's a {b.label} there"
        half = self.held.width_mm / 2.0
        if self.held.obstacles is not None:
            for gif in self.held.obstacles.obstacles:
                for g in gif.geometries:
                    ext = max(g.box.dims_mm.x, g.box.dims_mm.y) / 2.0 if g.HasField("box") else 30.0
                    if math.hypot(spot[0] - g.center.x, spot[1] - g.center.y) < ext + half + PLACE_OBSTACLE_MARGIN_MM:
                        return f"too close to the {g.label or 'object'} there"
        return None

    async def _place_at_pixel(self, u: float, v: float, boxes: list[Box]) -> None:
        spot = await table_point_from_pixel(self.io, u, v)
        problem = self.place_problem(spot, u, v, boxes)
        if problem:
            self.message = f"Can't place there: {problem}. Look at another spot"
            return
        place = pose_at([spot[0], spot[1], self.held.place_z], self.held.orientation)
        above = pose_at([spot[0], spot[1], self.held.place_z + APPROACH_ABOVE_PLACE_MM], self.held.orientation)
        self.screen = "menu"
        await self._release_at(above, place, "that spot")

    async def _let_go(self) -> None:
        await self.io.open_gripper(self.status)
        up = clamp_to_workspace(xyz(self.held.pose) + np.array([0.0, 0.0, LET_GO_LIFT_MM]),
                                self.held, self.axes, self.io.reach_mm - 30.0)
        await self._move(pose_at(up, self.held.orientation), "lift away", self.io.straight_line())
        await self._go_home_and_finish("Let go")

    async def _go_home_and_finish(self, msg: str) -> None:
        home = self.held.home
        if home is not None:
            await self.io.move(pose_at(xyz(home), self.held.orientation), "the observe pose",
                               self.status, None, self.io.upright())
        self.message = msg
        self.finished = True

    # ---- per-frame

    async def tick(self, gaze_pt, blinking: bool, head, obs, now: Optional[float] = None):
        now = time.monotonic() if now is None else now
        w, h = self.io.frame_w, self.io.frame_h
        if self.screen == "steer":
            return await self._tick_steer(gaze_pt, blinking, head, now)
        if self.busy:
            return await self._tick_busy(gaze_pt, blinking, now)
        if self.screen == "place_look":
            return self._tick_place_look(gaze_pt, blinking, obs, now)

        canvas = np.zeros((h, w, 3), np.uint8)
        tiles = self.confirm_tiles if self.screen == "confirm" else self.tiles
        self.selector.select_seconds = CONFIRM_DWELL_S if self.screen == "confirm" else MENU_DWELL_S
        leader, progress, selected = self._select([t.box for t in tiles if t.enabled], gaze_pt, blinking, now)
        if selected is not None:
            if self.screen == "confirm":
                self.screen = "menu"
                if selected.label == "yes":
                    self._run(self._let_go())
                else:
                    self.message = "Still holding it"
            else:
                self._start(selected.label)
        draw_tiles(canvas, tiles, leader, progress)
        if self.screen == "confirm":
            put_lines(canvas, [f"Let go of the {self.held.label} here?"], 40, 70, 1.0)
        else:
            x0, y0, _, _ = rest_zone(w, h)
            put_lines(canvas, [self.message[:44], "Look at a button and hold.", "Look here to rest."], x0 + 20, y0 + 50)
        draw_gaze(canvas, gaze_pt)
        return canvas

    def _select(self, boxes, gaze_pt, blinking: bool, now: float):
        if blinking:
            self.selector.hold(now)
            return self._leader, self._progress, None
        leader, progress, selected = self.selector.update(boxes, gaze_pt, now)
        self._leader, self._progress = leader, progress
        if selected is not None:
            self.selector.reset()
            self._leader, self._progress = None, 0.0
        return leader, progress, selected

    async def _tick_busy(self, gaze_pt, blinking: bool, now: float):
        w, h = self.io.frame_w, self.io.frame_h
        canvas = np.zeros((h, w, 3), np.uint8)
        put_lines(canvas, [self.status.status[:70] or "Moving...", "Look at STOP and hold to stop the arm."], 30, 60)
        if blinking:
            self.stop_selector.hold(now)
            progress, selected = 0.0, None
        else:
            _, progress, selected = self.stop_selector.update([self.stop_box], gaze_pt, now)
        b = self.stop_box
        cv2.rectangle(canvas, (b.x0, b.y0), (b.x1, b.y1), (0, 0, 150), -1)
        cv2.rectangle(canvas, (b.x0, b.y1 - 16), (int(b.x0 + (b.x1 - b.x0) * progress), b.y1), (0, 220, 255), -1)
        cv2.putText(canvas, "STOP", ((b.x0 + b.x1) // 2 - 90, (b.y0 + b.y1) // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
        draw_gaze(canvas, gaze_pt)
        if selected is not None:
            await self.stop("Stopped")
            self.screen = "menu"
        return canvas

    def _tick_place_look(self, gaze_pt, blinking: bool, obs, now: float):
        w, h = self.io.frame_w, self.io.frame_h
        canvas = obs.frame.copy() if obs is not None else np.zeros((h, w, 3), np.uint8)
        boxes = list(obs.boxes) if obs is not None else []
        for b in boxes:
            cv2.rectangle(canvas, (b.x0, b.y0), (b.x1, b.y1), (0, 0, 220), 2)
        in_cancel = gaze_pt is not None and self.cancel_tile.box.contains(*gaze_pt)
        leader, progress, selected = self._select([self.cancel_tile.box], gaze_pt, blinking, now)
        draw_tiles(canvas, [self.cancel_tile], leader, progress)
        if selected is not None:
            self.screen = "menu"
            self.message = f"Holding: {self.held.label}"
            return canvas
        put_lines(canvas, [self.message[:60]], 330, 70)
        if not blinking:
            anchor, prog, spot = self.point.update(None if in_cancel else gaze_pt, now)
            if anchor is not None:
                ax, ay = int(anchor[0]), int(anchor[1])
                cv2.circle(canvas, (ax, ay), int(POINT_RADIUS_PX), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.ellipse(canvas, (ax, ay), (int(POINT_RADIUS_PX) + 8,) * 2, -90, 0, 360 * prog,
                            (0, 220, 255), 4, cv2.LINE_AA)
            if spot is not None:
                self._run(self._place_at_pixel(spot[0], spot[1], boxes))
        draw_gaze(canvas, gaze_pt)
        return canvas

    async def _tick_steer(self, gaze_pt, blinking: bool, head, now: float):
        w, h = self.io.frame_w, self.io.frame_h
        canvas = np.zeros((h, w, 3), np.uint8)
        cmd: SteerCommand = self.steer.update(head, now)
        # Gaze buttons only while the head is centered, so steering and
        # choosing never fight each other.
        if cmd.state == "neutral" and not self.busy:
            leader, progress, selected = self._select([t.box for t in self.steer_tiles], gaze_pt, blinking, now)
        else:
            self.selector.hold(now)
            leader, progress, selected = None, 0.0, None
        if selected is not None:
            if selected.label == "steer_done":
                self.screen, self.steer = "menu", None
                self.message = f"Holding: {self.held.label}"
                return canvas
            self.plane = "vertical" if selected.label == "plane_vertical" else "horizontal"
        if cmd.state == "move" and not self.busy:
            here = xyz(self.held.pose)
            step = steer_vector(cmd, self.plane, self.axes.left, self.axes.toward)
            target = clamp_to_workspace(here + step, self.held, self.axes, self.io.reach_mm - 30.0)
            if np.linalg.norm(target - here) < MIN_MOVE_MM:
                self.message = "Edge of the safe area"
            else:
                self.message = "Moving"
                self._run(self._move(pose_at(target, self.held.orientation), "steer", self.io.straight_line()))
        tiles = [Tile(t.action, t.label + ("  *" if t.action == f"plane_{self.plane}" else ""), t.box)
                 for t in self.steer_tiles]
        draw_tiles(canvas, tiles, leader, progress)
        draw_steer_panel(canvas, cmd, self.plane, (w // 2, h // 2 + 60), 170)
        text = {
            "arming": "Hold your head still in the center...",
            "paused": f"Paused ({cmd.reason}): center your head to continue",
            "neutral": "Turn your head to move it. Center your head to stop.",
            "pending": "Hold that...",
            "move": "Moving",
        }[cmd.state]
        put_lines(canvas, [text, "Buttons work while your head is centered."], 30, 180)
        draw_gaze(canvas, gaze_pt)
        return canvas
