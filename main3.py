"""Gaze-selected pick with a Viam arm.

Laptop webcam -> gaze point on the RealSense feed window -> dwell on a 2D
detection box locks that object -> while the arm is still, its box bottom
center is back-projected through the wrist camera and intersected with the
known table plane. Manual dimensions supply the grasp height. The motion
service then moves the gripper above it, opens, descends, closes, and lifts.

SAFETY: the arm moves ONLY with --execute. Without it everything runs (camera,
vision, gaze, lock, camera-ray transforms, checks) and the poses are
printed, but no motion or gripper command is sent. Q cancels the grasp and
sends arm.stop(), best effort; the physical E-STOP is the real stop.

Demo mode (the default) has exactly two gaze actions:
  1. Look at an object and hold: the arm picks it up and carries it back to
     the observe pose, still holding it, so the camera sees the table again.
  2. While holding, look at a DIFFERENT object and hold: the arm puts the held
     one back down where it came from, then picks up the new one.
The new object is located (and every object frozen into world coordinates)
before anything moves; the put-back spot is then treated as an obstacle.

--menu instead shows a big gaze menu after the pick (delivery.py): bring it
to the user, raise/lower/closer/away, put it back, place it down or where the
user looks, let go, or (--user head, which implies --menu) steer it with head
motion (head_control.py).

Keys: Q stop+quit, R release the lock once the grasp attempt has finished, C recalibrate,
P (demo mode, operator) put the held object back without picking another.
Flags: --execute (allow motion), --menu, --user eyes|head, --skip-calibration (reuse
the last calibration), --set-home, --set-serve (record where objects are brought to the user).
"""

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# main3.py intentionally lives one folder above the shared implementation
# modules. Keep that folder importable without copying or modifying it.
HERE = Path(__file__).resolve().parent
ARM_CONTROL_DIR = HERE / "arm-control"
if str(ARM_CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(ARM_CONTROL_DIR))

import cv2
import numpy as np

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.proto.common import GeometriesInFrame, Geometry, Pose, PoseInFrame, WorldState
from viam.proto.service.motion import Constraints, LinearConstraint, OrientationConstraint

from delivery import PLACE_CLEARANCE_MM, DeliveryIO, DeliverySession, Held, user_axes
from gaze_lock import IGNORE_LABELS, Box, GazeLockController, draw_live, draw_locked, filter_background_boxes
from head_control import HeadRange, calibrate_head_range
from webcam_gaze import (
    CALIBRATION_PATH,
    GazeCalibration,
    GazeEstimator,
    WebcamGazeTracker,
    run_calibration,
)

ENV_FILES = (HERE / ".env", HERE.parent / ".env")   # arm-control/.env, then the repo root


def load_env() -> dict[str, str]:
    """Credentials come from a gitignored .env (see ../.env.example), never from source."""
    values: dict[str, str] = {}
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    for key in ("VIAM_MACHINE_ADDRESS", "VIAM_API_KEY", "VIAM_API_KEY_ID"):
        values[key] = os.environ.get(key) or values.get(key, "")
        if not values[key]:
            raise SystemExit(
                f"{key} is not set. Copy ../.env.example to {ENV_FILES[0]} and fill it in "
                "(the file is gitignored, so it stays off GitHub)."
            )
    return values


CAMERA_NAME = "cam"
# The configured machine service `vision-1` supplies the 2D detections.
DETECTOR_NAME = "vision-1"
MOTION_SERVICE_NAME = "motion"
GRIPPER_NAME = "gripper"
ARM_NAME = "arm"

MOTION_REFERENCE_FRAME = "world"

# --- Geometry of this machine (hackathon fragment, read with `viam fragment get`) ---
# The gripper frame sits 150 mm out from the arm flange (not 105 as in the
# crash-course deck), i.e. already near the finger pads. The fingertips of the
# UFactory gripper are ~165 mm out (workshop: 105 mm TCP + 60 mm fingers), so
# they extend FINGERTIP_FROM_FLANGE_MM - tcp beyond the TCP. The TCP distance is
# measured from the machine at startup; GRIPPER_TCP_FROM_FLANGE_MM is only the
# fallback. Measure the fingertip number with a tape if grasps land high/low.
FINGERTIP_FROM_FLANGE_MM = 165.0
GRIPPER_TCP_FROM_FLANGE_MM = 150.0
# Measured physical table surface in the world frame. The configured collision
# table is deliberately about 23 mm higher, which is conservative.
TABLE_TOP_Z_MM = -146.0
# `motion.move(component_name="gripper", ...)` targets the gripper origin,
# not the fingertips. With this top-down tool orientation the fingertips are
# 165 mm below that origin.
GRIPPER_ORIGIN_TO_FINGERTIPS_MM = 165.0
FINGERTIP_TABLE_CLEARANCE_MM = 10.0
ARM_REACH_MM = 700.0                 # xArm6
APPROACH_HEIGHT_MM = 120.0           # standoff above the grasp pose
GRASP_HEIGHT_FRACTION = 0.60         # grip a tall item 60% up from the table
DEFAULT_OBJECT_HEIGHT_MM = 100.0
MIN_TARGET_RADIUS_MM = 200.0
MAX_TARGET_RADIUS_MM = 600.0
# Temporary fixed-observe-pose correction. The mounted camera/frame system
# currently projects table targets about 3 in (76.2 mm) too close to the arm
# base. Positive values move a target radially away from the base.
RADIAL_OUTWARD_CORRECTION_MM = 76.2
# The arm's installed world-X direction is opposite the initial assumption.
# Positive world X moves the physical pinch point 3.5 in left while keeping
# the same north/away-from-base (Y) line.
LATERAL_X_CORRECTION_MM = 88.9

# Measured user objects. Values are millimetres (1 inch = 25.4 mm).
OBJECT_DIMENSIONS_MM = {
    "coke can": {"width": 63.5, "depth": 63.5, "height": 120.65},
    "cokecan": {"width": 63.5, "depth": 63.5, "height": 120.65},
    "juicebox": {"width": 63.5, "depth": 60.325, "height": 152.4},
    "juice box": {"width": 63.5, "depth": 60.325, "height": 152.4},
    "black block": {"width": 28.575, "depth": 28.575, "height": 60.325},
    "blackblock": {"width": 28.575, "depth": 28.575, "height": 60.325},
    "block": {"width": 28.575, "depth": 28.575, "height": 60.325},
}

# Orientation: grasp with the wrist orientation the arm already has at the
# observe pose when it points roughly down, so the planner never has to spin
# the wrist. (The old fixed target, theta=0 vs the arm's 142.7, asked for a
# ~143 degree wrist rotation on every grasp.) This is the fallback.
DEFAULT_GRASP_ORIENTATION = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
MIN_DOWNWARD_O_Z = -0.9              # current orientation reused only if o_z <= this
ORIENTATION_TOLERANCE_DEGS = 15.0    # held along approach and carry moves
LINE_TOLERANCE_MM = 10.0             # descent / lift stay on a straight line

# Width-based close (so a paper cup isn't crushed): the uFactory gripper takes a
# position on a 0-850 scale (850 = fully open, ~85 mm), i.e. ~10 units per mm.
GRIPPER_SQUEEZE_MM = 8.0
GRIPPER_POS_PER_MM = 10.0
GRIPPER_MAX_POS = 850

# Home / serve pose: record once with the arm parked there
# (python main.py --set-home -> home_pose.json). Without it, the object is
# carried back to where the gripper was when the object was locked.
HOME_POSE_PATH = HERE / "home_pose.json"
GO_HOME_ON_START = False
RETURN_TO_START = False

# Every other segmented object is frozen into world coordinates at lock time
# and handed to the planner as an obstacle, on top of the machine's static
# table/wall/ceiling obstacles. Direct arm moves ignore obstacles, so this
# code only ever moves through the motion service.
AVOID_DETECTED_OBJECTS = True
RETRY_WITHOUT_OBSTACLES = True       # retry once with only the static obstacles (constraints kept)

WEBCAM_INDEX = 0
RELEASE_AFTER_SECONDS = 4.0
WINDOW = "Gaze-selected pick (Q stop+quit, R release lock, C recalibrate)"

EXECUTE = "--execute" in sys.argv
DRY_RUN = not EXECUTE
SKIP_CALIBRATION = "--skip-calibration" in sys.argv
SET_HOME = "--set-home" in sys.argv
SET_SERVE = "--set-serve" in sys.argv     # record where objects are brought to the user, and exit

# --user eyes (default): the gaze menu after a pick. --user head: the same menu
# plus head-motion steering, with a per-user head-range calibration at startup.
USER_PROFILE = sys.argv[sys.argv.index("--user") + 1] if "--user" in sys.argv else "eyes"
if USER_PROFILE != "eyes":
    raise SystemExit("main3.py uses fixed-head nine-point gaze calibration only; use --user eyes")
# Default: the two-action demo (pick; look at another object to swap). --menu:
# the post-pick gaze menu. Head steering lives in the menu, so --user head implies it.
MENU = "--menu" in sys.argv or USER_PROFILE == "head"
SERVE_POSE_PATH = HERE / "serve_pose.json"
LIVE_COOLDOWN_S = 2.5          # after putting something down, don't immediately select again


async def connect():
    env = load_env()
    # The SDK's periodic health check has a 1 s deadline; a slow call on the
    # machine (slow perception) can trip it and the client tears the connection
    # down. Disable the check; failures surface on the call itself instead.
    opts = RobotClient.Options.with_api_key(
        api_key=env["VIAM_API_KEY"], api_key_id=env["VIAM_API_KEY_ID"],
        check_connection_interval=0, attempt_reconnect_interval=0,
    )
    return await RobotClient.at_address(env["VIAM_MACHINE_ADDRESS"], opts)


def decode_color_frame(images):
    """First image from get_images() that decodes as a color picture."""
    for img in images:
        if "depth" in img.name.lower():
            continue
        frame = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    return None


def decode_viam_image(img) -> Optional[np.ndarray]:
    if img is None or not img.data:
        return None
    return cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)


def box_from_detection(det, index: int, frame_w: int, frame_h: int) -> Box:
    """yolo-detector reports absolute pixels at the camera's native resolution;
    use the normalized fields only if a detector fills them in."""
    if det.x_max_normalized or det.y_max_normalized:
        x0, y0 = det.x_min_normalized * frame_w, det.y_min_normalized * frame_h
        x1, y1 = det.x_max_normalized * frame_w, det.y_max_normalized * frame_h
    else:
        x0, y0, x1, y1 = det.x_min, det.y_min, det.x_max, det.y_max
    return Box(int(x0), int(y0), int(x1), int(y1), det.class_name, float(det.confidence), index)


# --------------------------------------------------------------------------- live feed

@dataclass
class Observation:
    frame: np.ndarray
    boxes: list[Box]
    at: float
    paired: bool      # image and boxes came from one capture


COLOR_SOURCE_NAME = "color"


class RobotFeed:
    """One background task capturing image + detections TOGETHER
    (CaptureAllFromCamera), so the boxes drawn and hit-tested always belong to
    the exact image on screen, and the lock snapshot is that same image. Falls
    back to separate calls if the detector can't do a combined capture."""

    def __init__(self, cam: Camera, detector: VisionClient):
        self.cam, self.detector = cam, detector
        self.obs: Optional[Observation] = None
        self.frame_w = self.frame_h = 0
        self.paired = True
        self.paused = False
        self.fps = 0.0
        self.capture_ms = 0.0
        self._task: Optional[asyncio.Task] = None

    def _boxes(self, detections) -> list[Box]:
        return filter_background_boxes(
            [box_from_detection(d, i, self.frame_w, self.frame_h) for i, d in enumerate(detections)])

    async def _capture_paired(self) -> Observation:
        res = await self.detector.capture_all_from_camera(CAMERA_NAME, return_image=True, return_detections=True)
        frame = decode_viam_image(res.image)
        if frame is None:
            raise RuntimeError("combined capture returned no decodable image")
        if not self.frame_w:
            self.frame_h, self.frame_w = frame.shape[:2]
        return Observation(frame, self._boxes(res.detections or []), time.monotonic(), True)

    async def _capture_unpaired(self) -> Observation:
        try:
            images, _ = await self.cam.get_images(filter_source_names=[COLOR_SOURCE_NAME])
        except Exception:
            images = []
        if not images:
            images, _ = await self.cam.get_images()
        frame = decode_color_frame(images)
        if frame is None:
            raise RuntimeError(f"no color frame from camera '{CAMERA_NAME}'")
        if not self.frame_w:
            self.frame_h, self.frame_w = frame.shape[:2]
        detections = await self.detector.get_detections_from_camera(CAMERA_NAME)
        return Observation(frame, self._boxes(detections), time.monotonic(), False)

    async def capture(self) -> Observation:
        if self.paired:
            try:
                return await self._capture_paired()
            except Exception as e:
                self.paired = False
                print(f"[feed] '{DETECTOR_NAME}' can't do a combined capture ({type(e).__name__}: {e}); "
                      "falling back to separate image + detection calls, which can come from different frames")
        return await self._capture_unpaired()

    async def first(self) -> tuple[int, int]:
        self.obs = await self.capture()
        return self.frame_w, self.frame_h

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            if self.paused:
                await asyncio.sleep(0.05)
                continue
            try:
                t0 = time.monotonic()
                obs = await self.capture()
                dt = time.monotonic() - t0
                self.capture_ms = dt * 1000
                self.fps = 0.7 * self.fps + 0.3 / max(1e-3, dt)
                self.obs = obs
            except Exception as e:
                print(f"[feed] capture error: {e}")
                await asyncio.sleep(0.5)


# --------------------------------------------------------------------------- table-plane localization


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


async def get_intrinsics(cam: Camera) -> Optional[Intrinsics]:
    try:
        ip = (await cam.get_properties()).intrinsic_parameters
        if ip.focal_x_px > 0 and ip.width_px > 0:
            return Intrinsics(ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px,
                              ip.width_px, ip.height_px)
    except Exception as e:
        print(f"[main] camera intrinsics unavailable ({type(e).__name__}: {e})")
    return None


async def pose_in(robot: RobotClient, pose: Pose, ref: str, dest: str) -> Pose:
    if ref == dest:
        return pose
    return (await robot.transform_pose(PoseInFrame(reference_frame=ref, pose=pose), dest)).pose


def dimensions_for(label: str) -> dict[str, float]:
    key = label.casefold().replace("_", " ").replace("-", " ")
    dimensions = OBJECT_DIMENSIONS_MM.get(key)
    if dimensions is None:
        print(f"[pick] WARNING: no manual dimensions for {label!r}; using a "
              f"{DEFAULT_OBJECT_HEIGHT_MM:.0f} mm default height")
        return {"width": 60.0, "depth": 60.0, "height": DEFAULT_OBJECT_HEIGHT_MM}
    return dimensions


async def bottom_center_on_table(robot: RobotClient, box: Box, intr: Intrinsics,
                                 frame_w: int, frame_h: int) -> tuple[float, float, tuple[float, float]]:
    """Back-project a box bottom-center and intersect its world ray with table Z."""
    pixel = ((box.x0 + box.x1) / 2.0, float(box.y1))
    u, v = pixel[0] * intr.width / frame_w, pixel[1] * intr.height / frame_h
    ray_x, ray_y = (u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy
    origin = await pose_in(robot, Pose(x=0, y=0, z=0, o_z=1), CAMERA_NAME, MOTION_REFERENCE_FRAME)
    far = await pose_in(robot, Pose(x=ray_x * 1000, y=ray_y * 1000, z=1000, o_z=1),
                        CAMERA_NAME, MOTION_REFERENCE_FRAME)
    dx, dy, dz = far.x - origin.x, far.y - origin.y, far.z - origin.z
    if abs(dz) < 1e-6:
        raise RuntimeError("camera ray is parallel to the table; check the camera frame")
    t = (TABLE_TOP_Z_MM - origin.z) / dz
    if t <= 0:
        raise RuntimeError("table intersection is behind the camera; check camera orientation")
    return origin.x + t * dx, origin.y + t * dy, pixel


def apply_radial_correction(x: float, y: float) -> tuple[float, float]:
    """Move the target away from the arm base by the measured fixed miss."""
    radius = math.hypot(x, y)
    if radius < 1e-6:
        raise RuntimeError("cannot apply radial correction at the arm-base origin")
    scale = RADIAL_OUTWARD_CORRECTION_MM / radius
    outward_x, outward_y = x + x * scale, y + y * scale
    return outward_x + LATERAL_X_CORRECTION_MM, outward_y


@dataclass
class GraspPlan:
    orientation: dict
    approach: Pose
    grasp: Pose
    return_pose: Optional[Pose]
    width_mm: float
    fingertip_beyond_tcp_mm: float


def plan_grasp(x: float, y: float, dimensions: dict[str, float], home: Optional[Pose]) -> GraspPlan:
    """Make a table-relative top-down plan using hand-supplied dimensions."""
    orient = dict(DEFAULT_GRASP_ORIENTATION)
    fingertip_z = TABLE_TOP_Z_MM + dimensions["height"] * GRASP_HEIGHT_FRACTION
    grasp_z = fingertip_z + GRIPPER_ORIGIN_TO_FINGERTIPS_MM
    grasp = Pose(x=x, y=y, z=grasp_z, **orient)
    approach = Pose(x=x, y=y, z=grasp_z + APPROACH_HEIGHT_MM, **orient)
    width = min(dimensions["width"], dimensions["depth"])
    return GraspPlan(orient, approach, grasp, home if RETURN_TO_START else None, width, 0.0)


# --------------------------------------------------------------------------- motion

class GraspJob:
    """Status of the background grasp so the display loop can show progress."""

    def __init__(self):
        self.status = "locating selected object on the table..."
        self.task: Optional[asyncio.Task] = None
        self.finished_at: Optional[float] = None
        self.holding: Optional[bool] = None
        self.held: Optional[Held] = None      # what the gripper holds right now (kept current during a swap)
        self.ok = False                       # the whole job succeeded

    @property
    def done(self) -> bool:
        return self.finished_at is not None


def upright() -> Constraints:
    return Constraints(orientation_constraint=[
        OrientationConstraint(orientation_tolerance_degs=ORIENTATION_TOLERANCE_DEGS)])


def straight_line() -> Constraints:
    return Constraints(linear_constraint=[
        LinearConstraint(line_tolerance_mm=LINE_TOLERANCE_MM, orientation_tolerance_degs=ORIENTATION_TOLERANCE_DEGS)])


async def move_to(motion: MotionClient, pose: Pose, label: str, job: GraspJob,
                  world_state: Optional[WorldState] = None,
                  constraints: Optional[Constraints] = None) -> bool:
    prefix = "DRY RUN, would be " if DRY_RUN else ""
    extras = []
    if world_state is not None:
        extras.append(f"{sum(len(g.geometries) for g in world_state.obstacles)} obstacles")
    if constraints is not None:
        extras.append("straight line" if constraints.linear_constraint else "orientation held")
    job.status = (f"{prefix}moving to {label}: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f} mm"
                  + (f" ({', '.join(extras)})" if extras else ""))
    print(f"[grab] {job.status}")
    if DRY_RUN:
        await asyncio.sleep(0.4)
        return True
    # viam-sdk 0.80.0: component_name is the component's plain name (proto string).
    try:
        ok = await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame=MOTION_REFERENCE_FRAME, pose=pose),
            world_state=world_state,
            constraints=constraints,
        )
    except Exception as e:
        # A free retry is only useful when the planner cannot honor an
        # otherwise safe straight-line path. Never retry a reported collision:
        # that destination itself is unsafe regardless of path type.
        message = str(e).casefold()
        if (constraints is None or not constraints.linear_constraint
                or "linear with cbirrt" not in message):
            raise
        print(f"[grab] linear {label} failed ({e}); retrying that short segment as a free move")
        ok = await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame=MOTION_REFERENCE_FRAME, pose=pose),
            world_state=world_state,
        )
    print(f"[grab] {label} result: {ok}")
    if not ok and world_state is not None and RETRY_WITHOUT_OBSTACLES:
        print(f"[grab] planning to {label} failed with detected obstacles; retrying with static obstacles only")
        return await move_to(motion, pose, label, job, None, constraints)
    if not ok:
        job.status = f"motion.move() failed on {label}"
        print(f"[grab] {job.status}")
    return ok


def gripper_position_for_width(width_mm: float) -> int:
    return int(max(0.0, min(GRIPPER_MAX_POS, (width_mm - GRIPPER_SQUEEZE_MM) * GRIPPER_POS_PER_MM)))


async def open_gripper(gripper: Gripper, job: GraspJob) -> None:
    job.status = "DRY RUN, would open the gripper" if DRY_RUN else "opening the gripper"
    print(f"[grab] {job.status}")
    if not DRY_RUN:
        await gripper.open()
        await asyncio.sleep(0.3)   # let the fingers finish opening


async def close_gripper(gripper: Gripper, arm: Arm, width_mm: float, job: GraspJob) -> None:
    """Close to the object's width ({"set": pos} on the gripper, then the arm's
    move_gripper); grab() by force feedback if that isn't accepted or the width
    estimate is useless."""
    max_span_mm = GRIPPER_MAX_POS / GRIPPER_POS_PER_MM
    if DRY_RUN:
        job.status = (f"DRY RUN, would close the gripper to position {gripper_position_for_width(width_mm)} "
                      f"(object ~{width_mm:.0f} mm wide)" if 0 < width_mm <= max_span_mm
                      else "DRY RUN, would call grab()")
        print(f"[grab] {job.status}")
        return
    if width_mm > max_span_mm:
        job.status = f"estimated width {width_mm:.0f} mm exceeds the gripper's ~{max_span_mm:.0f} mm span; using grab()"
        print(f"[grab] {job.status}")
        await gripper.grab()
        return
    if width_mm > 0:
        target = gripper_position_for_width(width_mm)
        job.status = f"object ~{width_mm:.0f} mm wide, closing gripper to position {target}"
        print(f"[grab] {job.status}")
        attempts = (
            (gripper, "gripper", {"set": target}),
            (arm, "arm", {"setup_gripper": True, "move_gripper": target}),
        )
        for who, label, cmd in attempts:
            try:
                await who.do_command(cmd)
                await asyncio.sleep(0.3)
                return
            except Exception as e:
                print(f"[grab] {cmd} on {label} not accepted ({type(e).__name__}: {e})")
        print("[grab] width-based close unavailable; using grab()")
    else:
        job.status = "no size estimate, using grab()"
        print(f"[grab] {job.status}")
    await gripper.grab()
    await asyncio.sleep(0.3)


async def execute_grasp(motion: MotionClient, gripper: Gripper, arm: Arm, plan: GraspPlan,
                        obstacles: Optional[WorldState], job: GraspJob) -> bool:
    """Approach above (wrist held) -> open -> straight down -> close -> straight
    up, then hover there holding it: what happens next is the user's choice
    (delivery.py). A missed grasp opens and goes home. In a dry run every step
    is printed, nothing moves, and it continues as if holding."""
    # Long travel is a free motion plan; only the short vertical legs are linear.
    if not await move_to(motion, plan.approach, "free standoff", job, obstacles):
        return False
    await open_gripper(gripper, job)
    if not await move_to(motion, plan.grasp, "grasp (straight down)", job, obstacles, straight_line()):
        await move_to(motion, plan.approach, "back up", job, obstacles, straight_line())
        return False
    await close_gripper(gripper, arm, plan.width_mm, job)

    if not DRY_RUN:
        # is_holding_something() returns a HoldingStatus (truthy even when False).
        status = await gripper.is_holding_something()
        job.holding = bool(getattr(status, "is_holding_something", status))
        print(f"[grab] holding_something={job.holding} "
              f"(gripper position={getattr(status, 'meta', {}).get('position', '?')})")

    await move_to(motion, plan.approach, "lift (straight up)", job, obstacles, straight_line())

    if not DRY_RUN and not job.holding:
        await gripper.open()
        if plan.return_pose is not None:
            await move_to(motion, plan.return_pose, "home", job, obstacles, upright())
        job.status = "missed: nothing in the gripper. Look at the object again to retry"
        print(f"[grab] {job.status}")
        return False

    job.status = "DRY RUN: holding (pretend)" if DRY_RUN else "holding it"
    print(f"[grab] {job.status}")
    return True


async def put_back(motion: MotionClient, gripper: Gripper, held: Held,
                   obstacles: Optional[WorldState], job: GraspJob) -> bool:
    """Set the held object down where it was picked up: above -> straight down
    -> open -> straight up. The gripper opens only once the descent succeeded,
    so a failed plan never drops it. True once it's released."""
    g = held.pick_grasp
    above = Pose(x=g.x, y=g.y, z=held.pick_approach.z, **held.orientation)
    place = Pose(x=g.x, y=g.y, z=held.place_z, **held.orientation)
    if not await move_to(motion, above, f"above where the {held.label} was", job, obstacles, upright()):
        return False
    if not await move_to(motion, place, f"put the {held.label} down", job, obstacles, straight_line()):
        await move_to(motion, above, "back up", job, obstacles, straight_line())
        return False
    await open_gripper(gripper, job)
    job.held = None                       # released: from here on the gripper is empty
    await move_to(motion, above, "lift (straight up)", job, obstacles, straight_line())
    return True


async def run_put_back(motion: MotionClient, gripper: Gripper, held: Held, job: GraspJob) -> bool:
    """P in demo mode: put the held object back and return to the observe pose."""
    job.held = held
    try:
        if not await put_back(motion, gripper, held, held.obstacles, job):
            job.status = f"couldn't put the {held.label} back; still holding it"
            print(f"[grab] {job.status}")
            return False
        if held.home is not None:
            await move_to(motion, held.home, "the observe pose", job, held.obstacles, upright())
        job.ok = True
        job.status = f"put the {held.label} back"
        print(f"[grab] {job.status}")
        return True
    except asyncio.CancelledError:
        job.status = "cancelled"
        raise
    except Exception as e:
        job.status = f"error: {e}"
        print(f"[grab] {job.status}")
        return False
    finally:
        job.finished_at = time.monotonic()


async def gripper_world_pose(motion: MotionClient) -> Optional[Pose]:
    try:
        return (await motion.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)).pose
    except Exception as e:
        print(f"[grab] couldn't read the gripper pose ({type(e).__name__}: {e})")
        return None


async def measure_tcp_mm(motion: MotionClient) -> float:
    """Flange-to-gripper-frame distance from the live frame system."""
    try:
        g = (await motion.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)).pose
        a = (await motion.get_pose(ARM_NAME, MOTION_REFERENCE_FRAME)).pose
        d = math.dist((g.x, g.y, g.z), (a.x, a.y, a.z))
        if 20.0 < d < 400.0:
            return d
        print(f"[main] measured gripper TCP offset {d:.0f} mm looks wrong; using {GRIPPER_TCP_FROM_FLANGE_MM:.0f}")
    except Exception as e:
        print(f"[main] couldn't measure the gripper TCP offset ({type(e).__name__}: {e}); "
              f"using {GRIPPER_TCP_FROM_FLANGE_MM:.0f} mm")
    return GRIPPER_TCP_FROM_FLANGE_MM


async def run_grasp(robot: RobotClient, intr: Optional[Intrinsics], frame_w: int, frame_h: int,
                    motion: MotionClient, gripper: Gripper, arm: Arm,
                    box: Box, job: GraspJob, home: Optional[Pose], tcp_mm: float,
                    held: Optional[Held] = None, carry_back: bool = False) -> bool:
    """Locate the locked object and pick it. With `held` (demo mode, the
    gripper already has something), put that back where it came from first.
    carry_back: return to the observe pose holding it (demo mode) instead of
    hovering above the pick spot for the menu."""
    job.held = held
    try:
        if intr is None:
            raise RuntimeError("camera intrinsics unavailable; no arm motion sent")
        # The locked frame and box must be used before moving the wrist camera.
        raw_x, raw_y, pixel = await bottom_center_on_table(robot, box, intr, frame_w, frame_h)
        x, y = apply_radial_correction(raw_x, raw_y)
        radius = math.hypot(x, y)
        print(f"[pick] selected {box.label!r}; bottom-center pixel=({pixel[0]:.1f}, {pixel[1]:.1f})")
        print(f"[pick] raw table-plane x={raw_x:.1f} y={raw_y:.1f} mm; applying "
              f"{RADIAL_OUTWARD_CORRECTION_MM:.1f} mm outward and "
              f"{LATERAL_X_CORRECTION_MM:.1f} mm X -> x={x:.1f} y={y:.1f} mm; "
              f"radius={radius:.1f} mm")
        if not MIN_TARGET_RADIUS_MM <= radius <= MAX_TARGET_RADIUS_MM:
            raise RuntimeError(f"target radius {radius:.1f} mm is outside the safe "
                               f"{MIN_TARGET_RADIUS_MM:.0f}-{MAX_TARGET_RADIUS_MM:.0f} mm workspace")
        dimensions = dimensions_for(box.label)
        plan = plan_grasp(x, y, dimensions, home)
        print(f"[pick] manual dimensions={dimensions}; desired fingertip z="
              f"{plan.grasp.z - GRIPPER_ORIGIN_TO_FINGERTIPS_MM:.1f} mm; "
              f"gripper-origin grasp z={plan.grasp.z:.1f} mm; standoff z={plan.approach.z:.1f} mm")
        obstacles = None
        footprint = None
        if held is not None:
            # Swap: put the held object back first, avoiding everything on the
            # table (the new target included); afterwards its spot is occupied
            # again, so the new grasp avoids it too.
            print(f"[grab] swap: putting the {held.label} back before picking {box.label}")
            if not await put_back(motion, gripper, held, None, job):
                job.status = f"couldn't put the {held.label} back; still holding it, nothing else picked"
                print(f"[grab] {job.status}")
                return False
        try:
            if not await execute_grasp(motion, gripper, arm, plan, None, job):
                return False
        except Exception as e:
            if "self-collision" not in str(e).casefold():
                raise
            print("[pick] top-down approach self-collided; retrying once with a 15-degree tilt")
            tilted = dict(o_x=0.0, o_y=1.0, o_z=0.0, theta=15.0)
            plan = GraspPlan(tilted,
                             Pose(x=x, y=y, z=plan.approach.z, **tilted),
                             Pose(x=x, y=y, z=plan.grasp.z, **tilted),
                             plan.return_pose, plan.width_mm, 0.0)
            if not await execute_grasp(motion, gripper, arm, plan, None, job):
                return False
        # The object was resting on the table, so setting it down anywhere on
        # the table means putting the TCP back at the grasp height.
        job.held = Held(label=box.label, pick_grasp=plan.grasp,
                        pick_approach=plan.approach, orientation=plan.orientation, obstacles=obstacles,
                        place_z=plan.grasp.z + PLACE_CLEARANCE_MM, width_mm=plan.width_mm,
                        pose=plan.approach, home=plan.return_pose, footprint=None)
        if carry_back and plan.return_pose is not None:
            if await move_to(motion, plan.return_pose, "the observe pose (holding it)", job, obstacles, upright()):
                job.held.pose = plan.return_pose
            else:
                job.status = f"holding the {job.held.label}, but couldn't get back to the observe pose"
                print(f"[grab] {job.status}")
                return False
            job.status = (f"DRY RUN: holding the {job.held.label} (pretend)" if DRY_RUN
                          else f"holding the {job.held.label}")
        job.ok = True
        return True
    except asyncio.CancelledError:
        job.status = "cancelled"
        raise
    except Exception as e:
        job.status = f"error: {e}"
        print(f"[grab] {job.status}")
        return False
    finally:
        job.finished_at = time.monotonic()


def put_text(canvas, text: str, org: tuple[int, int], scale: float = 0.8,
             color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Text with a dark outline, readable on any camera image."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def report_task_exception(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        print(f"[grab] grasp task crashed: {task.exception()!r}")


async def stop_everything(arm: Arm, job: Optional[GraspJob]) -> None:
    """Cancel the grasp first (so it sends nothing new), then stop the arm."""
    if job is not None and job.task is not None and not job.task.done():
        job.task.cancel()
    if EXECUTE:
        print("[main] STOP: arm.stop() sent. The physical E-stop is the real stop.")
        try:
            await asyncio.wait_for(arm.stop(), timeout=2.0)
        except Exception as e:
            print(f"[main] arm.stop() failed ({type(e).__name__}: {e}): USE THE E-STOP")
    if job is not None and job.task is not None:
        await asyncio.gather(job.task, return_exceptions=True)


# --------------------------------------------------------------------------- setup

POSE_FIELDS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


def load_home_pose() -> Optional[Pose]:
    if not HOME_POSE_PATH.exists():
        return None
    return Pose(**json.loads(HOME_POSE_PATH.read_text()))


def save_home_pose(pose: Pose) -> None:
    HOME_POSE_PATH.write_text(json.dumps({k: getattr(pose, k) for k in POSE_FIELDS}, indent=2))


def load_serve_pose() -> Optional[Pose]:
    if not SERVE_POSE_PATH.exists():
        return None
    return Pose(**json.loads(SERVE_POSE_PATH.read_text()))


def serve_problem(pose: Pose) -> Optional[str]:
    if pose.z < TABLE_TOP_Z_MM + 60:
        return f"it's only {pose.z - TABLE_TOP_Z_MM:.0f} mm above the table"
    reach = math.hypot(pose.x, pose.y)
    if reach > ARM_REACH_MM or reach < 200:
        return f"it's {reach:.0f} mm from the arm base (needs 200-{ARM_REACH_MM:.0f})"
    return None


def make_delivery_io(robot: RobotClient, motion: MotionClient, gripper: Gripper, arm: Arm,
                     intr: Optional[Intrinsics], frame_w: int, frame_h: int) -> DeliveryIO:
    async def stop_arm():
        try:
            await asyncio.wait_for(arm.stop(), timeout=2.0)
        except Exception as e:
            print(f"[main] arm.stop() failed ({type(e).__name__}: {e}): USE THE E-STOP")

    return DeliveryIO(
        move=lambda pose, label, status, ws=None, c=None: move_to(motion, pose, label, status, ws, c),
        upright=upright, straight_line=straight_line,
        open_gripper=lambda status: open_gripper(gripper, status),
        stop_arm=stop_arm,
        read_pose=lambda: gripper_world_pose(motion),
        transform=lambda pose, src, dst: pose_in(robot, pose, src, dst),
        dry_run=DRY_RUN, reach_mm=ARM_REACH_MM, table_top_z=TABLE_TOP_Z_MM,
        camera_name=CAMERA_NAME, world_frame=MOTION_REFERENCE_FRAME,
        intrinsics=intr, frame_w=frame_w, frame_h=frame_h,
    )


def resolve_motion_name(machine) -> str:
    have = {r.name for r in machine.resource_names}
    missing = [n for n in (CAMERA_NAME, DETECTOR_NAME, GRIPPER_NAME, ARM_NAME) if n not in have]
    if missing:
        raise SystemExit(f"Not on this machine: {missing}. Resources found: {sorted(have)}. "
                         "Fix the names at the top of main.py.")
    for name in (MOTION_SERVICE_NAME, "builtin"):
        if name in have:
            return name
    raise SystemExit(f"No motion service named '{MOTION_SERVICE_NAME}' or 'builtin'. "
                     f"Resources found: {sorted(have)}")


def load_or_calibrate(gaze: WebcamGazeTracker, frame_w: int, frame_h: int) -> GazeCalibration:
    """Run only the fixed-head, full-face nine-point calibration."""
    if SKIP_CALIBRATION:
        calib = GazeCalibration.load()
        if calib is not None and (calib.frame_w, calib.frame_h) == (frame_w, frame_h):
            print(f"[main] --skip-calibration: reusing {CALIBRATION_PATH}")
            return calib
        print("[main] no compatible saved calibration (missing, older format, or other frame size); recalibrating")
    return run_calibration(
        gaze, frame_w, frame_h, window_name=WINDOW, keep_window=True,
        quick=True, simple_nine_point=True, full_face=True, relaxed_framing=True,
    )


async def main():
    machine = await connect()
    motion_name = resolve_motion_name(machine)
    print(f"[main] motion service '{motion_name}'. " + (
        "EXECUTE: the arm WILL move. Keep a hand on the E-stop." if EXECUTE
        else "DRY RUN (default): nothing will move. Pass --execute to allow motion."))
    cam = Camera.from_robot(machine, CAMERA_NAME)
    detector = VisionClient.from_robot(machine, DETECTOR_NAME)
    motion = MotionClient.from_robot(machine, motion_name)
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    arm = Arm.from_robot(machine, ARM_NAME)

    if SET_HOME:
        pose = (await motion.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)).pose
        save_home_pose(pose)
        print(f"[main] home pose saved to {HOME_POSE_PATH}: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f} "
              f"o=({pose.o_x:.2f}, {pose.o_y:.2f}, {pose.o_z:.2f}) theta={pose.theta:.0f}")
        await machine.close()
        return

    if SET_SERVE:
        pose = (await motion.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)).pose
        problem = serve_problem(pose)
        if problem:
            await machine.close()
            raise SystemExit(f"Not saving that serve pose: {problem}. Move the gripper to where the user "
                             "should receive objects, over the table, and run --set-serve again.")
        SERVE_POSE_PATH.write_text(json.dumps({k: getattr(pose, k) for k in POSE_FIELDS}, indent=2))
        print(f"[main] serve pose saved to {SERVE_POSE_PATH}: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f}. "
              "'Bring to me', Closer/Away and head steering use it; nothing goes past it toward the user.")
        await machine.close()
        return

    serve = load_serve_pose()
    if serve is None:
        print("[main] no serve_pose.json: 'Bring to me', Closer/Away and head steering are off until you run "
              "--set-serve with the gripper where the user should receive objects")
    elif serve_problem(serve):
        print(f"[main] ignoring serve_pose.json: {serve_problem(serve)}")
        serve = None

    tcp_mm = await measure_tcp_mm(motion)
    intr = await get_intrinsics(cam)
    print(f"[main] gripper TCP {tcp_mm:.0f} mm from the flange -> fingertips "
          f"{max(0.0, FINGERTIP_FROM_FLANGE_MM - tcp_mm):.0f} mm beyond it; camera intrinsics "
          + (f"fx={intr.fx:.0f} fy={intr.fy:.0f} at {intr.width}x{intr.height}" if intr else "UNAVAILABLE"))

    home = load_home_pose()
    if home is None:
        print("[main] no home_pose.json; objects are carried back to where the gripper was when locked")
    elif GO_HOME_ON_START and EXECUTE:
        await move_to(motion, home, "home", GraspJob(), constraints=upright())

    gaze = WebcamGazeTracker(camera_index=WEBCAM_INDEX)
    lock = GazeLockController()
    feed = RobotFeed(cam, detector)
    job: Optional[GraspJob] = None
    session: Optional[DeliverySession] = None
    held: Optional[Held] = None     # demo mode: what the gripper has between picks
    live_blocked_until = 0.0
    hovered, progress = None, 0.0

    try:
        frame_w, frame_h = await feed.first()
        # Calibrate in the same AUTOSIZE window the live loop uses. Background
        # capture starts afterwards: calibration blocks the event loop.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        calib = load_or_calibrate(gaze, frame_w, frame_h)
        head_range: Optional[HeadRange] = None
        if USER_PROFILE == "head":
            head_range = HeadRange.load() if SKIP_CALIBRATION else None
            if head_range is None:
                head_range = calibrate_head_range(gaze, frame_w, frame_h, WINDOW)
            else:
                print("[main] --skip-calibration: reusing head_range.json")
        estimator = GazeEstimator(gaze, calib)
        delivery_io = make_delivery_io(machine, motion, gripper, arm, intr, frame_w, frame_h)
        print(f"[main] {'menu' if MENU else 'demo (pick, then look at another object to swap)'} mode, "
              f"user profile '{USER_PROFILE}'"
              + (f", serving at ({serve.x:.0f}, {serve.y:.0f}, {serve.z:.0f})" if serve else ""))
        feed.start()

        while True:
            if session is not None:
                # HOLDING: the arm has the object; the menu (or steering) decides what happens.
                feed.paused = not session.wants_feed
                gaze_pt, ear, blinking = await asyncio.to_thread(estimator.read)
                canvas = await session.tick(gaze_pt, blinking, estimator.last_head, feed.obs)
                cv2.putText(canvas, f"{'EXECUTE' if EXECUTE else 'DRY RUN'} | Q stop+quit", (16, frame_h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if EXECUTE else (200, 200, 200), 1,
                            cv2.LINE_AA)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    await session.stop("Quit")
                    break
                if session.finished and not session.busy:
                    print(f"[deliver] {session.message}; back to choosing objects")
                    session = None
                    lock.release()
                    estimator.reset()
                    live_blocked_until = time.monotonic() + LIVE_COOLDOWN_S
                continue

            if job is not None and not lock.is_locked:
                # P (operator): putting the held object back; no gaze lock involved.
                feed.paused = True
                view = (feed.obs.frame * 0.5).astype(np.uint8)
                for i, line in enumerate([job.status, "Q stop+quit"]):
                    put_text(view, line, (16, 40 + 36 * i), 0.8)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    await stop_everything(arm, job)
                    break
                if job.done and (job.ok or time.monotonic() - job.finished_at > RELEASE_AFTER_SECONDS):
                    held = job.held
                    job = None
                    estimator.reset()
                    live_blocked_until = time.monotonic() + LIVE_COOLDOWN_S
                await asyncio.sleep(0.03)
                continue

            if lock.is_locked:
                feed.paused = True   # don't compete with segmentation/planning for the machine's CPU
                lines = [job.status] if job else []
                if job is not None and job.task is None:
                    lines.append("G confirm pick  |  R cancel selection  |  Q stop+quit")
                else:
                    lines.append("Q stop+quit" + ("  |  R release" if job and job.done else "  |  grasp in progress..."))
                cv2.imshow(WINDOW, draw_locked(lock.locked, lines))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    await stop_everything(arm, job)
                    break
                if job is not None and job.task is None:
                    if key == ord("r"):
                        lock.release()
                        job = None
                        estimator.reset()
                        feed.paused = False
                        continue
                    if key == ord("g"):
                        job.status = f"locating {lock.locked.box.label} from its locked 2D box"
                        job.task = asyncio.create_task(run_grasp(
                            machine, intr, frame_w, frame_h, motion, gripper, arm,
                            lock.locked.box, job, home, tcp_mm, held=held, carry_back=not MENU))
                        job.task.add_done_callback(report_task_exception)
                    await asyncio.sleep(0.03)
                    continue
                if job is not None and job.done and not MENU:
                    held = job.held   # what the gripper has now (a swap may have put one down)
                if job is not None and job.done and job.held is not None and MENU:
                    session = DeliverySession(delivery_io, job.held, serve,
                                              head_mode=USER_PROFILE == "head", head_range=head_range)
                    print(f"[deliver] holding the {job.held.label}: showing the menu")
                    job = None
                    continue
                if job is not None and job.done:
                    # Demo mode: straight back to choosing after a success;
                    # after a failure, leave the reason up for a few seconds.
                    if ((job.ok and not MENU) or key == ord("r")
                            or time.monotonic() - job.finished_at > RELEASE_AFTER_SECONDS):
                        lock.release()
                        estimator.reset()
                        job = None
                        live_blocked_until = time.monotonic() + LIVE_COOLDOWN_S
                await asyncio.sleep(0.03)
                continue
            feed.paused = False

            obs = feed.obs
            frame, boxes = obs.frame, obs.boxes
            gaze_pt, ear, blinking = await asyncio.to_thread(estimator.read)
            locked = None
            cooling = time.monotonic() < live_blocked_until
            if blinking or cooling:
                lock.hold()   # a blink (or the cooldown) neither adds nor removes attention
            else:
                hovered, progress, locked = lock.update(frame, boxes, gaze_pt)
            if locked is not None:
                feed.paused = True
                print(f"[lock] locked onto '{locked.box.label}' at ({locked.box.x0},{locked.box.y0})-"
                      f"({locked.box.x1},{locked.box.y1})")
                job = GraspJob()
                job.status = f"selected {locked.box.label}; press G to confirm the pick"
                continue

            view = draw_live(frame, boxes, hovered, progress, gaze_pt)
            if gaze_pt is None:
                cv2.putText(view, "No face detected", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 80, 255), 2, cv2.LINE_AA)
            elif blinking:
                cv2.putText(view, "BLINK", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
            elif cooling:
                cv2.putText(view, f"Ready in {live_blocked_until - time.monotonic():.1f}s", (16, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            if held is not None:
                put_text(view, f"Holding: {held.label}{' (pretend)' if DRY_RUN else ''}  -  look at another "
                               "object to swap   |   P: put it back", (16, 66), 0.7, (0, 230, 0))
            hud = (f"{'EXECUTE' if EXECUTE else 'DRY RUN'} | "
                   f"{'paired' if feed.paired else 'UNPAIRED'} capture {feed.fps:.1f}/s "
                   f"({feed.capture_ms:.0f} ms) | {len(boxes)} boxes")
            cv2.putText(view, hud, (16, frame_h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255) if EXECUTE else (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p") and held is not None and not MENU:
                lock.release()
                job = GraspJob()
                job.status = f"putting the {held.label} back"
                job.task = asyncio.create_task(run_put_back(motion, gripper, held, job))
                job.task.add_done_callback(report_task_exception)
                continue
            if key == ord("c"):
                feed.paused = True
                calib = run_calibration(
                    gaze, frame_w, frame_h, window_name=WINDOW, keep_window=True,
                    quick=True, simple_nine_point=True, full_face=True, relaxed_framing=True,
                )
                estimator = GazeEstimator(gaze, calib)
    finally:
        await feed.stop()
        if job is not None and job.task is not None and not job.task.done():
            await stop_everything(arm, job)
        if session is not None and session.busy:
            await session.stop("Quit")
        if held is not None and not DRY_RUN:
            print(f"[main] note: the gripper is still holding the {held.label}")
        gaze.close()
        cv2.destroyAllWindows()
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
