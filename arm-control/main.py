"""Gaze-selected pick with a Viam arm.

Laptop webcam -> gaze point on the RealSense feed window -> soft-targeted dwell
on a YOLO box locks that object (frozen snapshot of the exact image the boxes
came from) -> while the arm is still: two agreeing 3D segmentations, the object whose
center projects into the locked box is chosen, and it and every other object
are frozen into world coordinates -> sanity checks -> the motion service moves
the gripper above it, opens, descends in a straight line, closes, lifts, and
carries it back level.

SAFETY: the arm moves ONLY with --execute. Without it everything runs (camera,
YOLO, gaze, lock, 3D segmentation, transforms, checks) and the poses are
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
the last calibration), --quick-calibration (straight-head stage only), --detector NAME,
--set-home, --set-serve (record where objects are brought to the user).
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

import cv2
import numpy as np
from grpclib import GRPCError, Status
from grpclib.exceptions import StreamTerminatedError

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

HERE = Path(__file__).resolve().parent
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
# Any vision service works here; they all return boxes + labels. objects-3d's
# own `detector_name` attribute (Viam app config) must point at the same one,
# or its 3D objects won't correspond to the boxes on screen.
DETECTOR_CANDIDATES = ("yolo-detector", "shape-detector")   # viam-labs:yolov8, devrel:shape-finder
DETECTOR_NAME = DETECTOR_CANDIDATES[0]
if "--detector" in sys.argv:                               # python main.py --detector shape-detector
    DETECTOR_NAME = sys.argv[sys.argv.index("--detector") + 1]
SEGMENTER_NAME = "objects-3d"        # viam:vision:detections-to-segments
MOTION_SERVICE_NAME = "motion"       # falls back to the SDK default name "builtin"
GRIPPER_NAME = "gripper"
ARM_NAME = "arm"
SEGMENTER_TIMEOUT_S = 60.0           # 3D segmentation took ~4 s on this machine
CAPTURE_TIMEOUT_S = 30.0
RECONNECT_DELAY_S = 2.0

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
# The table obstacle is a 200 mm box centered at world z = -123: top at -23.
TABLE_TOP_Z_MM = -23.0
FINGERTIP_TABLE_CLEARANCE_MM = 10.0
ARM_REACH_MM = 700.0                 # xArm6
APPROACH_HEIGHT_MM = 100.0           # standoff above the grasp pose
GRASP_Z_OFFSET_MM = 0.0              # + raises the grasp (fingertips land at object center + this)

# Orientation: grasp with the wrist orientation the arm already has at the
# observe pose when it points roughly down, so the planner never has to spin
# the wrist. (The old fixed target, theta=0 vs the arm's 142.7, asked for a
# ~143 degree wrist rotation on every grasp.) This is the fallback.
DEFAULT_GRASP_ORIENTATION = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=142.7)
MIN_DOWNWARD_O_Z = -0.9              # current orientation reused only if o_z <= this
ORIENTATION_TOLERANCE_DEGS = 15.0    # held along approach and carry moves
LINE_TOLERANCE_MM = 10.0             # descent / lift stay on a straight line

# Width-based close (so a paper cup isn't crushed): the uFactory gripper takes a
# position on a 0-850 scale (850 = fully open, ~85 mm), i.e. ~10 units per mm.
GRIPPER_SQUEEZE_MM = 8.0
GRIPPER_POS_PER_MM = 10.0
GRIPPER_MAX_POS = 850
# Close fully and let the object stop the jaws. This is the merged behavior
# from the contributor's "gripper close fully" change.
GRIPPER_CLOSE_FULLY = True
GRIPPER_FULL_CLOSE_POS = 0

# Home / serve pose: record once with the arm parked there
# (python main.py --set-home -> home_pose.json). Without it, the object is
# carried back to where the gripper was when the object was locked.
HOME_POSE_PATH = HERE / "home_pose.json"
GO_HOME_ON_START = True
RETURN_TO_START = True

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
QUICK_CALIBRATION = "--quick-calibration" in sys.argv
SET_HOME = "--set-home" in sys.argv
SET_SERVE = "--set-serve" in sys.argv     # record where objects are brought to the user, and exit

# --user eyes (default): the gaze menu after a pick. --user head: the same menu
# plus head-motion steering, with a per-user head-range calibration at startup.
USER_PROFILE = sys.argv[sys.argv.index("--user") + 1] if "--user" in sys.argv else "eyes"
if USER_PROFILE not in ("eyes", "head"):
    raise SystemExit("--user must be 'eyes' or 'head'")
# Default: the two-action demo (pick; look at another object to swap). --menu:
# the post-pick gaze menu. Head steering lives in the menu, so --user head implies it.
MENU = "--menu" in sys.argv or USER_PROFILE == "head"
SERVE_POSE_PATH = HERE / "serve_pose.json"
LIVE_COOLDOWN_S = 2.5          # after putting something down, don't immediately select again
MAX_OBSERVATION_AGE_S = 3.0    # never select from an old camera/detection snapshot
# A segmented object whose center is this close to the gripper is the one in
# the gripper (the wrist camera can see it): never a target, never an obstacle.
IN_GRIPPER_RADIUS_MM = 120.0


async def connect():
    env = load_env()
    # SDK 0.80.0 hard-codes a 1 s ResourceNames health-check deadline. Slow
    # camera/vision calls can make that check tear down a working connection.
    # Both intervals must be zero: the SDK uses reconnect_interval as the
    # probe interval when check_connection_interval is zero. The application
    # rebuilds the client and all resource handles after a transport failure.
    opts = RobotClient.Options.with_api_key(
        api_key=env["VIAM_API_KEY"], api_key_id=env["VIAM_API_KEY_ID"],
        check_connection_interval=0, attempt_reconnect_interval=0,
    )
    return await RobotClient.at_address(env["VIAM_MACHINE_ADDRESS"], opts)


class ReconnectRequired(ConnectionError):
    """A failed transport requires a new client and new resource handles."""


def is_transport_error(error: Exception) -> bool:
    if isinstance(error, (ConnectionError, StreamTerminatedError)):
        return True
    if isinstance(error, GRPCError):
        if error.status == Status.UNAVAILABLE:
            return True
        # The Rust WebRTC bridge sometimes reports a closed channel as UNKNOWN.
        if error.status == Status.UNKNOWN:
            message = (error.message or "").lower()
            return not message or any(s in message for s in (
                "channel closed", "connection lost", "datachannel", "stream closed"))
    # A slow request alone is not evidence that the transport has closed.
    return False


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
        self.connection_error: Optional[Exception] = None
        self._task: Optional[asyncio.Task] = None

    def _boxes(self, detections) -> list[Box]:
        return filter_background_boxes(
            [box_from_detection(d, i, self.frame_w, self.frame_h) for i, d in enumerate(detections)])

    async def _capture_paired(self) -> Observation:
        res = await self.detector.capture_all_from_camera(
            CAMERA_NAME, return_image=True, return_detections=True, timeout=CAPTURE_TIMEOUT_S)
        frame = decode_viam_image(res.image)
        if frame is None:
            raise RuntimeError("combined capture returned no decodable image")
        if not self.frame_w:
            self.frame_h, self.frame_w = frame.shape[:2]
        return Observation(frame, self._boxes(res.detections or []), time.monotonic(), True)

    async def _capture_unpaired(self) -> Observation:
        try:
            images, _ = await self.cam.get_images(filter_source_names=[COLOR_SOURCE_NAME], timeout=CAPTURE_TIMEOUT_S)
        except Exception as error:
            if is_transport_error(error):
                raise
            images = []
        if not images:
            images, _ = await self.cam.get_images(timeout=CAPTURE_TIMEOUT_S)
        frame = decode_color_frame(images)
        if frame is None:
            raise RuntimeError(f"no color frame from camera '{CAMERA_NAME}'")
        if not self.frame_w:
            self.frame_h, self.frame_w = frame.shape[:2]
        detections = await self.detector.get_detections_from_camera(CAMERA_NAME, timeout=CAPTURE_TIMEOUT_S)
        return Observation(frame, self._boxes(detections), time.monotonic(), False)

    async def capture(self) -> Observation:
        if self.paired:
            try:
                return await self._capture_paired()
            except Exception as paired_error:
                if is_transport_error(paired_error):
                    raise
                # A closed connection makes both capture methods fail. Do not
                # mistake that for an unsupported combined capture forever.
                try:
                    obs = await self._capture_unpaired()
                except Exception as fallback_error:
                    if is_transport_error(fallback_error):
                        raise
                    raise paired_error
                self.paired = False
                print(f"[feed] '{DETECTOR_NAME}' can't do a combined capture "
                      f"({type(paired_error).__name__}: {paired_error}); falling back to separate "
                      "image + detection calls, which can come from different frames")
                return obs
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
                self.obs = None  # Never let gaze lock a box from the last good frame.
                print(f"[feed] capture error: {e}")
                if is_transport_error(e):
                    self.connection_error = e
                    return
                await asyncio.sleep(0.5)


# --------------------------------------------------------------------------- 3D target

def object_center_and_size(point_cloud_obj):
    geoms = point_cloud_obj.geometries.geometries
    if not geoms:
        return None
    g = geoms[0]
    kind = g.WhichOneof("geometry_type")
    if kind == "box":
        d = g.box.dims_mm
        size = (d.x, d.y, d.z)
    elif kind == "sphere":
        r = g.sphere.radius_mm
        size = (2 * r, 2 * r, 2 * r)
    else:
        size = (0.0, 0.0, 0.0)
    return g.center, size


def object_label(point_cloud_obj) -> str:
    for geom in point_cloud_obj.geometries.geometries:
        if geom.label:
            return geom.label
    return ""


def object_reference_frame(point_cloud_obj, default: str = CAMERA_NAME) -> str:
    """The segmenter says which frame its geometries are in ('cam' on this
    machine). Trust it: labeling camera-frame numbers as world sends the arm
    somewhere wrong."""
    return point_cloud_obj.geometries.reference_frame or default


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
        ip = (await cam.get_properties(timeout=CAPTURE_TIMEOUT_S)).intrinsic_parameters
        if ip.focal_x_px > 0 and ip.width_px > 0:
            return Intrinsics(ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px,
                              ip.width_px, ip.height_px)
    except Exception as e:
        if is_transport_error(e):
            raise
        print(f"[main] camera intrinsics unavailable ({type(e).__name__}: {e})")
    return None


def project_to_pixel(pt, intr: Intrinsics, frame_w: int, frame_h: int) -> Optional[tuple[float, float]]:
    """Camera-frame point (x right, y down, z forward, mm) -> display pixel."""
    if pt.z <= 1e-6:
        return None
    sx, sy = frame_w / intr.width, frame_h / intr.height
    return (intr.fx * pt.x / pt.z + intr.cx) * sx, (intr.fy * pt.y / pt.z + intr.cy) * sy


async def pose_in(robot: RobotClient, pose: Pose, ref: str, dest: str) -> Pose:
    if ref == dest:
        return pose
    return (await robot.transform_pose(PoseInFrame(reference_frame=ref, pose=pose), dest)).pose


async def match_object_to_box(robot: RobotClient, objs, box: Box, intr: Optional[Intrinsics],
                              frame_w: int, frame_h: int):
    """The 3D object the user locked: the one whose center projects into the
    locked box (same label preferred, then closest to the box center). Uses
    geometry, not list positions, so ordering and missing detections can't
    swap objects. None if nothing lands there (e.g. the object was moved)."""
    if intr is None:
        same = [o for o in objs if object_label(o) == box.label]
        if len(same) == 1:
            print("[grab] no camera intrinsics; matched by unique label")
            return same[0]
        print(f"[grab] no camera intrinsics and {len(same)} '{box.label}' objects; can't tell them apart")
        return None
    margin = 0.15 * max(box.x1 - box.x0, box.y1 - box.y0)
    bcx, bcy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    candidates = []
    for o in objs:
        if not o.geometries.geometries:
            continue
        c = await pose_in(robot, o.geometries.geometries[0].center, object_reference_frame(o), CAMERA_NAME)
        uv = project_to_pixel(c, intr, frame_w, frame_h)
        if uv is None:
            continue
        u, v = uv
        if box.x0 - margin <= u <= box.x1 + margin and box.y0 - margin <= v <= box.y1 + margin:
            candidates.append((object_label(o) != box.label, math.hypot(u - bcx, v - bcy), id(o), o))
    return min(candidates)[3] if candidates else None


async def obstacles_in_world(robot: RobotClient, objs, exclude) -> Optional[WorldState]:
    """Every other segmented object, frozen into WORLD coordinates now, while
    the arm is still. Left in the camera frame they would ride along with the
    wrist camera once the arm moves."""
    if not AVOID_DETECTED_OBJECTS:
        return None
    geoms = []
    for o in objs:
        if o is exclude or object_label(o).lower() in IGNORE_LABELS:
            continue
        ref = object_reference_frame(o)
        for g in o.geometries.geometries:
            ng = Geometry()
            ng.CopyFrom(g)
            ng.center.CopyFrom(await pose_in(robot, g.center, ref, MOTION_REFERENCE_FRAME))
            geoms.append(ng)
    if not geoms:
        return None
    return WorldState(obstacles=[GeometriesInFrame(reference_frame=MOTION_REFERENCE_FRAME, geometries=geoms)])


def merge_world_states(*states: Optional[WorldState]) -> Optional[WorldState]:
    obstacles = [gif for s in states if s is not None for gif in s.obstacles]
    return WorldState(obstacles=obstacles) if obstacles else None


async def objects_in_gripper(robot: RobotClient, objs, gripper_pose: Optional[Pose]) -> list:
    """Segmented objects sitting at the gripper: what it's holding, seen by the wrist camera."""
    if gripper_pose is None:
        return []
    found = []
    for o in objs:
        if not o.geometries.geometries:
            continue
        c = await pose_in(robot, o.geometries.geometries[0].center, object_reference_frame(o), MOTION_REFERENCE_FRAME)
        if math.dist((c.x, c.y, c.z), (gripper_pose.x, gripper_pose.y, gripper_pose.z)) < IN_GRIPPER_RADIUS_MM:
            found.append(o)
    return found


def target_problem(w: Pose) -> Optional[str]:
    """Reason not to move to this world-frame object center, or None."""
    if not (TABLE_TOP_Z_MM - 40 <= w.z <= TABLE_TOP_Z_MM + 400):
        return (f"its height z={w.z:.0f} mm isn't on the table (table top ~{TABLE_TOP_Z_MM:.0f}); "
                "check the camera frame / segmentation")
    reach = math.hypot(w.x, w.y)
    if reach > ARM_REACH_MM:
        return f"it is {reach:.0f} mm from the arm base, beyond reach (~{ARM_REACH_MM:.0f})"
    return None


@dataclass
class GraspPlan:
    orientation: dict
    approach: Pose
    grasp: Pose
    return_pose: Optional[Pose]
    width_mm: float
    fingertip_beyond_tcp_mm: float


def grasp_orientation(current: Optional[Pose]) -> dict:
    if current is not None and current.o_z <= MIN_DOWNWARD_O_Z:
        return dict(o_x=current.o_x, o_y=current.o_y, o_z=current.o_z, theta=current.theta)
    return dict(DEFAULT_GRASP_ORIENTATION)


def plan_grasp(w: Pose, size, current: Optional[Pose], tcp_mm: float,
               home: Optional[Pose]) -> GraspPlan:
    """Poses in world. The TCP stops so the fingertips land at the object's
    center height (never closer than the clearance to the table)."""
    orient = grasp_orientation(current)
    beyond = max(0.0, FINGERTIP_FROM_FLANGE_MM - tcp_mm)
    z = w.z + beyond + GRASP_Z_OFFSET_MM
    z = max(z, TABLE_TOP_Z_MM + FINGERTIP_TABLE_CLEARANCE_MM + beyond)
    grasp = Pose(x=w.x, y=w.y, z=z, **orient)
    approach = Pose(x=w.x, y=w.y, z=z + APPROACH_HEIGHT_MM, **orient)
    back = None
    if RETURN_TO_START:
        src = home or current
        if src is not None:
            back = Pose(x=src.x, y=src.y, z=src.z, **orient)   # carried back level, same wrist
    width = min((d for d in size[:2] if d > 0), default=0.0)
    return GraspPlan(orient, approach, grasp, back, width, beyond)


# --------------------------------------------------------------------------- motion

class GraspJob:
    """Status of the background grasp so the display loop can show progress."""

    def __init__(self):
        self.status = "segmenting 3D target..."
        self.task: Optional[asyncio.Task] = None
        self.finished_at: Optional[float] = None
        self.holding: Optional[bool] = None
        self.held: Optional[Held] = None      # what the gripper holds right now (kept current during a swap)
        self.ok = False                       # the whole job succeeded
        self.motion_started = False
        self.connection_error: Optional[Exception] = None

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
    job.motion_started = True
    ok = await motion.move(
        component_name=GRIPPER_NAME,
        destination=PoseInFrame(reference_frame=MOTION_REFERENCE_FRAME, pose=pose),
        world_state=world_state,
        constraints=constraints,
    )
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
        job.motion_started = True
        await gripper.open()
        await asyncio.sleep(0.3)   # let the fingers finish opening


async def close_gripper(gripper: Gripper, arm: Arm, width_mm: float, job: GraspJob) -> None:
    """Close to the object's width ({"set": pos} on the gripper, then the arm's
    move_gripper); grab() by force feedback if that isn't accepted or the width
    estimate is useless."""
    max_span_mm = GRIPPER_MAX_POS / GRIPPER_POS_PER_MM
    if DRY_RUN:
        target = GRIPPER_FULL_CLOSE_POS if GRIPPER_CLOSE_FULLY else gripper_position_for_width(width_mm)
        job.status = (f"DRY RUN, would close the gripper to position {target} "
                      f"(object ~{width_mm:.0f} mm wide)" if 0 < width_mm <= max_span_mm
                      else "DRY RUN, would call grab()")
        print(f"[grab] {job.status}")
        return
    job.motion_started = True
    if width_mm > max_span_mm:
        job.status = f"estimated width {width_mm:.0f} mm exceeds the gripper's ~{max_span_mm:.0f} mm span; using grab()"
        print(f"[grab] {job.status}")
        await gripper.grab()
        return
    if width_mm > 0:
        target = GRIPPER_FULL_CLOSE_POS if GRIPPER_CLOSE_FULLY else gripper_position_for_width(width_mm)
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
                if (is_transport_error(e) or isinstance(e, TimeoutError)
                        or (isinstance(e, GRPCError) and e.status in (Status.DEADLINE_EXCEEDED, Status.CANCELLED))):
                    # The command may have reached the gripper before its
                    # response was lost; do not issue a different close command.
                    raise
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
    if not await move_to(motion, plan.approach, "approach", job, obstacles, upright()):
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
            await move_to(motion, held.home, "the observe pose", job,
                          merge_world_states(held.obstacles, held.footprint), upright())
        job.ok = True
        job.status = f"put the {held.label} back"
        print(f"[grab] {job.status}")
        return True
    except asyncio.CancelledError:
        job.status = "cancelled"
        raise
    except Exception as e:
        if is_transport_error(e):
            job.connection_error = e
        job.status = f"error: {e}"
        print(f"[grab] {job.status}")
        return False
    finally:
        job.finished_at = time.monotonic()


async def capture_objects(segmenter: VisionClient):
    """3D objects from one segmenter capture."""
    try:
        res = await segmenter.capture_all_from_camera(CAMERA_NAME, return_object_point_clouds=True,
                                                      timeout=SEGMENTER_TIMEOUT_S)
        return list(res.objects or [])
    except Exception as e:
        if is_transport_error(e):
            raise
        print(f"[grab] combined segmenter capture unavailable ({type(e).__name__}); using get_object_point_clouds")
        return list(await segmenter.get_object_point_clouds(CAMERA_NAME, timeout=SEGMENTER_TIMEOUT_S))


async def gripper_world_pose(motion: MotionClient) -> Optional[Pose]:
    try:
        return (await motion.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME, timeout=CAPTURE_TIMEOUT_S)).pose
    except Exception as e:
        if is_transport_error(e):
            raise
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
        if is_transport_error(e):
            raise
        print(f"[main] couldn't measure the gripper TCP offset ({type(e).__name__}: {e}); "
              f"using {GRIPPER_TCP_FROM_FLANGE_MM:.0f} mm")
    return GRIPPER_TCP_FROM_FLANGE_MM


async def run_grasp(robot: RobotClient, intr: Optional[Intrinsics], frame_w: int, frame_h: int,
                    segmenter: VisionClient, motion: MotionClient, gripper: Gripper, arm: Arm,
                    box: Box, job: GraspJob, home: Optional[Pose], tcp_mm: float,
                    held: Optional[Held] = None, carry_back: bool = False) -> bool:
    """Locate the locked object and pick it. With `held` (demo mode, the
    gripper already has something), put that back where it came from first.
    carry_back: return to the observe pose holding it (demo mode) instead of
    hovering above the pick spot for the menu."""
    job.held = held
    try:
        # Everything that depends on where the wrist camera is happens now,
        # before anything moves.
        current = await gripper_world_pose(motion)
        objs = await capture_objects(segmenter)
        in_hand = await objects_in_gripper(robot, objs, current)
        target = await match_object_to_box(robot, objs, box, intr, frame_w, frame_h)
        if target is not None and any(target is o for o in in_hand):
            job.status = f"NOT MOVING: that's the {held.label if held else 'object'} already in the gripper"
            print(f"[grab] {job.status}")
            return False
        objs = [o for o in objs if not any(o is h for h in in_hand)]
        if target is None:
            job.status = f"NOT MOVING: no 3D object where you locked '{box.label}' ({len(objs)} segmented)"
            print(f"[grab] {job.status}")
            return False
        center, size = object_center_and_size(target)
        ref = object_reference_frame(target)
        w = await pose_in(robot, Pose(x=center.x, y=center.y, z=center.z, o_z=1.0), ref, MOTION_REFERENCE_FRAME)
        print(f"[grab] target '{object_label(target)}' center ({center.x:.0f}, {center.y:.0f}, {center.z:.0f}) "
              f"in '{ref}' -> ({w.x:.0f}, {w.y:.0f}, {w.z:.0f}) in '{MOTION_REFERENCE_FRAME}', "
              f"size {size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f} mm")
        problem = target_problem(w)
        if problem:
            job.status = f"NOT MOVING: {problem}"
            print(f"[grab] {job.status}")
            return False
        obstacles = await obstacles_in_world(robot, objs, target)
        footprint = await obstacles_in_world(robot, [target], None)   # the target itself, also frozen now
        n = sum(len(g.geometries) for g in obstacles.obstacles) if obstacles else 0
        plan = plan_grasp(w, size, current, tcp_mm, home)
        print(f"[grab] {n} other object(s) as obstacles; fingertips {plan.fingertip_beyond_tcp_mm:.0f} mm "
              f"beyond the TCP; wrist o=({plan.orientation['o_x']:.2f}, {plan.orientation['o_y']:.2f}, "
              f"{plan.orientation['o_z']:.2f}) theta={plan.orientation['theta']:.1f}")
        if held is not None:
            # Swap: put the held object back first, avoiding everything on the
            # table (the new target included); afterwards its spot is occupied
            # again, so the new grasp avoids it too.
            print(f"[grab] swap: putting the {held.label} back before picking the {object_label(target) or box.label}")
            if not await put_back(motion, gripper, held, merge_world_states(obstacles, footprint), job):
                job.status = f"couldn't put the {held.label} back; still holding it, nothing else picked"
                print(f"[grab] {job.status}")
                return False
            obstacles = merge_world_states(obstacles, held.footprint)
        if not await execute_grasp(motion, gripper, arm, plan, obstacles, job):
            return False
        # The object was resting on the table, so setting it down anywhere on
        # the table means putting the TCP back at the grasp height.
        job.held = Held(label=object_label(target) or box.label, pick_grasp=plan.grasp,
                        pick_approach=plan.approach, orientation=plan.orientation, obstacles=obstacles,
                        place_z=plan.grasp.z + PLACE_CLEARANCE_MM, width_mm=plan.width_mm,
                        pose=plan.approach, home=plan.return_pose, footprint=footprint)
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
        if is_transport_error(e):
            job.connection_error = e
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
    missing = [n for n in (CAMERA_NAME, DETECTOR_NAME, SEGMENTER_NAME, GRIPPER_NAME, ARM_NAME) if n not in have]
    if missing:
        raise SystemExit(f"Not on this machine: {missing}. Resources found: {sorted(have)}. "
                         "Fix the names at the top of main.py.")
    for name in (MOTION_SERVICE_NAME, "builtin"):
        if name in have:
            return name
    raise SystemExit(f"No motion service named '{MOTION_SERVICE_NAME}' or 'builtin'. "
                     f"Resources found: {sorted(have)}")


def load_or_calibrate(gaze: WebcamGazeTracker, frame_w: int, frame_h: int,
                      recovering: bool = False) -> GazeCalibration:
    """Calibrate every run; --skip-calibration reuses the last one; C recalibrates mid-run."""
    if SKIP_CALIBRATION or recovering:
        calib = GazeCalibration.load()
        if calib is not None and (calib.frame_w, calib.frame_h) == (frame_w, frame_h):
            print(f"[main] reusing {CALIBRATION_PATH}")
            return calib
        print("[main] no compatible saved calibration (missing, older format, or other frame size); recalibrating")
    return run_calibration(gaze, frame_w, frame_h, window_name=WINDOW, keep_window=True, quick=QUICK_CALIBRATION)


async def run_session(machine: RobotClient, recovering: bool = False):
    """All resource handles belong to this connection; none survive a reconnect."""
    motion_name = resolve_motion_name(machine)
    print(f"[main] motion service '{motion_name}'. " + (
        "EXECUTE: the arm WILL move. Keep a hand on the E-stop." if EXECUTE
        else "DRY RUN (default): nothing will move. Pass --execute to allow motion."))
    cam = Camera.from_robot(machine, CAMERA_NAME)
    detector = VisionClient.from_robot(machine, DETECTOR_NAME)
    segmenter = VisionClient.from_robot(machine, SEGMENTER_NAME)
    motion = MotionClient.from_robot(machine, motion_name)
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    arm = Arm.from_robot(machine, ARM_NAME)

    if SET_HOME:
        pose = (await motion.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)).pose
        save_home_pose(pose)
        print(f"[main] home pose saved to {HOME_POSE_PATH}: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f} "
              f"o=({pose.o_x:.2f}, {pose.o_y:.2f}, {pose.o_z:.2f}) theta={pose.theta:.0f}")
        return

    if SET_SERVE:
        pose = (await motion.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)).pose
        problem = serve_problem(pose)
        if problem:
            raise SystemExit(f"Not saving that serve pose: {problem}. Move the gripper to where the user "
                             "should receive objects, over the table, and run --set-serve again.")
        SERVE_POSE_PATH.write_text(json.dumps({k: getattr(pose, k) for k in POSE_FIELDS}, indent=2))
        print(f"[main] serve pose saved to {SERVE_POSE_PATH}: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f}. "
              "'Bring to me', Closer/Away and head steering use it; nothing goes past it toward the user.")
        return

    if recovering and EXECUTE:
        status = await gripper.is_holding_something(timeout=CAPTURE_TIMEOUT_S)
        if bool(getattr(status, "is_holding_something", status)):
            raise RuntimeError("Reconnected, but the gripper reports holding an object. "
                               "Check the arm before restarting; no action was replayed.")

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
    elif GO_HOME_ON_START and EXECUTE and not recovering:
        try:
            await move_to(motion, home, "home", GraspJob(), constraints=upright())
        except Exception as error:
            # A lost response cannot tell us whether a motion command ran.
            raise RuntimeError("Startup movement failed. Check the arm before restarting; "
                               "the movement will not be retried automatically.") from error

    gaze = WebcamGazeTracker(camera_index=WEBCAM_INDEX)
    lock = GazeLockController()
    feed = RobotFeed(cam, detector)
    job: Optional[GraspJob] = None
    session: Optional[DeliverySession] = None
    held: Optional[Held] = None     # demo mode: what the gripper has between picks
    live_blocked_until = 0.0
    hovered, progress = None, 0.0
    keep_window = False

    try:
        frame_w, frame_h = await feed.first()
        # Calibrate in the same AUTOSIZE window the live loop uses. Background
        # capture starts afterwards: calibration blocks the event loop.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        calib = load_or_calibrate(gaze, frame_w, frame_h, recovering=recovering)
        head_range: Optional[HeadRange] = None
        if USER_PROFILE == "head":
            head_range = HeadRange.load() if SKIP_CALIBRATION or recovering else None
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
        if recovering:
            live_blocked_until = time.monotonic() + LIVE_COOLDOWN_S

        while True:
            if session is not None and session.error is not None and is_transport_error(session.error):
                raise ReconnectRequired("Connection lost during delivery") from session.error
            if job is not None and job.connection_error is not None:
                raise ReconnectRequired("Connection lost during grasp processing") from job.connection_error
            if feed.connection_error is not None:
                raise ReconnectRequired("Connection lost during capture") from feed.connection_error
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
                view = ((feed.obs.frame * 0.5).astype(np.uint8) if feed.obs is not None
                        else np.zeros((frame_h, frame_w, 3), dtype=np.uint8))
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
                lines.append("Q stop+quit" + ("  |  R release" if job and job.done else "  |  grasp in progress..."))
                cv2.imshow(WINDOW, draw_locked(lock.locked, lines))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    await stop_everything(arm, job)
                    break
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
            if obs is None or time.monotonic() - obs.at > MAX_OBSERVATION_AGE_S:
                lock.release()
                estimator.reset()
                canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for a fresh robot frame", (24, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 160, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "No object can be selected until a fresh frame arrives", (24, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                await asyncio.sleep(0.05)
                continue
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
                job.task = asyncio.create_task(run_grasp(
                    machine, intr, frame_w, frame_h, segmenter, motion, gripper, arm,
                    locked.box, job, home, tcp_mm, held=held, carry_back=not MENU))
                job.task.add_done_callback(report_task_exception)
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
                calib = run_calibration(gaze, frame_w, frame_h, window_name=WINDOW, keep_window=True,
                                        quick=QUICK_CALIBRATION)
                estimator = GazeEstimator(gaze, calib)
    except Exception as error:
        if is_transport_error(error) and (
                held is not None or session is not None
                or (job is not None and (job.motion_started or job.held is not None))):
            await stop_everything(arm, job)
            raise RuntimeError("Connection lost during object handling. Check the arm before restarting; "
                               "the interrupted action will not be replayed.") from error
        # Calibration refers to the window's physical screen position. Keep
        # that window in place while replacing only the connection and handles.
        keep_window = is_transport_error(error)
        if keep_window and feed.frame_w and feed.frame_h:
            canvas = np.zeros((feed.frame_h, feed.frame_w, 3), dtype=np.uint8)
            put_text(canvas, "Reconnecting to the robot; selection cleared", (24, 55), 0.8)
            cv2.imshow(WINDOW, canvas)
            cv2.waitKey(1)
        raise
    finally:
        await feed.stop()
        if job is not None and job.task is not None and not job.task.done():
            await stop_everything(arm, job)
        if session is not None and session.busy:
            try:
                await session.stop("Quit")
            except Exception as error:
                print(f"[main] delivery cleanup failed: {type(error).__name__}: {error}")
        if held is not None and not DRY_RUN:
            print(f"[main] note: the gripper is still holding the {held.label}")
        gaze.close()
        if not keep_window:
            cv2.destroyAllWindows()


async def main():
    recovering = False
    while True:
        machine = None
        try:
            machine = await connect()
            await run_session(machine, recovering=recovering)
            return
        except Exception as error:
            if not is_transport_error(error):
                cv2.destroyAllWindows()
                raise
            print(f"[connection] {type(error).__name__}: {error}. Reconnecting with fresh resource handles; "
                  "the previous gaze selection is discarded.")
            recovering = True
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            cv2.destroyAllWindows()
            raise
        finally:
            # close() also cancels SDK background tasks: finish it BEFORE
            # creating a replacement client, or it may cancel the new client.
            if machine is not None:
                await machine.close()
        await asyncio.sleep(RECONNECT_DELAY_S)


if __name__ == "__main__":
    asyncio.run(main())
