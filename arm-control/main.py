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
--go-home-on-start (opt in to moving to the saved pose at launch).
--finger-clearance-mm N (override measured fingertip-to-body space),
--max-finger-extension-mm N (longest body-to-tip extension over the jaw stroke),
--gripper-body-from-flange-mm N (fixed flange-to-body-underside distance),
--grasp-z-offset-mm N (additional upward offset).
--object-width-mm LABEL=MM (explicit measured width for this label in this run).
"""

import argparse
import asyncio
import json
import math
import os
import sys
import textwrap
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
from viam.spatialmath import Quaternion

from delivery import PLACE_CLEARANCE_MM, DeliveryIO, DeliverySession, Held, user_axes
from gaze_lock import IGNORE_LABELS, Box, GazeLockController, draw_live, draw_locked, filter_background_boxes
from head_control import HeadRange, calibrate_head_range
from grasp_checks import load_grasp_model, check_model_clearance
from action_recovery import OperatorFault, check_action_recovery, is_session_expired
from joint_checks import JointRangeError, require_arm_joint_ranges
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
DETECTOR_CANDIDATES = ("vision-1", "yolo-detector", "shape-detector")   # mlmodel, viam-labs:yolov8, devrel:shape-finder
# vision-1 (the team's trained model) is what objects-3d's detector_name points at on this machine.
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
# The configured TCP sits 150 mm from the flange. The housing is rigid, but
# the hinged fingers change extension as they close. Calibrate the fixed body
# datum and the full finger-extension range; do not assume a 165 mm tool tip.
GRIPPER_TCP_FROM_FLANGE_MM = 150.0
# User measured the fixed mounting-face -> housing-underside distance as
# approximately 3.85 inches (97.79 mm). This is NOT a fingertip measurement.
MEASURED_GRIPPER_BODY_FROM_FLANGE_MM = 97.8
# Housing underside -> finger ends: 2.3 inches fully open, 2.75 closed.
# Round the minimum down and maximum up to 0.1 mm for the planning bounds.
MEASURED_MIN_FINGER_EXTENSION_MM = 58.4
MEASURED_MAX_FINGER_EXTENSION_MM = 69.9
# The table obstacle is a 200 mm box centered at world z = -123: top at -23.
TABLE_TOP_Z_MM = -23.0
# But the wrist depth camera sees the real table surface at z = +3..+7 mm
# (measured three times on 2026-09-19 from different arm poses). The object
# heights come from that same camera, so the fingertip floor uses the higher of
# the two; a too-high floor only makes flat objects ungraspable, never a crash.
MEASURED_TABLE_TOP_Z_MM = 5.0
FINGERTIP_TABLE_CLEARANCE_MM = 10.0
ARM_REACH_MM = 700.0                 # xArm6
APPROACH_HEIGHT_MM = 100.0           # standoff above the grasp pose
FINGER_INSERTION_MM = 15.0          # shallow engagement below the object's top, not its center
GRIPPER_BODY_CLEARANCE_MM = 15.0    # body must remain this far above the object's top
DEFAULT_RETURN_HEIGHT_ABOVE_TABLE_MM = 355.6   # 14 inches if no home was taught
MIN_RELEASE_RETREAT_MM = 40.0
MOTION_WAIT_REPORT_S = 5.0
GRASP_ARRIVAL_POSITION_MM = 5.0
GRASP_ARRIVAL_ANGLE_DEG = 2.0
TARGET_REPEAT_POSITION_MM = 10.0
TARGET_REPEAT_SIZE_MM = 10.0


def grasp_settings(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--grasp-z-offset-mm", type=float, default=0.0)
    parser.add_argument("--finger-clearance-mm", type=float, default=MEASURED_MIN_FINGER_EXTENSION_MM)
    parser.add_argument("--max-finger-extension-mm", type=float, default=MEASURED_MAX_FINGER_EXTENSION_MM)
    parser.add_argument("--gripper-body-from-flange-mm", type=float,
                        default=MEASURED_GRIPPER_BODY_FROM_FLANGE_MM)
    parser.add_argument("--object-width-mm", action="append", default=[], metavar="LABEL=MM")
    args, _ = parser.parse_known_args(argv)
    widths = {}
    for spec in args.object_width_mm:
        label, separator, raw = spec.partition("=")
        try:
            value = float(raw)
        except ValueError:
            parser.error("--object-width-mm must be LABEL=MM, for example block=28.6")
        if not separator or not label.strip() or not math.isfinite(value) or not 1 < value <= 85:
            parser.error("--object-width-mm needs a label and a measured width above 1 and at most 85 mm")
        widths[label.strip().lower()] = value
    args.object_width_mm = widths
    if not math.isfinite(args.grasp_z_offset_mm) or args.grasp_z_offset_mm < 0:
        parser.error("--grasp-z-offset-mm must be a finite, nonnegative upward offset")
    if args.finger_clearance_mm is not None and (
            not math.isfinite(args.finger_clearance_mm)
            or args.finger_clearance_mm <= GRIPPER_BODY_CLEARANCE_MM):
        parser.error(f"--finger-clearance-mm must exceed the {GRIPPER_BODY_CLEARANCE_MM:g} mm body clearance")
    for name in ("max_finger_extension_mm", "gripper_body_from_flange_mm"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if (args.max_finger_extension_mm is not None and args.finger_clearance_mm is not None
            and args.max_finger_extension_mm < args.finger_clearance_mm):
        parser.error("maximum finger extension cannot be less than minimum finger clearance")
    return args


_grasp_settings = grasp_settings(sys.argv[1:])
GRASP_Z_OFFSET_MM = _grasp_settings.grasp_z_offset_mm
# Minimum/maximum axial tip extension from the same fixed housing underside.
# These bounds must cover the complete open-to-close stroke being used.
FINGER_CLEARANCE_MM = _grasp_settings.finger_clearance_mm
MAX_FINGER_EXTENSION_MM = _grasp_settings.max_finger_extension_mm
GRIPPER_BODY_FROM_FLANGE_MM = _grasp_settings.gripper_body_from_flange_mm
MEASURED_OBJECT_WIDTHS_MM = _grasp_settings.object_width_mm


def require_grasp_calibration():
    values = (FINGER_CLEARANCE_MM, MAX_FINGER_EXTENSION_MM, GRIPPER_BODY_FROM_FLANGE_MM)
    if any(v is None for v in values):
        flags = ("--finger-clearance-mm", "--max-finger-extension-mm", "--gripper-body-from-flange-mm")
        missing = ", ".join(flag for flag, value in zip(flags, values) if value is None)
        raise ValueError(f"Missing measured gripper geometry: {missing}. "
                         "Finger measurements run from the housing underside to the finger ends, "
                         "not between the jaws. Supply the missing values before picking.")
    if (not all(math.isfinite(v) for v in values)
            or FINGER_CLEARANCE_MM <= GRIPPER_BODY_CLEARANCE_MM
            or MAX_FINGER_EXTENSION_MM < FINGER_CLEARANCE_MM
            or GRIPPER_BODY_FROM_FLANGE_MM <= 0):
        raise ValueError("Invalid measured gripper geometry")

# Orientation: grasp with the wrist orientation the arm already has at the
# observe pose when it points roughly down, so the planner never has to spin
# the wrist. (The old fixed target, theta=0 vs the arm's 142.7, asked for a
# ~143 degree wrist rotation on every grasp.) This is the fallback.
DEFAULT_GRASP_ORIENTATION = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=142.7)
MIN_DOWNWARD_O_Z = -0.9              # current orientation reused only if o_z <= this
ORIENTATION_TOLERANCE_DEGS = 15.0    # held along approach and carry moves
LINE_TOLERANCE_MM = 10.0             # descent / lift stay on a straight line

# Position-limited close: the uFactory gripper takes a
# position on a 0-850 scale (850 = fully open, ~85 mm), i.e. ~10 units per mm.
# This bounds requested closure; it is not force control or a crush guarantee.
GRIPPER_SQUEEZE_MM = 1.0
GRIPPER_POS_PER_MM = 10.0
GRIPPER_MAX_POS = 850
GRIPPER_OPEN_MIN_POS = 830
GRIPPER_CLOSED_MAX_POS = 10
GRIPPER_MIN_CLOSURE_POS = 10
GRIPPER_POSITION_TOLERANCE = 6
GRIPPER_VERIFY_TIMEOUT_S = 3.0
GRIPPER_MAX_WIDTH_ERROR_MM = 5.0

# Home / serve pose: record once with the arm parked there
# (python main.py --set-home -> home_pose.json). Without it, the object is
# carried back to where the gripper was when the object was locked.
HOME_POSE_PATH = HERE / "home_pose.json"
# The saved pose is primarily a return destination after picking/dropping.
# Moving there at launch requires an explicit flag.
GO_HOME_ON_START = "--go-home-on-start" in sys.argv
RETURN_TO_START = True

# Every other segmented object is frozen into world coordinates at lock time
# and handed to the planner as an obstacle, on top of the machine's static
# table/wall/ceiling obstacles. Direct arm moves ignore obstacles, so this
# code only ever moves through the motion service.
AVOID_DETECTED_OBJECTS = True

WEBCAM_INDEX = 0
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


TOP_FACE_BAND_MM = 12.0       # points this close below the highest ones count as the object's top face
TOP_FACE_MAX_SHIFT_MM = 30.0  # the top-face center may differ this much from the segmenter's center


def parse_pcd_xyz(data: bytes) -> np.ndarray:
    """XYZ in mm (N x 3) from a binary PCD, the format Viam returns point clouds in."""
    head, sep, body = data.partition(b"DATA binary\n")
    if not sep:
        raise ValueError("unsupported point cloud encoding (expected binary PCD)")
    fields = {}
    for line in head.decode("ascii", "replace").splitlines():
        parts = line.split()
        if parts:
            fields[parts[0].upper()] = parts[1:]
    names, sizes = fields["FIELDS"], [int(v) for v in fields["SIZE"]]
    counts = [int(v) for v in fields.get("COUNT", ["1"] * len(names))]
    n = int(fields["POINTS"][0])
    stride = sum(size * count for size, count in zip(sizes, counts))
    raw = np.frombuffer(body, dtype=np.uint8, count=n * stride).reshape(n, stride)
    cols, offset = {}, 0
    for name, size, count in zip(names, sizes, counts):
        if name in ("x", "y", "z"):
            cols[name] = raw[:, offset:offset + size].copy().view({4: "<f4", 8: "<f8"}[size]).ravel()
        offset += size * count
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(float)
    xyz = xyz[np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)]
    if len(xyz) and np.median(np.abs(xyz[:, 2])) < 20.0:   # Viam point clouds are in metres
        xyz *= 1000.0
    return xyz


async def frame_to_world(robot: RobotClient, ref: str) -> tuple[np.ndarray, np.ndarray]:
    """(origin, R) such that world = origin + R @ p for a point p in `ref`, as the
    frames are right now (so call it while the arm is still)."""
    async def world(x, y, z):
        p = await pose_in(robot, Pose(x=x, y=y, z=z, o_z=1.0), ref, MOTION_REFERENCE_FRAME)
        return np.array([p.x, p.y, p.z])
    origin = await world(0.0, 0.0, 0.0)
    axes = [await world(*e) - origin for e in ((1000.0, 0.0, 0.0), (0.0, 1000.0, 0.0), (0.0, 0.0, 1000.0))]
    return origin, np.stack(axes, axis=1) / 1000.0


def object_top_face(point_cloud_obj, to_world) -> Optional[tuple[float, float, float]]:
    """(x, y, top z) of the object's top face in world, from its own depth points.
    The segmenter's box center is the MEAN of the points, which the dense top
    face pulls upward, so center + height/2 overshoots the real top (by ~13 mm
    on a 60 mm block on this machine) and the fingers barely reach the object."""
    try:
        points = parse_pcd_xyz(point_cloud_obj.point_cloud)
    except Exception as e:
        print(f"[grab] couldn't read the object's depth points ({type(e).__name__}: {e})")
        return None
    if len(points) < 30:
        return None
    origin, rotation = to_world
    world = points @ rotation.T + origin
    top = float(np.percentile(world[:, 2], 98))          # robust to a few stray points
    face = world[world[:, 2] >= top - TOP_FACE_BAND_MM]
    if len(face) < 20:
        return None
    return float(np.median(face[:, 0])), float(np.median(face[:, 1])), top


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
    wrist camera once the arm moves. Planner names identify individual shapes;
    semantic labels on the source detections remain unchanged."""
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
            # A class label such as "can" is not unique when several objects or
            # duplicate detections are present. Viam requires unique names.
            ng.label = f"gaze_detected_{len(geoms)}"
            geoms.append(ng)
    if not geoms:
        return None
    return WorldState(obstacles=[GeometriesInFrame(reference_frame=MOTION_REFERENCE_FRAME, geometries=geoms)])


def merge_world_states(*states: Optional[WorldState]) -> Optional[WorldState]:
    """Copy obstacle frames and give every merged shape a unique planner name.

    Captures and retained object footprints may each start numbering at zero.
    Reassign names across the full request, including a single input state,
    without changing the stored states used by later actions.
    """
    obstacles = []
    geometry_index = 0
    for state in states:
        if state is None:
            continue
        for frame in state.obstacles:
            copied_frame = GeometriesInFrame()
            copied_frame.CopyFrom(frame)
            for geometry in copied_frame.geometries:
                geometry.label = f"gaze_obstacle_{geometry_index}"
                geometry_index += 1
            obstacles.append(copied_frame)
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
    object_top_z_mm: float
    fingertip_z_mm: float
    body_clearance_mm: float
    lowest_fingertip_z_mm: float
    body_z_mm: float


def pose_rotation(pose: Pose) -> np.ndarray:
    """Use Viam's orientation-vector convention, not an axis-angle formula."""
    if not all(math.isfinite(v) for v in (pose.o_x, pose.o_y, pose.o_z, pose.theta)):
        raise ValueError("Invalid object orientation")
    if math.hypot(pose.o_x, pose.o_y, pose.o_z) < 1e-9:
        return np.eye(3)
    return np.asarray(Quaternion.from_pose(pose).to_rotation_matrix().elements).reshape(3, 3)


def world_box_extents(world_center: Pose, size) -> np.ndarray:
    dims = np.asarray(size, dtype=float)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        raise ValueError("No usable 3D object dimensions; refusing to guess the object's top")
    # Box dimensions are measured along its local axes. After a camera->world
    # transform, camera depth is generally NOT the object's vertical height.
    return np.abs(pose_rotation(world_center)) @ dims


def pose_error(actual: Pose, expected: Pose) -> tuple[float, float]:
    coordinates = (actual.x, actual.y, actual.z, expected.x, expected.y, expected.z)
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("Nonfinite robot pose")
    distance = math.dist(coordinates[:3], coordinates[3:])
    rotation = pose_rotation(actual).T @ pose_rotation(expected)
    angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    return distance, angle


def target_repeat_problem(first: Pose, first_size, second: Pose, second_size) -> Optional[str]:
    """A stationary target must agree across two independent depth captures."""
    distance = math.dist((first.x, first.y, first.z), (second.x, second.y, second.z))
    a, b = world_box_extents(first, first_size), world_box_extents(second, second_size)
    change = float(np.max(np.abs(a - b)))
    top_change = abs(first.z + a[2] / 2 - second.z - b[2] / 2)
    if (not math.isfinite(distance) or distance > TARGET_REPEAT_POSITION_MM
            or change > TARGET_REPEAT_SIZE_MM or top_change > TARGET_REPEAT_POSITION_MM):
        return (f"depth captures disagree: center changed {distance:.1f} mm, "
                f"dimensions {change:.1f} mm, top {top_change:.1f} mm; "
                "check the depth image/camera frame before retrying")
    return None


def grasp_orientation(current: Optional[Pose]) -> dict:
    if current is not None and current.o_z <= MIN_DOWNWARD_O_Z:
        return dict(o_x=current.o_x, o_y=current.o_y, o_z=current.o_z, theta=current.theta)
    return dict(DEFAULT_GRASP_ORIENTATION)


def plan_grasp(w: Pose, size, current: Optional[Pose], tcp_mm: float,
               home: Optional[Pose], top_z: Optional[float] = None,
               table_z: Optional[float] = None) -> GraspPlan:
    """Use the world top of the object and keep the palm above it.
    top_z: the object's measured top (from its depth points); table_z: the
    table as the depth camera sees it (run_grasp passes MEASURED_TABLE_TOP_Z_MM).
    The floor uses the HIGHER of that and the configured table, so the
    fingertips never go below either."""
    require_grasp_calibration()
    if not all(math.isfinite(v) for v in (w.x, w.y, w.z, tcp_mm, FINGER_CLEARANCE_MM, GRASP_Z_OFFSET_MM)):
        raise ValueError("Invalid grasp geometry")
    if tcp_mm <= 0:
        raise ValueError("Invalid TCP reference offset")
    if current is None or not all(math.isfinite(v) for v in (
            current.x, current.y, current.z, current.o_x, current.o_y, current.o_z, current.theta)):
        raise ValueError("A valid current gripper pose is required before descent")
    if GRASP_Z_OFFSET_MM < 0 or FINGER_CLEARANCE_MM <= GRIPPER_BODY_CLEARANCE_MM:
        raise ValueError("Invalid upward offset or measured finger clearance")
    if current.o_z > MIN_DOWNWARD_O_Z:
        raise ValueError("Park the gripper pointing vertically down before picking")
    orient = grasp_orientation(current)
    norm = math.hypot(orient["o_x"], orient["o_y"], orient["o_z"])
    down = -orient["o_z"] / norm
    if down < math.cos(math.radians(5)):
        raise ValueError("Park the gripper pointing vertically down (within 5 degrees) before picking")
    extents = world_box_extents(w, size)
    top = top_z if top_z is not None else w.z + extents[2] / 2.0
    table = TABLE_TOP_Z_MM if table_z is None else max(TABLE_TOP_Z_MM, table_z)
    # Allow for the low edge of a slightly tilted body, using a conservative
    # 100 mm radius around the tool axis. This is separate from finger length.
    tilt_margin = 100.0 * math.sqrt(max(0.0, 1.0 - down * down))
    stroke_height = (MAX_FINGER_EXTENSION_MM - FINGER_CLEARANCE_MM) * down
    # Position the RIGID housing. A closing linkage may lift the object before
    # the arm lifts, so reserve the full axial stroke above its observed top.
    body_z = max(
        top + FINGER_CLEARANCE_MM * down - tilt_margin - FINGER_INSERTION_MM,
        top + GRIPPER_BODY_CLEARANCE_MM + tilt_margin + stroke_height,
        table + FINGERTIP_TABLE_CLEARANCE_MM + MAX_FINGER_EXTENSION_MM * down + tilt_margin,
    ) + GRASP_Z_OFFSET_MM
    tip_z = body_z - FINGER_CLEARANCE_MM * down + tilt_margin
    lowest_tip_z = body_z - MAX_FINGER_EXTENSION_MM * down - tilt_margin
    if tip_z >= top:
        raise ValueError("The upward offset/table clearance leaves the fingertips above the object; "
                         "refusing a grasp with no finger overlap")
    body_from_tcp = GRIPPER_BODY_FROM_FLANGE_MM - tcp_mm
    z = body_z + body_from_tcp * down
    x, y = w.x - body_from_tcp * orient["o_x"] / norm, w.y - body_from_tcp * orient["o_y"] / norm
    grasp = Pose(x=x, y=y, z=z, **orient)
    approach = Pose(x=x, y=y, z=z + APPROACH_HEIGHT_MM, **orient)
    back = None
    if RETURN_TO_START:
        src = home or current
        if src is not None:
            back = Pose(x=src.x, y=src.y, z=src.z, **orient)   # carried back level, same wrist
    # Existing width estimate remains approximate: it is not force feedback.
    # The close verifier rejects implausible gaps and unconfirmed contact.
    width = min((d for d in size[:2] if d > 0), default=0.0)
    beyond = GRIPPER_BODY_FROM_FLANGE_MM + MAX_FINGER_EXTENSION_MM - tcp_mm
    return GraspPlan(orient, approach, grasp, back, width, beyond, top, tip_z,
                     body_z - tilt_margin - top - stroke_height, lowest_tip_z, body_z)


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
        self.requires_operator = False
        self.action_error: Optional[Exception] = None
        self.plan_rejection: Optional[str] = None
        self.motion_failure: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.finished_at is not None


def upright() -> Constraints:
    return Constraints(orientation_constraint=[
        OrientationConstraint(orientation_tolerance_degs=ORIENTATION_TOLERANCE_DEGS)])


def straight_line() -> Constraints:
    return Constraints(linear_constraint=[
        LinearConstraint(line_tolerance_mm=LINE_TOLERANCE_MM, orientation_tolerance_degs=ORIENTATION_TOLERANCE_DEGS)])


def is_ik_constraint_rejection(error: Exception) -> bool:
    """Recognize the specific failure returned before motion execution starts."""
    return (isinstance(error, GRPCError) and error.status == Status.UNKNOWN
            and (error.message or "").lower().startswith("all ik solutions failed constraints."))


async def report_motion_wait(job, label: str, started: float) -> None:
    """Report a pending RPC without assuming the robot has begun moving."""
    while True:
        await asyncio.sleep(MOTION_WAIT_REPORT_S)
        elapsed = time.monotonic() - started
        job.status = f"{label}: motion request pending {elapsed:.0f}s (planning/execution)"
        print(f"[motion] {job.status}; movement not confirmed; Q stops", flush=True)


async def move_to(motion: MotionClient, pose: Pose, label: str, job: GraspJob,
                  world_state: Optional[WorldState] = None,
                  constraints: Optional[Constraints] = None) -> bool:
    if getattr(job, "plan_rejection", None) or getattr(job, "motion_failure", None):
        # A failed stage ends this job. Existing cleanup paths may request a
        # retreat, but no additional command is sent until a new job is chosen.
        job.status = getattr(job, "plan_rejection", None) or job.motion_failure
        return False
    prefix = "DRY RUN, would be moving" if DRY_RUN else "requesting move"
    extras = []
    if world_state is not None:
        extras.append(f"{sum(len(g.geometries) for g in world_state.obstacles)} obstacles")
    if constraints is not None:
        extras.append("straight line" if constraints.linear_constraint else "orientation held")
    job.status = (f"{prefix} to {label}: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f} mm"
                  + (f" ({', '.join(extras)})" if extras else ""))
    print(f"[grab] {job.status}")
    if DRY_RUN:
        await asyncio.sleep(0.4)
        return True
    # viam-sdk 0.80.0: component_name is the component's plain name (proto string).
    motion_started_before = getattr(job, "motion_started", False)
    job.motion_started = True
    started = time.monotonic()
    reporter = asyncio.create_task(report_motion_wait(job, label, started))
    try:
        ok = await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame=MOTION_REFERENCE_FRAME, pose=pose),
            world_state=world_state,
            constraints=constraints,
        )
    except asyncio.CancelledError:
        print(f"[motion] {label}: request cancelled after {time.monotonic() - started:.1f}s", flush=True)
        raise
    except Exception as error:
        print(f"[motion] {label}: request failed after {time.monotonic() - started:.1f}s: "
              f"{type(error).__name__}: {error}", flush=True)
        if not is_ik_constraint_rejection(error):
            raise
        # Viam plans before executing. This narrow error says this particular
        # command never executed; earlier completed actions remain recorded.
        job.motion_started = motion_started_before
        job.plan_rejection = f"PLAN REJECTED at {label}: {error.message}"
        job.status = job.plan_rejection
        print(f"[grab] {job.status}")
        return False
    finally:
        reporter.cancel()
        await asyncio.gather(reporter, return_exceptions=True)
    print(f"[motion] {label}: response success={ok} after {time.monotonic() - started:.1f}s", flush=True)
    if ok:
        job.status = f"Completed move to {label} (motion service reported success)"
    if not ok:
        job.status = f"motion.move() failed on {label}"
        # False does not establish whether the robot moved. Preserve the
        # physical-action flag and block automatic follow-up commands.
        job.motion_failure = job.status
        job.requires_operator = True
        print(f"[grab] {job.status}")
    return ok


def gripper_position_for_width(width_mm: float) -> int:
    return int(max(0.0, min(GRIPPER_MAX_POS, (width_mm - GRIPPER_SQUEEZE_MM) * GRIPPER_POS_PER_MM)))


def checked_gripper_position(response, key: str) -> float:
    """A missing or malformed readback leaves the physical state unknown."""
    value = response.get(key) if isinstance(response, dict) else None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= GRIPPER_MAX_POS):
        raise RuntimeError(f"gripper returned no valid {key!r} position; stopping with grip unconfirmed")
    return float(value)


async def read_gripper_position(gripper: Gripper, timeout: float = 3.0) -> float:
    response = await gripper.do_command({"get": True}, timeout=timeout)
    return checked_gripper_position(response, "pos")


async def open_gripper(gripper: Gripper, job: GraspJob) -> None:
    job.status = "DRY RUN, would open the gripper" if DRY_RUN else "opening the gripper"
    print(f"[grab] {job.status}")
    if not DRY_RUN:
        job.motion_started = True
        job.holding = None
        await gripper.open(timeout=CAPTURE_TIMEOUT_S)
        # A successful RPC does not establish that the fingers actually opened.
        # Keep the arm above the object until a fully open readback arrives.
        async with asyncio.timeout(2.0):
            while True:
                position = await read_gripper_position(gripper, timeout=2.0)
                moving = await gripper.is_moving(timeout=2.0)
                if not moving and position >= GRIPPER_OPEN_MIN_POS:
                    job.holding = False
                    print(f"[grab] opening verified at position {position:.0f}")
                    return
                await asyncio.sleep(0.1)


async def close_gripper(gripper: Gripper, arm: Arm, width_mm: float, job: GraspJob) -> bool:
    """Close once to a bounded width and require position evidence of contact.

    G1's holding boolean only means the jaws are between their endpoints. An
    empty partial close therefore cannot be called a grasp. Even the checks
    below are contact heuristics, not force sensing or proof of object retention.
    """
    job.holding = None

    def unconfirmed(reason: str) -> bool:
        job.holding = False
        job.status = f"grip unconfirmed: {reason}"
        print(f"[grab] {job.status}")
        return False

    max_span_mm = GRIPPER_MAX_POS / GRIPPER_POS_PER_MM
    if not math.isfinite(width_mm) or not 0 < width_mm <= max_span_mm:
        return unconfirmed(f"estimated width {width_mm:g} mm is outside the usable gripper span")
    target = gripper_position_for_width(width_mm)
    if DRY_RUN:
        job.status = (f"DRY RUN, would close once to position {target} "
                      f"(object ~{width_mm:.0f} mm wide) and verify contact")
        print(f"[grab] {job.status}")
        return True

    before = await read_gripper_position(gripper, timeout=GRIPPER_VERIFY_TIMEOUT_S)
    if before < GRIPPER_OPEN_MIN_POS:
        return unconfirmed(f"gripper was not fully open before closing (position {before:.0f})")

    job.motion_started = True
    job.status = f"object ~{width_mm:.0f} mm wide, closing once to position {target}"
    print(f"[grab] {job.status}")
    # A timeout may follow an executed command. Never replay through another
    # API, and never replace this bounded move with an unrestricted grab().
    response = await gripper.do_command({"set": target}, timeout=CAPTURE_TIMEOUT_S)
    if isinstance(response, dict) and not response:
        return unconfirmed("the driver did not acknowledge the width command")
    checked_gripper_position(response, "position")

    # Read measured positions after the acknowledgement. The driver's moving
    # flag tracks its RPC, so also require the actual position to settle.
    samples = []
    async with asyncio.timeout(GRIPPER_VERIFY_TIMEOUT_S):
        while True:
            position = await read_gripper_position(gripper, timeout=GRIPPER_VERIFY_TIMEOUT_S)
            moving = await gripper.is_moving(timeout=GRIPPER_VERIFY_TIMEOUT_S)
            samples.append(position)
            samples = samples[-3:]
            if (not moving and len(samples) == 3
                    and max(samples) - min(samples) <= GRIPPER_POSITION_TOLERANCE):
                break
            await asyncio.sleep(0.1)

    if before - position < GRIPPER_MIN_CLOSURE_POS:
        return unconfirmed(f"no measurable closure ({before:.0f} -> {position:.0f})")
    if not GRIPPER_CLOSED_MAX_POS < position < GRIPPER_OPEN_MIN_POS:
        return unconfirmed(f"jaws ended at an empty-gripper endpoint ({position:.0f})")
    if position - target <= GRIPPER_POSITION_TOLERANCE:
        return unconfirmed("jaws reached the requested width without evidence of object contact")
    if position / GRIPPER_POS_PER_MM > width_mm + GRIPPER_MAX_WIDTH_ERROR_MM:
        return unconfirmed("jaws stopped too far apart for the detected object's width")

    status = await gripper.is_holding_something(timeout=GRIPPER_VERIFY_TIMEOUT_S)
    holding = getattr(status, "is_holding_something", status)
    if not isinstance(holding, bool):
        raise RuntimeError("gripper returned an invalid holding status; stopping with grip unconfirmed")
    if not holding:
        return unconfirmed("the gripper reports no object after closing")

    job.holding = True
    job.status = f"contact inferred: jaws {before:.0f} -> {position:.0f}, target {target}"
    print(f"[grab] {job.status}")
    return True


async def verify_grasp_arrival(motion: MotionClient, arm: Arm, pose: Pose, job: GraspJob) -> bool:
    """Never close solely because the movement RPC returned success."""
    if DRY_RUN:
        return True
    for sample in range(2):
        moving = await arm.is_moving(timeout=GRIPPER_VERIFY_TIMEOUT_S)
        actual = await gripper_world_pose(motion)
        if actual is None:
            raise RuntimeError("Could not verify the actual grasp pose; jaws have not been closed")
        distance, angle = pose_error(actual, pose)
        print(f"[grab] arrival readback {sample + 1}: TCP ({actual.x:.1f}, {actual.y:.1f}, "
              f"{actual.z:.1f}); error {distance:.1f} mm / {angle:.2f} deg; moving={moving}")
        if moving or distance > GRASP_ARRIVAL_POSITION_MM or angle > GRASP_ARRIVAL_ANGLE_DEG:
            job.status = (f"NOT CLOSING: grasp pose not reached/stopped "
                          f"({distance:.1f} mm, {angle:.1f} deg, moving={moving})")
            print(f"[grab] {job.status}")
            return False
        if sample == 0:
            await asyncio.sleep(0.15)
    return True


async def execute_grasp(motion: MotionClient, gripper: Gripper, arm: Arm, plan: GraspPlan,
                        obstacles: Optional[WorldState], job: GraspJob) -> bool:
    """Approach above (wrist held) -> open -> straight down -> close -> straight
    up, then hover there holding it: what happens next is the user's choice
    (delivery.py). An unconfirmed grasp opens at the table and retreats. In a dry run every step
    is printed, nothing moves, and it continues as if holding."""
    if not await move_to(motion, plan.approach, "approach", job, obstacles, upright()):
        return False
    await open_gripper(gripper, job)
    if not await move_to(motion, plan.grasp, "grasp (straight down)", job, obstacles, straight_line()):
        await move_to(motion, plan.approach, "back up", job, obstacles, straight_line())
        return False
    if not await verify_grasp_arrival(motion, arm, plan.grasp, job):
        return False
    if not await close_gripper(gripper, arm, plan.width_mm, job):
        reason = job.status
        # Release at the table before retreating. Do not lift/carry an object
        # on the strength of a position-only HoldingStatus.
        await open_gripper(gripper, job)
        await move_to(motion, plan.approach, "retreat after unconfirmed grip", job, obstacles, straight_line())
        job.status = f"pickup NOT confirmed: {reason}"
        print(f"[grab] {job.status}")
        return False

    if not await move_to(motion, plan.approach, "lift (straight up)", job, obstacles, straight_line()):
        job.status = "grip confirmed, but lift failed; object may still be held"
        return False
    job.status = "DRY RUN: holding (pretend)" if DRY_RUN else "holding it"
    print(f"[grab] {job.status}")
    return True


def released_footprint(held: Held, release: Pose) -> Optional[WorldState]:
    """Keep the released object in the world model for the return across the table."""
    if held.footprint is None:
        return None
    footprint = WorldState()
    footprint.CopyFrom(held.footprint)
    delta = (release.x - held.pick_grasp.x, release.y - held.pick_grasp.y,
             release.z - held.pick_grasp.z)
    for frame in footprint.obstacles:
        if frame.reference_frame != MOTION_REFERENCE_FRAME:
            raise ValueError("Released-object footprint must already be in world coordinates")
        for geometry in frame.geometries:
            geometry.center.x += delta[0]
            geometry.center.y += delta[1]
            geometry.center.z += delta[2]
            if release.z > held.place_z + 10.0:
                # "Let go" can release above the table. Enclose the vertical
                # drop as well as the release position instead of leaving a
                # floating obstacle at the old wrist height.
                kind = geometry.WhichOneof("geometry_type")
                if kind == "box":
                    dims = geometry.box.dims_mm
                    extents = world_box_extents(geometry.center, (dims.x, dims.y, dims.z))
                elif kind == "sphere":
                    extents = np.full(3, 2.0 * geometry.sphere.radius_mm)
                else:
                    raise ValueError("Cannot bound the released object's drop; return paused")
                low = min(TABLE_TOP_Z_MM, geometry.center.z - extents[2] / 2.0)
                high = geometry.center.z + extents[2] / 2.0
                geometry.center.CopyFrom(Pose(x=geometry.center.x, y=geometry.center.y,
                                              z=(low + high) / 2.0, o_z=1.0))
                geometry.box.dims_mm.x = float(extents[0])
                geometry.box.dims_mm.y = float(extents[1])
                geometry.box.dims_mm.z = float(high - low)
    return footprint


async def return_after_release(motion: MotionClient, held: Held, job) -> bool:
    """After verified opening, rise at the actual drop XY, then return home."""
    release = held.pose if DRY_RUN else await gripper_world_pose(motion)
    if release is None:
        raise RuntimeError("Object released, but the actual wrist pose is unknown; return paused")
    home = held.home or Pose(x=held.pick_approach.x, y=held.pick_approach.y,
                             z=max(held.pick_approach.z, TABLE_TOP_Z_MM + DEFAULT_RETURN_HEIGHT_ABOVE_TABLE_MM),
                             **held.orientation)
    if not all(math.isfinite(value) for p in (release, home)
               for value in (p.x, p.y, p.z, p.o_x, p.o_y, p.o_z, p.theta)):
        raise ValueError("Invalid return/drop pose; no return movement sent")
    # Keep the current orientation while the fingertips clear the released
    # object. The horizontal return happens only after this ascent succeeds.
    up = Pose(x=release.x, y=release.y, z=max(home.z, release.z + MIN_RELEASE_RETREAT_MM),
              o_x=release.o_x, o_y=release.o_y, o_z=release.o_z, theta=release.theta)
    held.pose = release
    print(f"[grab] object released; vertical retreat at ({release.x:.1f}, {release.y:.1f}) "
          f"to z={up.z:.1f}, then saved return ({home.x:.1f}, {home.y:.1f}, {home.z:.1f}) mm")
    if not await move_to(motion, up, "rise vertically after release", job, held.obstacles, straight_line()):
        job.status = "Object released, but vertical retreat failed; return paused"
        job.requires_operator = True
        return False
    held.pose = up
    obstacles = merge_world_states(held.obstacles, released_footprint(held, release))
    if not await move_to(motion, home, "the saved return pose after release", job, obstacles, upright()):
        job.status = "Object released, but couldn't return to the saved pose"
        job.requires_operator = True
        return False
    held.pose = home
    return True


async def put_back(motion: MotionClient, gripper: Gripper, held: Held,
                   obstacles: Optional[WorldState], job: GraspJob) -> bool:
    """Set the held object down where it was picked up: above -> straight down
    -> open -> straight up. The gripper opens only once the descent succeeded,
    so a failed plan never drops it. True once released and the retreat succeeds."""
    # During a swap this is the newer scene, including the next pick target.
    # Keep it for the complete placement and return, including recovery.
    held.obstacles = obstacles
    g = held.pick_grasp
    above = Pose(x=g.x, y=g.y, z=held.pick_approach.z, **held.orientation)
    place = Pose(x=g.x, y=g.y, z=held.place_z, **held.orientation)
    if not await move_to(motion, above, f"above where the {held.label} was", job, obstacles, upright()):
        return False
    held.pose = above
    if not await move_to(motion, place, f"put the {held.label} down", job, obstacles, straight_line()):
        await move_to(motion, above, "back up", job, obstacles, straight_line())
        return False
    held.pose = place
    await open_gripper(gripper, job)
    job.held = None                       # released: from here on the gripper is empty
    return await return_after_release(motion, held, job)


async def run_put_back(motion: MotionClient, gripper: Gripper, held: Held, job: GraspJob) -> bool:
    """P in demo mode: put the held object back and return to the observe pose."""
    job.held = held
    try:
        if not await put_back(motion, gripper, held, held.obstacles, job):
            state = "still holding it" if job.held is not None else f"released it; {job.status}"
            job.status = f"couldn't finish putting the {held.label} back; {state}"
            print(f"[grab] {job.status}")
            return False
        job.ok = True
        job.status = f"put the {held.label} back"
        print(f"[grab] {job.status}")
        return True
    except asyncio.CancelledError:
        job.status = "cancelled"
        raise
    except Exception as e:
        job.action_error = e
        if is_transport_error(e):
            job.connection_error = e
        if job.motion_started:
            job.requires_operator = True
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
        if current is None or await arm.is_moving(timeout=GRIPPER_VERIFY_TIMEOUT_S):
            job.status = "NOT MOVING: the wrist must be stationary before measuring the target"
            print(f"[grab] {job.status}")
            return False
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
        # Preserve the box orientation so its world vertical extent is correct.
        w = await pose_in(robot, center, ref, MOTION_REFERENCE_FRAME)
        print(f"[grab] target '{object_label(target)}' center ({center.x:.0f}, {center.y:.0f}, {center.z:.0f}) "
              f"in '{ref}' -> ({w.x:.0f}, {w.y:.0f}, {w.z:.0f}) in '{MOTION_REFERENCE_FRAME}', "
              f"size {size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f} mm")
        problem = target_problem(w)
        if problem:
            job.status = f"NOT MOVING: {problem}"
            print(f"[grab] {job.status}")
            return False
        # A stale depth result following manual wrist adjustment can otherwise
        # acquire a plausible but completely wrong world height. Require a
        # second independent capture while the wrist remains stationary.
        await asyncio.sleep(0.3)
        repeated = await capture_objects(segmenter)
        repeated_target = await match_object_to_box(robot, repeated, box, intr, frame_w, frame_h)
        if repeated_target is None:
            job.status = "NOT MOVING: target missing from the second depth capture"
            print(f"[grab] {job.status}")
            return False
        repeated_center, repeated_size = object_center_and_size(repeated_target)
        repeated_world = await pose_in(robot, repeated_center,
                                       object_reference_frame(repeated_target), MOTION_REFERENCE_FRAME)
        after = await gripper_world_pose(motion)
        if after is None:
            raise RuntimeError("Cannot verify that the wrist stayed still during target capture")
        distance, angle = pose_error(after, current)
        problem = target_repeat_problem(w, size, repeated_world, repeated_size)
        if object_label(target) != object_label(repeated_target):
            problem = "the second depth capture matched a different object label"
        if distance > 2.0 or angle > 1.0 or await arm.is_moving(timeout=GRIPPER_VERIFY_TIMEOUT_S):
            problem = "wrist moved during depth capture; release the lock and select a fresh image"
        if problem:
            job.status = f"NOT MOVING: {problem}"
            print(f"[grab] {job.status}")
            return False
        print(f"[grab] repeated depth agrees: target world z={w.z:.1f} / {repeated_world.z:.1f} mm")
        # Use the more recent geometry after checking agreement.
        target, objs, w, size = repeated_target, repeated, repeated_world, repeated_size
        problem = target_problem(w)
        if problem:
            job.status = f"NOT MOVING: {problem}"
            print(f"[grab] {job.status}")
            return False
        in_hand = await objects_in_gripper(robot, objs, after)
        if any(target is o for o in in_hand):
            job.status = "NOT MOVING: the repeated target is already at the gripper"
            print(f"[grab] {job.status}")
            return False
        objs = [o for o in objs if not any(o is h for h in in_hand)]
        obstacles = await obstacles_in_world(robot, objs, target)
        footprint = await obstacles_in_world(robot, [target], None)   # the target itself, also frozen now
        n = sum(len(g.geometries) for g in obstacles.obstacles) if obstacles else 0
        # The top face from the object's own depth points (they came with the
        # segmentation; the frames are read now, while the arm is still).
        face = object_top_face(target, await frame_to_world(robot, object_reference_frame(target)))
        top_z = None
        if face is not None:
            fx, fy, top_z = face
            box_top = w.z + world_box_extents(w, size)[2] / 2.0
            if math.hypot(fx - w.x, fy - w.y) <= TOP_FACE_MAX_SHIFT_MM:
                w = Pose(x=fx, y=fy, z=w.z, o_x=w.o_x, o_y=w.o_y, o_z=w.o_z, theta=w.theta)
            print(f"[grab] top face from depth points: z={top_z:.1f} at ({fx:.0f}, {fy:.0f}) "
                  f"(segmenter box said top z={box_top:.1f})")
        plan = plan_grasp(w, size, current, tcp_mm, home, top_z=top_z, table_z=MEASURED_TABLE_TOP_Z_MM)
        measured_width = MEASURED_OBJECT_WIDTHS_MM.get((object_label(target) or box.label).lower())
        if measured_width is not None:
            print(f"[grab] using operator-measured width {measured_width:.1f} mm for "
                  f"'{object_label(target) or box.label}' (depth-box estimate {plan.width_mm:.1f} mm)")
            plan.width_mm = measured_width
        print(f"[grab] object top z={plan.object_top_z_mm:.1f}; fingertip z range="
              f"{plan.lowest_fingertip_z_mm:.1f}..{plan.fingertip_z_mm:.1f}; body z={plan.body_z_mm:.1f}; "
              f"TCP z={plan.grasp.z:.1f}; body clearance after possible jaw retraction="
              f"{plan.body_clearance_mm:.1f} mm; "
              f"extra upward offset={GRASP_Z_OFFSET_MM:.1f} mm")
        if not (0 < plan.width_mm <= GRIPPER_MAX_POS / GRIPPER_POS_PER_MM):
            job.status = f"NOT MOVING: estimated grasp width {plan.width_mm:.1f} mm is outside the gripper opening"
            print(f"[grab] {job.status}")
            return False
        print(f"[grab] {n} other object(s) as obstacles; maximum fingertip offset "
              f"{plan.fingertip_beyond_tcp_mm:.0f} mm from the TCP; "
              f"wrist o=({plan.orientation['o_x']:.2f}, {plan.orientation['o_y']:.2f}, "
              f"{plan.orientation['o_z']:.2f}) theta={plan.orientation['theta']:.1f}")
        model = await load_grasp_model(robot, gripper, world_frame=MOTION_REFERENCE_FRAME)
        if abs(model.table_top_z_mm - TABLE_TOP_Z_MM) > 1.0:
            raise ValueError(f"Configured table top {model.table_top_z_mm:.1f} mm differs from "
                             f"the grasp calculation's {TABLE_TOP_Z_MM:.1f} mm; update calibration first")
        clearance = check_model_clearance(model, plan.grasp)
        print(f"[grab] live collision model '{clearance.limiting_geometry}' table gap "
              f"{clearance.clearance_mm:.1f} mm; minimum modeled TCP z={clearance.required_tcp_z_mm:.1f} mm")
        if held is not None:
            # Swap: put the held object back first, avoiding everything on the
            # table (the new target included); afterwards its spot is occupied
            # again, so the new grasp avoids it too.
            print(f"[grab] swap: putting the {held.label} back before picking the {object_label(target) or box.label}")
            if not await put_back(motion, gripper, held, merge_world_states(obstacles, footprint), job):
                job.status = (f"couldn't finish putting the {held.label} back; "
                              + ("still holding it" if job.held is not None else "object released")
                              + ", nothing else picked")
                print(f"[grab] {job.status}")
                return False
            obstacles = merge_world_states(obstacles, held.footprint)
        picked = await execute_grasp(motion, gripper, arm, plan, obstacles, job)
        if not picked and not job.holding:
            return False
        # The object was resting on the table, so setting it down anywhere on
        # the table means putting the TCP back at the grasp height.
        job.held = Held(label=object_label(target) or box.label, pick_grasp=plan.grasp,
                        pick_approach=plan.approach, orientation=plan.orientation, obstacles=obstacles,
                        place_z=plan.grasp.z + PLACE_CLEARANCE_MM, width_mm=plan.width_mm,
                        pose=plan.approach if picked else plan.grasp, home=plan.return_pose, footprint=footprint)
        if not picked:
            return False
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
        job.action_error = e
        if is_transport_error(e):
            job.connection_error = e
        if job.motion_started:
            job.requires_operator = True
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


async def stop_everything(arm: Arm, job: Optional[GraspJob], *, reason: str = "stop requested") -> None:
    """Cancel the grasp first (so it sends nothing new), then stop the arm."""
    print(f"[main] STOP requested: {reason}" + (f"; last action: {job.status}" if job else ""), flush=True)
    if job is not None and job.task is not None and not job.task.done():
        job.task.cancel()
    if EXECUTE:
        print("[main] STOP: requesting arm.stop(). The physical E-stop is the real stop.", flush=True)
        try:
            await asyncio.wait_for(arm.stop(), timeout=2.0)
            print("[main] arm.stop() acknowledged", flush=True)
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
        return_after_release=lambda held, status: return_after_release(motion, held, status),
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

    require_grasp_calibration()  # Before any startup movement.
    if EXECUTE:
        try:
            await require_arm_joint_ranges(arm)
        except JointRangeError as error:
            raise SystemExit(f"START BLOCKED: {error}\nNo startup or grasp command was sent.") from None

    serve = load_serve_pose()
    if serve is None:
        print("[main] no serve_pose.json: 'Bring to me', Closer/Away and head steering are off until you run "
              "--set-serve with the gripper where the user should receive objects")
    elif serve_problem(serve):
        print(f"[main] ignoring serve_pose.json: {serve_problem(serve)}")
        serve = None

    tcp_mm = await measure_tcp_mm(motion)
    intr = await get_intrinsics(cam)
    print(f"[main] flange->TCP {tcp_mm:.0f} mm; flange->body {GRIPPER_BODY_FROM_FLANGE_MM:.1f} mm; "
          f"body->fingertips {FINGER_CLEARANCE_MM:.1f}..{MAX_FINGER_EXTENSION_MM:.1f} mm; camera intrinsics "
          + (f"fx={intr.fx:.0f} fy={intr.fy:.0f} at {intr.width}x{intr.height}" if intr else "UNAVAILABLE"))

    home = load_home_pose()
    if home is None:
        print("[main] no home_pose.json; objects are carried back to where the gripper was when locked")
    elif not GO_HOME_ON_START:
        print("[main] saved return pose loaded; startup movement is off (opt in with --go-home-on-start)")
    elif GO_HOME_ON_START and EXECUTE and not recovering:
        startup_job = GraspJob()
        try:
            startup_ok = await move_to(motion, home, "startup home", startup_job, constraints=upright())
        except Exception as error:
            # A lost response cannot tell us whether a motion command ran.
            raise RuntimeError("Startup movement failed. Check the arm before restarting; "
                               "the movement will not be retried automatically.") from error
        if not startup_ok:
            if startup_job.plan_rejection is not None:
                raise SystemExit(f"Startup paused: {startup_job.plan_rejection}\n"
                                 "Check the saved home pose and gripper/table geometry before restarting. "
                                 "No grasp was started; no movement will be retried automatically.")
            raise SystemExit(f"Startup stopped: {startup_job.status}. Check the arm before restarting; "
                             "no grasp was started and no movement will be retried automatically.")

    gaze = WebcamGazeTracker(camera_index=WEBCAM_INDEX)
    lock = GazeLockController()
    feed = RobotFeed(cam, detector)
    job: Optional[GraspJob] = None
    session: Optional[DeliverySession] = None
    held: Optional[Held] = None     # demo mode: what the gripper has between picks
    live_blocked_until = 0.0
    hovered, progress = None, 0.0
    keep_window = False
    operator_fault: Optional[OperatorFault] = None

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
            if operator_fault is None:
                fault_reason, fault_error = None, None
                if job is not None and (job.requires_operator or (
                        job.connection_error is not None and (job.motion_started or job.held is not None))):
                    fault_reason = f"Physical action interrupted: {job.status}"
                    fault_error = getattr(job, "action_error", None) or job.connection_error
                elif session is not None and (getattr(session, "requires_operator", False)
                                               or session.error is not None):
                    fault_reason = f"Delivery interrupted: {session.message}"
                    fault_error = session.error
                elif feed.connection_error is not None and (held is not None or session is not None
                        or (job is not None and (job.motion_started or job.held is not None))):
                    fault_reason = "Connection lost during object handling"
                    fault_error = feed.connection_error
                if fault_reason is not None:
                    if is_session_expired(fault_error):
                        fault_reason = "Viam safety session expired during the physical action"
                    operator_fault = OperatorFault(fault_reason)
                    feed.paused = True
                    # Cancel the action before one best-effort stop. Never
                    # replay its open/close/move after a session or RPC error.
                    if session is not None and session.busy:
                        session.task.cancel()
                        await asyncio.gather(session.task, return_exceptions=True)
                    await stop_everything(arm, job, reason=fault_reason)
                    print(f"[main] PAUSED: {fault_reason}. No command will be replayed. "
                          "Use robot controls to put down any object and open the jaws; "
                          "R then verifies stopped/open/empty state.")
            if operator_fault is not None:
                feed.paused = True
                if operator_fault.poll_check():
                    print(f"[main] {operator_fault.detail}; failed action discarded, waiting for fresh selection")
                    # Only verified open/empty recovery may discard a known
                    # held object. Until here all object state is retained.
                    held, job, session, operator_fault = None, None, None, None
                    lock.release()
                    estimator.reset()
                    feed.obs = None
                    live_blocked_until = time.monotonic() + LIVE_COOLDOWN_S
                    continue
                lines = ["ACTION PAUSED - no automatic retry"]
                lines.extend(textwrap.wrap(operator_fault.reason, width=90))
                lines.extend(textwrap.wrap(operator_fault.detail, width=90))
                lines.append("R check cleared gripper | Q quit")
                if lock.is_locked:
                    canvas = draw_locked(lock.locked, lines)
                else:
                    canvas = ((feed.obs.frame * 0.5).astype(np.uint8) if feed.obs is not None
                              else np.zeros((frame_h, frame_w, 3), dtype=np.uint8))
                    for i, line in enumerate(lines):
                        put_text(canvas, line, (16, 40 + 36 * i), 0.65)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break  # The interrupted action was already stopped once.
                if key == ord("r"):
                    operator_fault.start_check(lambda: check_action_recovery(
                        arm, gripper, lambda: read_gripper_position(gripper),
                        lambda: gripper_world_pose(motion)))
                await asyncio.sleep(0.03)
                continue
            if session is not None and session.error is not None and is_transport_error(session.error):
                raise ReconnectRequired("Connection lost during delivery") from session.error
            if job is not None and job.connection_error is not None:
                raise ReconnectRequired("Connection lost during grasp processing") from job.connection_error
            if feed.connection_error is not None:
                raise ReconnectRequired("Connection lost during capture") from feed.connection_error
            if job is not None and job.done and not job.ok:
                # Keep failed attempts visible until the operator acknowledges
                # them. A timeout must not reselect the same object by gaze.
                feed.paused = True
                lines = [job.status]
                failure_detail = job.plan_rejection or job.motion_failure
                if failure_detail is not None and failure_detail != job.status:
                    lines.append(failure_detail)
                lines.append("Paused after failed attempt | R acknowledge | Q stop+quit")
                if lock.is_locked:
                    canvas = draw_locked(lock.locked, lines)
                else:
                    canvas = ((feed.obs.frame * 0.5).astype(np.uint8) if feed.obs is not None
                              else np.zeros((frame_h, frame_w, 3), dtype=np.uint8))
                    for i, line in enumerate(lines):
                        put_text(canvas, line, (16, 40 + 36 * i), 0.65)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    await stop_everything(arm, job, reason="Q pressed on failed-attempt screen")
                    break
                if key == ord("r"):
                    held = job.held
                    if MENU and held is not None:
                        session = DeliverySession(delivery_io, held, serve,
                                                  head_mode=USER_PROFILE == "head", head_range=head_range)
                        print(f"[deliver] acknowledged failed attempt; holding the {held.label}: showing the menu")
                    else:
                        lock.release()
                        estimator.reset()
                        live_blocked_until = time.monotonic() + LIVE_COOLDOWN_S
                    job = None
                await asyncio.sleep(0.03)
                continue
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
                    await stop_everything(arm, job, reason="Q pressed during put-back")
                    break
                if job.done and job.ok:
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
                    await stop_everything(arm, job, reason="Q pressed during selected-object action")
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
                    # Failed attempts are handled by the acknowledgement gate
                    # above. Successful demo picks can resume selection.
                    if (job.ok and not MENU) or key == ord("r"):
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
            await stop_everything(arm, job, reason=f"connection error: {type(error).__name__}: {error}")
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
        if operator_fault is not None:
            await operator_fault.cancel_check()
        await feed.stop()
        if job is not None and job.task is not None and not job.task.done():
            pending_error = sys.exc_info()[1]
            reason = (f"session interrupted by {type(pending_error).__name__}: {pending_error}"
                      if pending_error is not None else "session closed while action was pending")
            await stop_everything(arm, job, reason=reason)
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
