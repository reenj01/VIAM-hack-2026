"""Gaze-selected pick with a Viam arm.

Laptop webcam -> gaze point on the RealSense feed window -> dwell on a YOLO box
locks that object (frozen snapshot, since the camera rides the arm) -> the
segmenter deprojects it to a world-frame pose ONCE while the arm is still ->
motion service moves the gripper there -> gripper closes.

Keys: Q quit, R release the lock once the grasp attempt has finished, C recalibrate.
Flags: --dry-run (no movement), --skip-calibration (reuse the last calibration),
--detector NAME.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame

from gaze_lock import Box, GazeLockController, draw_live, draw_locked, filter_background_boxes
from webcam_gaze import (
    BLINK_EAR_THRESHOLD,
    CALIBRATION_PATH,
    GazeCalibration,
    GazeSmoother,
    WebcamGazeTracker,
    estimate_gaze,
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
MOTION_SERVICE_NAME = "motion"       # if this doesn't resolve, the SDK default name is "builtin"
GRIPPER_NAME = "gripper"
ARM_NAME = "arm"
SEGMENTER_TIMEOUT_S = 60.0           # get_object_point_clouds can take several seconds

# objects-3d transforms point clouds from the camera frame to the world frame,
# so the poses it returns are already world-frame.
MOTION_REFERENCE_FRAME = "world"

# A 2-finger gripper doesn't need a computed grasp orientation, just a
# consistent approach angle. o_x/o_y/o_z is the orientation vector, theta the
# rotation about it in degrees. (0, 0, -1, 0) points the gripper straight down.
DEFAULT_GRASP_ORIENTATION = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)

# The gripper's frame sits 105 mm out from the end of the arm (crash course,
# "Frames"), so moving the *gripper* to a pose puts the finger center there.
APPROACH_HEIGHT_MM = 100.0
GRASP_Z_OFFSET_MM = 0.0        # + lifts the grasp point above the object's center

# Width-based close (so a paper cup isn't crushed): the uFactory gripper takes a
# position on a 0-850 scale (850 = fully open, ~85 mm), i.e. ~10 units per mm.
# Target = (object width - squeeze) * units/mm; falls back to grab() if the
# do_command isn't accepted.
GRIPPER_SQUEEZE_MM = 8.0
GRIPPER_POS_PER_MM = 10.0
GRIPPER_MAX_POS = 850

# After a grasp attempt, lift back to the approach height and return to the
# pose the gripper was at when the object was locked, so the camera sees the
# table again for the next selection (the deck's arm-position-saver idea).
RETURN_TO_START = True

# Every hackathon machine has table + wall obstacles (erh:vmodutils:obstacle).
# The motion service plans around them; direct arm moves (MoveToPosition /
# MoveToJointPositions) go straight through them, so only ever move via motion.

WEBCAM_INDEX = 0
RELEASE_AFTER_SECONDS = 4.0   # show the grasp result on the frozen frame, then go live again
WINDOW = "Gaze-selected pick (Q quit, R release lock, C recalibrate)"

# `python main.py --dry-run`: full gaze -> lock -> 3D pose flow, prints what it
# would send to the motion service and gripper, moves nothing.
DRY_RUN = "--dry-run" in sys.argv
SKIP_CALIBRATION = "--skip-calibration" in sys.argv


async def connect():
    env = load_env()
    # The SDK's periodic health check has a 1 s deadline; a slow call on the
    # machine (3D segmentation) trips it and the client tears the connection
    # down. Disable the check; failures surface on the call itself instead.
    opts = RobotClient.Options.with_api_key(
        api_key=env["VIAM_API_KEY"], api_key_id=env["VIAM_API_KEY_ID"],
        check_connection_interval=0, attempt_reconnect_interval=0,
    )
    return await RobotClient.at_address(env["VIAM_MACHINE_ADDRESS"], opts)


def decode_color_frame(images):
    """First image from get_images() that decodes as a color picture.

    viam-sdk 0.80.0 returns NamedImage(name, data, mime_type); the depth stream
    is raw bytes that cv2 can't decode, so it's skipped either by name or by
    failing to decode.
    """
    for img in images:
        if "depth" in img.name.lower():
            continue
        frame = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    return None


COLOR_SOURCE_NAME = "color"   # only fetch this from get_images(); the raw depth frame is ~1.8 MB


class RobotFeed:
    """Pulls camera frames and detections in background tasks, each at its own
    pace, so the UI loop runs at webcam rate instead of blocking on a network
    round trip plus a YOLO inference for every frame it draws."""

    def __init__(self, cam: Camera, detector: VisionClient):
        self.cam, self.detector = cam, detector
        self.frame: Optional[np.ndarray] = None
        self.boxes: list[Box] = []
        self.frame_w = self.frame_h = 0
        self.paused = False
        self.frame_fps = 0.0
        self.detect_ms = 0.0
        self._boxes_at = 0.0
        self._tasks: list[asyncio.Task] = []

    async def fetch_frame(self) -> Optional[np.ndarray]:
        try:
            images, _ = await self.cam.get_images(filter_source_names=[COLOR_SOURCE_NAME])
        except Exception:
            images = []
        if not images:
            images, _ = await self.cam.get_images()
        frame = decode_color_frame(images)
        if frame is not None and self.frame_w and frame.shape[1::-1] != (self.frame_w, self.frame_h):
            frame = cv2.resize(frame, (self.frame_w, self.frame_h))
        return frame

    async def first_frame(self) -> tuple[int, int]:
        frame = await self.fetch_frame()
        if frame is None:
            raise RuntimeError(f"Could not get a color frame from camera '{CAMERA_NAME}'")
        self.frame_h, self.frame_w = frame.shape[:2]
        self.frame = frame
        return self.frame_w, self.frame_h

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._frame_loop()), asyncio.create_task(self._detect_loop())]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    @property
    def boxes_age_s(self) -> float:
        return time.monotonic() - self._boxes_at if self._boxes_at else 0.0

    async def _frame_loop(self) -> None:
        last = time.monotonic()
        while True:
            if self.paused:
                await asyncio.sleep(0.05)
                continue
            try:
                frame = await self.fetch_frame()
                if frame is not None:
                    self.frame = frame
                    now = time.monotonic()
                    self.frame_fps = 0.8 * self.frame_fps + 0.2 / max(1e-3, now - last)
                    last = now
            except Exception as e:
                print(f"[feed] camera error: {e}")
                await asyncio.sleep(0.5)

    async def _detect_loop(self) -> None:
        while True:
            if self.paused:
                await asyncio.sleep(0.05)
                continue
            try:
                t0 = time.monotonic()
                dets = await self.detector.get_detections_from_camera(CAMERA_NAME)
                self.detect_ms = (time.monotonic() - t0) * 1000
                self.boxes = filter_background_boxes(
                    [box_from_detection(d, i, self.frame_w, self.frame_h) for i, d in enumerate(dets)])
                self._boxes_at = time.monotonic()
            except Exception as e:
                print(f"[feed] detector error: {e}")
                await asyncio.sleep(0.5)


def box_from_detection(det, index: int, frame_w: int, frame_h: int) -> Box:
    """yolo-detector reports absolute pixels at the camera's native resolution;
    prefer the normalized fields when present so a resized display still lines up."""
    if det.x_max_normalized or det.y_max_normalized:
        x0, y0 = det.x_min_normalized * frame_w, det.y_min_normalized * frame_h
        x1, y1 = det.x_max_normalized * frame_w, det.y_max_normalized * frame_h
    else:
        x0, y0, x1, y1 = det.x_min, det.y_min, det.x_max, det.y_max
    return Box(int(x0), int(y0), int(x1), int(y1), det.class_name, float(det.confidence), index)


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


def select_object_for_box(point_cloud_objects, class_name: str, index: int):
    """objects-3d consumes yolo-detector's output and keeps its label, so match
    on label first; fall back to list order when ambiguous or unlabeled."""
    matches = [o for o in point_cloud_objects if class_name in object_label(o)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[min(index, len(matches) - 1)]
    if 0 <= index < len(point_cloud_objects):
        return point_cloud_objects[index]
    return None


class GraspJob:
    """Status of the background grasp so the display loop can show progress."""

    def __init__(self):
        self.status = "segmenting 3D target..."
        self.task: Optional[asyncio.Task] = None
        self.finished_at: Optional[float] = None
        self.holding: Optional[bool] = None

    @property
    def done(self) -> bool:
        return self.finished_at is not None


async def move_to(motion: MotionClient, pose: Pose, label: str, job: GraspJob) -> bool:
    prefix = "DRY RUN, would be " if DRY_RUN else ""
    job.status = f"{prefix}moving to {label}: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f} mm"
    print(f"[grab] {job.status}")
    if DRY_RUN:
        await asyncio.sleep(0.5)
        return True
    # viam-sdk 0.80.0: component_name is the component's plain name (proto string).
    ok = await motion.move(
        component_name=GRIPPER_NAME,
        destination=PoseInFrame(reference_frame=MOTION_REFERENCE_FRAME, pose=pose),
    )
    if not ok:
        job.status = f"motion.move() failed on {label}"
        print(f"[grab] {job.status}")
    return ok


def gripper_position_for_width(width_mm: float) -> int:
    return int(max(0.0, min(GRIPPER_MAX_POS, (width_mm - GRIPPER_SQUEEZE_MM) * GRIPPER_POS_PER_MM)))


async def close_gripper(gripper: Gripper, arm: Arm, width_mm: float, job: GraspJob) -> None:
    """Close to the object's width via the uFactory module's move_gripper
    do_command (tried on the gripper, then the arm, which is where the module
    documents it); grab() if neither accepts it or there's no size estimate."""
    if width_mm > 0:
        target = gripper_position_for_width(width_mm)
        job.status = f"object ~{width_mm:.0f} mm wide, closing gripper to position {target}"
        print(f"[grab] {job.status}")
        # Command names per viam-modules/viam-ufactory-xarm: the standalone
        # gripper component takes {"set": pos}; the arm takes
        # {"setup_gripper": true, "move_gripper": pos}.
        attempts = (
            (gripper, "gripper", {"set": target}),
            (arm, "arm", {"setup_gripper": True, "move_gripper": target}),
        )
        for who, label, cmd in attempts:
            try:
                await who.do_command(cmd)
                return
            except Exception as e:
                print(f"[grab] {cmd} on {label} not accepted ({type(e).__name__}: {e})")
        print("[grab] width-based close unavailable; using grab()")
    else:
        job.status = "no size estimate, using grab()"
        print(f"[grab] {job.status}")
    await gripper.grab()


async def grasp_object(motion: MotionClient, gripper: Gripper, arm: Arm, point_cloud_obj, job: GraspJob) -> bool:
    cs = object_center_and_size(point_cloud_obj)
    if cs is None:
        job.status = "selected object has no geometry, cannot compute a pose"
        return False
    center, size = cs

    grasp = Pose(x=center.x, y=center.y, z=center.z + GRASP_Z_OFFSET_MM, **DEFAULT_GRASP_ORIENTATION)
    approach = Pose(x=grasp.x, y=grasp.y, z=grasp.z + APPROACH_HEIGHT_MM, **DEFAULT_GRASP_ORIENTATION)

    start = None
    if RETURN_TO_START and not DRY_RUN:
        start = (await motion.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)).pose

    if not await move_to(motion, approach, "approach", job):
        return False
    if not await move_to(motion, grasp, "grasp", job):
        if start is not None:
            await move_to(motion, start, "start (retreat)", job)
        return False

    width = min((d for d in size[:2] if d > 0), default=0.0)
    if DRY_RUN:
        job.status = (f"DRY RUN: would close gripper to position {gripper_position_for_width(width)} "
                      f"(object ~{width:.0f} mm wide)" if width > 0 else "DRY RUN: would call grab()")
        print(f"[grab] {job.status}")
        return True
    await close_gripper(gripper, arm, width, job)

    # is_holding_something() returns a HoldingStatus (truthy even when False);
    # its meta carries the gripper position on the 0-850 scale.
    status = await gripper.is_holding_something()
    job.holding = bool(getattr(status, "is_holding_something", status))
    print(f"[grab] grab complete, holding_something={job.holding} "
          f"(gripper position={getattr(status, 'meta', {}).get('position', '?')})")

    if start is not None:
        await move_to(motion, approach, "approach (lift)", job)
        await move_to(motion, start, "start", job)

    job.status = f"done, holding_something={job.holding}"
    print(f"[grab] {job.status}")
    return bool(job.holding)


async def run_grasp(segmenter: VisionClient, motion: MotionClient, gripper: Gripper, arm: Arm,
                    box: Box, job: GraspJob) -> bool:
    try:
        # Freeze the 3D target NOW, before the arm (and the camera on it) moves.
        objs = await segmenter.get_object_point_clouds(CAMERA_NAME, timeout=SEGMENTER_TIMEOUT_S)
        match = select_object_for_box(objs, box.label, box.index)
        if match is None:
            job.status = f"no matching 3D object for '{box.label}'"
            print(f"[grab] {job.status}")
            return False
        return await grasp_object(motion, gripper, arm, match, job)
    except Exception as e:
        job.status = f"error: {e}"
        print(f"[grab] {job.status}")
        return False
    finally:
        job.finished_at = time.monotonic()


def resolve_motion_name(machine) -> str:
    """Check the configured names against the machine; the built-in motion
    service is usually named 'builtin', so fall back to that."""
    have = {r.name for r in machine.resource_names}
    missing = [n for n in (CAMERA_NAME, DETECTOR_NAME, SEGMENTER_NAME, GRIPPER_NAME) if n not in have]
    if missing:
        raise SystemExit(f"Not on this machine: {missing}. Resources found: {sorted(have)}. "
                         "Fix the names at the top of main.py.")
    for name in (MOTION_SERVICE_NAME, "builtin"):
        if name in have:
            return name
    raise SystemExit(f"No motion service named '{MOTION_SERVICE_NAME}' or 'builtin'. "
                     f"Resources found: {sorted(have)}")


def load_or_calibrate(gaze: WebcamGazeTracker, frame_w: int, frame_h: int) -> GazeCalibration:
    """Calibrate every run, like gaze_dot.py: a saved calibration from a
    different sitting position is what makes the gaze feel off.
    `--skip-calibration` reuses the last one; C recalibrates mid-run."""
    if SKIP_CALIBRATION and CALIBRATION_PATH.exists():
        calib = GazeCalibration.load()
        if (calib.frame_w, calib.frame_h) == (frame_w, frame_h):
            print(f"[main] --skip-calibration: reusing {CALIBRATION_PATH}")
            return calib
        print("[main] saved calibration is for a different frame size, recalibrating")
    return run_calibration(gaze, frame_w, frame_h, window_name=WINDOW, keep_window=True)


async def main():
    machine = await connect()
    motion_name = resolve_motion_name(machine)
    print(f"[main] using motion service '{motion_name}'" + (" (DRY RUN: no movement)" if DRY_RUN else ""))
    cam = Camera.from_robot(machine, CAMERA_NAME)
    detector = VisionClient.from_robot(machine, DETECTOR_NAME)
    segmenter = VisionClient.from_robot(machine, SEGMENTER_NAME)
    motion = MotionClient.from_robot(machine, motion_name)
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    arm = Arm.from_robot(machine, ARM_NAME)

    gaze = WebcamGazeTracker(camera_index=WEBCAM_INDEX)
    smoother = GazeSmoother()
    lock = GazeLockController()
    feed = RobotFeed(cam, detector)
    job: Optional[GraspJob] = None

    try:
        frame_w, frame_h = await feed.first_frame()

        # Calibrate in the same AUTOSIZE window the live loop uses, so the gaze
        # mapping is to the exact screen position the feed is shown at. The
        # background fetching starts afterwards: calibration blocks the event
        # loop for ~25 s and shouldn't leave robot calls hanging mid-flight.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        calib = load_or_calibrate(gaze, frame_w, frame_h)
        feed.start()

        while True:
            if lock.is_locked:
                feed.paused = True  # don't compete with segmentation/planning for the machine's CPU
                lines = [job.status] if job else []
                lines.append("Q quit" + ("  |  R release" if job and job.done else "  |  grasp in progress..."))
                cv2.imshow(WINDOW, draw_locked(lock.locked, lines))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if job is not None and job.done:
                    if key == ord("r") or time.monotonic() - job.finished_at > RELEASE_AFTER_SECONDS:
                        lock.release()
                        smoother.reset()
                        job = None
                await asyncio.sleep(0.03)  # let the grasp task run
                continue
            feed.paused = False

            frame, boxes = feed.frame, feed.boxes
            # Webcam read + MediaPipe take ~30 ms; run them off the event loop so
            # the feed tasks keep receiving data meanwhile.
            gaze_pt, ear = await asyncio.to_thread(estimate_gaze, gaze, calib, smoother)
            hovered, progress, locked = lock.update(frame, boxes, gaze_pt)
            if locked is not None:
                feed.paused = True
                print(f"[lock] locked onto '{locked.box.label}' (index {locked.box.index})")
                job = GraspJob()
                job.task = asyncio.create_task(run_grasp(segmenter, motion, gripper, arm, locked.box, job))
                continue

            view = draw_live(frame, boxes, hovered, progress, gaze_pt)
            if gaze_pt is None:
                cv2.putText(view, "No face detected", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 80, 255), 2, cv2.LINE_AA)
            elif ear is not None and ear < BLINK_EAR_THRESHOLD:
                cv2.putText(view, "BLINK", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
            hud = (f"camera {feed.frame_fps:.0f} fps | detector {feed.detect_ms:.0f} ms "
                   f"({feed.boxes_age_s:.1f}s old) | {len(boxes)} boxes")
            cv2.putText(view, hud, (16, frame_h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                feed.paused = True
                calib = run_calibration(gaze, frame_w, frame_h, window_name=WINDOW, keep_window=True)
                smoother.reset()
    finally:
        await feed.stop()
        if job is not None and job.task is not None and not job.task.done():
            print("[main] waiting for the in-progress grasp to finish before exiting...")
            await job.task
        gaze.close()
        cv2.destroyAllWindows()
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
