"""Gaze-select, confirm, then pick and lift an object with the Viam arm.

Run normally for dry-run diagnostics. Add --execute only with the E-stop in
reach. A gaze selection alone never moves the arm: G must confirm the pick.
"""

import asyncio
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
HELPERS = ROOT / "arm-control"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

import cv2
import numpy as np
from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.component.arm import JointPositions
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from gaze_lock import Box, filter_background_boxes
from webcam_gaze import GazeCalibration, GazeEstimator, WebcamGazeTracker, run_calibration


# --- Private connection values. Never put the values themselves in source. ---
load_dotenv(ROOT / ".env")
MACHINE_ADDRESS = os.getenv("MACHINE_ADDRESS") or os.getenv("VIAM_MACHINE_ADDRESS")
API_KEY = os.getenv("API_KEY") or os.getenv("VIAM_API_KEY")
API_KEY_ID = os.getenv("API_KEY_ID") or os.getenv("VIAM_API_KEY_ID")

# --- Exact machine resource names ---
ARM_NAME, GRIPPER_NAME, CAMERA_NAME = "arm", "gripper", "cam"
DETECTOR_NAME, MOTION_NAME = "vision-1", "motion"
OBSERVE_JOINTS = [294.952, -60.135, -19.090, 0.009, 79.179, 165.257]

# Geometry values are in millimeters. Object height is supplied by hand; this
# script never uses point-cloud segmentation.
TABLE_SURFACE_Z_MM = -123.0
DEFAULT_OBJECT_HEIGHT_MM = 100.0
OBJECT_DIMENSIONS_MM = {
    # Measured values supplied in inches, converted with 1 in = 25.4 mm.
    "coke can": {"width": 63.5, "depth": 63.5, "height": 120.65},
    "cokecan": {"width": 63.5, "depth": 63.5, "height": 120.65},
    "juicebox": {"width": 63.5, "depth": 60.325, "height": 152.4},
    "juice box": {"width": 63.5, "depth": 60.325, "height": 152.4},
    "black block": {"width": 28.575, "depth": 28.575, "height": 60.325},
    "blackblock": {"width": 28.575, "depth": 28.575, "height": 60.325},
}
GRASP_HEIGHT_FRACTION = 0.60
APPROACH_STANDOFF_MM = 120.0
LIFT_MM = 120.0
MIN_TARGET_RADIUS_MM = 200.0
MAX_TARGET_RADIUS_MM = 600.0

SCENE_REFRESH_SECONDS = 0.30
GAZE_DWELL_SECONDS = 2.0
WEBCAM_INDEX = 0
WINDOW = "Gaze pick — G confirms | C recalibrate | Q quits"
# Real pick motions run after the G confirmation. Use --dry-run for testing.
DRY_RUN = "--dry-run" in sys.argv
SKIP_CALIBRATION = "--skip-calibration" in sys.argv
MOVE_TO_OBSERVE = "--observe" in sys.argv


async def connect() -> RobotClient:
    if not all((MACHINE_ADDRESS, API_KEY, API_KEY_ID)):
        raise SystemExit("Missing MACHINE_ADDRESS, API_KEY, or API_KEY_ID in .env")
    options = RobotClient.Options.with_api_key(
        api_key=API_KEY, api_key_id=API_KEY_ID,
        check_connection_interval=0, attempt_reconnect_interval=0,
    )
    return await RobotClient.at_address(MACHINE_ADDRESS, options)


# --------------------------------------------------------------------------- cached camera scene

@dataclass
class Observation:
    frame: np.ndarray                 # native camera-image pixels
    boxes: list[Box]                  # native camera-image pixels


@dataclass
class DisplayTransform:
    """Explicit image <-> virtual fullscreen-window coordinate mapping."""
    image_w: int
    image_h: int
    window_w: int
    window_h: int
    scale: float
    left: float
    top: float

    @classmethod
    def fit(cls, iw: int, ih: int, ww: int, wh: int):
        scale = min(ww / iw, wh / ih)
        return cls(iw, ih, ww, wh, scale, (ww - iw * scale) / 2, (wh - ih * scale) / 2)

    def image_to_window(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.left, y * self.scale + self.top

    def window_to_image(self, x: float, y: float) -> Optional[tuple[float, float]]:
        ix, iy = (x - self.left) / self.scale, (y - self.top) / self.scale
        return (ix, iy) if 0 <= ix < self.image_w and 0 <= iy < self.image_h else None

    def render(self, frame: np.ndarray) -> np.ndarray:
        canvas = np.zeros((self.window_h, self.window_w, 3), np.uint8)
        w, h = round(self.image_w * self.scale), round(self.image_h * self.scale)
        x, y = round(self.left), round(self.top)
        canvas[y:y + h, x:x + w] = cv2.resize(frame, (w, h))
        return canvas


def box_from_detection(detection, index: int, width: int, height: int) -> Box:
    if detection.x_max_normalized or detection.y_max_normalized:
        x0, y0 = detection.x_min_normalized * width, detection.y_min_normalized * height
        x1, y1 = detection.x_max_normalized * width, detection.y_max_normalized * height
    else:
        x0, y0, x1, y1 = detection.x_min, detection.y_min, detection.x_max, detection.y_max
    return Box(int(x0), int(y0), int(x1), int(y1), detection.class_name,
               float(detection.confidence), index)


def decode_color(images) -> Optional[np.ndarray]:
    for image in images:
        if "depth" not in image.name.lower():
            frame = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
    return None


class RobotFeed:
    """Fetches the scene a few times per second; display loop uses the cache."""

    def __init__(self, camera: Camera, detector: VisionClient):
        self.camera, self.detector = camera, detector
        self.latest: Optional[Observation] = None
        self.paused = False
        self._task: Optional[asyncio.Task] = None

    async def capture(self) -> Observation:
        images, _ = await self.camera.get_images()
        frame = decode_color(images)
        if frame is None:
            raise RuntimeError("cam returned no color image")
        h, w = frame.shape[:2]
        detections = await self.detector.get_detections_from_camera(CAMERA_NAME)
        boxes = filter_background_boxes([box_from_detection(d, i, w, h) for i, d in enumerate(detections)])
        return Observation(frame, boxes)

    async def first(self) -> Observation:
        self.latest = await self.capture()
        return self.latest

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            if not self.paused:
                try:
                    self.latest = await self.capture()
                except Exception as exc:
                    print(f"[scene] capture failed: {exc}")
            await asyncio.sleep(SCENE_REFRESH_SECONDS)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


class DirectBoxDwell:
    """Select only when the gaze remains inside one actual detection box.

    Unlike the old soft-evidence selector, nearby boxes cannot steal enough
    score to prevent a selection. Brief missed detections and blinks preserve
    the timer; looking outside the box for longer resets it.
    """

    def __init__(self, dwell_seconds: float, grace_seconds: float = 0.35):
        self.dwell_seconds, self.grace_seconds = dwell_seconds, grace_seconds
        self.box: Optional[Box] = None
        self.elapsed = 0.0
        self.last_tick: Optional[float] = None
        self.last_hit: Optional[float] = None

    def reset(self) -> None:
        self.box, self.elapsed, self.last_tick, self.last_hit = None, 0.0, None, None

    def hold(self) -> None:
        """Pause time during a blink, rather than counting it as attention."""
        if self.box is not None:
            self.last_tick = time.monotonic()

    @staticmethod
    def _same_object(old: Box, new: Box) -> bool:
        if old.label != new.label:
            return False
        ix0, iy0 = max(old.x0, new.x0), max(old.y0, new.y0)
        ix1, iy1 = min(old.x1, new.x1), min(old.y1, new.y1)
        intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = old.area + new.area - intersection
        return union > 0 and intersection / union >= 0.25

    def update(self, boxes: list[Box], gaze_image: Optional[tuple[float, float]]):
        now = time.monotonic()
        hit = None
        if gaze_image is not None:
            inside = [box for box in boxes if box.contains(*gaze_image)]
            hit = min(inside, key=lambda box: box.area) if inside else None
        if hit is None:
            if self.box is not None and self.last_hit is not None and now - self.last_hit <= self.grace_seconds:
                self.last_tick = now
                return self.box, min(1.0, self.elapsed / self.dwell_seconds), None
            self.reset()
            return None, 0.0, None
        if self.box is None or not self._same_object(self.box, hit):
            self.box, self.elapsed, self.last_tick, self.last_hit = hit, 0.0, now, now
            return hit, 0.0, None
        self.elapsed += min(0.15, now - (self.last_tick or now))
        self.box, self.last_tick, self.last_hit = hit, now, now
        progress = min(1.0, self.elapsed / self.dwell_seconds)
        return hit, progress, hit if progress >= 1.0 else None


def draw_scene(obs: Observation, transform: DisplayTransform, gaze: Optional[tuple[float, float]],
               leader: Optional[Box], progress: float) -> np.ndarray:
    canvas = transform.render(obs.frame)
    for box in obs.boxes:
        x0, y0 = transform.image_to_window(box.x0, box.y0)
        x1, y1 = transform.image_to_window(box.x1, box.y1)
        color = (0, 220, 0) if leader and leader.key == box.key else (160, 160, 160)
        cv2.rectangle(canvas, (round(x0), round(y0)), (round(x1), round(y1)), color, 3)
        cv2.putText(canvas, f"{box.label} {box.confidence:.2f}", (round(x0), max(22, round(y0) - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    if gaze:
        gx, gy = round(gaze[0]), round(gaze[1])
        cv2.circle(canvas, (gx, gy), 9, (0, 0, 255), -1)
        cv2.circle(canvas, (gx, gy), 15, (255, 255, 255), 2)
        if leader:
            cv2.ellipse(canvas, (gx, gy), (24, 24), -90, 0, 360 * progress, (0, 255, 255), 3)
    return canvas


# --------------------------------------------------------------------------- 3-D association and arm motion

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


async def get_intrinsics(camera: Camera) -> Optional[Intrinsics]:
    try:
        p = (await camera.get_properties()).intrinsic_parameters
        if p and p.focal_x_px and p.width_px:
            return Intrinsics(p.focal_x_px, p.focal_y_px, p.center_x_px, p.center_y_px, p.width_px, p.height_px)
    except Exception as exc:
        print(f"[3d] no intrinsics: {exc}")
    return None


async def transformed(robot: RobotClient, pose: Pose, source: str, destination: str) -> Pose:
    if source == destination:
        return pose
    return (await robot.transform_pose(PoseInFrame(reference_frame=source, pose=pose), destination)).pose


def dimensions_for(label: str) -> dict[str, float]:
    normalized = label.casefold().replace("_", " ").replace("-", " ")
    dimensions = OBJECT_DIMENSIONS_MM.get(normalized)
    if dimensions is None:
        print(f"[pick] WARNING: no manual dimensions for {label!r}; using default "
              f"height {DEFAULT_OBJECT_HEIGHT_MM:.0f} mm")
        return {"width": 60.0, "depth": 60.0, "height": DEFAULT_OBJECT_HEIGHT_MM}
    return dimensions


async def bottom_center_on_table(robot: RobotClient, selected: Box, intr: Intrinsics,
                                 image_w: int, image_h: int) -> tuple[float, float, tuple[float, float]]:
    """Back-project a box bottom-center and intersect its world ray with table Z."""
    pixel = ((selected.x0 + selected.x1) / 2, float(selected.y1))
    u, v = pixel[0] * intr.width / image_w, pixel[1] * intr.height / image_h
    ray_x, ray_y = (u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy
    origin = await transformed(robot, Pose(x=0, y=0, z=0, o_z=1), CAMERA_NAME, "world")
    far = await transformed(robot, Pose(x=ray_x * 1000, y=ray_y * 1000, z=1000, o_z=1), CAMERA_NAME, "world")
    dx, dy, dz = far.x - origin.x, far.y - origin.y, far.z - origin.z
    if abs(dz) < 1e-6:
        raise RuntimeError("camera ray is parallel to the table plane")
    distance = (TABLE_SURFACE_Z_MM - origin.z) / dz
    if distance <= 0:
        raise RuntimeError("table-plane intersection is behind cam; check cam frame orientation")
    return origin.x + distance * dx, origin.y + distance * dy, pixel


def top_down(x: float, y: float, z: float) -> Pose:
    return Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0)


def tilted_top_down(x: float, y: float, z: float) -> Pose:
    """15-degree pitch fallback for a strict top-down self-collision."""
    return Pose(x=x, y=y, z=z, o_x=0, o_y=1, o_z=0, theta=15)


async def arm_stop(arm: Arm) -> None:
    if not DRY_RUN:
        try:
            await asyncio.wait_for(arm.stop(), timeout=2)
        except Exception as exc:
            print(f"[safety] arm.stop failed: {exc}; use physical E-stop")


async def move(motion: MotionClient, target: Pose, label: str, linear: bool = False) -> bool:
    destination = PoseInFrame(reference_frame="world", pose=target)
    print(f"[pick] {'DRY RUN: would ' if DRY_RUN else ''}{label}: world "
          f"({target.x:.1f}, {target.y:.1f}, {target.z:.1f}), "
          f"o=({target.o_x:.2f},{target.o_y:.2f},{target.o_z:.2f},{target.theta:.1f})")
    if DRY_RUN:
        await asyncio.sleep(0.2)
        return True
    constraints = Constraints(linear_constraint=[LinearConstraint()]) if linear else None
    try:
        ok = await motion.move(component_name=GRIPPER_NAME, destination=destination, constraints=constraints)
        print(f"[pick] {label} result: {ok}")
    except Exception as exc:
        if not linear:
            raise
        print(f"[pick] linear {label} infeasible ({exc}); retrying free")
        ok = await motion.move(component_name=GRIPPER_NAME, destination=destination)
        print(f"[pick] free retry {label} result: {ok}")
    if not ok and linear:
        print(f"[pick] linear {label} returned false; retrying free")
        ok = await motion.move(component_name=GRIPPER_NAME, destination=destination)
        print(f"[pick] free retry {label} result: {ok}")
    return bool(ok)


class PickJob:
    def __init__(self):
        self.status, self.task, self.done, self.success = "waiting", None, False, False


async def pick(robot: RobotClient, motion: MotionClient, gripper: Gripper,
               selected: Box, intr: Optional[Intrinsics], image_w: int, image_h: int, job: PickJob) -> None:
    try:
        if intr is None:
            raise RuntimeError("cam intrinsics unavailable; cannot calculate a table-plane intersection")
        job.status = f"locating {selected.label} on table"
        x, y, pixel = await bottom_center_on_table(robot, selected, intr, image_w, image_h)
        radius = (x * x + y * y) ** 0.5
        print(f"[pick] selected label: {selected.label!r}; bottom-center pixel=({pixel[0]:.1f}, {pixel[1]:.1f})")
        print(f"[pick] table-plane world position: x={x:.1f} y={y:.1f} mm; radius={radius:.1f} mm")
        if not MIN_TARGET_RADIUS_MM <= radius <= MAX_TARGET_RADIUS_MM:
            raise RuntimeError(f"target radius {radius:.1f} mm is outside safe {MIN_TARGET_RADIUS_MM:.0f}-"
                               f"{MAX_TARGET_RADIUS_MM:.0f} mm workspace; no motion sent")
        dims = dimensions_for(selected.label)
        grasp_z = TABLE_SURFACE_Z_MM + dims["height"] * GRASP_HEIGHT_FRACTION
        print(f"[pick] manual dimensions={dims}; grasp z={grasp_z:.1f} mm")
        if not DRY_RUN:
            job.status = "opening gripper"
            await gripper.open()
        job.status = "approaching"
        pose = top_down
        try:
            if not await move(motion, pose(x, y, grasp_z + APPROACH_STANDOFF_MM), "free standoff"):
                raise RuntimeError("approach rejected")
        except Exception as exc:
            if "self-collision" not in str(exc).casefold():
                raise
            print("[pick] strict top-down self-collided; retrying once with a 15-degree tilted approach")
            pose = tilted_top_down
            if not await move(motion, pose(x, y, grasp_z + APPROACH_STANDOFF_MM), "tilted free standoff"):
                raise RuntimeError("tilted approach rejected")
        job.status = "descending"
        if not await move(motion, pose(x, y, grasp_z), "linear descent", linear=True):
            raise RuntimeError("descent rejected")
        job.status = "grabbing"
        if not DRY_RUN:
            await gripper.grab()
        job.status = "lifting"
        if not await move(motion, pose(x, y, grasp_z + LIFT_MM), "linear lift", linear=True):
            raise RuntimeError("lift rejected")
        job.status, job.success = f"lifted {selected.label}", True
    except asyncio.CancelledError:
        job.status = "cancelled"
        raise
    except Exception as exc:
        job.status = f"failed: {exc}"
        print(f"[pick] {job.status}")
        traceback.print_exc()
    finally:
        job.done = True


# --------------------------------------------------------------------------- gaze application

def calibration(gaze: WebcamGazeTracker, width: int, height: int) -> GazeCalibration:
    if SKIP_CALIBRATION:
        saved = GazeCalibration.load()
        if saved and (saved.frame_w, saved.frame_h) == (width, height):
            return saved
    return run_calibration(gaze, width, height, window_name=WINDOW, keep_window=True,
                           quick=True, simple_nine_point=True, full_face=True, relaxed_framing=True)


def confirmation_view(snapshot: np.ndarray, transform: DisplayTransform, box: Box, status: str) -> np.ndarray:
    canvas = transform.render(snapshot)
    x0, y0 = transform.image_to_window(box.x0, box.y0)
    x1, y1 = transform.image_to_window(box.x1, box.y1)
    cv2.rectangle(canvas, (round(x0), round(y0)), (round(x1), round(y1)), (0, 220, 0), 4)
    cv2.putText(canvas, f"Selected: {box.label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 220, 0), 2)
    cv2.putText(canvas, "Press G to confirm pick. R returns to selection. Q cancels.",
                (20, 76), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
    cv2.putText(canvas, status, (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .62, (40, 230, 255), 2)
    return canvas


async def main() -> None:
    machine = await connect()
    arm = Arm.from_robot(machine, ARM_NAME)
    camera = Camera.from_robot(machine, CAMERA_NAME)
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    detector = VisionClient.from_robot(machine, DETECTOR_NAME)
    motion = MotionClient.from_robot(machine, MOTION_NAME)
    feed, gaze, job = RobotFeed(camera, detector), WebcamGazeTracker(WEBCAM_INDEX), None
    try:
        print("[main] " + ("DRY RUN: no arm commands" if DRY_RUN else
                           "LIVE MOTION: E-stop must be reachable"))
        if MOVE_TO_OBSERVE and DRY_RUN:
            print(f"[main] DRY RUN: would move to observe joints {OBSERVE_JOINTS}")
        elif MOVE_TO_OBSERVE:
            await arm.move_to_joint_positions(JointPositions(values=OBSERVE_JOINTS))
            await asyncio.sleep(1.0)
        else:
            print("[main] using the arm's current pose; it will not move to the saved observe pose")
        first = await feed.first()
        image_h, image_w = first.frame.shape[:2]
        transform = DisplayTransform.fit(image_w, image_h, image_w, image_h)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        estimator = GazeEstimator(gaze, calibration(gaze, transform.window_w, transform.window_h))
        dwell = DirectBoxDwell(dwell_seconds=GAZE_DWELL_SECONDS)
        intr = await get_intrinsics(camera)
        feed.start()
        pending, snapshot = None, None
        while True:
            if job is not None:
                cv2.imshow(WINDOW, confirmation_view(snapshot, transform, pending, job.status))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    if job.task and not job.task.done(): job.task.cancel()
                    await arm_stop(arm)
                    break
                if job.done:
                    # A completed failure used to disappear immediately,
                    # making it impossible to read the actual Viam error.
                    # Keep it on screen until the user intentionally resets.
                    if key == ord("r"):
                        print(f"[pick] clearing result: {job.status}")
                        job, pending, snapshot = None, None, None
                        dwell.reset(); feed.paused = False
                await asyncio.sleep(.02)
                continue
            if pending is not None:
                feed.paused = True
                cv2.imshow(WINDOW, confirmation_view(snapshot, transform, pending, "No arm motion until G"))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    pending, snapshot = None, None
                    dwell.reset(); feed.paused = False
                elif key == ord("g"):
                    job = PickJob()
                    job.task = asyncio.create_task(pick(machine, motion, gripper, pending, intr, image_w, image_h, job))
                await asyncio.sleep(.02)
                continue
            obs = feed.latest
            if obs is None:
                await asyncio.sleep(.03); continue
            gaze_window, _, blinking = await asyncio.to_thread(estimator.read)
            # Inverse transform is deliberate: hit tests always use image-space boxes.
            gaze_image = transform.window_to_image(*gaze_window) if gaze_window else None
            if blinking:
                dwell.hold(); leader, progress, locked = dwell.box, 0.0, None
            else:
                leader, progress, locked = dwell.update(obs.boxes, gaze_image)
            if locked:
                pending, snapshot = locked, obs.frame.copy()
                feed.paused = True
                print(f"[gaze] selected {pending.label!r}; press G to confirm")
                continue
            view = draw_scene(obs, transform, gaze_window, leader, progress)
            if not obs.boxes:
                cv2.putText(view, "No objects detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 180, 255), 2)
            elif gaze_window is None:
                cv2.putText(view, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 180, 255), 2)
            elif leader is not None:
                cv2.putText(view, f"Looking at {leader.label}: {progress:.1f}/{GAZE_DWELL_SECONDS:.1f}s",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("c"):
                estimator = GazeEstimator(gaze, calibration(gaze, transform.window_w, transform.window_h))
                dwell.reset()
    finally:
        if job and job.task and not job.task.done():
            job.task.cancel(); await asyncio.gather(job.task, return_exceptions=True)
        await arm_stop(arm)
        await feed.stop(); gaze.close(); cv2.destroyAllWindows(); await machine.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted. If the arm is moving, use the physical E-stop.")
