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

from gaze_lock import Box, GazeLockController, filter_background_boxes
from webcam_gaze import GazeCalibration, GazeEstimator, WebcamGazeTracker, run_calibration


# --- Private connection values. Never put the values themselves in source. ---
load_dotenv(ROOT / ".env")
MACHINE_ADDRESS = os.getenv("MACHINE_ADDRESS") or os.getenv("VIAM_MACHINE_ADDRESS")
API_KEY = os.getenv("API_KEY") or os.getenv("VIAM_API_KEY")
API_KEY_ID = os.getenv("API_KEY_ID") or os.getenv("VIAM_API_KEY_ID")

# --- Exact machine resource names ---
ARM_NAME, GRIPPER_NAME, CAMERA_NAME = "arm", "gripper", "cam"
DETECTOR_NAME, SEGMENTER_NAME, MOTION_NAME = "vision-1", "objects-3d", "motion"
OBSERVE_JOINTS = [294.952, -60.135, -19.090, 0.009, 79.179, 165.257]

# Geometry values are in millimeters. Tune object height only after a safe hover.
TABLE_SURFACE_Z_MM = -123.0
ESTIMATED_OBJECT_HEIGHT_MM = 50.0
GRASP_HEIGHT_OFFSET_MM = 0.0
APPROACH_STANDOFF_MM = 120.0
LIFT_MM = 120.0

SCENE_REFRESH_SECONDS = 0.30
GAZE_DWELL_SECONDS = 2.0
WEBCAM_INDEX = 0
WINDOW = "Gaze pick — G confirms | C recalibrate | Q quits"
EXECUTE = "--execute" in sys.argv
DRY_RUN = not EXECUTE
SKIP_CALIBRATION = "--skip-calibration" in sys.argv


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


def segment_label(segment) -> str:
    return next((g.label for g in segment.geometries.geometries if g.label), "")


def project(point: Pose, intr: Intrinsics, image_w: int, image_h: int) -> Optional[tuple[float, float]]:
    # Geometry.center is millimeters. Raw point-cloud bytes are meters, but we
    # deliberately never use them, avoiding a 1000x coordinate conversion bug.
    if point.z <= 0:
        return None
    return ((intr.fx * point.x / point.z + intr.cx) * image_w / intr.width,
            (intr.fy * point.y / point.z + intr.cy) * image_h / intr.height)


async def matching_segment(robot: RobotClient, segmenter: VisionClient, selected: Box,
                           intr: Optional[Intrinsics], image_w: int, image_h: int):
    """GetObjectPointClouds, match label, then projected center if repeated."""
    segments = await segmenter.get_object_point_clouds(CAMERA_NAME, timeout=90)
    candidates = [s for s in segments if s.geometries.geometries and segment_label(s) == selected.label]
    if not candidates:
        labels = sorted({segment_label(s) for s in segments if segment_label(s)})
        raise RuntimeError(f"no 3-D segment for {selected.label!r}; got {labels or 'none'}")
    if len(candidates) == 1:
        return candidates[0]
    if intr is None:
        raise RuntimeError(f"cannot distinguish {len(candidates)} {selected.label!r} segments without intrinsics")
    cx, cy = (selected.x0 + selected.x1) / 2, (selected.y0 + selected.y1) / 2
    matches = []
    for segment in candidates:
        geometry = segment.geometries.geometries[0]
        in_camera = await transformed(robot, geometry.center, segment.geometries.reference_frame or ARM_NAME, CAMERA_NAME)
        pixel = project(in_camera, intr, image_w, image_h)
        if pixel and selected.x0 <= pixel[0] <= selected.x1 and selected.y0 <= pixel[1] <= selected.y1:
            matches.append((np.hypot(pixel[0] - cx, pixel[1] - cy), segment))
    if not matches:
        raise RuntimeError("no repeated-label 3-D center projected into the selected box")
    return min(matches, key=lambda pair: pair[0])[1]


def top_down(x: float, y: float, z: float) -> Pose:
    return Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0)


async def arm_stop(arm: Arm) -> None:
    if EXECUTE:
        try:
            await asyncio.wait_for(arm.stop(), timeout=2)
        except Exception as exc:
            print(f"[safety] arm.stop failed: {exc}; use physical E-stop")


async def move(motion: MotionClient, target: Pose, label: str, linear: bool = False) -> bool:
    destination = PoseInFrame(reference_frame=ARM_NAME, pose=target)
    print(f"[pick] {'DRY RUN: would ' if DRY_RUN else ''}{label}: arm ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")
    if DRY_RUN:
        await asyncio.sleep(0.2)
        return True
    constraints = Constraints(linear_constraint=[LinearConstraint()]) if linear else None
    try:
        ok = await motion.move(component_name=GRIPPER_NAME, destination=destination, constraints=constraints)
    except Exception as exc:
        if not linear:
            raise
        print(f"[pick] linear {label} infeasible ({exc}); retrying free")
        ok = await motion.move(component_name=GRIPPER_NAME, destination=destination)
    if not ok and linear:
        print(f"[pick] linear {label} returned false; retrying free")
        ok = await motion.move(component_name=GRIPPER_NAME, destination=destination)
    return bool(ok)


class PickJob:
    def __init__(self):
        self.status, self.task, self.done, self.success = "waiting", None, False, False


async def pick(robot: RobotClient, segmenter: VisionClient, motion: MotionClient, gripper: Gripper,
               selected: Box, intr: Optional[Intrinsics], image_w: int, image_h: int, job: PickJob) -> None:
    try:
        job.status = f"matching 3-D {selected.label}"
        segment = await matching_segment(robot, segmenter, selected, intr, image_w, image_h)
        geometry = segment.geometries.geometries[0]
        center = await transformed(robot, geometry.center, segment.geometries.reference_frame or ARM_NAME, ARM_NAME)
        print(f"[3d] {selected.label}: center in arm frame = ({center.x:.1f}, {center.y:.1f}, {center.z:.1f}) mm")
        # Never use the inflated segment center z for the grasp plane.
        grasp_z = TABLE_SURFACE_Z_MM + ESTIMATED_OBJECT_HEIGHT_MM + GRASP_HEIGHT_OFFSET_MM
        if not DRY_RUN:
            job.status = "opening gripper"
            await gripper.open()
        job.status = "approaching"
        if not await move(motion, top_down(center.x, center.y, grasp_z + APPROACH_STANDOFF_MM), "free standoff"):
            raise RuntimeError("approach rejected")
        job.status = "descending"
        if not await move(motion, top_down(center.x, center.y, grasp_z), "linear descent", linear=True):
            raise RuntimeError("descent rejected")
        job.status = "grabbing"
        if not DRY_RUN:
            await gripper.grab()
        job.status = "lifting"
        if not await move(motion, top_down(center.x, center.y, grasp_z + LIFT_MM), "linear lift", linear=True):
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
    segmenter = VisionClient.from_robot(machine, SEGMENTER_NAME)
    motion = MotionClient.from_robot(machine, MOTION_NAME)
    feed, gaze, job = RobotFeed(camera, detector), WebcamGazeTracker(WEBCAM_INDEX), None
    try:
        print("[main] " + ("EXECUTE: E-stop must be reachable" if EXECUTE else "DRY RUN: no arm commands"))
        if DRY_RUN:
            print(f"[main] DRY RUN: observe joints {OBSERVE_JOINTS}")
        else:
            await arm.move_to_joint_positions(JointPositions(values=OBSERVE_JOINTS))
            await asyncio.sleep(1.0)
        first = await feed.first()
        image_h, image_w = first.frame.shape[:2]
        transform = DisplayTransform.fit(image_w, image_h, image_w, image_h)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        estimator = GazeEstimator(gaze, calibration(gaze, transform.window_w, transform.window_h))
        selector = GazeLockController(select_seconds=GAZE_DWELL_SECONDS)
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
                        selector.release(); feed.paused = False
                await asyncio.sleep(.02)
                continue
            if pending is not None:
                feed.paused = True
                cv2.imshow(WINDOW, confirmation_view(snapshot, transform, pending, "No arm motion until G"))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    pending, snapshot = None, None
                    selector.release(); feed.paused = False
                elif key == ord("g"):
                    job = PickJob()
                    job.task = asyncio.create_task(pick(machine, segmenter, motion, gripper, pending, intr, image_w, image_h, job))
                await asyncio.sleep(.02)
                continue
            obs = feed.latest
            if obs is None:
                await asyncio.sleep(.03); continue
            gaze_window, _, blinking = await asyncio.to_thread(estimator.read)
            # Inverse transform is deliberate: hit tests always use image-space boxes.
            gaze_image = transform.window_to_image(*gaze_window) if gaze_window else None
            if blinking:
                selector.hold(); leader, progress, locked = None, 0.0, None
            else:
                leader, progress, locked = selector.update(obs.frame, obs.boxes, gaze_image)
            if locked:
                pending, snapshot = locked.box, locked.snapshot
                feed.paused = True
                print(f"[gaze] selected {pending.label!r}; press G to confirm")
                continue
            view = draw_scene(obs, transform, gaze_window, leader, progress)
            if not obs.boxes:
                cv2.putText(view, "No objects detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 180, 255), 2)
            elif gaze_window is None:
                cv2.putText(view, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 180, 255), 2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("c"):
                estimator = GazeEstimator(gaze, calibration(gaze, transform.window_w, transform.window_h))
                selector.release()
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
