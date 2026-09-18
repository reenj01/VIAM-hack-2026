"""Gaze-selected pick with a Viam arm.

Laptop webcam -> gaze point on the RealSense feed window -> dwell on a YOLO box
locks that object (frozen snapshot, since the camera rides the arm) -> the
segmenter deprojects it to a world-frame pose ONCE while the arm is still ->
motion service moves the gripper there -> gripper closes.

Keys: Q quit, R release the lock once the grasp attempt has finished.
"""

import asyncio
import time
from typing import Optional

import cv2
import numpy as np

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame

from gaze_lock import Box, GazeLockController, draw_live, draw_locked
from webcam_gaze import (
    BLINK_EAR_THRESHOLD,
    CALIBRATION_PATH,
    GazeCalibration,
    GazeSmoother,
    WebcamGazeTracker,
    estimate_gaze,
    run_calibration,
)

API_KEY = "<from Connect tab>"
API_KEY_ID = "<from Connect tab>"
ADDRESS = "<your-machine-address.viam.cloud>"

CAMERA_NAME = "cam"
DETECTOR_NAME = "yolo-detector"      # viam-labs:vision:yolov8 (yolov8m)
SEGMENTER_NAME = "objects-3d"        # viam:vision:detections-to-segments
MOTION_SERVICE_NAME = "motion"       # if this doesn't resolve, the SDK default name is "builtin"
GRIPPER_NAME = "gripper"

# objects-3d transforms point clouds from the camera frame to the world frame,
# so the poses it returns are already world-frame.
MOTION_REFERENCE_FRAME = "world"

# A 2-finger gripper doesn't need a computed grasp orientation, just a
# consistent approach angle. o_x/o_y/o_z is the orientation vector, theta the
# rotation about it in degrees. (0, 0, -1, 0) points the gripper straight down.
DEFAULT_GRASP_ORIENTATION = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)

APPROACH_HEIGHT_MM = 100.0
GRASP_Z_OFFSET_MM = 0.0
GRIPPER_CLEARANCE_MM = 15.0

WEBCAM_INDEX = 0
RELEASE_AFTER_SECONDS = 4.0   # show the grasp result on the frozen frame, then go live again
WINDOW = "Gaze-selected pick (Q quit, R release lock)"


async def connect():
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    return await RobotClient.at_address(ADDRESS, opts)


def decode_color_frame(images):
    for img in images:
        if "depth" in img.source_name.lower():
            continue
        return cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
    return None


async def grab_frame(cam: Camera, frame_w: Optional[int] = None, frame_h: Optional[int] = None):
    images, _ = await cam.get_images()
    frame = decode_color_frame(images)
    if frame is not None and frame_w is not None and frame.shape[1::-1] != (frame_w, frame_h):
        frame = cv2.resize(frame, (frame_w, frame_h))
    return frame


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


async def grasp_object(motion: MotionClient, gripper: Gripper, point_cloud_obj, job: GraspJob) -> bool:
    cs = object_center_and_size(point_cloud_obj)
    if cs is None:
        job.status = "selected object has no geometry, cannot compute a pose"
        return False
    center, size = cs

    grasp = Pose(x=center.x, y=center.y, z=center.z + GRASP_Z_OFFSET_MM, **DEFAULT_GRASP_ORIENTATION)
    approach = Pose(x=grasp.x, y=grasp.y, z=grasp.z + APPROACH_HEIGHT_MM, **DEFAULT_GRASP_ORIENTATION)

    for label, pose in (("approach", approach), ("grasp", grasp)):
        job.status = f"moving to {label}: x={pose.x:.0f} y={pose.y:.0f} z={pose.z:.0f} mm"
        print(f"[grab] {job.status}")
        # viam-sdk 0.80.0: component_name is the component's plain name (proto string).
        ok = await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame=MOTION_REFERENCE_FRAME, pose=pose),
        )
        if not ok:
            job.status = f"motion.move() failed on {label}"
            return False

    width = min((d for d in size[:2] if d > 0), default=0.0)
    if width > 0:
        target_mm = max(0.0, width - GRIPPER_CLEARANCE_MM)
        job.status = f"object ~{width:.0f}mm wide, closing to {target_mm:.0f}mm"
        print(f"[grab] {job.status}")
        try:
            # TODO verify against the gripper module's do_command docs: the command
            # key and units (the demo gripper's position scale was 0-850, not mm).
            await gripper.do_command({"move_to_position": {"position_mm": target_mm}})
        except Exception as e:
            print(f"[grab] width-based close unavailable ({e}); using grab()")
            await gripper.grab()
    else:
        job.status = "no size estimate, using grab()"
        await gripper.grab()

    job.holding = await gripper.is_holding_something()
    job.status = f"grab complete, holding_something={job.holding}"
    print(f"[grab] {job.status}")
    return bool(job.holding)


async def run_grasp(segmenter: VisionClient, motion: MotionClient, gripper: Gripper,
                    box: Box, job: GraspJob) -> bool:
    try:
        # Freeze the 3D target NOW, before the arm (and the camera on it) moves.
        objs = await segmenter.get_object_point_clouds(CAMERA_NAME)
        match = select_object_for_box(objs, box.label, box.index)
        if match is None:
            job.status = f"no matching 3D object for '{box.label}'"
            print(f"[grab] {job.status}")
            return False
        return await grasp_object(motion, gripper, match, job)
    except Exception as e:
        job.status = f"error: {e}"
        print(f"[grab] {job.status}")
        return False
    finally:
        job.finished_at = time.monotonic()


def load_or_calibrate(gaze: WebcamGazeTracker, frame_w: int, frame_h: int) -> GazeCalibration:
    if CALIBRATION_PATH.exists():
        calib = GazeCalibration.load()
        if (calib.frame_w, calib.frame_h) == (frame_w, frame_h):
            print(f"[main] loaded calibration from {CALIBRATION_PATH}")
            return calib
        print("[main] saved calibration is for a different frame size, recalibrating")
    return run_calibration(gaze, frame_w, frame_h, window_name=WINDOW, keep_window=True)


async def main():
    machine = await connect()
    cam = Camera.from_robot(machine, CAMERA_NAME)
    detector = VisionClient.from_robot(machine, DETECTOR_NAME)
    segmenter = VisionClient.from_robot(machine, SEGMENTER_NAME)
    motion = MotionClient.from_robot(machine, MOTION_SERVICE_NAME)
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)

    gaze = WebcamGazeTracker(camera_index=WEBCAM_INDEX)
    smoother = GazeSmoother()
    lock = GazeLockController()
    job: Optional[GraspJob] = None

    try:
        frame = await grab_frame(cam)
        if frame is None:
            raise RuntimeError(f"Could not get a color frame from camera '{CAMERA_NAME}'")
        frame_h, frame_w = frame.shape[:2]

        # Calibrate in the same AUTOSIZE window the live loop uses, so the gaze
        # mapping is to the exact screen position the feed is shown at.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        calib = load_or_calibrate(gaze, frame_w, frame_h)

        while True:
            if lock.is_locked:
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

            frame = await grab_frame(cam, frame_w, frame_h)
            if frame is None:
                continue
            detections = await detector.get_detections_from_camera(CAMERA_NAME)
            boxes = [box_from_detection(d, i, frame_w, frame_h) for i, d in enumerate(detections)]

            gaze_pt, ear = estimate_gaze(gaze, calib, smoother)
            hovered, progress, locked = lock.update(frame, boxes, gaze_pt)
            if locked is not None:
                print(f"[lock] locked onto '{locked.box.label}' (index {locked.box.index})")
                job = GraspJob()
                job.task = asyncio.create_task(run_grasp(segmenter, motion, gripper, locked.box, job))
                continue

            view = draw_live(frame, boxes, hovered, progress, gaze_pt)
            if gaze_pt is None:
                cv2.putText(view, "No face detected", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 80, 255), 2, cv2.LINE_AA)
            elif ear is not None and ear < BLINK_EAR_THRESHOLD:
                cv2.putText(view, "BLINK", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if job is not None and job.task is not None and not job.task.done():
            print("[main] waiting for the in-progress grasp to finish before exiting...")
            await job.task
        gaze.close()
        cv2.destroyAllWindows()
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
