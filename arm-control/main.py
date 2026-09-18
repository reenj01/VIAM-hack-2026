import asyncio
import time

import cv2
import numpy as np

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame

from webcam_gaze import (
    WebcamGazeTracker,
    GazeCalibration,
    DwellSelector,
    CALIBRATION_PATH,
    run_calibration,
)

API_KEY = "<from Connect tab>"
API_KEY_ID = "<from Connect tab>"
ADDRESS = "<your-machine-address.viam.cloud>"

# --- Fill these in with the exact names configured in the Viam app -----------
CAMERA_NAME = "cam"
VISION_SERVICE_NAME = "vision-1"       # the YOLO detector + segmenter vision service
GRIPPER_NAME = "gripper"
MOTION_REFERENCE_FRAME = "world"       # frame the target pose is expressed in
# Fixed approach orientation for the gripper (see design discussion: a 2-finger
# gripper doesn't need a computed grasp orientation, just a consistent approach
# angle). o_x/o_y/o_z is the orientation vector, theta is rotation about it, in
# degrees. (0, 0, -1, 0) points the gripper straight down. Adjust for your rig.
DEFAULT_GRASP_ORIENTATION = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)

WEBCAM_INDEX = 0
GAZE_COOLDOWN_SECONDS = 3.0  # after acting on a selection, ignore new ones briefly


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=API_KEY, api_key_id=API_KEY_ID
    )
    return await RobotClient.at_address(ADDRESS, opts)


def decode_color_frame(images):
    for img in images:
        if "depth" in img.source_name.lower():
            continue
        return cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
    return None


def box_from_detection(det, frame_w, frame_h) -> tuple[int, int, int, int]:
    """Detections carry both pixel and normalized coords; prefer normalized
    since the vision service may run inference at a different resolution than
    the frame we're displaying."""
    if det.x_max_normalized or det.y_max_normalized:
        return (
            int(det.x_min_normalized * frame_w), int(det.y_min_normalized * frame_h),
            int(det.x_max_normalized * frame_w), int(det.y_max_normalized * frame_h),
        )
    return int(det.x_min), int(det.y_min), int(det.x_max), int(det.y_max)


def find_matching_point_cloud_object(point_cloud_objects, class_name: str, index: int):
    """Match a 2D detection to its deprojected 3D point-cloud object.

    TODO: verify against your configured segmenter. Viam's "detection to
    segments" pattern typically returns one PointCloudObject per detection,
    either in the same order as get_detections_from_camera() or labeled with
    the detector's class name on the geometry. This tries label matching
    first and falls back to positional matching.
    """
    for obj in point_cloud_objects:
        for geom in obj.geometries.geometries:
            if geom.label == class_name:
                return obj
    if 0 <= index < len(point_cloud_objects):
        return point_cloud_objects[index]
    return None


async def move_gripper_to_object(motion: MotionClient, point_cloud_obj) -> bool:
    geometries = point_cloud_obj.geometries.geometries
    if not geometries:
        print("[grab] selected object has no geometry, cannot compute a pose")
        return False
    center = geometries[0].center

    target_pose = Pose(
        x=center.x, y=center.y, z=center.z,
        **DEFAULT_GRASP_ORIENTATION,
    )
    destination = PoseInFrame(reference_frame=MOTION_REFERENCE_FRAME, pose=target_pose)

    print(f"[grab] moving gripper to x={center.x:.1f} y={center.y:.1f} z={center.z:.1f} (mm)")
    return await motion.move(component_name=GRIPPER_NAME, destination=destination)


async def main():
    machine = await connect()
    cam = Camera.from_robot(machine, CAMERA_NAME)
    vision = VisionClient.from_robot(machine, VISION_SERVICE_NAME)
    motion = MotionClient.from_robot(machine, "builtin")
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)

    gaze = WebcamGazeTracker(camera_index=WEBCAM_INDEX)
    dwell = DwellSelector()
    last_action_at = 0.0

    try:
        # Prime one RealSense frame to know the display resolution, then
        # calibrate gaze directly against that same pixel space.
        images, _ = await cam.get_images()
        frame = decode_color_frame(images)
        if frame is None:
            raise RuntimeError(f"Could not get a color frame from camera '{CAMERA_NAME}'")
        frame_h, frame_w = frame.shape[:2]

        if CALIBRATION_PATH.exists():
            calib = GazeCalibration.load()
            if (calib.frame_w, calib.frame_h) != (frame_w, frame_h):
                print("[main] saved calibration is for a different frame size, recalibrating")
                calib = run_calibration(gaze, frame_w, frame_h)
        else:
            calib = run_calibration(gaze, frame_w, frame_h)

        window = "Gaze-selected pick (Q to quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, frame_w, frame_h)

        while True:
            images, _ = await cam.get_images()
            frame = decode_color_frame(images)
            if frame is None:
                continue
            frame = cv2.resize(frame, (frame_w, frame_h))

            detections = await vision.get_detections_from_camera(CAMERA_NAME)

            _, _, gaze_feats, ear = gaze.read()
            hovered_index = None

            for i, det in enumerate(detections):
                x0, y0, x1, y1 = box_from_detection(det, frame_w, frame_h)
                inside = False
                if gaze_feats is not None:
                    gx, gy = calib.predict(gaze_feats)
                    inside = x0 <= gx <= x1 and y0 <= gy <= y1
                    if inside:
                        hovered_index = i
                color = (0, 220, 0) if inside else (100, 100, 100)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}",
                            (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if gaze_feats is not None:
                gx, gy = calib.predict(gaze_feats)
                cv2.circle(frame, (int(gx), int(gy)), 8, (0, 0, 255), -1)

            hovered_label = f"{hovered_index}:{detections[hovered_index].class_name}" if hovered_index is not None else None
            selected, progress = dwell.update(hovered_label)
            if hovered_index is not None:
                gx, gy = calib.predict(gaze_feats)
                cv2.ellipse(frame, (int(gx), int(gy)), (20, 20), -90, 0, 360 * progress,
                            (0, 255, 255), 3)

            now = time.monotonic()
            if selected is not None and hovered_index is not None and now - last_action_at > GAZE_COOLDOWN_SECONDS:
                last_action_at = now
                chosen = detections[hovered_index]
                print(f"[select] gaze-selected '{chosen.class_name}' (index {hovered_index})")
                point_cloud_objects = await vision.get_object_point_clouds(CAMERA_NAME)
                match = find_matching_point_cloud_object(point_cloud_objects, chosen.class_name, hovered_index)
                if match is None:
                    print("[grab] could not find a matching 3D object for this detection")
                else:
                    moved = await move_gripper_to_object(motion, match)
                    if moved:
                        await gripper.grab()
                        holding = await gripper.is_holding_something()
                        print(f"[grab] grab complete, holding_something={holding}")
                    else:
                        print("[grab] motion.move() reported failure")

            if ear is not None and ear < 0.17:
                cv2.putText(frame, "BLINK", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        gaze.close()
        cv2.destroyAllWindows()
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
