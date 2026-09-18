"""Fullscreen gaze selection for one frozen Viam camera snapshot.

This program reads a Viam camera image and YOLO detections after calibration.
It does not send any arm or gripper commands. Put the arm at its safe observe
pose manually before starting it, then press N to capture a fresh scene.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

from gaze_dot import (
    CALIBRATION_SECONDS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    FRAME_HOLD_SECONDS,
    FRAME_MESSAGES,
    FRAME_POSITION_TOLERANCE,
    FRAME_SIZE_RATIO_HIGH,
    FRAME_SIZE_RATIO_LOW,
    FRAME_TARGET_CX,
    FRAME_TARGET_CY,
    FRAME_TARGET_H,
    FRAME_TARGET_W,
    LEFT_EYE_CORNERS,
    LEFT_IRIS,
    MIN_SAMPLES_PER_POINT,
    MODEL_PATH,
    RIGHT_EYE_CORNERS,
    RIGHT_IRIS,
    SETTLE_SECONDS,
    TARGETS,
    draw_id_frame,
    evaluate_framing,
    face_bbox_normalized,
    fit_calibration,
    gaze_features,
)


WINDOW_NAME = "Viam gaze selection"
DOT_RADIUS = 14
SMOOTHING = 0.80
BOX_EDGE_MARGIN_PX = 15


@dataclass(frozen=True)
class DisplayLayout:
    """Exact mapping between camera-image pixels and the displayed canvas."""

    scale: float
    offset_x: int
    offset_y: int
    image_width: int
    image_height: int

    def image_to_canvas(self, x: float, y: float) -> tuple[int, int]:
        return (
            round(self.offset_x + x * self.scale),
            round(self.offset_y + y * self.scale),
        )

    def canvas_to_image(self, x: float, y: float) -> tuple[float, float] | None:
        if not (
            self.offset_x <= x < self.offset_x + self.image_width * self.scale
            and self.offset_y <= y < self.offset_y + self.image_height * self.scale
        ):
            return None
        return ((x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale)


@dataclass
class SceneSnapshot:
    image: np.ndarray
    detections: list
    object_point_clouds: list
    captured_at: float


def decode_viam_image(named_image) -> np.ndarray:
    """Decode the JPEG/PNG bytes returned by Viam into an OpenCV BGR image."""
    image = cv2.imdecode(np.frombuffer(named_image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Viam returned a camera image that OpenCV could not decode.")
    return image


async def connect() -> RobotClient:
    load_dotenv()
    required = ("VIAM_MACHINE_ADDRESS", "VIAM_API_KEY", "VIAM_API_KEY_ID")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} in .env. Copy .env.example first.")
    options = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], options)


async def capture_snapshot(detector: VisionClient, camera_name: str) -> SceneSnapshot:
    """Capture the display image and boxes without transferring bulky PCD data."""
    result = await detector.capture_all_from_camera(
        camera_name,
        return_image=True,
        return_detections=True,
        return_object_point_clouds=False,
        timeout=15,
    )
    if result.image is None:
        raise RuntimeError("objects-3d returned no image. Check its camera configuration.")
    return SceneSnapshot(
        image=decode_viam_image(result.image),
        detections=list(result.detections or []),
        object_point_clouds=[],
        captured_at=time.monotonic(),
    )


async def capture_fresh_snapshot(camera_name: str, detector_name: str) -> SceneSnapshot:
    """Open a short-lived Viam connection and retry one transient disconnect."""
    last_error: Exception | None = None
    for attempt in range(2):
        robot: RobotClient | None = None
        try:
            robot = await connect()
            detector = VisionClient.from_robot(robot, detector_name)
            return await capture_snapshot(detector, camera_name)
        except Exception as error:
            last_error = error
            if attempt == 0:
                await asyncio.sleep(1)
        finally:
            if robot is not None:
                await robot.close()
    raise RuntimeError(
        "Could not capture the RealSense image and YOLO boxes after two attempts. "
        "Check that the Viam machine, cam, and yolo-detector are online."
    ) from last_error


def fit_image_to_canvas(image: np.ndarray, canvas_width: int, canvas_height: int) -> tuple[np.ndarray, DisplayLayout]:
    image_height, image_width = image.shape[:2]
    scale = min(canvas_width / image_width, canvas_height / image_height)
    render_width, render_height = round(image_width * scale), round(image_height * scale)
    offset_x = (canvas_width - render_width) // 2
    offset_y = (canvas_height - render_height) // 2
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[offset_y : offset_y + render_height, offset_x : offset_x + render_width] = cv2.resize(
        image, (render_width, render_height), interpolation=cv2.INTER_AREA
    )
    return canvas, DisplayLayout(scale, offset_x, offset_y, image_width, image_height)


def detection_contains(detection, image_point: tuple[float, float]) -> bool:
    x, y = image_point
    return (
        detection.x_min + BOX_EDGE_MARGIN_PX <= x <= detection.x_max - BOX_EDGE_MARGIN_PX
        and detection.y_min + BOX_EDGE_MARGIN_PX <= y <= detection.y_max - BOX_EDGE_MARGIN_PX
    )


def draw_detections(canvas: np.ndarray, detections: list, layout: DisplayLayout, active_index: int | None) -> None:
    for index, detection in enumerate(detections):
        x1, y1 = layout.image_to_canvas(detection.x_min, detection.y_min)
        x2, y2 = layout.image_to_canvas(detection.x_max, detection.y_max)
        color = (0, 255, 0) if index == active_index else (0, 190, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = f"{detection.class_name} {detection.confidence:.0%}"
        cv2.putText(canvas, label, (x1, max(26, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def draw_status(canvas: np.ndarray, message: str) -> None:
    # Start at y=0 so no light image strip remains above the status bar.
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (0, 0, 0), -1)
    cv2.putText(canvas, message, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def draw_target(canvas: np.ndarray, target: tuple[float, float], number: int, progress: float) -> None:
    x, y = round(target[0] * canvas.shape[1]), round(target[1] * canvas.shape[0])
    cv2.circle(canvas, (x, y), 28, (0, 255, 255), 3)
    cv2.circle(canvas, (x, y), max(1, round(23 * progress)), (0, 255, 255), -1)
    cv2.putText(canvas, str(number), (x - 9, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


async def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name}. See README.md for the download command.")

    camera_name = os.getenv("VIAM_CAMERA_NAME", "cam")
    detector_name = os.getenv("VIAM_DETECTOR_NAME", "yolo-detector")

    laptop_camera = cv2.VideoCapture(CAMERA_INDEX)
    laptop_camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    laptop_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not laptop_camera.isOpened():
        raise RuntimeError("Could not open the laptop webcam. Try CAMERA_INDEX = 0 in gaze_dot.py.")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # The scene is deliberately not captured yet. Calibration happens against
    # the laptop-webcam view first; only afterwards do we freeze the arm-camera
    # image and its detections for selection.
    snapshot: SceneSnapshot | None = None
    calibration_samples: list[np.ndarray] = []
    calibration_targets: list[np.ndarray] = []
    calibrating = False
    framing = False
    frame_hold_started = 0.0
    calibrated = False
    target_index = 0
    target_started_at = 0.0
    target_samples: list[np.ndarray] = []
    mapping: np.ndarray | None = None
    smoothed_dot: np.ndarray | None = None
    last_timestamp_ms = -1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen_applied = False

    try:
        print("Face calibration ready. Press C to begin; Q quits.")

        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, laptop_frame = laptop_camera.read()
                if not ok:
                    raise RuntimeError("Could not read the laptop webcam.")
                laptop_frame = cv2.flip(laptop_frame, 1)
                rgb = cv2.cvtColor(laptop_frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(last_timestamp_ms + 1, int(time.monotonic() * 1000))
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                landmarks = result.face_landmarks[0] if result.face_landmarks else None
                features = gaze_features(landmarks) if landmarks is not None else None

                canvas_height, canvas_width = laptop_frame.shape[:2]
                # Until calibration finishes, the user sees their mirrored
                # laptop-webcam feed. The calibration targets are in the same
                # fullscreen canvas coordinates used later for the RealSense view.
                canvas = laptop_frame.copy()
                layout: DisplayLayout | None = None
                active_index: int | None = None

                if calibrated and snapshot is not None:
                    canvas, layout = fit_image_to_canvas(snapshot.image, canvas_width, canvas_height)

                if calibrated and snapshot is not None and features is not None and mapping is not None:
                    predicted = features @ mapping
                    predicted[0] = np.clip(predicted[0], 0, canvas_width - 1)
                    predicted[1] = np.clip(predicted[1], 0, canvas_height - 1)
                    smoothed_dot = predicted if smoothed_dot is None else 0.80 * smoothed_dot + 0.20 * predicted
                    image_point = layout.canvas_to_image(*smoothed_dot)
                    if image_point is not None:
                        hits = [i for i, detection in enumerate(snapshot.detections) if detection_contains(detection, image_point)]
                        active_index = hits[0] if hits else None

                if calibrated and snapshot is not None and layout is not None:
                    draw_detections(canvas, snapshot.detections, layout, active_index)
                now = time.monotonic()
                if framing:
                    # This is deliberately identical to gaze_dot.py's
                    # pre-calibration posture gate: the same oval, target
                    # dimensions, feedback, and one-second steady hold.
                    if landmarks is None:
                        frame_hold_started = 0.0
                        draw_id_frame(canvas, FRAME_TARGET_CX, FRAME_TARGET_CY,
                                      FRAME_TARGET_W, FRAME_TARGET_H, aligned=False)
                        draw_status(canvas, "Face not found. Center your face in the frame.")
                    else:
                        bbox = face_bbox_normalized(landmarks)
                        status, aligned = evaluate_framing(
                            bbox, FRAME_TARGET_CX, FRAME_TARGET_CY, FRAME_TARGET_W, FRAME_TARGET_H,
                            FRAME_POSITION_TOLERANCE, FRAME_SIZE_RATIO_LOW, FRAME_SIZE_RATIO_HIGH)
                        draw_id_frame(canvas, FRAME_TARGET_CX, FRAME_TARGET_CY,
                                      FRAME_TARGET_W, FRAME_TARGET_H, aligned)
                        if aligned:
                            if frame_hold_started == 0.0:
                                frame_hold_started = now
                            remaining = max(0.0, FRAME_HOLD_SECONDS - (now - frame_hold_started))
                            draw_status(canvas, "Hold still..." if remaining > 0 else "Starting calibration...")
                            if remaining <= 0.0:
                                framing = False
                                calibrating = True
                                target_index = 0
                                target_started_at = 0.0
                        else:
                            frame_hold_started = 0.0
                            draw_status(canvas, FRAME_MESSAGES[status])

                elif calibrating:
                    if target_started_at == 0.0:
                        target_started_at = now
                    elapsed = now - target_started_at
                    draw_target(canvas, TARGETS[target_index], target_index + 1, min(elapsed / (SETTLE_SECONDS + CALIBRATION_SECONDS), 1.0))
                    draw_status(canvas, f"Look at yellow target {target_index + 1} of {len(TARGETS)}")
                    if features is not None and elapsed >= SETTLE_SECONDS:
                        target_samples.append(features)
                    if elapsed >= SETTLE_SECONDS + CALIBRATION_SECONDS:
                        if len(target_samples) >= MIN_SAMPLES_PER_POINT:
                            calibration_samples.append(np.median(target_samples, axis=0))
                            calibration_targets.append(np.array([TARGETS[target_index][0] * canvas_width, TARGETS[target_index][1] * canvas_height]))
                        target_index += 1
                        target_started_at = 0.0
                        target_samples = []
                        if target_index == len(TARGETS):
                            if len(calibration_samples) < 6:
                                raise RuntimeError("Too few reliable calibration targets. Press C and try again.")
                            mapping = fit_calibration(calibration_samples, calibration_targets)
                            calibrating, calibrated, smoothed_dot = False, True, None
                            # Do not move the arm here. The operator must have
                            # already placed it at the safe observe pose.
                            snapshot = await capture_fresh_snapshot(camera_name, detector_name)
                            print("Calibration complete. RealSense scene captured for gaze selection.")
                elif calibrated and snapshot is not None and smoothed_dot is not None:
                    center = tuple(np.round(smoothed_dot).astype(int))
                    cv2.circle(canvas, center, DOT_RADIUS, (0, 0, 255), -1)
                    cv2.circle(canvas, center, DOT_RADIUS + 3, (255, 255, 255), 2)
                    if active_index is not None:
                        draw_status(canvas, f"Gaze is inside: {snapshot.detections[active_index].class_name}. No robot command is sent.")
                    else:
                        draw_status(canvas, "Gaze is outside a selectable box. N refreshes the frozen scene; R recalibrates.")
                else:
                    draw_id_frame(canvas, FRAME_TARGET_CX, FRAME_TARGET_CY,
                                  FRAME_TARGET_W, FRAME_TARGET_H, aligned=False)
                    draw_status(canvas, "Press C, center your face in the oval, then keep your head still.")

                cv2.imshow(WINDOW_NAME, canvas)
                # See the matching gaze_dot.py call: this second application
                # removes the macOS title-bar strip after first render.
                if not fullscreen_applied:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    fullscreen_applied = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("c"), ord("r")):
                    calibration_samples, calibration_targets, target_samples = [], [], []
                    calibrating, framing, calibrated, target_index = False, True, False, 0
                    frame_hold_started = 0.0
                    target_started_at, mapping, smoothed_dot = 0.0, None, None
                if key == ord("n"):
                    if calibrated:
                        snapshot = await capture_fresh_snapshot(camera_name, detector_name)
                        print("RealSense scene refreshed. The arm must remain at its observe pose.")
    finally:
        laptop_camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
