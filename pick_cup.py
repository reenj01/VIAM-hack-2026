"""Use YOLO to detect a bowl, then approach and pick its 3-D segment.

Run ``--stage hover`` first. It moves only to the camera-frame standoff.
Use ``--stage grab`` only after that hover position is visibly safe.

Viam requirement: objects-3d must have ``detector_name: yolo-detector`` in
the machine configuration, so its 3-D geometry has the same labels as YOLO.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.component.arm import JointPositions
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

load_dotenv()

# Read the private .env values used by the rest of this project. The short
# names are only compatibility fallbacks for an older Viam tutorial template.
ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS") or os.getenv("MACHINE_ADDRESS")
API_KEY = os.getenv("VIAM_API_KEY") or os.getenv("API_KEY")
API_KEY_ID = os.getenv("VIAM_API_KEY_ID") or os.getenv("API_KEY_ID")
CAMERA_NAME = os.getenv("VIAM_CAMERA_NAME", "cam")
ARM_NAME = os.getenv("VIAM_ARM_NAME", "arm")
GRIPPER_NAME = os.getenv("VIAM_GRIPPER_NAME", "gripper")
DETECTOR_NAME = os.getenv("VIAM_DETECTOR_NAME", "yolo-detector")
SEGMENTER_NAME = os.getenv("VIAM_SEGMENTER_NAME", "objects-3d")
MOTION_NAME = os.getenv("VIAM_MOTION_NAME", "builtin")
TARGET_LABEL = os.getenv("VIAM_PICK_TARGET_LABEL", "bowl").casefold()

# Fixed pose that gives the wrist camera a stable, known view of the table.
OBSERVE_JOINTS = [294.952, -60.135, -19.090, 0.009, 79.179, 165.257]

# Viam tutorial offsets, in millimeters. They are relative to camera/gripper
# frames, not absolute world coordinates.
APPROACH_MM = float(os.getenv("VIAM_APPROACH_MM", "-100"))
GRIPPER_CENTER_OFFSET_MM = float(os.getenv("VIAM_GRIPPER_CENTER_OFFSET_MM", "-60"))
LIFT_MM = float(os.getenv("VIAM_LIFT_MM", "150"))
GRASP_DESCENT_MM = (APPROACH_MM - GRIPPER_CENTER_OFFSET_MM) * -1


async def connect() -> RobotClient:
    if not all((ADDRESS, API_KEY, API_KEY_ID)):
        raise RuntimeError(
            "Missing VIAM_MACHINE_ADDRESS, VIAM_API_KEY, or VIAM_API_KEY_ID in .env"
        )
    options = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    return await RobotClient.at_address(ADDRESS, options)


def offset_pose(pose: Pose, z_offset_mm: float) -> Pose:
    """Offset a pose along Z in the pose's own reference frame."""
    return Pose(
        x=pose.x,
        y=pose.y,
        z=pose.z + z_offset_mm,
        o_x=pose.o_x,
        o_y=pose.o_y,
        o_z=pose.o_z,
        theta=pose.theta,
    )


def gripper_relative_pose(z_mm: float) -> PoseInFrame:
    """Pure Z translation from the gripper's current TCP frame."""
    return PoseInFrame(
        reference_frame=GRIPPER_NAME,
        pose=Pose(x=0, y=0, z=z_mm, o_x=0, o_y=0, o_z=1, theta=0),
    )


async def find_yolo_target(detector: VisionClient, segmenter: VisionClient) -> PoseInFrame | None:
    """Confirm a YOLO bowl, then retrieve the matching depth-derived object."""
    detections = await detector.get_detections_from_camera(CAMERA_NAME)
    bowl_detections = [
        detection
        for detection in detections
        if detection.class_name.casefold() == TARGET_LABEL
    ]
    if not bowl_detections:
        labels = [detection.class_name for detection in detections]
        print(f"YOLO did not detect {TARGET_LABEL!r}. Current YOLO labels: {labels}")
        return None

    selected = max(bowl_detections, key=lambda detection: detection.confidence)
    print(
        f"YOLO selected {selected.class_name!r} ({selected.confidence:.0%}), "
        f"box=({selected.x_min}, {selected.y_min})-({selected.x_max}, {selected.y_max})"
    )

    objects = await segmenter.get_object_point_clouds(CAMERA_NAME, timeout=90)
    matches = []
    for obj in objects:
        if not obj.geometries.geometries:
            continue
        geometry = obj.geometries.geometries[0]
        print(
            f"3-D {geometry.label!r}: center=({geometry.center.x:.1f}, "
            f"{geometry.center.y:.1f}, {geometry.center.z:.1f}) mm"
        )
        if geometry.label.casefold() == TARGET_LABEL:
            matches.append((len(obj.point_cloud), geometry))

    if not matches:
        print(
            f"YOLO detected {TARGET_LABEL!r}, but {SEGMENTER_NAME!r} returned no matching "
            "3-D segment. In Viam Configure, set objects-3d's detector_name to "
            f"{DETECTOR_NAME!r}."
        )
        return None

    _, geometry = max(matches, key=lambda match: match[0])
    return PoseInFrame(reference_frame=CAMERA_NAME, pose=geometry.center)


async def main(stage: str) -> None:
    machine = await connect()
    arm = Arm.from_robot(machine, ARM_NAME)
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    detector = VisionClient.from_robot(machine, DETECTOR_NAME)
    segmenter = VisionClient.from_robot(machine, SEGMENTER_NAME)
    motion = MotionClient.from_robot(machine, MOTION_NAME)

    try:
        print("Moving to the fixed observe pose...")
        await arm.move_to_joint_positions(JointPositions(values=OBSERVE_JOINTS))
        await asyncio.sleep(1)

        target_in_camera = await find_yolo_target(detector, segmenter)
        if target_in_camera is None:
            return

        # First move only: the wrist camera is still in the same fixed pose
        # from which the bowl was detected, so its coordinates remain valid.
        approach = PoseInFrame(
            reference_frame=CAMERA_NAME,
            pose=offset_pose(target_in_camera.pose, APPROACH_MM),
        )
        print(f"Moving to a {APPROACH_MM:.0f} mm standoff above the {TARGET_LABEL}...")
        await motion.move(component_name=GRIPPER_NAME, destination=approach)

        if stage == "hover":
            print(f"At standoff. Confirm the gripper is safely centered over the {TARGET_LABEL}.")
            return

        # The camera moved during approach. Descend relative to the gripper,
        # not the old camera frame.
        await gripper.open()
        await asyncio.sleep(0.3)
        input(f"Press Enter to descend and grab {TARGET_LABEL}, or Ctrl-C to abort: ")
        await motion.move(
            component_name=GRIPPER_NAME,
            destination=gripper_relative_pose(GRASP_DESCENT_MM),
        )
        grabbed = await gripper.grab()
        print("Gripper reports object grasped:", grabbed)
        if not grabbed:
            print("No object was confirmed; not lifting.")
            return

        await motion.move(
            component_name=GRIPPER_NAME,
            destination=gripper_relative_pose(-LIFT_MM),
        )
        print(f"{TARGET_LABEL} lifted and held in the gripper.")
    finally:
        await machine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO bowl pick with a Viam arm.")
    parser.add_argument("--stage", choices=["hover", "grab"], default="hover")
    args = parser.parse_args()
    asyncio.run(main(args.stage))
