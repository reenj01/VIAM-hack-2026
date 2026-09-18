"""Viam Phase 5 perception-guided pick-and-place tutorial, adapted to this machine.

This intentionally follows the tutorial's sequence:
observe from one repeatable wrist-camera pose -> get 3-D segments -> approach
in the camera frame -> descend in the gripper frame -> grab -> travel -> place
-> return home.

Before ``--stage full`` can run, save all four position-saver switches in the
Viam app: pose-home, pose-observe, pose-travel, and pose-place. This machine
currently has only pose-home and pose-observe, so use --stage detect and
--stage approach until the remaining two switches are configured.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.errors import ResourceNotFoundError
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

load_dotenv()

ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS") or os.getenv("MACHINE_ADDRESS")
API_KEY = os.getenv("VIAM_API_KEY") or os.getenv("API_KEY")
API_KEY_ID = os.getenv("VIAM_API_KEY_ID") or os.getenv("API_KEY_ID")

CAMERA_NAME = os.getenv("VIAM_CAMERA_NAME", "cam")
GRIPPER_NAME = os.getenv("VIAM_GRIPPER_NAME", "gripper")
VISION_NAME = os.getenv("VIAM_SEGMENTER_NAME", "objects-3d")
MOTION_NAME = os.getenv("VIAM_MOTION_NAME", "builtin")
TARGET_LABEL = os.getenv("VIAM_TUTORIAL_TARGET_LABEL", "rectangle-red").casefold()

# Pose savers configured in the Viam app. In this project, pose-observe is
# the tutorial's fixed camera-viewing "home" pose.
HOME_SWITCH_NAME = os.getenv("VIAM_HOME_POSE_NAME", "pose-home")
OBSERVE_SWITCH_NAME = os.getenv("VIAM_OBSERVE_POSE_NAME", "pose-observe")
TRAVEL_SWITCH_NAME = os.getenv("VIAM_TRAVEL_POSE_NAME", "pose-travel")
PLACE_SWITCH_NAME = os.getenv("VIAM_PLACE_POSE_NAME", "pose-place")

# Tutorial values. These are millimetres in their respective camera/gripper
# frames, not world-frame coordinates.
APPROACH_MM = float(os.getenv("VIAM_APPROACH_MM", "-100"))
GRIPPER_LENGTH_MM = float(os.getenv("VIAM_GRIPPER_CENTER_OFFSET_MM", "-60"))
GRASP_DISTANCE_MM = (APPROACH_MM - GRIPPER_LENGTH_MM) * -1


async def connect() -> RobotClient:
    if not all((ADDRESS, API_KEY, API_KEY_ID)):
        raise RuntimeError(
            "Missing VIAM_MACHINE_ADDRESS, VIAM_API_KEY, or VIAM_API_KEY_ID in .env"
        )
    options = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    return await RobotClient.at_address(ADDRESS, options)


def offset_pose(pose: Pose, z_offset_mm: float) -> Pose:
    """Raise/lower a pose in Z while retaining X/Y and orientation."""
    return Pose(
        x=pose.x,
        y=pose.y,
        z=pose.z + z_offset_mm,
        o_x=pose.o_x,
        o_y=pose.o_y,
        o_z=pose.o_z,
        theta=pose.theta,
    )


async def detect_target_object(vision: VisionClient) -> PoseInFrame | None:
    """Find the configured shape-detector label in the tutorial's 3-D output."""
    objects = await vision.get_object_point_clouds(CAMERA_NAME, timeout=90)
    if not objects:
        print("No objects detected.")
        return None

    objects_with_geometry = [obj for obj in objects if obj.geometries.geometries]
    if not objects_with_geometry:
        print("Objects were returned, but none included a usable geometry.")
        return None

    matching_objects = [
        obj
        for obj in objects_with_geometry
        if obj.geometries.geometries[0].label.casefold() == TARGET_LABEL
    ]
    if not matching_objects:
        labels = [obj.geometries.geometries[0].label for obj in objects_with_geometry]
        print(f"No {TARGET_LABEL!r} detected. Available shape labels: {labels}")
        return None

    # If two matching blocks are visible, use the one with the most depth
    # points, which is the same selection rule used by the tutorial.
    obj = max(matching_objects, key=lambda candidate: len(candidate.point_cloud))
    geometry = obj.geometries.geometries[0]
    print(f"Detected: {geometry.label}")
    print(
        "Object center in camera frame: "
        f"x={geometry.center.x:.1f}, y={geometry.center.y:.1f}, z={geometry.center.z:.1f} mm"
    )
    return PoseInFrame(reference_frame=CAMERA_NAME, pose=geometry.center)


async def main(stage: str) -> None:
    machine = await connect()
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    vision = VisionClient.from_robot(machine, VISION_NAME)
    motion = MotionClient.from_robot(machine, MOTION_NAME)
    home = Switch.from_robot(machine, HOME_SWITCH_NAME)
    observe = Switch.from_robot(machine, OBSERVE_SWITCH_NAME)
    travel = None
    place_pose = None

    # Check the two fixed placement switches before moving toward a detected
    # object. A missing resource should fail while the arm is still parked.
    if stage == "full":
        try:
            travel = Switch.from_robot(machine, TRAVEL_SWITCH_NAME)
            place_pose = Switch.from_robot(machine, PLACE_SWITCH_NAME)
            await travel.get_position()
            await place_pose.get_position()
        except ResourceNotFoundError as error:
            raise RuntimeError(
                "The full tutorial sequence needs two saved arm-position-saver switches that "
                f"are not configured: {TRAVEL_SWITCH_NAME!r} and/or {PLACE_SWITCH_NAME!r}. "
                "In the Viam Configure tab, add two copies of the same arm-position-saver "
                "model used for pose-home and pose-observe, save safe travel and place poses "
                "with switch position 1, then retry --stage full. Until then, use --stage "
                "detect or --stage approach."
            ) from error

    try:
        # Wrist-camera rule: detect only from this saved, repeatable pose.
        print(f"Moving to observation pose: {OBSERVE_SWITCH_NAME}")
        await observe.set_position(2)
        await asyncio.sleep(0.5)

        obj_in_cam = await detect_target_object(vision)
        if obj_in_cam is None:
            return

        approach_pose = offset_pose(obj_in_cam.pose, APPROACH_MM)
        print(
            f"Approach in {CAMERA_NAME!r} frame: "
            f"x={approach_pose.x:.1f}, y={approach_pose.y:.1f}, z={approach_pose.z:.1f} mm"
        )
        if stage == "detect":
            print("Detection test complete; no approach motion sent.")
            return

        # First move: the camera has not moved since the detection, so its
        # coordinates are valid for the approach pose.
        await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame=CAMERA_NAME, pose=approach_pose),
        )
        print("Reached approach standoff.")
        if stage == "approach":
            print("Approach test complete; no gripper or descent motion sent.")
            return

        # From here on, the camera rides with the arm. Descend in gripper
        # coordinates exactly as the tutorial prescribes.
        await gripper.open()
        await asyncio.sleep(0.3)
        input("Press Enter to descend and grab rectangle-red, or Ctrl-C to abort: ")
        await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(
                reference_frame=GRIPPER_NAME,
                pose=Pose(
                    x=0, y=0, z=GRASP_DISTANCE_MM,
                    o_x=0, o_y=0, o_z=1, theta=0,
                ),
            ),
        )
        grabbed = await gripper.grab()
        print("Gripper reports grasped:", grabbed)
        if not grabbed:
            print("No object was confirmed. Not travelling to the place pose.")
            return
        await asyncio.sleep(0.3)

        if stage == "grab":
            print("rectangle-red is grasped. The arm is holding position; no place move was sent.")
            return

        await travel.set_position(2)
        await place_pose.set_position(2)
        await gripper.open()
        await home.set_position(2)
        print("Tutorial pick-and-place cycle complete.")
    finally:
        await machine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Viam Phase 5 pick-and-place flow.")
    parser.add_argument(
        "--stage",
        choices=["detect", "approach", "grab", "full"],
        default="detect",
        help="Start with detect, then approach; grab stops after closing; full needs travel/place switches.",
    )
    asyncio.run(main(parser.parse_args().stage))
