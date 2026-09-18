"""Detect a cup and pick it up, following the Phase 5 approach structure.

Run --stage hover first and check by eye, then --stage grab.
"""

import argparse, asyncio, os
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.proto.component.arm import JointPositions
from viam.proto.common import Pose, PoseInFrame

load_dotenv()

# Connection and resource names. The short fallback names keep compatibility
# with an older tutorial .env, while this project uses the VIAM_* names.
ADDR = os.getenv("VIAM_MACHINE_ADDRESS") or os.getenv("MACHINE_ADDRESS")
KEY = os.getenv("VIAM_API_KEY") or os.getenv("API_KEY")
KEY_ID = os.getenv("VIAM_API_KEY_ID") or os.getenv("API_KEY_ID")
CAMERA_NAME = os.getenv("VIAM_CAMERA_NAME", "cam")
ARM_NAME = os.getenv("VIAM_ARM_NAME", "arm")
GRIPPER_NAME = os.getenv("VIAM_GRIPPER_NAME", "gripper")
SEGMENTER_NAME = os.getenv("VIAM_SEGMENTER_NAME", "objects-3d")
MOTION_NAME = os.getenv("VIAM_MOTION_NAME", "builtin")
TARGET_LABEL = os.getenv("VIAM_TARGET_LABEL", "cup").lower()


def float_env(name, default):
    """Read a numeric .env setting with a useful error for typos."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number in .env.") from error


def joint_list_env(name, default):
    """Read six comma-separated joint angles from .env."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        values = [float(value.strip()) for value in raw.split(",")]
    except ValueError as error:
        raise RuntimeError(f"{name} must contain comma-separated numbers.") from error
    if len(values) != 6:
        raise RuntimeError(f"{name} must contain exactly 6 joint angles, not {len(values)}.")
    return values


OBSERVE_JOINTS = joint_list_env(
    "VIAM_OBSERVE_JOINTS", [294.952, -60.135, -19.090, 0.009, 79.179, 165.257]
)
# These distances are measured in the relevant reference frame, not in world.
# The first approach is in the wrist-camera frame. The next two translations
# are in the gripper TCP frame after the wrist camera has started moving.
APPROACH_MM = float_env("VIAM_APPROACH_MM", -100.0)
GRIPPER_CENTER_OFFSET_MM = float_env("VIAM_GRIPPER_CENTER_OFFSET_MM", -60.0)
LIFT_MM = float_env("VIAM_LIFT_MM", 150.0)
GRASP_DESCENT_MM = (APPROACH_MM - GRIPPER_CENTER_OFFSET_MM) * -1


def offset_pose(pose, z_offset):
    """Copy a pose, offsetting only Z in its own reference frame."""
    return Pose(
        x=pose.x, y=pose.y, z=pose.z + z_offset,
        o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta,
    )


def gripper_relative_pose(z_mm):
    """A straight translation relative to the configured gripper TCP."""
    return PoseInFrame(
        reference_frame=GRIPPER_NAME,
        pose=Pose(x=0, y=0, z=z_mm, o_x=0, o_y=0, o_z=1, theta=0),
    )


async def find_cup(seg):
    objs = await seg.get_object_point_clouds(CAMERA_NAME)
    print(f"{len(objs)} object(s) detected")
    for o in objs:
        if not o.geometries.geometries:
            continue
        g = o.geometries.geometries[0]
        print(f"  {g.label}: center=({g.center.x:.1f}, {g.center.y:.1f}, {g.center.z:.1f})")
        if TARGET_LABEL in g.label.lower():
            return g.center
    return None


async def main(stage):
    if not all((ADDR, KEY, KEY_ID)):
        raise RuntimeError(
            "Missing VIAM_MACHINE_ADDRESS, VIAM_API_KEY, or VIAM_API_KEY_ID in .env"
        )
    machine = await RobotClient.at_address(ADDR, RobotClient.Options.with_api_key(
        api_key=KEY, api_key_id=KEY_ID))
    arm = Arm.from_robot(machine, ARM_NAME)
    gripper = Gripper.from_robot(machine, GRIPPER_NAME)
    seg = VisionClient.from_robot(machine, SEGMENTER_NAME)
    motion = MotionClient.from_robot(machine, MOTION_NAME)

    try:
        print("moving to observe pose...")
        await arm.move_to_joint_positions(JointPositions(values=OBSERVE_JOINTS))
        await asyncio.sleep(1.0)

        cup_in_camera = await find_cup(seg)
        if cup_in_camera is None:
            print(f"no '{TARGET_LABEL}' found")
            return

        # GetObjectPointClouds reports its center in CAMERA_NAME's coordinate
        # frame. The old code incorrectly reused its x/y as world coordinates
        # and invented a world Z value. That yielded an impossible IK target.
        approach_in_camera = PoseInFrame(
            reference_frame=CAMERA_NAME,
            pose=offset_pose(cup_in_camera, APPROACH_MM),
        )
        print(
            f"moving to standoff: {APPROACH_MM:.0f} mm from the cup in "
            f"the {CAMERA_NAME!r} frame..."
        )
        await motion.move(
            component_name=GRIPPER_NAME,
            destination=approach_in_camera)

        if stage == "hover":
            print("stopped at standoff. Check the gripper is centered over the cup.")
            return

        await gripper.open()
        input("descend and grab? Enter to continue, Ctrl-C to abort: ")

        # The wrist camera moves during the approach, so never reuse its
        # original coordinates here. Descend relative to the gripper instead.
        print(f"descending {GRASP_DESCENT_MM:.0f} mm relative to the gripper...")
        await motion.move(
            component_name=GRIPPER_NAME,
            destination=gripper_relative_pose(GRASP_DESCENT_MM))

        grabbed = await gripper.grab()
        print("grabbed:", grabbed)
        if not grabbed:
            print("No object was confirmed in the gripper; not lifting.")
            return

        # Positive Z above was the descent direction; negative Z lifts.
        print(f"lifting {LIFT_MM:.0f} mm...")
        await motion.move(
            component_name=GRIPPER_NAME,
            destination=gripper_relative_pose(-LIFT_MM))
        print("done — cup held. Call gripper.open() to release.")

    finally:
        await machine.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["hover", "grab"], default="hover")
    asyncio.run(main(p.parse_args().stage))
