"""Move the xArm through the two saved "face" joint-5 positions.

This is a direct joint-space test. Clear the work area and test values slowly
in Viam Control before running it on the physical arm.
"""

import asyncio
import os

from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.proto.component.arm import JointPositions
from viam.robot.client import RobotClient

load_dotenv()

ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS")
API_KEY = os.getenv("VIAM_API_KEY")
API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
ARM_NAME = os.getenv("VIAM_ARM_NAME", "arm")

# Python lists are zero-indexed: index 4 is the fifth physical joint.
JOINT_INDEX = 4
FACING_OUT_DEG = -0.34
FACING_DOWN_DEG = 74.59
HOLD_SECONDS = 9


async def main() -> None:
    if not all((ADDRESS, API_KEY, API_KEY_ID)):
        raise RuntimeError(
            "Missing VIAM_MACHINE_ADDRESS, VIAM_API_KEY, or VIAM_API_KEY_ID in .env"
        )

    options = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    machine = await RobotClient.at_address(ADDRESS, options)
    arm = Arm.from_robot(machine, ARM_NAME)
    moved_from_start = False

    try:
        start = list((await arm.get_joint_positions()).values)
        if len(start) <= JOINT_INDEX:
            raise RuntimeError(
                f"Arm {ARM_NAME!r} has {len(start)} joints; joint index {JOINT_INDEX} is unavailable."
            )
        print("start:", [round(value, 2) for value in start])

        facing_out = list(start)
        facing_out[JOINT_INDEX] = FACING_OUT_DEG
        print(f"moving joint {JOINT_INDEX + 1} to {FACING_OUT_DEG} degrees...")
        await arm.move_to_joint_positions(JointPositions(values=facing_out))
        moved_from_start = True

        print(f"facing out, holding {HOLD_SECONDS} seconds...")
        await asyncio.sleep(HOLD_SECONDS)

        facing_down = list(start)
        facing_down[JOINT_INDEX] = FACING_DOWN_DEG
        print(f"moving joint {JOINT_INDEX + 1} to {FACING_DOWN_DEG} degrees...")
        await arm.move_to_joint_positions(JointPositions(values=facing_down))
        moved_from_start = False
        print("facing down")
    finally:
        # If the program is interrupted during the 9-second hold, try to put
        # the arm back at the pose it had when this script began.
        if moved_from_start:
            try:
                print("returning to the starting joint pose before closing...")
                await arm.move_to_joint_positions(JointPositions(values=start))
            except Exception as error:
                print(f"could not return the arm automatically: {error}")
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
