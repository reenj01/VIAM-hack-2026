"""Move the arm to its saved fixed standoff/observation pose."""

import asyncio
import os

from dotenv import load_dotenv
from viam.components.switch import Switch
from viam.robot.client import RobotClient

load_dotenv()

ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS") or os.getenv("MACHINE_ADDRESS")
API_KEY = os.getenv("VIAM_API_KEY") or os.getenv("API_KEY")
API_KEY_ID = os.getenv("VIAM_API_KEY_ID") or os.getenv("API_KEY_ID")

# This is the pose saver you created for the arm's table-viewing position.
STANDOFF_POSE_NAME = os.getenv("VIAM_DEFAULT_STANDOFF_POSE_NAME", "pose-observe")


async def main() -> None:
    if not all((ADDRESS, API_KEY, API_KEY_ID)):
        raise RuntimeError(
            "Missing VIAM_MACHINE_ADDRESS, VIAM_API_KEY, or VIAM_API_KEY_ID in .env"
        )

    options = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    machine = await RobotClient.at_address(ADDRESS, options)
    try:
        standoff_pose = Switch.from_robot(machine, STANDOFF_POSE_NAME)
        print(f"Moving to saved standoff pose: {STANDOFF_POSE_NAME}")
        # Position 2 means "go to" for Viam's arm-position-saver switch.
        await standoff_pose.set_position(2)
        print("Standoff pose reached.")
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
