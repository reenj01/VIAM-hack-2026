"""Read-only checks before acknowledging an interrupted physical action.

An expired safety session does not authorize replaying a motor command. These
checks only allow a new selection once the operator has cleared the gripper.
"""

import asyncio
import math
from dataclasses import dataclass
from typing import Optional

from grpclib import GRPCError, Status
from viam.proto.common import Pose


def is_session_expired(error: Exception) -> bool:
    return (isinstance(error, GRPCError)
            and error.status == Status.INVALID_ARGUMENT
            and error.message == "SESSION_EXPIRED")


@dataclass
class RecoveryCheck:
    safe: bool
    message: str
    pose: Optional[Pose] = None


async def check_action_recovery(arm, gripper, read_position, read_pose,
                                *, timeout: float = 3.0, open_position: float = 830.0) -> RecoveryCheck:
    """Only read state; never open, close, move, or replay the failed action."""
    try:
        if await arm.is_moving(timeout=timeout) is not False:
            return RecoveryCheck(False, "Arm is moving or its stopped state is unknown")
        if await gripper.is_moving(timeout=timeout) is not False:
            return RecoveryCheck(False, "Gripper is moving or its stopped state is unknown")
        position = await asyncio.wait_for(read_position(), timeout)
        if (isinstance(position, bool) or not isinstance(position, (int, float))
                or not math.isfinite(position) or not open_position <= position <= 850.0):
            return RecoveryCheck(False, "Gripper not verified fully open; use robot controls to open it, then R")
        holding = await gripper.is_holding_something(timeout=timeout)
        if getattr(holding, "is_holding_something", None) is not False:
            return RecoveryCheck(False, "Gripper is not verified empty; use robot controls to place the object, then R")
        pose = await asyncio.wait_for(read_pose(), timeout)
        fields = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")
        if pose is None or not all(math.isfinite(getattr(pose, key)) for key in fields):
            return RecoveryCheck(False, "Current gripper pose could not be verified")
        if math.hypot(pose.o_x, pose.o_y, pose.o_z) == 0 and pose.theta != 0:
            return RecoveryCheck(False, "Current gripper orientation is invalid")
        # A second stopped read prevents accepting a snapshot collected while
        # either component began moving between the earlier reads.
        if await arm.is_moving(timeout=timeout) is not False:
            return RecoveryCheck(False, "Arm moved during the recovery checks")
        if await gripper.is_moving(timeout=timeout) is not False:
            return RecoveryCheck(False, "Gripper moved during the recovery checks")
        final_position = await asyncio.wait_for(read_position(), timeout)
        if (isinstance(final_position, bool) or not isinstance(final_position, (int, float))
                or not math.isfinite(final_position) or not open_position <= final_position <= 850.0
                or abs(final_position - position) > 6.0):
            return RecoveryCheck(False, "Gripper opening changed during the checks; still paused")
        return RecoveryCheck(True, "Stopped arm and empty, open gripper verified", pose)
    except Exception as error:
        return RecoveryCheck(False, f"Readback failed ({type(error).__name__}: {error}); still paused")


@dataclass
class OperatorFault:
    reason: str
    detail: str = "Use robot controls to put down any object and open jaws; R checks stopped/open/empty"
    check_task: Optional[asyncio.Task] = None

    def start_check(self, check) -> None:
        if self.check_task is None:
            self.detail = "Checking stopped arm and empty, open gripper..."
            self.check_task = asyncio.create_task(check())

    def poll_check(self) -> bool:
        if self.check_task is None or not self.check_task.done():
            return False
        try:
            result = self.check_task.result()
        except (Exception, asyncio.CancelledError) as error:
            self.detail = f"Recovery check interrupted ({type(error).__name__}); still paused"
            return False
        finally:
            self.check_task = None
        self.detail = result.message
        return result.safe

    async def cancel_check(self) -> None:
        if self.check_task is not None:
            self.check_task.cancel()
            await asyncio.gather(self.check_task, return_exceptions=True)
            self.check_task = None
