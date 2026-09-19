"""Session expiry and operator acknowledgement checks; no hardware access."""

import asyncio
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, sentinel

from grpclib import GRPCError, Status
from viam.proto.common import Pose

import main as app
from action_recovery import OperatorFault, RecoveryCheck, check_action_recovery, is_session_expired
import test_connection_recovery as connection_tests


class SessionExpiryTests(unittest.TestCase):
    def test_only_exact_sdk_session_expiry_is_classified(self):
        self.assertTrue(is_session_expired(GRPCError(Status.INVALID_ARGUMENT, "SESSION_EXPIRED")))
        for error in (GRPCError(Status.UNKNOWN, "SESSION_EXPIRED"),
                      GRPCError(Status.INVALID_ARGUMENT, "invalid pose"),
                      RuntimeError("SESSION_EXPIRED"), None):
            self.assertFalse(is_session_expired(error))


class ReadOnlyRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.arm = SimpleNamespace(is_moving=AsyncMock(return_value=False), move=AsyncMock(), stop=AsyncMock())
        self.gripper = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                                       is_holding_something=AsyncMock(return_value=SimpleNamespace(is_holding_something=False)),
                                       open=AsyncMock(), grab=AsyncMock(), do_command=AsyncMock())
        self.position = AsyncMock(return_value=840.0)
        self.pose = AsyncMock(return_value=Pose(x=200, y=-400, z=300, o_z=-1))

    async def check(self):
        result = await check_action_recovery(self.arm, self.gripper, self.position, self.pose)
        self.arm.move.assert_not_awaited()
        self.arm.stop.assert_not_awaited()
        self.gripper.open.assert_not_awaited()
        self.gripper.grab.assert_not_awaited()
        self.gripper.do_command.assert_not_awaited()
        return result

    async def test_stopped_open_empty_gripper_allows_acknowledgement(self):
        result = await self.check()
        self.assertTrue(result.safe)
        self.assertEqual(self.arm.is_moving.await_count, 2)
        self.assertEqual(self.gripper.is_moving.await_count, 2)

    async def test_partial_close_even_with_holding_false_stays_paused(self):
        self.position.return_value = 492.0
        result = await self.check()
        self.assertFalse(result.safe)
        self.assertIn("fully open", result.message)

    async def test_movement_unknown_empty_and_invalid_pose_stay_paused(self):
        for moving in (True, None, 0):
            self.arm.is_moving.return_value = moving
            self.assertFalse((await self.check()).safe)
        self.arm.is_moving.return_value = False
        self.gripper.is_holding_something.return_value = False  # malformed SDK response
        self.assertFalse((await self.check()).safe)
        self.gripper.is_holding_something.return_value = SimpleNamespace(is_holding_something=False)
        self.pose.return_value = Pose(z=float("nan"), o_z=-1)
        self.assertFalse((await self.check()).safe)

    async def test_motion_starting_during_checks_stays_paused(self):
        self.arm.is_moving.side_effect = [False, True]
        self.assertFalse((await self.check()).safe)

    async def test_jaws_changing_after_open_readback_stay_paused(self):
        self.position.side_effect = [840.0, 492.0]
        self.assertFalse((await self.check()).safe)

    async def test_readback_failure_does_not_escape_or_retry(self):
        self.position.side_effect = GRPCError(Status.INVALID_ARGUMENT, "SESSION_EXPIRED")
        result = await self.check()
        self.assertFalse(result.safe)
        self.assertIn("Readback failed", result.message)
        self.position.assert_awaited_once()

    async def test_pending_read_checks_are_cancelled_on_quit(self):
        # Use an actual coroutine so cancellation reaches the read operation.
        async def wait():
            await asyncio.sleep(10)
            return RecoveryCheck(True, "ok")
        fault = OperatorFault("session expired")
        fault.start_check(wait)
        task = fault.check_task
        fault.start_check(wait)
        self.assertIs(fault.check_task, task)
        await fault.cancel_check()
        self.assertTrue(task.cancelled())
        self.assertIsNone(fault.check_task)


class FaultWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_expiry_remains_in_gui_and_r_does_not_replay_action(self):
        failed = False

        async def fail_grasp(*args, **kwargs):
            nonlocal failed
            job = args[9]
            job.motion_started = True
            job.requires_operator = True
            job.action_error = GRPCError(Status.INVALID_ARGUMENT, "SESSION_EXPIRED")
            job.status = f"error: {job.action_error}"
            failed = True

        with ExitStack() as stack:
            state = connection_tests.SessionRecoveryTests().mocked_session(stack)
            selected = SimpleNamespace(box=SimpleNamespace(label="block", x0=1, y0=2, x1=3, y1=4))
            state.lock.update.return_value = (None, 1.0, selected)
            grasp = stack.enter_context(patch.object(app, "run_grasp", new=AsyncMock(side_effect=fail_grasp)))
            stop = stack.enter_context(patch.object(app, "stop_everything", new=AsyncMock()))
            checks = stack.enter_context(patch.object(app, "check_action_recovery", new=AsyncMock(
                return_value=RecoveryCheck(False, "Gripper is not fully open"))))
            stack.enter_context(patch.object(app.cv2, "waitKey", side_effect=lambda *_:
                ord("q") if checks.await_count else (ord("r") if failed else 0)))
            await asyncio.wait_for(app.run_session(sentinel.machine), timeout=2)

        grasp.assert_awaited_once()
        stop.assert_awaited_once()
        checks.assert_awaited_once()
        state.lock.release.assert_not_called()
        state.feed.stop.assert_awaited_once()

    async def test_verified_recovery_discards_failed_action_and_stale_frame(self):
        failed = False

        async def fail_grasp(*args, **kwargs):
            nonlocal failed
            job = args[9]
            job.motion_started = True
            job.requires_operator = True
            job.action_error = GRPCError(Status.INVALID_ARGUMENT, "SESSION_EXPIRED")
            job.status = "reopening interrupted"
            failed = True

        with ExitStack() as stack:
            state = connection_tests.SessionRecoveryTests().mocked_session(stack)
            selected = SimpleNamespace(box=SimpleNamespace(label="block", x0=1, y0=2, x1=3, y1=4))
            state.lock.update.return_value = (None, 1.0, selected)
            grasp = stack.enter_context(patch.object(app, "run_grasp", new=AsyncMock(side_effect=fail_grasp)))
            stop = stack.enter_context(patch.object(app, "stop_everything", new=AsyncMock()))
            checks = stack.enter_context(patch.object(app, "check_action_recovery", new=AsyncMock(
                return_value=RecoveryCheck(True, "Stopped and empty", Pose(z=300, o_z=-1)))))
            stack.enter_context(patch.object(app.cv2, "waitKey", side_effect=lambda *_:
                ord("q") if checks.await_count else (ord("r") if failed else 0)))
            await asyncio.wait_for(app.run_session(sentinel.machine), timeout=2)

        grasp.assert_awaited_once()
        stop.assert_awaited_once()
        checks.assert_awaited_once()
        state.lock.release.assert_called()
        self.assertIsNone(state.feed.obs)


if __name__ == "__main__":
    unittest.main()
