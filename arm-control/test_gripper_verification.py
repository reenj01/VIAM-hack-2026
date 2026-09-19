"""Grasp verification regressions; every device operation is mocked."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main as app


class GripperVerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(app, "DRY_RUN", False))
        self.arm = SimpleNamespace(do_command=AsyncMock())
        self.job = app.GraspJob()

    def gripper(self, positions=(840, 400, 400, 400), response=None, holding=True, moving=False):
        readings = iter(positions)
        acknowledgement = {"position": 400} if response is None else response

        async def command(payload, **kwargs):
            if payload == {"get": True}:
                return {"pos": next(readings)}
            if "set" in payload:
                return acknowledgement
            raise AssertionError(f"unexpected command {payload}")

        return SimpleNamespace(
            do_command=AsyncMock(side_effect=command),
            is_moving=AsyncMock(return_value=moving),
            is_holding_something=AsyncMock(
                return_value=SimpleNamespace(is_holding_something=holding)),
            open=AsyncMock(), grab=AsyncMock(),
        )

    async def close(self, gripper, width=40.0):
        result = await app.close_gripper(gripper, self.arm, width, self.job)
        self.arm.do_command.assert_not_awaited()
        gripper.grab.assert_not_awaited()
        return result

    async def test_contact_requires_measured_closure_and_stopping_before_target(self):
        gripper = self.gripper()
        self.assertTrue(await self.close(gripper))
        self.assertTrue(self.job.holding)
        commands = [call.args[0] for call in gripper.do_command.await_args_list]
        self.assertEqual(commands, [{"get": True}, {"set": 390}] + [{"get": True}] * 3)

    async def test_empty_partial_close_is_not_holding_even_if_driver_says_true(self):
        gripper = self.gripper(positions=(840, 390, 390, 390), response={"position": 390})
        self.assertFalse(await self.close(gripper))
        self.assertFalse(self.job.holding)
        self.assertIn("without evidence", self.job.status)

    async def test_unchanged_open_jaws_are_not_a_grasp(self):
        gripper = self.gripper(positions=(840, 840, 840, 840), response={"position": 840})
        self.assertFalse(await self.close(gripper))
        self.assertIn("no measurable closure", self.job.status)

    async def test_fully_closed_empty_jaws_are_not_a_grasp(self):
        gripper = self.gripper(positions=(840, 0, 0, 0), response={"position": 0})
        self.assertFalse(await self.close(gripper))
        self.assertIn("endpoint", self.job.status)

    async def test_far_away_obstruction_is_not_the_target(self):
        gripper = self.gripper(positions=(840, 600, 600, 600), response={"position": 600})
        self.assertFalse(await self.close(gripper))
        self.assertIn("too far apart", self.job.status)

    async def test_close_requires_previously_opened_jaws(self):
        gripper = self.gripper(positions=(820,))
        self.assertFalse(await self.close(gripper))
        self.assertFalse(self.job.motion_started)
        gripper.do_command.assert_awaited_once()

    async def test_unacknowledged_command_does_not_trigger_another_close(self):
        gripper = self.gripper(positions=(840,), response={})
        self.assertFalse(await self.close(gripper))
        self.assertIn("did not acknowledge", self.job.status)
        self.assertEqual(gripper.do_command.await_count, 2)

    async def test_invalid_initial_position_stops_before_write(self):
        for position in (None, "840", True, float("nan"), float("inf"), -1, 851):
            with self.subTest(position=position):
                gripper = self.gripper(positions=(position,))
                with self.assertRaisesRegex(RuntimeError, "no valid"):
                    await self.close(gripper)
                gripper.do_command.assert_awaited_once()
                gripper.grab.assert_not_awaited()

    async def test_malformed_acknowledgement_leaves_holding_unknown(self):
        gripper = self.gripper(positions=(840,), response={"position": float("nan")})
        with self.assertRaisesRegex(RuntimeError, "no valid"):
            await self.close(gripper)
        self.assertIsNone(self.job.holding)
        self.assertEqual(gripper.do_command.await_count, 2)

    async def test_invalid_size_never_falls_back_to_unrestricted_grab(self):
        for width in (0, -1, 100, float("nan"), float("inf")):
            with self.subTest(width=width):
                gripper = self.gripper()
                self.assertFalse(await self.close(gripper, width))
                gripper.do_command.assert_not_awaited()

    async def test_false_holding_field_is_used_instead_of_truthy_wrapper(self):
        gripper = self.gripper(holding=False)
        self.assertFalse(await self.close(gripper))
        self.assertFalse(self.job.holding)
        self.assertIn("reports no object", self.job.status)

    async def test_position_must_settle_before_contact_is_accepted(self):
        gripper = self.gripper(positions=(840, 600, 480, 430, 400, 400, 400))
        self.assertTrue(await self.close(gripper))
        self.assertEqual(gripper.is_moving.await_count, 6)

    async def test_unknown_post_command_readback_cannot_be_called_holding(self):
        gripper = self.gripper(positions=(840, None))
        with self.assertRaisesRegex(RuntimeError, "no valid"):
            await self.close(gripper)
        self.assertIsNone(self.job.holding)

    async def test_verification_is_bounded_while_motion_continues(self):
        gripper = self.gripper(positions=(840, 400), moving=True)
        with patch.object(app, "GRIPPER_VERIFY_TIMEOUT_S", 0.01):
            with self.assertRaises(TimeoutError):
                await self.close(gripper)
        self.assertIsNone(self.job.holding)
        gripper.grab.assert_not_awaited()

    async def test_open_waits_for_actual_open_position(self):
        gripper = self.gripper(positions=(100, 600, 840))
        await app.open_gripper(gripper, self.job)
        gripper.open.assert_awaited_once()
        self.assertEqual(gripper.do_command.await_count, 3)
        self.assertFalse(self.job.holding)

    async def test_unknown_open_position_prevents_descent(self):
        gripper = self.gripper(positions=(None,))
        with self.assertRaisesRegex(RuntimeError, "no valid"):
            await app.open_gripper(gripper, self.job)
        self.assertIsNone(self.job.holding)

    async def test_dry_run_does_not_access_hardware_or_claim_verified_holding(self):
        gripper = self.gripper()
        with patch.object(app, "DRY_RUN", True):
            self.assertTrue(await self.close(gripper))
            await app.open_gripper(gripper, self.job)
        self.assertIsNone(self.job.holding)
        gripper.do_command.assert_not_awaited()
        gripper.open.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
