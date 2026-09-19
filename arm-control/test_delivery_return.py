"""Release bookkeeping and return sequencing; all robot commands are mocked."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from viam.proto.common import Pose

import delivery as app


class DeliveryReturnTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.held = app.Held(
            label="block", pick_grasp=Pose(x=250, y=-300, z=20, o_z=-1),
            pick_approach=Pose(x=250, y=-300, z=120, o_z=-1),
            orientation={"o_z": -1}, obstacles=None, place_z=25,
            width_mm=28.6, pose=Pose(x=275, y=-325, z=150, o_z=-1),
            home=Pose(x=300, y=-300, z=350, o_z=-1))
        self.io = app.DeliveryIO(
            move=AsyncMock(return_value=True), upright=lambda: "upright",
            straight_line=lambda: "straight", open_gripper=AsyncMock(),
            stop_arm=AsyncMock(), read_pose=AsyncMock(return_value=self.held.pose),
            transform=AsyncMock(), dry_run=False, reach_mm=700, table_top_z=-23,
            camera_name="cam", world_frame="world", intrinsics=None,
            frame_w=1280, frame_h=720, return_after_release=AsyncMock(return_value=True))
        self.session = app.DeliverySession(self.io, self.held, serve=None)

    async def test_put_back_opens_only_after_descent_then_returns(self):
        async def returned(held, status):
            self.assertTrue(self.session.released)
            self.io.open_gripper.assert_awaited_once_with(status)
            self.assertEqual(held.pose.z, self.held.place_z)
            return True

        self.io.return_after_release.side_effect = returned
        await self.session._put_back()

        self.assertEqual(self.io.move.await_count, 2)
        self.assertEqual(self.io.move.await_args_list[-1].args[-1], "straight")
        self.io.return_after_release.assert_awaited_once_with(self.held, self.session.status)
        self.assertTrue(self.session.finished)
        self.assertFalse(self.session.requires_operator)

    async def test_place_down_uses_shared_return_after_release(self):
        await self.session._place_down()

        self.io.move.assert_awaited_once()
        self.assertEqual(self.io.move.await_args.args[0].z, self.held.place_z)
        self.io.return_after_release.assert_awaited_once()
        self.assertTrue(self.session.released)
        self.assertTrue(self.session.finished)

    async def test_gaze_placement_uses_shared_return_after_release(self):
        with patch.object(app, "table_point_from_pixel", new=AsyncMock(return_value=[300, -300, -23])):
            await self.session._place_at_pixel(500, 500, [])

        self.io.return_after_release.assert_awaited_once()
        self.assertTrue(self.session.released)
        self.assertTrue(self.session.finished)

    async def test_let_go_uses_shared_return_without_an_independent_lift(self):
        await self.session._let_go()

        self.io.open_gripper.assert_awaited_once()
        self.io.move.assert_not_awaited()
        self.io.return_after_release.assert_awaited_once()
        self.assertTrue(self.session.released)
        self.assertTrue(self.session.finished)

    async def test_failed_descent_never_opens_or_returns(self):
        self.io.move.side_effect = [True, False]
        await self.session._put_back()

        self.assertEqual(self.io.move.await_count, 2)
        self.io.open_gripper.assert_not_awaited()
        self.io.return_after_release.assert_not_awaited()
        self.assertFalse(self.session.released)
        self.assertFalse(self.session.finished)
        self.assertTrue(self.session.requires_operator)

    async def test_failed_open_does_not_claim_release_or_send_return(self):
        self.io.open_gripper.side_effect = RuntimeError("open failed")
        await self.session._guard(self.session._let_go())

        self.assertFalse(self.session.released)
        self.assertFalse(self.session.finished)
        self.assertTrue(self.session.requires_operator)
        self.assertIsInstance(self.session.error, RuntimeError)
        self.io.return_after_release.assert_not_awaited()

    async def test_failed_return_keeps_released_state_and_pauses(self):
        self.io.return_after_release.return_value = False
        await self.session._let_go()

        self.assertTrue(self.session.released)
        self.assertFalse(self.session.finished)
        self.assertTrue(self.session.requires_operator)
        self.assertIn("return failed", self.session.message)

    async def test_exception_after_release_keeps_released_state_and_pauses(self):
        error = RuntimeError("SESSION_EXPIRED")
        self.io.return_after_release.side_effect = error
        await self.session._guard(self.session._let_go())

        self.assertTrue(self.session.released)
        self.assertFalse(self.session.finished)
        self.assertTrue(self.session.requires_operator)
        self.assertIs(self.session.error, error)

    async def test_missing_return_callback_pauses_instead_of_reporting_success(self):
        self.io.return_after_release = None
        await self.session._let_go()

        self.assertTrue(self.session.released)
        self.assertFalse(self.session.finished)
        self.assertTrue(self.session.requires_operator)
        self.io.move.assert_not_awaited()

    async def test_released_session_cannot_schedule_another_action(self):
        await self.session._let_go()
        self.session._start("raise")
        self.session._run(self.session._let_go())
        await self.session._let_go()

        self.io.open_gripper.assert_awaited_once()
        self.io.return_after_release.assert_awaited_once()
        self.assertIsNone(self.session.task)

    async def test_released_idle_session_does_not_show_or_select_holding_menu(self):
        self.session.released = True
        with patch.object(self.session, "_select") as select:
            canvas = await self.session.tick((50, 50), False, None, None)

        select.assert_not_called()
        self.assertEqual(canvas.shape, (720, 1280, 3))

    async def test_busy_return_keeps_stop_control_available(self):
        self.session.released = True
        self.session.task = asyncio.create_task(asyncio.Event().wait())
        try:
            with patch.object(self.session, "_tick_busy", new=AsyncMock(return_value="stop screen")) as busy:
                self.assertEqual(await self.session.tick(None, False, None, None), "stop screen")
            busy.assert_awaited_once()
        finally:
            self.session.task.cancel()
            await asyncio.gather(self.session.task, return_exceptions=True)

    async def test_stop_after_release_requires_operator_and_preserves_release(self):
        self.session.released = True
        await self.session.stop()

        self.assertTrue(self.session.released)
        self.assertTrue(self.session.requires_operator)
        self.assertFalse(self.session.finished)
        self.io.stop_arm.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
