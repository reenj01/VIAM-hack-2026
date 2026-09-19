"""Robot pose and depth consistency checks, with no hardware access."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from viam.proto.common import Pose
import main as app


class GraspArrivalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(app, 'DRY_RUN', False))
        self.target = Pose(x=250, y=-380, z=30, o_z=-1, theta=20)
        self.read = self.enterContext(patch.object(app, 'gripper_world_pose', new=AsyncMock(return_value=self.target)))
        self.arm = SimpleNamespace(is_moving=AsyncMock(return_value=False))
        self.job = app.GraspJob()

    async def check(self):
        return await app.verify_grasp_arrival(object(), self.arm, self.target, self.job)

    async def test_arrival_requires_two_stationary_correct_readbacks(self):
        self.assertTrue(await self.check())
        self.assertEqual(self.read.await_count, 2)

    async def test_success_response_does_not_allow_close_above_target(self):
        self.read.return_value = Pose(x=250, y=-380, z=130, o_z=-1, theta=20)
        self.assertFalse(await self.check())
        self.assertIn('NOT CLOSING', self.job.status)

    async def test_motion_or_wrong_rotation_prevents_close(self):
        self.arm.is_moving.return_value = True
        self.assertFalse(await self.check())
        self.arm.is_moving.return_value = False
        self.read.return_value = Pose(x=250, y=-380, z=30, o_z=-1, theta=35)
        self.assertFalse(await self.check())

    async def test_second_readback_can_invalidate_first(self):
        self.read.side_effect = [self.target, Pose(x=250, y=-380, z=40, o_z=-1, theta=20)]
        self.assertFalse(await self.check())

    async def test_unknown_pose_remains_an_error(self):
        self.read.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'verify'):
            await self.check()

    async def test_dry_run_does_not_read_robot(self):
        with patch.object(app, 'DRY_RUN', True):
            self.assertTrue(await self.check())
        self.read.assert_not_awaited()
        self.arm.is_moving.assert_not_awaited()


class TargetRepeatTests(unittest.TestCase):
    def test_large_recorded_height_change_is_rejected(self):
        a = Pose(x=250, y=-380, z=150, o_z=1)
        b = Pose(x=250, y=-380, z=1, o_z=1)
        self.assertIn('disagree', app.target_repeat_problem(a, (49,49,60), b, (49,49,60)))

    def test_stable_box_passes_but_size_change_does_not(self):
        a = Pose(x=250, y=-380, z=20, o_z=1)
        b = Pose(x=251, y=-381, z=22, o_z=1)
        self.assertIsNone(app.target_repeat_problem(a, (30,30,60), b, (31,29,61)))
        self.assertIsNotNone(app.target_repeat_problem(a, (30,30,60), b, (49,49,60)))

    def test_operator_width_is_explicit_label_scoped_and_gives_correct_close(self):
        self.assertEqual(app.grasp_settings([]).object_width_mm, {})
        widths = app.grasp_settings(['--object-width-mm', 'block=28.6']).object_width_mm
        self.assertEqual(widths, {'block':28.6})
        self.assertEqual(app.gripper_position_for_width(widths['block']), 276)


if __name__ == '__main__':
    unittest.main()
