"""Joint-range diagnostics and startup guards, with all hardware mocked."""
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, sentinel

import main as app
from joint_checks import JointRangeError, read_arm_joint_ranges, require_arm_joint_ranges
import test_connection_recovery as connection_tests


def model():
    names = ['waist', 'shoulder', 'elbow', 'forearm_rot', 'wrist', 'gripper_rot']
    limits = [(-359,359), (-117,116), (-225,10), (-359,359), (-97,179), (-359,359)]
    return dict(links=[dict(id='base', parent='world')], joints=[
        dict(id=name, parent=names[i-1] if i else 'base', type='revolute', min=low, max=high)
        for i, (name, (low, high)) in enumerate(zip(names, limits))])


class JointChecksTests(unittest.IsolatedAsyncioTestCase):
    def arm(self, *, last=-350, data=None, fmt=1):
        return SimpleNamespace(
            get_kinematics=AsyncMock(return_value=(fmt, json.dumps(data or model()).encode(), {})),
            get_joint_positions=AsyncMock(return_value=SimpleNamespace(values=[299,-2,-89,179,-84,last])),
            move_to_joint_positions=AsyncMock())

    async def test_actual_error_readback_is_rejected_without_wrapping_or_movement(self):
        arm = self.arm(last=-359.9936169608495)
        with self.assertRaisesRegex(JointRangeError, r'J6 \(gripper_rot\) = -359.994 deg'):
            await require_arm_joint_ranges(arm)
        arm.move_to_joint_positions.assert_not_awaited()

    async def test_valid_degrees_inside_model_are_accepted(self):
        readings = await require_arm_joint_ranges(self.arm())
        self.assertEqual(readings[-1].degrees, -350)
        self.assertTrue(all(r.in_range for r in readings))

    async def test_positive_full_turn_is_also_rejected(self):
        with self.assertRaises(JointRangeError):
            await require_arm_joint_ranges(self.arm(last=360))

    async def test_joint_chain_order_does_not_depend_on_json_array_order(self):
        data = model()
        data['joints'].reverse()
        readings = await read_arm_joint_ranges(self.arm(data=data))
        self.assertEqual(readings[0].name, 'waist')
        self.assertEqual(readings[-1].name, 'gripper_rot')

    async def test_unknown_format_never_guesses_radians_or_degrees(self):
        with self.assertRaisesRegex(JointRangeError, 'expected.*JSON'):
            await require_arm_joint_ranges(self.arm(fmt=2))

    async def test_bad_readbacks_are_not_accepted(self):
        for values in ([0]*5, [0,0,0,0,0,float('nan')]):
            with self.subTest(values=values):
                arm = self.arm()
                arm.get_joint_positions.return_value.values = values
                with self.assertRaises(JointRangeError):
                    await require_arm_joint_ranges(arm)

    async def test_invalid_limits_and_disconnected_chains_fail_closed(self):
        for field, value in (('min', 400), ('max', float('nan')), ('parent', 'missing')):
            with self.subTest(field=field):
                data = model()
                data['joints'][-1][field] = value
                with self.assertRaises(JointRangeError):
                    await require_arm_joint_ranges(self.arm(data=data))


class StartupJointChecksTests(unittest.IsolatedAsyncioTestCase):
    async def test_out_of_range_joint_blocks_startup_before_camera_or_motion(self):
        with ExitStack() as stack:
            connection_tests.SessionRecoveryTests().mocked_session(stack, home=app.Pose(z=383))
            app.require_arm_joint_ranges.side_effect = JointRangeError('J6 outside [-359, 359] deg')
            move = stack.enter_context(patch.object(app, 'move_to', new=AsyncMock()))
            with self.assertRaisesRegex(SystemExit, 'START BLOCKED: J6'):
                await app.run_session(sentinel.machine)
            move.assert_not_awaited()
            app.WebcamGazeTracker.assert_not_called()

    async def test_home_is_loaded_without_launch_movement_when_opt_in_is_off(self):
        with ExitStack() as stack:
            state = connection_tests.SessionRecoveryTests().mocked_session(stack, home=app.Pose(z=383))
            state.lock.update.return_value = (None, 0.0, None)
            stack.enter_context(patch.object(app, 'GO_HOME_ON_START', False))
            stack.enter_context(patch.object(app.cv2, 'waitKey', return_value=ord('q')))
            move = stack.enter_context(patch.object(app, 'move_to', new=AsyncMock()))
            await app.run_session(sentinel.machine)
            app.require_arm_joint_ranges.assert_awaited_once()
            state.feed.first.assert_awaited_once()
            move.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
