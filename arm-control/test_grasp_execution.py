"""Exercise grasp sequencing without connecting to any robot hardware."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, sentinel

import main as app


class GraspExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(app, "DRY_RUN", False))
        self.events = []
        self.job = app.GraspJob()
        self.plan = SimpleNamespace(approach=sentinel.approach, grasp=sentinel.grasp, width_mm=40.0)
        self.close_confirmed = True
        self.open_error = None
        self.lift_ok = True

        async def move(motion, pose, label, job, *args):
            self.events.append(("move", label, pose))
            return self.lift_ok if label == "lift (straight up)" else True

        async def open_jaws(gripper, job):
            self.events.append(("open",))
            if self.open_error is not None:
                raise self.open_error
            job.holding = False

        async def close_jaws(gripper, arm, width_mm, job):
            self.events.append(("close", width_mm))
            job.holding = self.close_confirmed
            job.status = "contact confirmed" if self.close_confirmed else "no measured closure"
            return self.close_confirmed

        self.move = self.enterContext(patch.object(app, "move_to", new=AsyncMock(side_effect=move)))
        self.open = self.enterContext(patch.object(app, "open_gripper", new=AsyncMock(side_effect=open_jaws)))
        self.close = self.enterContext(patch.object(app, "close_gripper", new=AsyncMock(side_effect=close_jaws)))
        self.arrival = self.enterContext(patch.object(app, "verify_grasp_arrival", new=AsyncMock(return_value=True)))

    async def execute(self):
        return await app.execute_grasp(
            sentinel.motion, sentinel.gripper, sentinel.arm, self.plan, None, self.job)

    async def test_unconfirmed_close_releases_at_table_then_retreats_without_lifting(self):
        self.close_confirmed = False
        self.assertFalse(await self.execute())
        self.assertEqual(self.events, [
            ("move", "approach", sentinel.approach),
            ("open",),
            ("move", "grasp (straight down)", sentinel.grasp),
            ("close", 40.0),
            ("open",),
            ("move", "retreat after unconfirmed grip", sentinel.approach),
        ])
        self.assertFalse(self.job.holding)
        self.assertIn("pickup NOT confirmed", self.job.status)
        self.assertIn("no measured closure", self.job.status)

    async def test_failed_open_verification_prevents_descent_and_close(self):
        self.open_error = RuntimeError("opening was not verified")
        with self.assertRaisesRegex(RuntimeError, "opening was not verified"):
            await self.execute()
        self.assertEqual(self.events, [
            ("move", "approach", sentinel.approach), ("open",),
        ])
        self.close.assert_not_awaited()

    async def test_failed_lift_does_not_report_success_or_release_held_object(self):
        self.lift_ok = False
        self.assertFalse(await self.execute())
        self.assertTrue(self.job.holding)
        self.assertIn("lift failed", self.job.status)
        self.open.assert_awaited_once()
        self.assertEqual(self.events[-1], ("move", "lift (straight up)", sentinel.approach))

    async def test_verified_close_allows_lift_and_success(self):
        self.assertTrue(await self.execute())
        self.assertTrue(self.job.holding)
        self.assertEqual(self.events, [
            ("move", "approach", sentinel.approach),
            ("open",),
            ("move", "grasp (straight down)", sentinel.grasp),
            ("close", 40.0),
            ("move", "lift (straight up)", sentinel.approach),
        ])

    async def test_unreached_grasp_pose_never_closes_or_lifts(self):
        self.arrival.return_value = False
        self.assertFalse(await self.execute())
        self.close.assert_not_awaited()
        self.assertEqual(self.events[-1], ("move", "grasp (straight down)", sentinel.grasp))


if __name__ == "__main__":
    unittest.main()
