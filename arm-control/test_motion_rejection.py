"""Planner rejection and failed-job handling with all hardware access mocked."""

import asyncio
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, sentinel

from grpclib import GRPCError, Status

import main as app
import test_connection_recovery as recovery_tests


IK_MESSAGE = ("all IK solutions failed constraints. Failures: { robot constraint: "
              "violation between gripper:claws and table:table geometries: 100.00% },")


class MotionRejectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(app, "DRY_RUN", False))
        self.pose = app.Pose(x=250, y=-300, z=10, o_z=-1)
        self.obstacles = app.WorldState()
        self.job = app.GraspJob()

    async def test_ik_rejection_preserves_obstacles_and_blocks_further_commands(self):
        motion = SimpleNamespace(move=AsyncMock(side_effect=GRPCError(Status.UNKNOWN, IK_MESSAGE)))

        self.assertFalse(await app.move_to(motion, self.pose, "grasp", self.job, self.obstacles))
        self.assertFalse(await app.move_to(motion, self.pose, "back up", self.job, self.obstacles))

        motion.move.assert_awaited_once()
        self.assertIs(motion.move.await_args.kwargs["world_state"], self.obstacles)
        self.assertFalse(self.job.motion_started)
        self.assertFalse(self.job.requires_operator)
        self.assertIn("PLAN REJECTED at grasp", self.job.plan_rejection)
        self.assertIn("gripper:claws", self.job.status)

    async def test_rejected_later_stage_preserves_previous_physical_action_record(self):
        self.job.motion_started = True
        self.job.holding = True
        motion = SimpleNamespace(move=AsyncMock(side_effect=GRPCError(Status.UNKNOWN, IK_MESSAGE)))

        self.assertFalse(await app.move_to(motion, self.pose, "lift", self.job))

        self.assertTrue(self.job.motion_started)
        self.assertTrue(self.job.holding)

    async def test_uncertain_failures_are_not_classified_as_planning_rejections(self):
        for error in (GRPCError(Status.UNKNOWN, "arm command failed"),
                      GRPCError(Status.UNKNOWN, "channel closed"),
                      GRPCError(Status.UNAVAILABLE, IK_MESSAGE),
                      TimeoutError("deadline")):
            with self.subTest(error=repr(error)):
                job = app.GraspJob()
                motion = SimpleNamespace(move=AsyncMock(side_effect=error))
                with self.assertRaises(type(error)) as caught:
                    await app.move_to(motion, self.pose, "grasp", job)
                self.assertIs(caught.exception, error)
                self.assertTrue(job.motion_started)
                self.assertIsNone(job.plan_rejection)

    async def test_false_result_does_not_retry_without_obstacles(self):
        motion = SimpleNamespace(move=AsyncMock(return_value=False))

        self.assertFalse(await app.move_to(motion, self.pose, "grasp", self.job, self.obstacles))
        self.assertFalse(await app.move_to(motion, self.pose, "back up", self.job, self.obstacles))

        motion.move.assert_awaited_once()
        self.assertIs(motion.move.await_args.kwargs["world_state"], self.obstacles)
        self.assertTrue(self.job.motion_started)
        self.assertIsNone(self.job.plan_rejection)
        self.assertEqual(self.job.motion_failure, "motion.move() failed on grasp")

    async def test_rejected_descent_never_closes_or_sends_a_retreat_command(self):
        motion = SimpleNamespace(move=AsyncMock(side_effect=[True, GRPCError(Status.UNKNOWN, IK_MESSAGE)]))
        plan = SimpleNamespace(approach=app.Pose(z=110), grasp=self.pose, width_mm=40)

        async def opened(gripper, job):
            job.motion_started = True
            job.holding = False

        with patch.object(app, "open_gripper", new=AsyncMock(side_effect=opened)), \
                patch.object(app, "verify_grasp_arrival", new=AsyncMock()) as arrival, \
                patch.object(app, "close_gripper", new=AsyncMock()) as close:
            self.assertFalse(await app.execute_grasp(motion, sentinel.gripper, sentinel.arm,
                                                     plan, self.obstacles, self.job))

        self.assertEqual(motion.move.await_count, 2)
        arrival.assert_not_awaited()
        close.assert_not_awaited()
        self.assertFalse(self.job.holding)
        self.assertIn("grasp (straight down)", self.job.plan_rejection)


class PutBackOutcomeTests(unittest.IsolatedAsyncioTestCase):
    def held(self):
        return SimpleNamespace(label="block", pick_grasp=app.Pose(x=250, y=-300, z=10),
                               pick_approach=app.Pose(x=250, y=-300, z=110),
                               place_z=15, orientation={"o_z": -1}, obstacles=None,
                               footprint=None, home=app.Pose(z=300), pose=app.Pose(z=110))

    async def test_rejected_retreat_keeps_released_state_and_reports_failure(self):
        job = app.GraspJob()
        held = self.held()
        with patch.object(app, "move_to", new=AsyncMock(side_effect=[True, True, False])) as move, \
                patch.object(app, "open_gripper", new=AsyncMock()) as opening:
            self.assertFalse(await app.run_put_back(sentinel.motion, sentinel.gripper, held, job))

        self.assertEqual(move.await_count, 3)
        opening.assert_awaited_once()
        self.assertIsNone(job.held)
        self.assertFalse(job.ok)
        self.assertIn("released it", job.status)

    async def test_rejected_home_return_is_not_reported_as_success(self):
        job = app.GraspJob()
        held = self.held()
        with patch.object(app, "move_to", new=AsyncMock(side_effect=[True, True, True, False])), \
                patch.object(app, "open_gripper", new=AsyncMock()):
            self.assertFalse(await app.run_put_back(sentinel.motion, sentinel.gripper, held, job))

        self.assertIsNone(job.held)
        self.assertFalse(job.ok)
        self.assertIn("couldn't return", job.status)


class StartupMotionTests(unittest.IsolatedAsyncioTestCase):
    async def test_refused_home_stops_before_selection_with_concise_stage_message(self):
        with ExitStack() as stack:
            state = recovery_tests.SessionRecoveryTests().mocked_session(stack, home=app.Pose(z=300))
            motion = app.MotionClient.from_robot.return_value
            motion.move = AsyncMock(side_effect=GRPCError(Status.UNKNOWN, IK_MESSAGE))

            with self.assertRaisesRegex(SystemExit, "PLAN REJECTED at startup home") as caught:
                await app.run_session(sentinel.machine)

            app.WebcamGazeTracker.assert_not_called()
            app.RobotFeed.assert_not_called()

        self.assertIn("saved home pose and gripper/table geometry", str(caught.exception))
        motion.move.assert_awaited_once()
        state.feed.first.assert_not_awaited()

    async def test_false_home_result_stops_without_claiming_no_execution(self):
        with ExitStack() as stack:
            recovery_tests.SessionRecoveryTests().mocked_session(stack, home=app.Pose(z=300))
            motion = app.MotionClient.from_robot.return_value
            motion.move = AsyncMock(return_value=False)

            with self.assertRaisesRegex(SystemExit, "motion.move\\(\\) failed on startup home") as caught:
                await app.run_session(sentinel.machine)

            app.WebcamGazeTracker.assert_not_called()

        self.assertNotIn("PLAN REJECTED", str(caught.exception))
        motion.move.assert_awaited_once()

    async def test_unknown_home_exception_keeps_runtime_error_and_original_cause(self):
        failure = GRPCError(Status.UNKNOWN, "arm execution failed")
        with ExitStack() as stack:
            recovery_tests.SessionRecoveryTests().mocked_session(stack, home=app.Pose(z=300))
            motion = app.MotionClient.from_robot.return_value
            motion.move = AsyncMock(side_effect=failure)

            with self.assertRaisesRegex(RuntimeError, "Startup movement failed") as caught:
                await app.run_session(sentinel.machine)

            app.WebcamGazeTracker.assert_not_called()

        self.assertIs(caught.exception.__cause__, failure)
        motion.move.assert_awaited_once()


class FailedJobPauseTests(unittest.IsolatedAsyncioTestCase):
    def setup_session(self, stack, *, held=None):
        state = recovery_tests.SessionRecoveryTests().mocked_session(stack)
        selected = SimpleNamespace(box=SimpleNamespace(label="block", x0=1, y0=2, x1=3, y1=4))
        created = []

        def select(*args):
            state.lock.is_locked = True
            return None, 1.0, selected

        def release():
            state.lock.is_locked = False

        async def reject(*args, **kwargs):
            job = args[9]
            created.append(job)
            job.finished_at = app.time.monotonic() - 60
            job.status = "grip not confirmed"
            job.held = held
            return False

        state.lock.update.side_effect = select
        state.lock.release.side_effect = release
        state.job = created
        state.grasp = stack.enter_context(patch.object(app, "run_grasp", new=AsyncMock(side_effect=reject)))
        stack.enter_context(patch.object(app, "draw_locked", return_value=state.feed.obs.frame))
        stack.enter_context(patch.object(app, "stop_everything", new=AsyncMock()))
        return state

    async def test_failed_attempt_stays_paused_past_old_timeout_without_reselecting(self):
        with ExitStack() as stack:
            state = self.setup_session(stack)
            paused_frames = 0

            def key(_):
                nonlocal paused_frames
                if state.job:
                    paused_frames += 1
                    if paused_frames >= 3:
                        return ord("q")
                return 0

            stack.enter_context(patch.object(app.cv2, "waitKey", side_effect=key))
            await asyncio.wait_for(app.run_session(sentinel.machine), timeout=2)

        self.assertEqual(paused_frames, 3)
        state.grasp.assert_awaited_once()
        state.lock.update.assert_called_once()
        state.lock.release.assert_not_called()

    async def test_failed_attempt_requires_r_before_releasing_selection(self):
        with ExitStack() as stack:
            state = self.setup_session(stack)

            def key(_):
                if state.lock.release.called:
                    return ord("q")
                return ord("r") if state.job else 0

            stack.enter_context(patch.object(app.cv2, "waitKey", side_effect=key))
            await asyncio.wait_for(app.run_session(sentinel.machine), timeout=2)

        state.grasp.assert_awaited_once()
        state.lock.release.assert_called_once()

    async def test_failed_lift_does_not_automatically_open_delivery_menu(self):
        with ExitStack() as stack:
            state = self.setup_session(stack, held=SimpleNamespace(label="block"))
            stack.enter_context(patch.object(app, "MENU", True))
            menu = stack.enter_context(patch.object(app, "DeliverySession"))
            stack.enter_context(patch.object(app.cv2, "waitKey", side_effect=lambda _: ord("q") if state.job else 0))
            await asyncio.wait_for(app.run_session(sentinel.machine), timeout=2)

        menu.assert_not_called()
        self.assertEqual(state.job[0].held.label, "block")

    async def test_acknowledgement_preserves_held_object_when_opening_delivery_menu(self):
        held = SimpleNamespace(label="block")
        with ExitStack() as stack:
            state = self.setup_session(stack, held=held)
            stack.enter_context(patch.object(app, "MENU", True))
            session = SimpleNamespace(error=None, wants_feed=False, finished=False, busy=False,
                                      tick=AsyncMock(return_value=state.feed.obs.frame), stop=AsyncMock())
            menu = stack.enter_context(patch.object(app, "DeliverySession", return_value=session))

            def key(_):
                if menu.called:
                    return ord("q")
                return ord("r") if state.job else 0

            stack.enter_context(patch.object(app.cv2, "waitKey", side_effect=key))
            await asyncio.wait_for(app.run_session(sentinel.machine), timeout=2)

        menu.assert_called_once()
        self.assertIs(menu.call_args.args[1], held)
        session.stop.assert_awaited_once_with("Quit")


if __name__ == "__main__":
    unittest.main()
