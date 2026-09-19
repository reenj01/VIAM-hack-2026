"""Connection recovery regression checks; all robot and camera access is mocked.

Run from arm-control with: python -m unittest test_connection_recovery -v
"""

import asyncio
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch, sentinel

from grpclib import GRPCError, Status
from grpclib.exceptions import StreamTerminatedError

import main as app


class ConnectionOptionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_disables_both_sdk_probe_intervals(self):
        credentials = {
            "VIAM_MACHINE_ADDRESS": "test-machine.invalid",
            "VIAM_API_KEY": "test-key",
            "VIAM_API_KEY_ID": "00000000-0000-0000-0000-000000000001",
        }
        with patch.object(app, "load_env", return_value=credentials), patch.object(
            app.RobotClient, "at_address", new_callable=AsyncMock, return_value=sentinel.machine
        ) as dial:
            machine = await app.connect()

        self.assertIs(machine, sentinel.machine)
        dial.assert_awaited_once()
        address, options = dial.await_args.args
        self.assertEqual(address, credentials["VIAM_MACHINE_ADDRESS"])
        self.assertEqual(options.check_connection_interval, 0)
        self.assertEqual(options.attempt_reconnect_interval, 0)


class TransportClassificationTests(unittest.TestCase):
    def test_transport_failures_require_a_new_connection(self):
        for error in (
            ConnectionError("closed"),
            StreamTerminatedError("connection closed"),
            GRPCError(Status.UNAVAILABLE, "DataChannel is not opened"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(app.is_transport_error(error))

    def test_timeouts_and_application_errors_do_not_imply_disconnect(self):
        for error in (
            TimeoutError("segmentation took too long"),
            GRPCError(Status.INVALID_ARGUMENT, "invalid pose"),
            RuntimeError("no decodable image"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(app.is_transport_error(error))


class RobotFeedRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_unpaired_transport_failure_does_not_retry_camera_capture(self):
        camera = SimpleNamespace(get_images=AsyncMock(side_effect=ConnectionError("closed transport")))
        detector = SimpleNamespace(get_detections_from_camera=AsyncMock())
        feed = app.RobotFeed(camera, detector)

        with self.assertRaises(ConnectionError):
            await feed._capture_unpaired()

        camera.get_images.assert_awaited_once()
        detector.get_detections_from_camera.assert_not_awaited()

    async def test_paired_transport_failure_does_not_try_separate_captures(self):
        feed = app.RobotFeed(Mock(), Mock())
        failure = ConnectionError("closed transport")
        with patch.object(feed, "_capture_paired", new=AsyncMock(side_effect=failure)), patch.object(
            feed, "_capture_unpaired", new=AsyncMock(return_value=sentinel.observation)
        ) as fallback:
            with self.assertRaises(ConnectionError):
                await feed.capture()

        fallback.assert_not_awaited()
        self.assertTrue(feed.paired)

    async def test_loop_clears_observation_and_stops_on_transport_failure(self):
        feed = app.RobotFeed(Mock(), Mock())
        feed.obs = sentinel.old_observation
        failure = ConnectionError("closed transport")
        with patch.object(feed, "capture", new=AsyncMock(side_effect=failure)) as capture:
            await asyncio.wait_for(feed._loop(), timeout=1)

        capture.assert_awaited_once()
        self.assertIsNone(feed.obs)
        self.assertIsNotNone(feed.connection_error)
        self.assertTrue(app.is_transport_error(feed.connection_error))


class GripperRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_uncertain_command_result_does_not_replay_close_through_another_api(self):
        for failure in (
            ConnectionError("closed transport"),
            TimeoutError("response deadline elapsed"),
            GRPCError(Status.DEADLINE_EXCEEDED, "response deadline elapsed"),
            GRPCError(Status.CANCELLED, "request cancelled"),
        ):
            with self.subTest(error=type(failure).__name__, message=str(failure)):
                gripper = SimpleNamespace(do_command=AsyncMock(side_effect=failure), grab=AsyncMock())
                arm = SimpleNamespace(do_command=AsyncMock())
                job = app.GraspJob()

                with patch.object(app, "DRY_RUN", False):
                    with self.assertRaises(type(failure)) as caught:
                        await app.close_gripper(gripper, arm, 40.0, job)

                self.assertIs(caught.exception, failure)
                self.assertTrue(job.motion_started)
                gripper.do_command.assert_awaited_once()
                arm.do_command.assert_not_awaited()
                gripper.grab.assert_not_awaited()


class DeliveryRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_menu_action_records_transport_error_for_session_supervisor(self):
        session = app.DeliverySession.__new__(app.DeliverySession)
        session.error = None
        failure = ConnectionError("delivery connection closed")
        action = AsyncMock(side_effect=failure)

        await session._guard(action())

        action.assert_awaited_once()
        self.assertIs(session.error, failure)
        self.assertIn(str(failure), session.message)


class SessionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def mocked_session(self, stack, *, home=None, feed_error=None):
        """Patch every device constructor and UI operation before run_session."""
        camera, detector, segmenter, motion, arm = (Mock() for _ in range(5))
        gripper = SimpleNamespace(is_holding_something=AsyncMock(return_value=False))
        gaze = Mock()
        frame = app.np.zeros((480, 640, 3), dtype=app.np.uint8)
        observation = SimpleNamespace(frame=frame, boxes=[], at=app.time.monotonic())
        feed = SimpleNamespace(
            first=AsyncMock(return_value=(640, 480), side_effect=feed_error),
            start=Mock(), stop=AsyncMock(), connection_error=None, obs=observation,
            paused=False, paired=True, fps=1.0, capture_ms=1.0,
        )
        lock = Mock(is_locked=False)
        estimator = Mock(read=Mock(return_value=((100, 100), 1.0, False)))
        patches = (
            patch.object(app, "EXECUTE", True), patch.object(app, "DRY_RUN", False),
            patch.object(app, "SET_HOME", False), patch.object(app, "SET_SERVE", False),
            patch.object(app, "GO_HOME_ON_START", True), patch.object(app, "MENU", False),
            patch.object(app, "USER_PROFILE", "eyes"),
            patch.object(app, "resolve_motion_name", return_value="motion"),
            patch.object(app.Camera, "from_robot", return_value=camera),
            patch.object(app.VisionClient, "from_robot", side_effect=(detector, segmenter)),
            patch.object(app.MotionClient, "from_robot", return_value=motion),
            patch.object(app.Gripper, "from_robot", return_value=gripper),
            patch.object(app.Arm, "from_robot", return_value=arm),
            patch.object(app, "load_serve_pose", return_value=None),
            patch.object(app, "measure_tcp_mm", new=AsyncMock(return_value=150.0)),
            patch.object(app, "get_intrinsics", new=AsyncMock(return_value=None)),
            patch.object(app, "load_home_pose", return_value=home),
            patch.object(app, "WebcamGazeTracker", return_value=gaze),
            patch.object(app, "GazeLockController", return_value=lock),
            patch.object(app, "RobotFeed", return_value=feed),
            patch.object(app, "load_or_calibrate", return_value=sentinel.calibration),
            patch.object(app, "GazeEstimator", return_value=estimator),
            patch.object(app, "make_delivery_io", return_value=sentinel.delivery),
            patch.object(app, "report_task_exception"),
            patch.object(app, "put_text"),
            patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "imshow"),
            patch.object(app.cv2, "waitKey", return_value=0), patch.object(app.cv2, "putText"),
            patch.object(app.cv2, "destroyAllWindows"),
        )
        for item in patches:
            stack.enter_context(item)
        return SimpleNamespace(feed=feed, lock=lock, gaze=gaze, gripper=gripper)

    async def test_recovery_does_not_repeat_startup_home_movement(self):
        failure = RuntimeError("stop before UI loop")
        with ExitStack() as stack:
            state = self.mocked_session(stack, home=app.Pose(o_z=-1.0), feed_error=failure)
            move = stack.enter_context(patch.object(app, "move_to", new=AsyncMock()))
            with self.assertRaises(RuntimeError) as caught:
                await app.run_session(sentinel.machine, recovering=True)

        self.assertIs(caught.exception, failure)
        move.assert_not_awaited()
        state.gripper.is_holding_something.assert_awaited_once()
        state.feed.stop.assert_awaited_once()
        state.gaze.close.assert_called_once()

    async def test_disconnect_after_motion_started_requires_operator_restart(self):
        async def fail_grasp(*args, **kwargs):
            job = args[9]
            job.motion_started = True
            job.connection_error = ConnectionError("response lost after command")

        with ExitStack() as stack:
            state = self.mocked_session(stack)
            selected = SimpleNamespace(box=SimpleNamespace(label="cup", x0=1, y0=2, x1=3, y1=4))
            state.lock.update.return_value = (None, 1.0, selected)
            stack.enter_context(patch.object(app, "run_grasp", new=AsyncMock(side_effect=fail_grasp)))
            stop = stack.enter_context(patch.object(app, "stop_everything", new=AsyncMock()))

            with self.assertRaisesRegex(RuntimeError, "will not be replayed") as caught:
                await asyncio.wait_for(app.run_session(sentinel.machine), timeout=2)

        self.assertFalse(app.is_transport_error(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, app.ReconnectRequired)
        stop.assert_awaited()
        state.feed.stop.assert_awaited_once()
        state.gaze.close.assert_called_once()


class ReconnectSupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def test_closes_old_client_before_redial_and_marks_recovered_session(self):
        events = []

        async def old_close():
            events.append("close old")

        async def new_close():
            events.append("close new")

        old_machine = SimpleNamespace(close=AsyncMock(side_effect=old_close))
        new_machine = SimpleNamespace(close=AsyncMock(side_effect=new_close))
        clients = iter((old_machine, new_machine))

        async def connect():
            machine = next(clients)
            events.append("connect old" if machine is old_machine else "connect new")
            return machine

        async def run_session(machine, recovering=False):
            events.append(("session", machine is new_machine, recovering))
            if machine is old_machine:
                raise app.ReconnectRequired("closed transport")

        with patch.object(app, "connect", new=AsyncMock(side_effect=connect)) as dial, patch.object(
            app, "run_session", new=AsyncMock(side_effect=run_session)
        ), patch.object(app.asyncio, "sleep", new=AsyncMock()):
            await app.main()

        self.assertEqual(dial.await_count, 2)
        old_machine.close.assert_awaited_once()
        new_machine.close.assert_awaited_once()
        self.assertEqual(
            events,
            ["connect old", ("session", False, False), "close old",
             "connect new", ("session", True, True), "close new"],
        )

    async def test_application_error_closes_client_and_does_not_retry(self):
        machine = SimpleNamespace(close=AsyncMock())
        failure = RuntimeError("invalid application state")
        with patch.object(app, "connect", new=AsyncMock(return_value=machine)) as dial, patch.object(
            app, "run_session", new=AsyncMock(side_effect=failure)
        ) as session, patch.object(app.asyncio, "sleep", new=AsyncMock()):
            with self.assertRaises(RuntimeError) as caught:
                await app.main()

        self.assertIs(caught.exception, failure)
        dial.assert_awaited_once()
        session.assert_awaited_once()
        machine.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
