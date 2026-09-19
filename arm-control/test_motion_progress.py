"""A pending/cancelled motion RPC is observable and is never replayed."""
import asyncio
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main as app


class MotionProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_rpc_reports_elapsed_time_and_cleans_up_on_success(self):
        called, release = asyncio.Event(), asyncio.Event()
        async def pending(**kwargs):
            called.set()
            await release.wait()
            return True
        motion = SimpleNamespace(move=AsyncMock(side_effect=pending))
        job, output = app.GraspJob(), io.StringIO()
        with patch.object(app, 'DRY_RUN', False), patch.object(app, 'MOTION_WAIT_REPORT_S', 0.01), redirect_stdout(output):
            task = asyncio.create_task(app.move_to(motion, app.Pose(z=164), 'approach', job))
            try:
                await asyncio.wait_for(called.wait(), 1)
                for _ in range(100):
                    if 'motion request pending' in output.getvalue():
                        break
                    await asyncio.sleep(0.01)
                self.assertIn('movement not confirmed', output.getvalue())
                release.set()
                self.assertTrue(await asyncio.wait_for(task, 1))
                self.assertIn('response success=True', output.getvalue())
                finished = output.getvalue()
                await asyncio.sleep(0.03)
                self.assertEqual(output.getvalue(), finished)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        motion.move.assert_awaited_once()

    async def test_stop_logs_reason_and_cancels_pending_request_without_replay(self):
        called = asyncio.Event()
        async def pending(**kwargs):
            called.set()
            await asyncio.Event().wait()
        motion = SimpleNamespace(move=AsyncMock(side_effect=pending))
        arm = SimpleNamespace(stop=AsyncMock())
        job, output = app.GraspJob(), io.StringIO()
        with patch.object(app, 'DRY_RUN', False), patch.object(app, 'EXECUTE', True), redirect_stdout(output):
            job.task = asyncio.create_task(app.move_to(motion, app.Pose(z=164), 'approach', job))
            await asyncio.wait_for(called.wait(), 1)
            await app.stop_everything(arm, job, reason='Q pressed during selected-object action')
        self.assertTrue(job.task.cancelled())
        self.assertIn('Q pressed during selected-object action', output.getvalue())
        self.assertIn('request cancelled after', output.getvalue())
        self.assertIn('arm.stop() acknowledged', output.getvalue())
        motion.move.assert_awaited_once()
        arm.stop.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
