"""All checks must finish before the first physical action; hardware is mocked."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, sentinel
from viam.proto.common import Pose, Geometry, GeometriesInFrame, RectangularPrism, Vector3
import main as app
from grasp_checks import GraspCollisionModel


def object_at(z=27, height=100):
    return SimpleNamespace(geometries=GeometriesInFrame(reference_frame='world', geometries=[
        Geometry(center=Pose(x=250, y=-380, z=z, o_z=1),
                 box=RectangularPrism(dims_mm=Vector3(x=49,y=49,z=height)), label='block')]))


class TargetPrecheckTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(app, 'DRY_RUN', False))
        self.enterContext(patch.object(app, 'MEASURED_OBJECT_WIDTHS_MM', {'block':28.6}))
        # These scenarios put objects on the configured table (-23); keep the floor there too.
        self.enterContext(patch.object(app, 'MEASURED_TABLE_TOP_Z_MM', app.TABLE_TOP_Z_MM))
        self.current = Pose(x=250,y=-380,z=300,o_z=-1)
        self.enterContext(patch.object(app, 'gripper_world_pose', new=AsyncMock(return_value=self.current)))
        self.first, self.second = object_at(), object_at()
        self.capture = self.enterContext(patch.object(app, 'capture_objects', new=AsyncMock(side_effect=[[self.first],[self.second]])))
        self.match = self.enterContext(patch.object(app, 'match_object_to_box', new=AsyncMock(side_effect=[self.first,self.second])))
        self.enterContext(patch.object(app, 'objects_in_gripper', new=AsyncMock(return_value=[])))
        self.enterContext(patch.object(app, 'obstacles_in_world', new=AsyncMock(return_value=None)))
        claws = Geometry(center=Pose(z=-2.5, o_z=1),
                         box=RectangularPrism(dims_mm=Vector3(x=40,y=170,z=105)),label='claws')
        self.model = self.enterContext(patch.object(app, 'load_grasp_model', new=AsyncMock(
            return_value=GraspCollisionModel((claws,), -23))))
        self.execute = self.enterContext(patch.object(app, 'execute_grasp', new=AsyncMock(return_value=False)))
        self.arm = SimpleNamespace(is_moving=AsyncMock(return_value=False))
        self.job = app.GraspJob()

    async def run_grasp(self):
        return await app.run_grasp(sentinel.robot, None, 1280,720,sentinel.segmenter,
                                   sentinel.motion,sentinel.gripper,self.arm,
                                   SimpleNamespace(label='block'),self.job,None,150,carry_back=False)

    async def test_valid_captures_use_measured_width_before_execution(self):
        await self.run_grasp()
        self.execute.assert_awaited_once()
        self.assertAlmostEqual(self.execute.await_args.args[3].width_mm,28.6)
        self.assertEqual(self.capture.await_count,2)

    async def test_inconsistent_height_never_reaches_execution(self):
        self.second.geometries.geometries[0].center.z = 176
        self.assertFalse(await self.run_grasp())
        self.execute.assert_not_awaited()
        self.assertIn('disagree',self.job.status)

    async def test_model_conflict_never_reaches_approach_or_close(self):
        self.first = object_at(2,50)
        self.second = object_at(2,50)
        self.capture.side_effect = [[self.first],[self.second]]
        self.match.side_effect = [self.first,self.second]
        self.assertFalse(await self.run_grasp())
        self.execute.assert_not_awaited()
        self.assertIn('collision shape',self.job.status)
        self.assertFalse(self.job.motion_started)
        self.assertFalse(self.job.requires_operator)

    async def test_moving_wrist_does_not_even_capture(self):
        self.arm.is_moving.return_value = True
        self.assertFalse(await self.run_grasp())
        self.capture.assert_not_awaited()
        self.execute.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
