"""Post-drop retreat/return sequencing with mocked robot operations."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, sentinel
from viam.proto.common import Pose, WorldState, GeometriesInFrame, Geometry, RectangularPrism, Vector3
import main as app


class ReturnAfterReleaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(app, 'DRY_RUN', False))
        self.actual = Pose(x=270,y=-410,z=65,o_z=-1,theta=20)
        self.home = Pose(x=190.5,y=-304,z=382.8,o_z=-1,theta=20)
        self.read = self.enterContext(patch.object(app, 'gripper_world_pose',new=AsyncMock(return_value=self.actual)))
        self.move = self.enterContext(patch.object(app, 'move_to',new=AsyncMock(return_value=True)))
        obstacle = Geometry(center=Pose(x=400,y=-400,z=10,o_z=1),
                            box=RectangularPrism(dims_mm=Vector3(x=30,y=30,z=50)),label='can')
        target = Geometry(center=Pose(x=250,y=-380,z=10,o_z=1),
                          box=RectangularPrism(dims_mm=Vector3(x=30,y=30,z=50)),label='can')
        self.obstacles = WorldState(obstacles=[GeometriesInFrame(reference_frame='world',geometries=[obstacle])])
        self.footprint = WorldState(obstacles=[GeometriesInFrame(reference_frame='world',geometries=[target])])
        self.held = SimpleNamespace(label='block',pose=Pose(x=999,y=999,z=999),home=self.home,
            pick_grasp=Pose(x=250,y=-380,z=60,o_z=-1),pick_approach=Pose(x=250,y=-380,z=160,o_z=-1),
            place_z=65,orientation=dict(o_z=-1,theta=20),obstacles=self.obstacles,footprint=self.footprint)
        self.job=app.GraspJob()

    async def run_return(self):
        return await app.return_after_release(sentinel.motion,self.held,self.job)

    async def test_rises_at_actual_drop_xy_before_returning_to_taught_pose(self):
        self.assertTrue(await self.run_return())
        self.assertEqual(self.move.await_count,2)
        rise,home=self.move.await_args_list
        self.assertAlmostEqual(rise.args[1].x,270)
        self.assertAlmostEqual(rise.args[1].y,-410)
        self.assertAlmostEqual(rise.args[1].z,382.8)
        self.assertTrue(rise.args[5].linear_constraint)
        self.assertEqual(home.args[1],self.home)
        self.assertEqual(self.held.pose,self.home)

    async def test_failed_vertical_retreat_never_sends_lateral_return(self):
        self.move.return_value=False
        self.assertFalse(await self.run_return())
        self.move.assert_awaited_once()
        self.assertIn('vertical retreat failed',self.job.status)
        self.assertTrue(self.job.requires_operator)

    async def test_failed_home_return_is_not_success(self):
        self.move.side_effect=[True,False]
        self.assertFalse(await self.run_return())
        self.assertIn("couldn't return",self.job.status)
        self.assertNotEqual(self.held.pose,self.home)
        self.assertTrue(self.job.requires_operator)

    async def test_release_above_home_rises_before_any_descent(self):
        self.actual.z=420
        self.assertTrue(await self.run_return())
        self.assertEqual(self.move.await_args_list[0].args[1].z,460)

    async def test_elevated_release_obstacle_covers_drop_to_table(self):
        self.actual.z=420
        self.assertTrue(await self.run_return())
        returned=self.move.await_args_list[1].args[4]
        placed=next(g for frame in returned.obstacles for g in frame.geometries if g.center.x==270)
        self.assertAlmostEqual(placed.center.z-placed.box.dims_mm.z/2,app.TABLE_TOP_Z_MM)
        self.assertAlmostEqual(placed.center.z+placed.box.dims_mm.z/2,395)

    async def test_swap_put_back_keeps_fresh_obstacles_through_return(self):
        fresh=WorldState()
        fresh.CopyFrom(self.obstacles)
        fresh.obstacles[0].geometries[0].center.x=500
        with patch.object(app,'open_gripper',new=AsyncMock()):
            self.assertTrue(await app.put_back(sentinel.motion,sentinel.gripper,self.held,fresh,self.job))
        self.assertIsNone(self.job.held)
        self.assertIs(self.move.await_args_list[2].args[4],fresh)
        returned=self.move.await_args_list[3].args[4]
        self.assertTrue(any(g.center.x==500 for frame in returned.obstacles for g in frame.geometries))

    async def test_return_preserves_other_obstacles_and_moves_released_footprint(self):
        before=self.footprint.SerializeToString()
        self.assertTrue(await self.run_return())
        self.assertIs(self.move.await_args_list[0].args[4],self.obstacles)
        returned=self.move.await_args_list[1].args[4]
        geometries=[g for frame in returned.obstacles for g in frame.geometries]
        self.assertEqual(len(geometries),2)
        self.assertEqual(len({g.label for g in geometries}),2)
        placed=next(g for g in geometries if g.center.x==270)
        self.assertEqual((placed.center.x,placed.center.y,placed.center.z),(270,-410,15))
        self.assertEqual(self.footprint.SerializeToString(),before)

    async def test_unknown_pose_blocks_all_return_commands(self):
        self.read.return_value=None
        with self.assertRaisesRegex(RuntimeError,'unknown'):
            await self.run_return()
        self.move.assert_not_awaited()

    async def test_dry_run_uses_recorded_pose_without_reading_robot(self):
        self.held.pose=self.actual
        with patch.object(app,'DRY_RUN',True):
            self.assertTrue(await self.run_return())
        self.read.assert_not_awaited()

    async def test_no_home_uses_fourteen_inch_fallback_above_table(self):
        self.held.home=None
        self.assertTrue(await self.run_return())
        home=self.move.await_args_list[1].args[1]
        self.assertAlmostEqual(home.z-app.TABLE_TOP_Z_MM,355.6)


if __name__=='__main__':
    unittest.main()
