import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from viam.proto.common import Geometry, Pose, PoseInFrame, RectangularPrism, Vector3

from grasp_checks import (
    GraspCollisionModel, box_world_bounds, check_model_clearance, geometry_z_bounds,
    load_grasp_model,
)


def box(x, y, z, center=None, label="test"):
    return Geometry(center=center or Pose(o_z=1), label=label,
                    box=RectangularPrism(dims_mm=Vector3(x=x, y=y, z=z)))


def live_gripper_boxes():
    return (box(50, 100, 100, Pose(z=-50, o_z=1), "case-gripper"),
            box(40, 170, 105, Pose(z=-2.5, o_z=1), "claws"))


class GeometryChecks(unittest.TestCase):
    def test_live_claws_at_rejected_video_height_overlap_table(self):
        model = GraspCollisionModel(live_gripper_boxes(), -23)
        with self.assertRaisesRegex(ValueError, "gripper:claws.*z=-32.0"):
            check_model_clearance(model, Pose(z=18, o_z=-1))

    def test_live_claws_require_29mm_tcp_for_2mm_clearance_when_vertical(self):
        model = GraspCollisionModel(live_gripper_boxes(), -23)
        with self.assertRaises(ValueError):
            check_model_clearance(model, Pose(z=28.9, o_z=-1))
        result = check_model_clearance(model, Pose(z=29, o_z=-1))
        self.assertAlmostEqual(result.lowest_gripper_z_mm, -21)
        self.assertAlmostEqual(result.clearance_mm, 2)
        self.assertAlmostEqual(result.required_tcp_z_mm, 29)
        self.assertEqual(result.limiting_geometry, "claws")

    def test_exact_live_orientation_finds_lower_corner(self):
        model = GraspCollisionModel(live_gripper_boxes(), -23)
        pose = Pose(z=100, o_x=-0.020850411815630295, o_y=0.03895153589456142,
                    o_z=-0.9990235423545198, theta=-23.043856775731037)
        result = check_model_clearance(model, pose)
        self.assertAlmostEqual(result.lowest_gripper_z_mm, 47.7657, places=3)
        self.assertGreater(result.required_tcp_z_mm, 29)

    def test_local_shape_rotation_contributes_to_bounds(self):
        # Local Z points along world X: the long local Z dimension now spans X.
        geometry = box(20, 20, 100, Pose(x=5, z=3, o_x=1))
        low, high = box_world_bounds(geometry, Pose(x=10, y=20, z=30, o_z=1))
        np.testing.assert_allclose(low, [-35, 10, 23], atol=1e-6)
        np.testing.assert_allclose(high, [65, 30, 43], atol=1e-6)

    def test_tilted_cube_low_edge_is_not_center_minus_half_height(self):
        geometry = box(40, 40, 40)
        low, high = geometry_z_bounds(geometry, Pose(z=100, o_x=math.sqrt(0.5), o_z=math.sqrt(0.5)))
        self.assertAlmostEqual(low, 100-20*math.sqrt(2))
        self.assertAlmostEqual(high, 100+20*math.sqrt(2))

    def test_empty_and_invalid_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no gripper"):
            check_model_clearance(GraspCollisionModel((), -23), Pose(o_z=-1))
        for geometry in (Geometry(), box(0, 20, 30), box(float("nan"), 20, 30),
                         box(20, 30, 40, Pose(z=float("nan")))):
            with self.subTest(geometry=geometry):
                with self.assertRaises(ValueError):
                    box_world_bounds(geometry, Pose(o_z=1))
        with self.assertRaises(ValueError):
            check_model_clearance(GraspCollisionModel(live_gripper_boxes(), float("nan")), Pose(o_z=-1))


class ReadOnlyModelChecks(unittest.IsolatedAsyncioTestCase):
    async def test_reads_configured_table_pose_and_shapes(self):
        gripper = SimpleNamespace(name="gripper", get_geometries=AsyncMock(return_value=live_gripper_boxes()))
        table = SimpleNamespace(get_geometries=AsyncMock(return_value=[box(3000, 3000, 200)]))
        robot = SimpleNamespace(transform_pose=AsyncMock(return_value=PoseInFrame(
            reference_frame="world", pose=Pose(z=-123, o_z=1))))
        with patch("grasp_checks.Gripper.from_robot", return_value=table):
            model = await load_grasp_model(robot, gripper)
        self.assertEqual(model.table_top_z_mm, -23)
        self.assertEqual(len(model.gripper_geometries), 2)
        self.assertEqual(robot.transform_pose.await_args.args[0].reference_frame, "table")
        gripper.get_geometries.assert_awaited_once()
        table.get_geometries.assert_awaited_once()

    async def test_missing_or_tilted_table_is_rejected(self):
        gripper = SimpleNamespace(name="gripper", get_geometries=AsyncMock(return_value=live_gripper_boxes()))
        cases = (([], Pose(o_z=1)), ([box(3000, 3000, 200)], Pose(o_x=1)))
        for geometries, pose in cases:
            table = SimpleNamespace(get_geometries=AsyncMock(return_value=geometries))
            robot = SimpleNamespace(transform_pose=AsyncMock(return_value=PoseInFrame(reference_frame="world", pose=pose)))
            with patch("grasp_checks.Gripper.from_robot", return_value=table):
                with self.assertRaises(ValueError):
                    await load_grasp_model(robot, gripper)


if __name__ == "__main__":
    unittest.main()
