"""Grasp geometry regression tests; no robot, camera, or motion calls.

Run from arm-control: python -m unittest test_grasp_geometry -v
"""

import contextlib
import io
import math
import unittest
from unittest.mock import patch

import numpy as np
from viam.proto.common import Pose

import main as app


class WorldBoxTests(unittest.TestCase):
    def test_camera_depth_is_not_world_height_after_quarter_turn(self):
        # Local Z points along world X. Local X supplies the vertical extent.
        world_center = Pose(o_x=1.0, o_y=0.0, o_z=0.0, theta=0.0)
        extents = app.world_box_extents(world_center, (120.0, 40.0, 30.0))
        np.testing.assert_allclose(extents, (30.0, 40.0, 120.0), atol=1e-6)

    def test_horizontal_rotation_expands_world_bounds_without_changing_height(self):
        world_center = Pose(o_z=1.0, theta=45.0)
        extents = app.world_box_extents(world_center, (80.0, 40.0, 100.0))
        np.testing.assert_allclose(extents, (120 / math.sqrt(2), 120 / math.sqrt(2), 100), atol=1e-6)

    def test_unusable_dimensions_are_rejected(self):
        for dimensions in ((40, 20), (40, 20, 0), (40, -20, 80),
                           (40, math.nan, 80), (math.inf, 20, 80)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    app.world_box_extents(Pose(o_z=1.0), dimensions)


class GraspHeightTests(unittest.TestCase):
    def setUp(self):
        for field, value in (("FINGER_CLEARANCE_MM", 60.0),
                             ("MAX_FINGER_EXTENSION_MM", 60.0),
                             ("GRIPPER_BODY_FROM_FLANGE_MM", 105.0),
                             ("GRASP_Z_OFFSET_MM", 0.0)):
            setting = patch.object(app, field, value)
            setting.start()
            self.addCleanup(setting.stop)

    def plan(self, height=100.0, *, current=None, tcp=150.0, center=None, size=None):
        current = current if current is not None else Pose(x=300, y=100, z=300, o_z=-1.0)
        center = center if center is not None else Pose(
            x=300.0, y=100.0, z=app.TABLE_TOP_Z_MM + height / 2, o_z=1.0
        )
        size = size if size is not None else (40.0, 50.0, height)
        return app.plan_grasp(center, size, current, tcp, None)

    def test_short_and_tall_objects_use_equal_shallow_insertion_below_their_tops(self):
        short = self.plan(height=40.0)
        tall = self.plan(height=160.0)

        for plan, height in ((short, 40.0), (tall, 160.0)):
            with self.subTest(height=height):
                self.assertAlmostEqual(plan.object_top_z_mm, app.TABLE_TOP_Z_MM + height)
                self.assertAlmostEqual(plan.object_top_z_mm - plan.fingertip_z_mm, 15.0)
                self.assertGreaterEqual(plan.body_clearance_mm, 15.0)
                self.assertAlmostEqual(plan.grasp.z - plan.fingertip_z_mm, 15.0)
                self.assertAlmostEqual(plan.approach.z - plan.grasp.z, app.APPROACH_HEIGHT_MM)
        # Their centers differ by only 60 mm, but their grasp heights differ by 120 mm.
        self.assertAlmostEqual(tall.grasp.z - short.grasp.z, 120.0)

    def test_rotated_object_uses_world_vertical_extent_to_find_top(self):
        center = Pose(x=300, y=100, z=app.TABLE_TOP_Z_MM + 60, o_x=1.0)
        plan = self.plan(center=center, size=(120.0, 40.0, 30.0))
        self.assertAlmostEqual(plan.object_top_z_mm, app.TABLE_TOP_Z_MM + 120)
        self.assertAlmostEqual(plan.fingertip_z_mm, app.TABLE_TOP_Z_MM + 105)

    def test_low_finger_clearance_reduces_insertion_to_keep_body_above_object(self):
        with patch.object(app, "FINGER_CLEARANCE_MM", 20.0), \
                patch.object(app, "MAX_FINGER_EXTENSION_MM", 20.0):
            plan = self.plan()
        self.assertAlmostEqual(plan.object_top_z_mm - plan.fingertip_z_mm, 5.0)
        self.assertAlmostEqual(plan.body_clearance_mm, 15.0)

    def test_upward_offset_raises_tcp_and_fingertips_by_exact_requested_amount(self):
        original = self.plan()
        with patch.object(app, "GRASP_Z_OFFSET_MM", 4.0):
            raised = self.plan()
        self.assertAlmostEqual(raised.grasp.z - original.grasp.z, 4.0)
        self.assertAlmostEqual(raised.fingertip_z_mm - original.fingertip_z_mm, 4.0)
        self.assertAlmostEqual(raised.lowest_fingertip_z_mm - original.lowest_fingertip_z_mm, 4.0)
        self.assertAlmostEqual(raised.body_z_mm - original.body_z_mm, 4.0)
        self.assertAlmostEqual(raised.body_clearance_mm - original.body_clearance_mm, 4.0)
        self.assertAlmostEqual(raised.grasp.x, original.grasp.x)
        self.assertAlmostEqual(raised.grasp.y, original.grasp.y)

    def test_upward_offset_without_finger_overlap_is_rejected(self):
        with patch.object(app, "GRASP_Z_OFFSET_MM", 15.0):
            with self.assertRaisesRegex(ValueError, "overlap|above the object"):
                self.plan()

    def test_table_clearance_is_preserved_and_too_short_object_is_rejected(self):
        plan = self.plan(height=12.0)
        self.assertGreaterEqual(plan.fingertip_z_mm, app.TABLE_TOP_Z_MM + app.FINGERTIP_TABLE_CLEARANCE_MM)
        self.assertGreaterEqual(plan.body_clearance_mm, 15.0)
        self.assertLess(plan.fingertip_z_mm, plan.object_top_z_mm)
        with self.assertRaises(ValueError):
            self.plan(height=8.0)

    def test_missing_geometry_measurements_are_rejected(self):
        for field in ("FINGER_CLEARANCE_MM", "MAX_FINGER_EXTENSION_MM", "GRIPPER_BODY_FROM_FLANGE_MM"):
            with self.subTest(field=field), patch.object(app, field, None):
                with self.assertRaises(ValueError):
                    self.plan()

    def test_hinge_range_protects_body_and_table_at_both_extensions(self):
        with patch.object(app, "MAX_FINGER_EXTENSION_MM", 75.0):
            plan = self.plan()
        self.assertAlmostEqual(plan.body_z_mm - plan.fingertip_z_mm, 60.0)
        self.assertAlmostEqual(plan.body_z_mm - plan.lowest_fingertip_z_mm, 75.0)
        self.assertAlmostEqual(plan.fingertip_beyond_tcp_mm, 30.0)
        self.assertAlmostEqual(plan.object_top_z_mm - plan.fingertip_z_mm, 15.0)
        # A pinched object may rise 15 mm as the linkage retracts. The remaining
        # body clearance must cover that rise, not just its starting height.
        self.assertAlmostEqual(plan.body_clearance_mm, 30.0)
        self.assertGreaterEqual(plan.lowest_fingertip_z_mm,
                                app.TABLE_TOP_Z_MM + app.FINGERTIP_TABLE_CLEARANCE_MM)

    def test_hinge_retraction_limits_insertion_to_protect_object_top(self):
        with patch.object(app, "FINGER_CLEARANCE_MM", 40.0), \
                patch.object(app, "MAX_FINGER_EXTENSION_MM", 60.0):
            plan = self.plan()
        self.assertAlmostEqual(plan.object_top_z_mm - plan.fingertip_z_mm, 5.0)
        self.assertAlmostEqual(plan.body_clearance_mm, 15.0)

    def test_hinge_range_can_make_short_object_unreachable_above_table(self):
        with patch.object(app, "MAX_FINGER_EXTENSION_MM", 75.0):
            plan = self.plan(height=26.0)
            self.assertAlmostEqual(plan.lowest_fingertip_z_mm,
                                   app.TABLE_TOP_Z_MM + app.FINGERTIP_TABLE_CLEARANCE_MM)
            self.assertLess(plan.fingertip_z_mm, plan.object_top_z_mm)
            for height in (25.0, 20.0):
                with self.subTest(height=height), self.assertRaises(ValueError):
                    self.plan(height=height)

    def test_changing_tcp_reference_preserves_physical_body_and_finger_heights(self):
        first = self.plan(tcp=130.0)
        second = self.plan(tcp=150.0)
        self.assertAlmostEqual(first.grasp.z - second.grasp.z, 20.0)
        self.assertAlmostEqual(first.body_z_mm, second.body_z_mm)
        self.assertAlmostEqual(first.fingertip_z_mm, second.fingertip_z_mm)
        self.assertAlmostEqual(first.lowest_fingertip_z_mm, second.lowest_fingertip_z_mm)
        self.assertAlmostEqual(first.body_clearance_mm, second.body_clearance_mm)

    def test_measured_body_datum_changes_tcp_without_changing_body_target(self):
        original = self.plan()
        with patch.object(app, "GRIPPER_BODY_FROM_FLANGE_MM", 160.0):
            measured = self.plan()
        self.assertAlmostEqual(measured.grasp.z - original.grasp.z, 55.0)
        self.assertAlmostEqual(measured.body_z_mm, original.body_z_mm)
        self.assertAlmostEqual(measured.body_clearance_mm, original.body_clearance_mm)
        # Reconstruct the physical body position from the commanded TCP and
        # the measured mount geometry. A fixed 165 mm tip assumption fails this.
        self.assertAlmostEqual(measured.grasp.z + 150.0 - 160.0, measured.body_z_mm)

    def test_tilted_tcp_compensation_targets_body_center_in_xy(self):
        tilt = math.radians(4.0)
        current = Pose(o_x=math.sin(tilt), o_z=-math.cos(tilt))
        plan = self.plan(current=current)
        body_offset_from_tcp = 105.0 - 150.0
        self.assertAlmostEqual(plan.grasp.x + body_offset_from_tcp * math.sin(tilt), 300.0)
        self.assertAlmostEqual(plan.grasp.y, 100.0)
        self.assertAlmostEqual(plan.grasp.z - body_offset_from_tcp * math.cos(tilt), plan.body_z_mm)
        self.assertGreaterEqual(plan.body_clearance_mm, 15.0 - 1e-9)

    def test_inverted_finger_extension_range_is_rejected(self):
        with patch.object(app, "MAX_FINGER_EXTENSION_MM", 59.0):
            with self.assertRaises(ValueError):
                self.plan()

    def test_nonfinite_geometry_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value, field="position"):
                with self.assertRaises(ValueError):
                    self.plan(center=Pose(x=value, y=100, z=27, o_z=1))
            with self.subTest(value=value, field="tcp"):
                with self.assertRaises(ValueError):
                    self.plan(tcp=value)
            with self.subTest(value=value, field="size"):
                with self.assertRaises(ValueError):
                    self.plan(size=(40, 50, value))
            for field in ("FINGER_CLEARANCE_MM", "MAX_FINGER_EXTENSION_MM", "GRIPPER_BODY_FROM_FLANGE_MM"):
                with self.subTest(value=value, field=field), patch.object(app, field, value):
                    with self.assertRaises(ValueError):
                        self.plan()

    def test_nonfinite_world_or_wrist_orientation_is_rejected(self):
        for field in ("o_x", "o_y", "o_z", "theta"):
            for value in (math.nan, math.inf):
                with self.subTest(field=field, value=value, role="world object"):
                    center = Pose(x=300, y=100, z=27, o_z=1.0)
                    setattr(center, field, value)
                    with self.assertRaises(ValueError):
                        self.plan(center=center)
                with self.subTest(field=field, value=value, role="wrist"):
                    current = Pose(o_z=-1.0)
                    setattr(current, field, value)
                    with self.assertRaises(ValueError):
                        self.plan(current=current)

    def test_missing_current_pose_or_nonpositive_tcp_is_rejected(self):
        with self.assertRaises(ValueError):
            app.plan_grasp(Pose(x=300, y=100, z=27, o_z=1), (40, 50, 100), None, 150, None)
        for tcp in (0.0, -1.0):
            with self.subTest(tcp=tcp), self.assertRaises(ValueError):
                self.plan(tcp=tcp)

    def test_tcp_reference_beyond_fingertips_still_places_body_correctly(self):
        plan = self.plan(tcp=200.0)
        self.assertAlmostEqual(plan.grasp.z + 200.0 - 105.0, plan.body_z_mm)

    def test_ten_degree_wrist_tilt_is_rejected_before_descent(self):
        tilt = math.radians(10)
        current = Pose(o_x=math.sin(tilt), o_z=-math.cos(tilt))
        with self.assertRaisesRegex(ValueError, "vertically down|5 degrees"):
            self.plan(current=current)

    def test_sideways_wrist_is_rejected_instead_of_using_a_default_orientation(self):
        with self.assertRaisesRegex(ValueError, "vertically down"):
            self.plan(current=Pose(o_x=1.0, o_z=0.0))

    def test_small_tilt_with_insufficient_body_clearance_is_rejected(self):
        tilt = math.radians(4)
        current = Pose(o_x=math.sin(tilt), o_z=-math.cos(tilt))
        with patch.object(app, "FINGER_CLEARANCE_MM", 20.0), \
                patch.object(app, "MAX_FINGER_EXTENSION_MM", 20.0):
            with self.assertRaises(ValueError):
                self.plan(current=current)


class GraspSettingsTests(unittest.TestCase):
    def test_valid_measurement_and_positive_offset(self):
        settings = app.grasp_settings(["--finger-clearance-mm", "60", "--max-finger-extension-mm", "75",
                                       "--gripper-body-from-flange-mm", "105", "--grasp-z-offset-mm", "4"])
        self.assertEqual(settings.finger_clearance_mm, 60.0)
        self.assertEqual(settings.max_finger_extension_mm, 75.0)
        self.assertEqual(settings.gripper_body_from_flange_mm, 105.0)
        self.assertEqual(settings.grasp_z_offset_mm, 4.0)
        empty = app.grasp_settings([])
        self.assertAlmostEqual(empty.finger_clearance_mm, 58.4)
        self.assertAlmostEqual(empty.max_finger_extension_mm, 69.9)
        self.assertAlmostEqual(empty.gripper_body_from_flange_mm, 97.8)

    def test_invalid_offset_or_inadequate_measurement_is_rejected(self):
        cases = (
            ["--grasp-z-offset-mm", "nan"], ["--grasp-z-offset-mm", "inf"],
            ["--grasp-z-offset-mm", "-1"], ["--finger-clearance-mm", "nan"],
            ["--finger-clearance-mm", "inf"], ["--finger-clearance-mm", "15"],
            ["--finger-clearance-mm", "0"], ["--finger-clearance-mm", "-1"],
            ["--max-finger-extension-mm", "nan"], ["--max-finger-extension-mm", "inf"],
            ["--max-finger-extension-mm", "0"], ["--max-finger-extension-mm", "-1"],
            ["--gripper-body-from-flange-mm", "nan"], ["--gripper-body-from-flange-mm", "inf"],
            ["--gripper-body-from-flange-mm", "0"], ["--gripper-body-from-flange-mm", "-1"],
            ["--finger-clearance-mm", "60", "--max-finger-extension-mm", "59"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    app.grasp_settings(arguments)
                self.assertNotEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
