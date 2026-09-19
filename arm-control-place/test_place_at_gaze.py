"""Look-to-place: gaze pixel -> table spot -> checked put-down at the new x, y.
Every device is faked: a camera looking straight down from (300, -150, 400)."""

import unittest
from unittest.mock import patch

import main as app
from delivery import Held
from gaze_lock import Box
from viam.proto.common import GeometriesInFrame, Geometry, Pose, PoseInFrame, RectangularPrism, Vector3, WorldState

X0, Y0, Z0 = 300.0, -150.0, 400.0
INTR = app.Intrinsics(640, 640, 640, 360, 1280, 720)


class Robot:
    async def transform_pose(self, query, destination):
        p = query.pose
        assert (query.reference_frame, destination) == ("cam", "world")
        return PoseInFrame(reference_frame="world",
                           pose=Pose(x=X0 + p.x, y=Y0 - p.y, z=Z0 - p.z, o_z=1))


class Motion:
    def __init__(self, fail_z=None):
        self.moves, self.worlds, self.fail_z = [], [], fail_z
        self.at = Pose(x=191, y=-304, z=383, o_z=-1)

    async def get_pose(self, name, frame, timeout=None, **kw):
        return PoseInFrame(reference_frame=frame, pose=self.at)

    async def move(self, component_name, destination, world_state=None, constraints=None, **kw):
        p = destination.pose
        kind = "line" if constraints and constraints.linear_constraint else "upright"
        self.moves.append((round(p.x), round(p.y), round(p.z), kind))
        self.worlds.append(world_state)
        if self.fail_z is not None and round(p.z) == self.fail_z:
            return False
        self.at = p
        return True


class Gripper:
    def __init__(self):
        self.opened = 0

    async def open(self, **kw):
        self.opened += 1

    async def do_command(self, cmd, **kw):
        return {"pos": 850}

    async def is_moving(self, **kw):
        return False


def box_at(x, y, label, size=60.0):
    return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=[
        Geometry(center=Pose(x=x, y=y, z=35, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=size, y=size, z=60)),
                 label=label)])])


def held_block():
    down = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
    return Held(label="block", pick_grasp=Pose(x=400, y=-200, z=60, **down),
                pick_approach=Pose(x=400, y=-200, z=160, **down), orientation=down,
                obstacles=box_at(250, -300, "can"), place_z=65.0, width_mm=60.0,
                pose=Pose(x=191, y=-304, z=383, **down), home=Pose(x=191, y=-304, z=383, **down),
                footprint=box_at(400, -200, "block"))


def centers(ws):
    return sorted((round(g.center.x), round(g.center.y)) for f in (ws.obstacles if ws else []) for g in f.geometries)


class PlaceAtGazeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(app, "DRY_RUN", False))
        self.job = app.GraspJob()

    async def place(self, u, v, motion=None, gripper=None):
        self.motion, self.gripper = motion or Motion(), gripper or Gripper()
        held = held_block()
        ok = await app.run_place_at(Robot(), INTR, 1280, 720, self.motion, self.gripper, held, u, v, self.job)
        return ok, held

    async def test_center_pixel_hits_the_table_straight_below_the_camera(self):
        spot = await app.table_point_at_pixel(Robot(), INTR, 1280, 720, 640, 360)
        self.assertEqual([round(v) for v in spot], [300, -150, round(app.MEASURED_TABLE_TOP_Z_MM)])

    async def test_places_at_the_new_xy_then_returns(self):
        ok, held = await self.place(640, 360)
        self.assertTrue(ok, self.job.status)
        self.assertEqual(self.motion.moves, [
            (300, -150, 160, "upright"),     # above the looked-at spot, same height as the pick's approach
            (300, -150, 65, "line"),         # straight down to the height it was resting at
            (300, -150, 383, "line"),        # rise after the jaws opened
            (191, -304, 383, "upright"),     # back to the observe pose
        ])
        self.assertEqual(self.gripper.opened, 1)
        self.assertIsNone(self.job.held)
        self.assertEqual(centers(self.motion.worlds[-1]), [(250, -300), (300, -150)],
                         "the return avoids the other object and the block where it now stands")
        self.assertEqual(round(held.pick_grasp.x), 400, "the caller's record of where it came from is untouched")

    async def test_spot_next_to_another_object_is_refused_before_moving(self):
        ok, _ = await self.place(575, 603)          # the table right next to the can at (250, -300)
        self.assertFalse(ok)
        self.assertIn("too close to the can", self.job.status)
        self.assertEqual(self.motion.moves, [])
        self.assertIsNotNone(self.job.held)

    async def test_spot_next_to_the_arm_base_is_refused(self):
        ok, _ = await self.place(160, 40)           # world (4, 48): on top of the base
        self.assertFalse(ok)
        self.assertIn("too close to it", self.job.status)
        self.assertEqual(self.motion.moves, [])

    async def test_spot_out_of_reach_is_refused(self):
        ok, _ = await self.place(1270, 10)
        self.assertFalse(ok)
        self.assertIn("out of reach", self.job.status)
        self.assertEqual(self.motion.moves, [])

    async def test_failed_descent_never_opens_and_keeps_holding(self):
        ok, _ = await self.place(640, 360, motion=Motion(fail_z=65))
        self.assertFalse(ok)
        self.assertEqual(self.gripper.opened, 0)
        self.assertIsNotNone(self.job.held)
        self.assertIn("still holding", self.job.status)

    def test_spot_dwell_only_counts_clear_of_boxes(self):
        boxes = [Box(600, 300, 700, 400, "block", 0.9, 0)]
        self.assertTrue(app.near_any_box(boxes, (650, 350), app.PLACE_SPOT_BOX_MARGIN_PX))
        self.assertTrue(app.near_any_box(boxes, (730, 350), app.PLACE_SPOT_BOX_MARGIN_PX))
        self.assertFalse(app.near_any_box(boxes, (800, 350), app.PLACE_SPOT_BOX_MARGIN_PX))


if __name__ == "__main__":
    unittest.main()
