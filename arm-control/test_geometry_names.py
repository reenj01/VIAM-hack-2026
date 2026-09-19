"""Planner obstacle names are unique; detector data and shapes stay intact."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, sentinel

from viam.proto.common import (
    GeometriesInFrame, Geometry, Pose, RectangularPrism, Vector3, WorldState,
)

import main as app


def geometry(label="can", x=0):
    return Geometry(
        label=label, center=Pose(x=x, y=20, z=40, o_z=1),
        box=RectangularPrism(dims_mm=Vector3(x=30, y=40, z=80)),
    )


def detected_object(label="can", x=0, geometries=None):
    return SimpleNamespace(geometries=GeometriesInFrame(
        reference_frame="cam",
        geometries=geometries if geometries is not None else [geometry(label, x)],
    ))


def labels(state):
    return [g.label for frame in state.obstacles for g in frame.geometries]


class DetectedObstacleNamesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(app, "AVOID_DETECTED_OBJECTS", True))
        self.transform = self.enterContext(patch.object(
            app, "pose_in", new=AsyncMock(side_effect=lambda _robot, pose, _ref, _world:
                Pose(x=pose.x + 100, y=pose.y, z=pose.z, o_z=1)),
        ))

    async def test_two_cans_and_multiple_shapes_get_unique_names_without_source_changes(self):
        first = detected_object(geometries=[geometry("can", 1), geometry("can", 2)])
        second = detected_object("can", 3)
        before = [o.geometries.SerializeToString() for o in (first, second)]

        result = await app.obstacles_in_world(sentinel.robot, [first, second], None)

        self.assertEqual(len(labels(result)), 3)
        self.assertEqual(len(set(labels(result))), 3)
        self.assertTrue(all(labels(result)))
        self.assertEqual([o.geometries.SerializeToString() for o in (first, second)], before)
        self.assertEqual(app.object_label(first), "can")
        self.assertEqual(app.object_label(second), "can")
        self.assertEqual(result.obstacles[0].reference_frame, app.MOTION_REFERENCE_FRAME)
        self.assertEqual([g.center.x for g in result.obstacles[0].geometries], [101, 102, 103])
        for cloned in result.obstacles[0].geometries:
            self.assertEqual(cloned.box, geometry().box)

    async def test_target_exclusion_and_ignored_labels_still_apply(self):
        target = detected_object("can", 1)
        other = detected_object("can", 2)
        ignored = detected_object("dining table", 3)

        result = await app.obstacles_in_world(sentinel.robot, [target, other, ignored], target)

        self.assertEqual(len(labels(result)), 1)
        self.assertEqual(result.obstacles[0].geometries[0].center.x, 102)
        self.transform.assert_awaited_once()

    async def test_disabled_or_empty_obstacles_return_none(self):
        self.assertIsNone(await app.obstacles_in_world(sentinel.robot, [], None))
        with patch.object(app, "AVOID_DETECTED_OBJECTS", False):
            self.assertIsNone(await app.obstacles_in_world(
                sentinel.robot, [detected_object()], None))
        self.transform.assert_not_awaited()


class MergedObstacleNamesTests(unittest.TestCase):
    def test_repeated_names_across_frames_and_states_are_unique(self):
        first = WorldState(obstacles=[
            GeometriesInFrame(reference_frame="world", geometries=[geometry("can", 1)]),
            GeometriesInFrame(reference_frame="cam", geometries=[geometry("can", 2)]),
        ])
        footprint = WorldState(obstacles=[GeometriesInFrame(
            reference_frame="world", geometries=[geometry("can", 3), geometry("", 4)],
        )])
        before = [s.SerializeToString() for s in (first, footprint)]

        result = app.merge_world_states(first, None, footprint)

        self.assertEqual(len(labels(result)), 4)
        self.assertEqual(len(set(labels(result))), 4)
        self.assertTrue(all(labels(result)))
        self.assertEqual([f.reference_frame for f in result.obstacles], ["world", "cam", "world"])
        self.assertEqual([g.center.x for f in result.obstacles for g in f.geometries], [1, 2, 3, 4])
        self.assertEqual([s.SerializeToString() for s in (first, footprint)], before)
        result.obstacles[0].geometries[0].center.x = 999
        self.assertEqual(first.obstacles[0].geometries[0].center.x, 1)

    def test_single_state_is_copied_and_duplicate_names_are_fixed(self):
        state = WorldState(obstacles=[GeometriesInFrame(
            reference_frame="world", geometries=[geometry("can"), geometry("can", 5)],
        )])
        before = state.SerializeToString()

        result = app.merge_world_states(state)

        self.assertEqual(len(set(labels(result))), 2)
        self.assertEqual(state.SerializeToString(), before)
        result.obstacles[0].geometries[0].label = "changed"
        self.assertEqual(state.SerializeToString(), before)

    def test_remerging_prior_merge_with_footprint_does_not_reintroduce_names(self):
        state = WorldState(obstacles=[GeometriesInFrame(
            reference_frame="world", geometries=[geometry("can")],
        )])
        merged = app.merge_world_states(state, state)

        result = app.merge_world_states(merged, state, merged)

        self.assertEqual(len(labels(result)), 5)
        self.assertEqual(len(set(labels(result))), 5)
        self.assertEqual(len(labels(merged)), 2)

    def test_empty_states_return_none(self):
        self.assertIsNone(app.merge_world_states())
        self.assertIsNone(app.merge_world_states(None, WorldState()))


if __name__ == "__main__":
    unittest.main()
