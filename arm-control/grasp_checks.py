"""Read-only checks against the collision geometry actually reported by Viam.

These checks do not replace Viam's planner. They detect an infeasible grasp
height before the approach move, without changing any collision shapes.
"""

import asyncio
import math
from dataclasses import dataclass

import numpy as np

from viam.components.gripper import Gripper
from viam.proto.common import Geometry, Pose, PoseInFrame
from viam.spatialmath import Quaternion


def _rotation(pose: Pose) -> np.ndarray:
    values = (pose.x, pose.y, pose.z, pose.o_x, pose.o_y, pose.o_z, pose.theta)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Collision geometry contains a nonfinite pose")
    if math.hypot(pose.o_x, pose.o_y, pose.o_z) == 0:
        if pose.theta != 0:
            raise ValueError("Collision geometry has an angle without an orientation axis")
        return np.eye(3)
    return np.asarray(Quaternion.from_pose(pose).to_rotation_matrix().elements).reshape(3, 3)


def box_world_bounds(geometry: Geometry, frame_pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """World min/max XYZ of an oriented box, including its local center pose."""
    if not geometry.HasField("box"):
        raise ValueError(f"Collision shape '{geometry.label}' is not a supported box")
    dims = geometry.box.dims_mm
    lengths = np.asarray((dims.x, dims.y, dims.z), dtype=float)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError(f"Collision shape '{geometry.label}' has invalid box dimensions")
    frame_rotation = _rotation(frame_pose)
    local_rotation = _rotation(geometry.center)
    center = np.asarray((frame_pose.x, frame_pose.y, frame_pose.z)) + frame_rotation @ np.asarray(
        (geometry.center.x, geometry.center.y, geometry.center.z))
    half_extents = np.abs(frame_rotation @ local_rotation) @ lengths / 2.0
    return center - half_extents, center + half_extents


def geometry_z_bounds(geometry: Geometry, frame_pose: Pose) -> tuple[float, float]:
    low, high = box_world_bounds(geometry, frame_pose)
    return float(low[2]), float(high[2])


@dataclass(frozen=True)
class GraspCollisionModel:
    gripper_geometries: tuple[Geometry, ...]
    table_top_z_mm: float
    gripper_name: str = "gripper"
    table_name: str = "table"


@dataclass(frozen=True)
class ModelClearance:
    table_top_z_mm: float
    lowest_gripper_z_mm: float
    clearance_mm: float
    required_tcp_z_mm: float
    limiting_geometry: str


def check_model_clearance(model: GraspCollisionModel, grasp_pose: Pose,
                          clearance_mm: float = 2.0) -> ModelClearance:
    """Require all modeled gripper boxes to remain above the table's top.

    This is conservative over the configured tabletop plane. It never raises
    the requested pose automatically: doing so can leave the jaws above a short
    object. The full Viam planner remains responsible for every other obstacle.
    """
    if not math.isfinite(clearance_mm) or clearance_mm < 0:
        raise ValueError("Model clearance must be a finite, nonnegative distance")
    if not math.isfinite(model.table_top_z_mm):
        raise ValueError("The configured table height is invalid")
    if not model.gripper_geometries:
        raise ValueError("Viam returned no gripper collision geometry; cannot verify grasp height")
    bottoms = [(geometry_z_bounds(g, grasp_pose)[0], g.label or "unlabeled")
               for g in model.gripper_geometries]
    lowest, label = min(bottoms)
    gap = lowest - model.table_top_z_mm
    required_z = grasp_pose.z + clearance_mm - gap
    result = ModelClearance(model.table_top_z_mm, lowest, gap, required_z, label)
    if gap < clearance_mm - 1e-6:
        raise ValueError(
            f"Grasp rejected before approach: Viam collision shape '{model.gripper_name}:{label}' "
            f"would reach z={lowest:.1f} mm, with table top z={model.table_top_z_mm:.1f} mm "
            f"(gap {gap:.1f} mm; required {clearance_mm:.1f} mm). "
            f"At this orientation its model requires TCP z>={required_z:.1f} mm, "
            f"but the requested grasp is z={grasp_pose.z:.1f} mm. "
            "Measured fingertip lengths do not update Viam's collision model; verify the "
            "gripper model and camera/table calibration before another physical attempt. "
            "Raising this grasp automatically could leave the fingers above the object."
        )
    return result


async def load_grasp_model(robot, gripper, table_name: str = "table", world_frame: str = "world",
                           timeout: float = 20.0) -> GraspCollisionModel:
    """Read actual gripper boxes and the configured table frame, moving nothing."""
    table = Gripper.from_robot(robot, table_name)
    geometries, table_geometries, table_frame = await asyncio.gather(
        gripper.get_geometries(timeout=timeout),
        table.get_geometries(timeout=timeout),
        asyncio.wait_for(robot.transform_pose(
            PoseInFrame(reference_frame=table_name, pose=Pose(o_z=1.0)), world_frame), timeout),
    )
    if not geometries:
        raise ValueError("Viam returned no gripper collision geometry; cannot verify grasp height")
    if not table_geometries:
        raise ValueError("Viam returned no table collision geometry; cannot verify grasp height")
    tops = []
    for geometry in table_geometries:
        rotation = _rotation(table_frame.pose) @ _rotation(geometry.center)
        if abs(rotation[2, 2]) < math.cos(math.radians(1.0)):
            raise ValueError("Configured table collision box is tilted; a horizontal tabletop is required")
        _, top = geometry_z_bounds(geometry, table_frame.pose)
        tops.append(top)
    # Validate the gripper shapes while loading too, so preflight catches bad data.
    for geometry in geometries:
        box_world_bounds(geometry, Pose(o_z=1.0))
    return GraspCollisionModel(tuple(geometries), max(tops), getattr(gripper, "name", "gripper"), table_name)


async def check_grasp_model(robot, gripper, grasp_pose: Pose, table_name: str = "table",
                            world_frame: str = "world", clearance_mm: float = 2.0,
                            timeout: float = 20.0) -> ModelClearance:
    model = await load_grasp_model(robot, gripper, table_name, world_frame, timeout)
    return check_model_clearance(model, grasp_pose, clearance_mm)
