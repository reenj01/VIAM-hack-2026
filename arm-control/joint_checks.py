"""Read-only checks of joint readback against the arm's own kinematic limits."""
import json
import math
from dataclasses import dataclass


class JointRangeError(ValueError):
    """Current joint state cannot be used as a valid planning start."""


@dataclass(frozen=True)
class JointReading:
    name: str
    degrees: float
    minimum: float
    maximum: float

    @property
    def in_range(self):
        return self.minimum <= self.degrees <= self.maximum


def read_joint_model(data: bytes):
    """Viam JSON uses degrees for revolute limits, unlike a URDF's radians.

    Follow the serial chain from world rather than assuming the JSON arrays
    are ordered. Unsupported models fail closed instead of guessing units.
    """
    try:
        model = json.loads(data)
        joints = model['joints']
        nodes = model['links'] + joints
        joint_ids = {j['id'] for j in joints}
        if not joints or len({n['id'] for n in nodes}) != len(nodes):
            raise ValueError('empty or duplicate joint/link names')
        ordered, visited, parent = [], set(), 'world'
        while True:
            children = [n for n in nodes if n['parent'] == parent]
            if not children:
                break
            if len(children) != 1 or children[0]['id'] in visited:
                raise ValueError('expected one serial arm chain')
            node = children[0]
            parent = node['id']
            visited.add(parent)
            if parent in joint_ids:
                if node['type'] != 'revolute':
                    raise ValueError('only revolute joints are supported by this check')
                low, high = node['min'], node['max']
                if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                        or not math.isfinite(v) for v in (low, high)) or low >= high):
                    raise ValueError(f'invalid limits for {parent}')
                ordered.append((parent, float(low), float(high)))
        if len(visited) != len(nodes):
            raise ValueError('disconnected arm model')
        return ordered
    except (KeyError, TypeError, ValueError) as error:
        raise JointRangeError(f'Cannot verify joint limits: {error}') from error


async def read_arm_joint_ranges(arm, timeout=5.0):
    result = await arm.get_kinematics(timeout=timeout)
    if result[0] != 1:  # KinematicsFileFormat.SVA: Viam JSON, limits in degrees.
        raise JointRangeError('Cannot verify joint limits: expected the configured Viam JSON arm model')
    limits = read_joint_model(result[1])
    positions = list((await arm.get_joint_positions(timeout=timeout)).values)
    if len(positions) != len(limits):
        raise JointRangeError('Joint readback count does not match the arm model')
    readings = []
    for (name, low, high), position in zip(limits, positions):
        if (isinstance(position, bool) or not isinstance(position, (int, float))
                or not math.isfinite(position)):
            raise JointRangeError(f'Invalid joint readback for {name}')
        # Deliberately preserve winding: -360 degrees must not become zero.
        readings.append(JointReading(name, position, low, high))
    return readings


async def require_arm_joint_ranges(arm):
    readings = await read_arm_joint_ranges(arm)
    failures = [f"J{i} ({r.name}) = {r.degrees:.3f} deg, allowed [{r.minimum:g}, {r.maximum:g}] deg"
                for i, r in enumerate(readings, 1) if not r.in_range]
    if failures:
        raise JointRangeError('; '.join(failures) + '. Use operator joint controls to bring the affected '
                              'joint safely inside its range, then restart. No automatic unwinding.')
    return readings
