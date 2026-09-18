"""Connect to the robot and check everything main.py depends on. Moves nothing.

    python preflight.py

Prints: the machine's resources vs. the names configured in main.py, what the
camera returns, YOLO detections, the segmenter's 3D objects (label, frame,
center, size), the gripper state, and the gripper's current world-frame pose
(copy its orientation into DEFAULT_GRASP_ORIENTATION once the arm is parked
in a good top-down position).
"""

import asyncio

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient

from main import (
    CAMERA_NAME, DETECTOR_NAME, SEGMENTER_NAME, MOTION_SERVICE_NAME, GRIPPER_NAME,
    MOTION_REFERENCE_FRAME, decode_color_frame, load_env, object_center_and_size, object_label,
)


async def step(title, coro):
    print(f"\n== {title}")
    try:
        return await coro
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        return None


async def check_names(machine):
    have = {r.name for r in machine.resource_names}
    print("   resources on the machine:")
    for r in sorted(machine.resource_names, key=lambda r: (r.type, r.subtype, r.name)):
        print(f"     {r.type}:{r.subtype}/{r.name}")
    for role, name in (("camera", CAMERA_NAME), ("detector", DETECTOR_NAME),
                       ("segmenter", SEGMENTER_NAME), ("motion", MOTION_SERVICE_NAME),
                       ("gripper", GRIPPER_NAME)):
        status = "OK" if name in have else "NOT FOUND -> fix this name at the top of main.py"
        print(f"   {role:9} '{name}': {status}")


async def check_camera(machine):
    cam = Camera.from_robot(machine, CAMERA_NAME)
    images, _ = await cam.get_images()
    for img in images:
        print(f"   image '{img.name}' {img.mime_type} {img.width}x{img.height}")
    frame = decode_color_frame(images)
    print("   color frame:", "NONE (nothing decoded)" if frame is None else f"{frame.shape[1]}x{frame.shape[0]}")


async def check_detector(machine):
    det = VisionClient.from_robot(machine, DETECTOR_NAME)
    dets = await det.get_detections_from_camera(CAMERA_NAME)
    print(f"   {len(dets)} detection(s)")
    for d in dets:
        print(f"     {d.class_name:14} {d.confidence:.2f}  px=({d.x_min},{d.y_min})-({d.x_max},{d.y_max})"
              f"  norm=({d.x_min_normalized:.2f},{d.y_min_normalized:.2f})-({d.x_max_normalized:.2f},{d.y_max_normalized:.2f})")


async def check_segmenter(machine):
    seg = VisionClient.from_robot(machine, SEGMENTER_NAME)
    objs = await seg.get_object_point_clouds(CAMERA_NAME)
    print(f"   {len(objs)} 3D object(s)")
    for i, o in enumerate(objs):
        cs = object_center_and_size(o)
        frame = o.geometries.reference_frame
        if cs is None:
            print(f"     [{i}] label='{object_label(o)}' frame='{frame}' (no geometry)")
            continue
        c, s = cs
        print(f"     [{i}] label='{object_label(o)}' frame='{frame}' "
              f"center=({c.x:.0f}, {c.y:.0f}, {c.z:.0f}) mm  size=({s[0]:.0f}, {s[1]:.0f}, {s[2]:.0f}) mm")
    if objs and objs[0].geometries.reference_frame != MOTION_REFERENCE_FRAME:
        print(f"   NOTE: objects are in frame '{objs[0].geometries.reference_frame}' but main.py sends "
              f"poses in '{MOTION_REFERENCE_FRAME}' -> set MOTION_REFERENCE_FRAME to match")


async def check_gripper(machine):
    gr = Gripper.from_robot(machine, GRIPPER_NAME)
    print("   is_moving:", await gr.is_moving())
    print("   is_holding_something:", await gr.is_holding_something())


async def check_motion(machine):
    mo = MotionClient.from_robot(machine, MOTION_SERVICE_NAME)
    pif = await mo.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)
    p = pif.pose
    print(f"   gripper pose in '{pif.reference_frame}': x={p.x:.0f} y={p.y:.0f} z={p.z:.0f} mm")
    print(f"   orientation: o_x={p.o_x:.3f} o_y={p.o_y:.3f} o_z={p.o_z:.3f} theta={p.theta:.1f}")
    print("   -> park the arm in a good top-down grasp pose, re-run this, and copy that orientation "
          "into DEFAULT_GRASP_ORIENTATION in main.py")


async def main():
    env = load_env()
    opts = RobotClient.Options.with_api_key(api_key=env["VIAM_API_KEY"], api_key_id=env["VIAM_API_KEY_ID"])
    machine = await RobotClient.at_address(env["VIAM_MACHINE_ADDRESS"], opts)
    try:
        await step("Resource names", check_names(machine))
        await step("Camera", check_camera(machine))
        await step("YOLO detector", check_detector(machine))
        await step("3D segmenter", check_segmenter(machine))
        await step("Gripper", check_gripper(machine))
        await step("Motion / gripper pose", check_motion(machine))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
