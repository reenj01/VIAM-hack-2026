"""Connect to the robot and check everything main.py depends on. Moves nothing.

    python preflight.py

Checks the resource names, the camera and its intrinsics, whether the detector
can capture image + boxes together, the segmenter's 3D objects AND where each
one lands in world coordinates (a coarse workspace check), the gripper, and
the gripper/TCP geometry main.py will use. Run it with the arm parked at the
pose you observe from, with an object on the table.
"""

import asyncio
import math
import time

from grpclib import GRPCError, Status

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient

from main import (
    ARM_NAME, CAMERA_NAME, DETECTOR_CANDIDATES, DETECTOR_NAME, FINGERTIP_FROM_FLANGE_MM, GRIPPER_NAME,
    MIN_DOWNWARD_O_Z, MOTION_REFERENCE_FRAME, MOTION_SERVICE_NAME, SEGMENTER_NAME, SEGMENTER_TIMEOUT_S,
    TABLE_TOP_Z_MM, connect, decode_color_frame, decode_viam_image, get_intrinsics, measure_tcp_mm,
    object_center_and_size, object_label, object_reference_frame, pose_in, project_to_pixel, target_problem,
)


async def step(title, coro, failures=None):
    print(f"\n== {title}")
    try:
        result = await coro
        if result is False and failures is not None:
            failures.append(title)
        return result
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        if failures is not None:
            failures.append(title)
        return None


def motion_name(have: set[str]):
    return next((n for n in (MOTION_SERVICE_NAME, "builtin") if n in have), None)


async def check_names(machine):
    have = {r.name for r in machine.resource_names}
    complete = True
    print("   resources on the machine:")
    for r in sorted(machine.resource_names, key=lambda r: (r.type, r.subtype, r.name)):
        print(f"     {r.type}:{r.subtype}/{r.name}")
    for role, name in (("camera", CAMERA_NAME), ("detector", DETECTOR_NAME), ("segmenter", SEGMENTER_NAME),
                       ("gripper", GRIPPER_NAME), ("arm", ARM_NAME)):
        status = "OK" if name in have else "NOT FOUND -> fix this name at the top of main.py"
        complete = complete and name in have
        print(f"   {role:9} '{name}': {status}")
    mn = motion_name(have)
    print(f"   motion    '{mn}': OK" if mn else f"   motion: NOT FOUND ('{MOTION_SERVICE_NAME}' or 'builtin')")
    return complete and mn is not None


async def check_camera(machine):
    cam = Camera.from_robot(machine, CAMERA_NAME)
    images, _ = await cam.get_images()
    for img in images:
        print(f"   image '{img.name}' {img.mime_type} {img.width}x{img.height}")
    frame = decode_color_frame(images)
    print("   color frame:", "NONE (nothing decoded)" if frame is None else f"{frame.shape[1]}x{frame.shape[0]}")
    if frame is None:
        raise RuntimeError("Camera returned no decodable color frame")
    intr = await get_intrinsics(cam)
    if intr:
        print(f"   intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f} "
              f"at {intr.width}x{intr.height}  <- used to match the locked box to its 3D object")
    else:
        print("   intrinsics: UNAVAILABLE -> main.py can only match 3D objects by a unique label")
    return intr


async def check_detector(machine, name, failures=None):
    det = VisionClient.from_robot(machine, name)
    dets = await det.get_detections_from_camera(CAMERA_NAME)
    print(f"   {len(dets)} detection(s)")
    for d in dets:
        print(f"     {d.class_name:14} {d.confidence:.2f}  px=({d.x_min},{d.y_min})-({d.x_max},{d.y_max})")
    if name == DETECTOR_NAME:
        try:
            t0 = time.monotonic()
            res = await det.capture_all_from_camera(CAMERA_NAME, return_image=True, return_detections=True)
            frame = decode_viam_image(res.image)
            if frame is None:
                raise RuntimeError("Combined capture returned no decodable image")
            print(f"   combined capture: OK in {time.monotonic() - t0:.2f}s, image "
                  f"{'decoded' if frame is not None else 'NOT decodable'}, {len(res.detections or [])} detection(s)"
                  "  <- image and boxes come from the same frame")
        except Exception as e:
            if isinstance(e, GRPCError) and e.status == Status.UNIMPLEMENTED:
                print("   combined capture: UNIMPLEMENTED -> main.py falls back to separate image + detection calls")
            else:
                print(f"   combined capture: FAILED ({type(e).__name__}: {e}); support could not be verified")
                if failures is not None:
                    failures.append(f"Detector '{name}' combined capture")
    return {d.class_name for d in dets}


async def check_segmenter(machine, detector_labels, intr):
    seg = VisionClient.from_robot(machine, SEGMENTER_NAME)
    t0 = time.monotonic()
    objs = await seg.get_object_point_clouds(CAMERA_NAME, timeout=SEGMENTER_TIMEOUT_S)
    print(f"   {len(objs)} 3D object(s) in {time.monotonic() - t0:.1f}s")
    seg_labels = {object_label(o) for o in objs} - {""}
    if seg_labels:
        valid_labels = {n: labels for n, labels in detector_labels.items() if labels is not None}
        if detector_labels.get(DETECTOR_NAME) is None:
            print(f"   detector comparison: SKIPPED; '{DETECTOR_NAME}' did not return a successful result")
        else:
            matched = [n for n, labels in valid_labels.items() if seg_labels & labels]
            print(f"   segmenter labels: {sorted(seg_labels)}; overlap with successful detector results: {matched or 'none'}")
            print("   Label overlap does not verify segmenter wiring; these calls also capture different frames. "
                  "Check detector_name in the segmenter's configuration to verify its source.")
    verdicts = []
    for i, o in enumerate(objs):
        cs = object_center_and_size(o)
        ref = object_reference_frame(o)
        if cs is None:
            print(f"     [{i}] '{object_label(o)}' frame='{ref}' (no geometry)")
            verdicts.append(False)
            continue
        c, s = cs
        w = await pose_in(machine, Pose(x=c.x, y=c.y, z=c.z, o_z=1.0), ref, MOTION_REFERENCE_FRAME)
        problem = target_problem(w)
        verdicts.append(problem is None)
        pix = ""
        if intr is not None:
            cc = await pose_in(machine, c, ref, CAMERA_NAME)
            uv = project_to_pixel(cc, intr, intr.width, intr.height)
            pix = f"  pixel=({uv[0]:.0f}, {uv[1]:.0f})" if uv else ""
        print(f"     [{i}] '{object_label(o)}' in '{ref}' ({c.x:.0f}, {c.y:.0f}, {c.z:.0f}) -> world "
              f"({w.x:.0f}, {w.y:.0f}, {w.z:.0f})  size {s[0]:.0f}x{s[1]:.0f}x{s[2]:.0f}{pix}  "
              + ("PASS" if problem is None else f"FAIL: {problem}"))
    if verdicts:
        print(f"   coarse workspace range check: {'PASS' if all(verdicts) else 'FAIL'} "
              f"(configured table top z~{TABLE_TOP_Z_MM:.0f} mm). "
              "This does not validate camera calibration, mounting transforms, or grasp accuracy. "
              "Compare the pixel column with the object boxes and world positions with measured locations.")
    else:
        print("   INCOMPLETE: no usable 3D objects; put an object on the table in view and re-run")
    return bool(verdicts) and all(verdicts)


async def check_gripper(machine):
    gr = Gripper.from_robot(machine, GRIPPER_NAME)
    print("   is_moving:", await gr.is_moving())
    print("   is_holding_something:", await gr.is_holding_something())
    arm = Arm.from_robot(machine, ARM_NAME)
    for who, label, cmd in ((gr, f"gripper '{GRIPPER_NAME}'", {"get": True}),
                            (arm, f"arm '{ARM_NAME}'", {"get_gripper": True})):
        try:
            resp = await who.do_command(cmd)
            print(f"   do_command {cmd} on {label}: {resp}  <- width-based close works via this component")
            break
        except Exception as e:
            print(f"   do_command {cmd} on {label}: not accepted ({type(e).__name__})")
    else:
        print("   no component accepts gripper do_commands; main.py will fall back to grab()")


async def check_motion(machine):
    mn = motion_name({r.name for r in machine.resource_names})
    if mn is None:
        print("   no motion service found")
        return False
    mo = MotionClient.from_robot(machine, mn)
    p = (await mo.get_pose(GRIPPER_NAME, MOTION_REFERENCE_FRAME)).pose
    print(f"   motion service '{mn}'")
    print(f"   gripper TCP in world: ({p.x:.0f}, {p.y:.0f}, {p.z:.0f}) mm, "
          f"orientation o=({p.o_x:.3f}, {p.o_y:.3f}, {p.o_z:.3f}) theta={p.theta:.1f}")
    print("   -> grasps reuse this wrist orientation" if p.o_z <= MIN_DOWNWARD_O_Z else
          "   -> not pointing down: grasps will use DEFAULT_GRASP_ORIENTATION instead")
    tcp = await measure_tcp_mm(mo)
    print(f"   gripper TCP is {tcp:.0f} mm from the flange; fingertips assumed {FINGERTIP_FROM_FLANGE_MM:.0f} mm "
          f"-> {max(0.0, FINGERTIP_FROM_FLANGE_MM - tcp):.0f} mm beyond the TCP (tape-measure flange->fingertip "
          "and set FINGERTIP_FROM_FLANGE_MM if grasps land high or low)")
    print(f"   height above the table top: TCP {p.z - TABLE_TOP_Z_MM:.0f} mm, reach {math.hypot(p.x, p.y):.0f} mm")


async def main():
    failures = []
    machine = await step("Connection", connect(), failures)
    if machine is None:
        print("\n== Preflight FAILED: connection could not be established")
        return 1
    try:
        await step("Resource names", check_names(machine), failures)
        intr = await step("Camera", check_camera(machine), failures)
        have = {r.name for r in machine.resource_names}
        detector_labels = {}
        for name in dict.fromkeys((DETECTOR_NAME, *DETECTOR_CANDIDATES)):
            if name in have:
                tag = " (the one main.py uses)" if name == DETECTOR_NAME else ""
                essential_failures = failures if name == DETECTOR_NAME else None
                detector_labels[name] = await step(
                    f"Detector '{name}'{tag}", check_detector(machine, name, essential_failures), essential_failures
                )
        await step("3D segmenter -> world", check_segmenter(machine, detector_labels, intr), failures)
        await step("Gripper", check_gripper(machine), failures)
        await step("Motion / gripper geometry", check_motion(machine), failures)
    finally:
        await machine.close()
    if failures:
        print(f"\n== Preflight FAILED: {len(failures)} essential check(s) failed or incomplete")
        for title in failures:
            print(f"   - {title}")
        return 1
    print("\n== Preflight PASS: essential checks completed. Physical geometry and picking accuracy remain unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
