"""Step 1 of training a custom YOLO: collect images of the object(s).

Pulls frames from the robot's camera (the one the model will run on), so the
training data matches deployment: same lens, same height, same lighting.

    python dataset_capture.py                # SPACE saves a frame, Q quits
    python dataset_capture.py --auto 0.7     # also saves a frame every 0.7 s
    python dataset_capture.py --local 1      # use a local camera index instead of the robot

Aim for 100-200 images. Between shots, move the cup around the table, rotate
it, add the other objects that will be around it, put your hand in some shots,
change what's behind it. Variety beats quantity.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import cv2

from main import CAMERA_NAME, connect, decode_color_frame

RAW_DIR = Path(__file__).resolve().parent / "dataset" / "raw"


async def robot_frames():
    from viam.components.camera import Camera
    machine = await connect()
    cam = Camera.from_robot(machine, CAMERA_NAME)
    try:
        while True:
            images, _ = await cam.get_images(filter_source_names=["color"])
            frame = decode_color_frame(images)
            if frame is not None:
                yield frame
    finally:
        await machine.close()


async def local_frames(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                yield frame
            await asyncio.sleep(0)
    finally:
        cap.release()


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto", type=float, default=0.0, help="also save a frame every N seconds")
    ap.add_argument("--local", type=int, default=None, help="local camera index instead of the robot camera")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(RAW_DIR.glob("*.jpg")))
    print(f"[capture] saving to {RAW_DIR} ({existing} images already there)")

    source = local_frames(args.local) if args.local is not None else robot_frames()
    count = existing
    last_auto = time.monotonic()
    window = "Dataset capture (SPACE save, Q quit)"
    async for frame in source:
        now = time.monotonic()
        save = False
        view = frame.copy()
        cv2.putText(view, f"{count} images  |  SPACE = save  Q = quit" + (f"  |  auto every {args.auto}s" if args.auto else ""),
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            save = True
        if args.auto and now - last_auto >= args.auto:
            save, last_auto = True, now
        if save:
            path = RAW_DIR / f"img_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(path), frame)
            count += 1
            print(f"[capture] saved {path.name} ({count})")
    cv2.destroyAllWindows()
    print(f"[capture] done: {count} images in {RAW_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
