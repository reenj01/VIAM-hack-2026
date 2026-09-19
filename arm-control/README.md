# arm-control

Connects webcam gaze tracking to the Viam-controlled arm: YOLO detections from
the robot's vision service are hit-tested against your gaze; holding your gaze
on one **locks** it (a frozen snapshot, since the camera rides the arm); the
segmenter deprojects it to a world-frame pose once while the arm is still; the
motion service moves the gripper there and it closes.

Builds on the gaze-tracking approach from `gaze_dot.py` at the repo root (same
feature extraction, calibration math, and face-framing gate), reused here as
an importable module.

| file | role |
| --- | --- |
| `webcam_gaze.py` | webcam + MediaPipe iris tracking, framing gate, calibration, smoothing, dwell timer |
| `gaze_lock.py` | gaze vs. detection boxes: hover, dwell, lock with frozen snapshot, drawing |
| `main.py` | the robot pipeline (Viam camera, YOLO detector, segmenter, motion, gripper) |
| `local_demo.py` | the same gaze → box lock-on with a local YOLO on a photo/video, no robot needed |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

For `local_demo.py` also `pip install ultralytics` (it downloads `yolov8n.pt`
on first run).

## Test the gaze lock-on without the robot

```bash
python local_demo.py --scene path/to/photo-with-objects.jpg
python local_demo.py --scene path/to/clip.mp4
```

Center your face in the oval, hold, then look at each of the 9 dots. After
that: look at a detected object and hold ~1 s to lock it. R releases, C
recalibrates, Q quits.

## Run against the robot

Copy `../.env.example` to `arm-control/.env` and fill in
`VIAM_MACHINE_ADDRESS`, `VIAM_API_KEY_ID` and `VIAM_API_KEY` from your Viam app
(plain values, no quotes or `<>`; the file is gitignored). Check the
service/component names at the top of `main.py`, then:

```bash
python main.py              # dry run: everything except motion, prints the poses
python main.py --execute    # the arm moves
```

**Motion is opt-in.** Without `--execute`, the camera, YOLO, gaze, lock, 3D
segmentation, transforms and safety checks all run and every pose is printed,
but nothing is sent to the arm or gripper. `Q` cancels a grasp and sends
`arm.stop()`, best effort; the physical E-stop is the real stop.

Every run starts with the face-framing oval, then calibration (~50 s):

1. **Head still**, eyes only: a 5x5 grid of dots.
2. **Eight directions** (upper left, top, upper right, right, lower right,
   bottom, lower left, left), two dots each: the midpoint toward that side,
   then the edge or corner (e.g. upper-middle-left, then upper-left). Look at
   them and let your head turn *slightly* with your eyes, the way it does
   when your attention moves somewhere. No full head turns.

Nothing measures or waits on the head angle; the user is trusted to look at
the dots. Because the same points are also in the head-still grid, the model
learns how much of a look comes from the eyes and how much from the head.
`S` skips a stage. `--quick-calibration` does the head-still stage only;
`--skip-calibration` reuses the last one; `C` recalibrates mid-run.
Blinks neither add nor remove selection evidence, and the cursor holds until
the eyes are fully open again. Keep the window where it is after calibrating:
the gaze mapping is to pixels on your physical screen.

## How a grasp works

1. The detector returns the image and its boxes from **one capture**
   (`CaptureAllFromCamera`), so the boxes on screen and the lock snapshot
   always belong to the same frame. The HUD says `UNPAIRED` if the detector
   can't do that and it falls back to separate calls.
2. On lock, while the arm is still: one 3D segmentation; the target is the
   object whose center **projects into the locked box** (camera intrinsics),
   not a list position, so ordering and missing detections can't swap
   objects. If nothing lands there (e.g. the object moved), it doesn't move.
3. The target and every other object are **frozen into world coordinates**
   before anything moves (the camera rides the wrist, so camera-frame values
   go stale the moment the arm moves).
4. It refuses to move if the target isn't on the table or is out of reach.
5. Approach 100 mm above, wrist orientation held (it reuses the wrist
   orientation the arm already has, so the wrist never spins) -> open ->
   straight-line descent -> close to the object's width -> straight up ->
   carried back level.

## Demo mode (default): pick, then swap

For the demo there are exactly two gaze actions:

1. **Look at an object and hold**: the arm picks it up and carries it back to
   the observe pose, still holding it, so the camera sees the table again.
2. **While holding, look at a different object and hold**: the arm puts the
   held one back down where it came from, then picks up the new one.

It uses the same dwell as a normal pick, so a passing glance doesn't
trigger a swap. The new object (and every other one) is located and frozen
into world coordinates *before* anything moves; the put-back avoids the new
object, and the new grasp avoids the spot the old one went back to. The
object in the gripper, if the wrist camera sees it, is never a target or an
obstacle. If the put-back can't be planned the arm backs up and keeps
holding. The gripper only opens once the object is down.

`P` (keyboard, for the operator) puts the held object back without picking
another, e.g. to reset between demos. Q stops and quits; the arm keeps
holding whatever it has.

## `--menu`: the post-pick menu and the two user profiles

With `--menu`, after a successful grasp the arm hovers, holding the object,
and the screen turns into a big gaze menu (`delivery.py`). Live object
selection is off while holding, so a second selection can't drop what's in
the gripper.

```
Bring to me   Raise          Closer   Put it back
Place down    [ rest zone ]           Place where I look
Let go        Lower          Away     Steer with head   (--user head)
```

- Look at a tile and hold (~1 s). The middle is a rest zone: looking there
  does nothing. Blinks don't count.
- **Bring to me** carries it level to the serve pose. **Closer / Away** move
  50 mm toward/away from the user, **Raise / Lower** 50 mm up/down.
- **Put it back** returns it to where it was picked. **Place down** sets it
  down right here. **Place where I look** goes back to the observe pose,
  shows the camera view; look at an empty spot and hold (spots on other
  objects or out of reach are refused). **Let go** asks Yes/No first.
- While the arm moves, the bottom of the screen is a big **STOP** (short
  dwell). Q also stops; the E-stop is the real stop.
- Every target is clamped: never below the height that rests the object on
  the table, never past the serve pose toward the user, within reach.

`--user head` (for users with some head movement; implies `--menu`) adds a per-user head-range
calibration at startup (hold still, then turn left/right and tilt up/down as
far as is comfortable; S if a direction isn't possible) and **Steer with
head** (`head_control.py`): turn/tilt your head to move the object in small
planned steps, center your head to stop. Two planes, switched with gaze
buttons: up/down + left/right, or closer/away + left/right (tilt up = away).
Buttons only react while the head is centered, so steering and choosing never
fight. A sudden jerk or losing your face pauses it until you re-center.
The design follows AMiCUS (head-motion arm control used by tetraplegic
users): personal range calibration, dead zone + smooth ramp, 2 DOF at a time,
on-screen dwell switching (their nod gesture failed most of the time for
users with restricted neck motion).

Setup, once, with the arm parked where the user should receive objects
(over the table, in front of them):

```bash
python main.py --set-serve
python main.py --menu             # eyes profile (dry run)
python main.py --user head        # head + eyes profile (dry run)
```

## This machine (read from its config with the Viam CLI)

- Hardware: UFactory xArm6 + UFactory two-finger gripper, RealSense D435 on
  the wrist, arm at 60 deg/s. Frames come from the hackathon fragment.
- The **gripper frame is 150 mm** from the flange (the deck says 105): the TCP
  is already near the finger pads. Fingertips are assumed 165 mm out, so
  they end ~15 mm beyond the TCP; main.py measures the TCP distance at
  startup. If grasps land high or low, tape-measure flange-to-fingertip and
  set `FINGERTIP_FROM_FLANGE_MM`.
- The camera frame is a calibrated value, `(83, -14, 18)` mm, theta -97.7.
  `preflight.py` checks it numerically: every segmented object should land on
  the table in world coordinates.
- The table obstacle is a 200 mm box centered at z = -123, so the **table top
  is at world z = -23**; there are also wall and ceiling obstacles. The
  motion service plans around them; direct arm moves ignore them, so this
  code only moves through the motion service.
- A `pose-home` switch already stores the observe pose. `--set-home` saves
  our own `home_pose.json`; without it, objects are carried back to where
  the gripper was when the object was locked.
- **Recommended config change:** the motion service has no joint limits, so
  the planner may pick solutions that spin the wrist or flip the elbow. On
  the `motion` service's config, add
  `{"input_range_override": {"arm": {"5": {"min": -3.1416, "max": 3.1416}}}}`
  (radians) to keep the last joint within one turn.

## Things to verify on real hardware

- `FINGERTIP_FROM_FLANGE_MM` (grasp height).
- The width-based gripper close (`{"set": pos}`, 0-850 scale); it falls back
  to `grab()` if rejected.
- `preflight.py`'s world-transform check passes with objects on the table.
