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
python main.py --detector vision-1
# Add --execute to enable physical movement after calibration is checked.
```

Calibration uses three axial distances in millimeters:

- Fixed flange-to-housing distance: the user measured approximately **3.85
  inches**, saved as **97.8 mm**. `--gripper-body-from-flange-mm` overrides
  this value if the hardware or mounting changes.
- Housing underside to fingertips, fully open: **2.3 inches**, saved as
  **58.4 mm** (`--finger-clearance-mm` overrides the minimum).
- Housing underside to fingertips, fully closed: **2.75 inches**, saved as
  **69.9 mm** (`--max-finger-extension-mm` overrides the maximum).

All three measured defaults are now saved, so no measurement flags are
required for this setup. They bound the complete open/close stroke used in
planning. The later direct measurements above supersede the initial approximate
3-inch reading in the photo; the 3.3-inch sideways jaw opening is a different
dimension. The previous assumed 165 mm flange-to-tip
distance has been removed from planning. Hinged tips do not have one fixed
distance from the flange.

Run `python preflight.py` first for read-only camera, detector, segmentation,
gripper, frame, and proposed grasp/model-clearance checks. Use the same Python
environment as `main.py`.

For the current test block, the measured distance between gripping faces is
**1.125 inches = 28.575 mm**. The recording estimated 49 mm and requested a
48 mm opening, which is too wide to contact these faces. Use an explicit
measurement for this test setup:

```powershell
python preflight.py --detector vision-1 --object-width-mm block=28.6
python main.py --detector vision-1 --object-width-mm block=28.6
```

The second command is a dry run. The width option applies to every selected
object with that label during that run; only use it when those objects have
the measured width along the jaws' closing direction. It is not a universal
width for all blocks. Without an override, width is still an approximate depth
box estimate. A measured width does not correct the object's depth or position.

The live collision model checked on 2026-09-19 extends the claws **50 mm below
the TCP**, while the measured closed fingertips extend **17.7 mm**. With a
table top at z=-23 mm, a requested TCP z=18 mm overlaps that model with the
table. Picking now reads the actual modeled boxes and refuses such a pose
before approaching. It does not shrink the geometry or shift the table.
Resolve the physical/model calibration if this check blocks a grasp; raising
the pose blindly can leave the fingers above a short object.

The display detector must correspond to the segmenter's detections. On the
2026-09-19 live check, `vision-1`'s can box contained the projected 3D can
center; `yolo-detector`'s cup box did not. For that setup, use
`python preflight.py --detector vision-1`, then
`python main.py --detector vision-1`. Add `--execute` to
enable motion. The default remains `yolo-detector`; `--detector` explicitly
selects the service to use.

Grasp height positions the **rigid housing above the 3D object's world top**.
It targets at most 15 mm insertion for the shortest finger extension. The
longest extension must clear the table; the body margin includes possible
object lift caused by retracting fingers during closure. If the measured
range cannot satisfy both table clearance and finger overlap, picking is
refused. A 15 mm body margin also allows for small wrist tilt. The gripper
must point down within 5 degrees. Add
`--grasp-z-offset-mm 5` for another 5 mm upward adjustment. An offset that
leaves no finger overlap is rejected. These margins depend on accurate depth,
camera calibration, the fixed housing datum, and measured extension bounds.

The gripper now verifies that it opened before descending and actually
reached a stationary grasp pose before closing, then checks closure before
lifting. Two pose readbacks must be within 5 mm and 2 degrees of the requested
grasp. It uses a single position-limited close, with 1 mm
requested squeeze instead of 8 mm, and verifies stable contact evidence.
An empty partial close, unchanged jaws, or an unacknowledged command cannot
become a successful pickup. A failed verification opens at the table and
retreats; an exception during a physical action stops automatic selection.
The specific IK constraint rejection shown in the recording now pauses on
screen instead of crashing. All unsuccessful attempts require **R** to
acknowledge before selection resumes; no attempt automatically repeats after
a timeout. No further movement is sent after a rejected movement, and detected
obstacles are never discarded to retry a plan.

### Expired sessions and duplicate geometry names

`requesting move to approach` means the motion RPC was submitted, not that the
arm has physically started moving. A pending request prints elapsed time every
five seconds; planning and execution share the same RPC. Completion, errors,
and cancellation report elapsed time. STOP logs its cause (Q, a detected fault,
or session cleanup) and whether `arm.stop()` acknowledged the request.

`INVALID_ARGUMENT: SESSION_EXPIRED` is a Viam safety-session failure, separate
from detection. The failed motor command is not replayed. The app stops once
and keeps a fault screen open. **R** performs read-only checks: arm and jaws
stopped, gripper fully open, holding status false, and a valid current pose.
Selection resumes only if these checks succeed; otherwise the fault stays
visible. Use the robot controls to put down any held object and open the jaws
before acknowledging an uncertain grip. A dead connection can be rebuilt only after these checks; startup home
movement is not replayed during connection recovery.

Multiple detected cans used to create identically named `can` collision
geometries. Planner geometries now receive unique IDs, including after merging
obstacles with a saved object footprint. Detection labels remain unchanged.
This fixes the duplicate-name error; it does not remove overlapping detections
or validate their depth. Implausible depth results are still rejected.

### Return after placing or releasing an object

The requested pose was read from the robot on 2026-09-19 and saved locally in
`home_pose.json`: approximately **(191, -304, 383) mm** in world coordinates.
The TCP is **406 mm (16.0 inches) above the configured table** at this pose.
This measured pose takes precedence over the approximate 14-inch request.
The file is ignored by Git because it belongs to this robot setup.
Starting the app does not move to this pose by default. `--go-home-on-start`
explicitly enables the startup move; post-pick and post-drop returns still use
the saved pose without that flag.

After each verified release, including a swap, the arm first rises vertically
at the actual drop X/Y to the saved return height, then returns to the saved
X/Y. For a release already near or above that height, it rises at least 40 mm
before returning. Other detected obstacles remain active, and the released
object is included in the model for the return across the table. An elevated
"Let go" includes the vertical drop region down to the table; this assumes a
vertical fall and does not predict bouncing or rolling. A rejected
ascent never proceeds to the lateral return. A failed return leaves the object
marked released and pauses instead of claiming success or offering holding
actions. No physical return was executed during development.

Teach a replacement pose with `python main.py --set-home --detector vision-1`
after manually parking the robot where it should return; teaching only reads
and saves the pose. Without a taught pose, post-release return uses at least
355.6 mm above the configured table at the pick approach X/Y.
There is no full-close `grab()` fallback. Position checks and estimated width
do **not** provide force control or guarantee that delicate objects cannot
be crushed sideways.

The SDK's one-second connection probe is disabled because slow vision requests
can trigger it. If the transport closes while choosing an object, the app
reconnects with fresh resource handles, clears the selection, and keeps the
calibration window in place. It does not repeat the startup home movement.
A connection loss during object handling pauses for operator recovery;
the interrupted action is not replayed.

**Motion is opt-in.** Without `--execute`, the camera, YOLO, gaze, lock, 3D
segmentation, transforms and safety checks all run and every pose is printed,
but nothing is sent to the arm or gripper. `Q` cancels a grasp and sends
`arm.stop()`, best effort; the physical E-stop is the real stop.

Every run starts with the face-framing oval, then calibration (~55 s):

1. **Head still**, eyes only: nine dots (corners, edge middles, center).
2. **Eight directions** (upper left, top, upper right, right, lower right,
   bottom, lower left, left). Each starts with a "turn your head" screen:
   your webcam view, an arrow and the instruction to turn your head
   *slightly* that way (a small attention turn, not a full one). It moves on
   by itself after 3 s (Space = sooner). Then two dots: the midpoint toward
   that side, then the edge or corner (e.g. upper-middle-left, then
   upper-left); keep the head there and follow them with your eyes.

Nothing measures or waits on the head angle. Because the same points are
also in the head-still grid, the model learns how much of a look comes from
the eyes and how much from the head.
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
2. On lock, while the arm is still: two independent 3D segmentations must agree
   within 10 mm in position, top height, and world box dimensions. The wrist
   must remain stationary. The target is the
   object whose center **projects into the locked box** (camera intrinsics),
   not a list position, so ordering and missing detections can't swap
   objects. If nothing lands there (e.g. the object moved), it doesn't move.
3. The target and every other object are **frozen into world coordinates**
   before anything moves (the camera rides the wrist, so camera-frame values
   go stale the moment the arm moves).
4. It refuses to move if the target isn't on the table or is out of reach.
5. Compute a shallow grasp below the object's world top with body clearance.
   Approach 100 mm above, wrist orientation held -> verify open jaws ->
   straight-line descent -> verify stationary arrival -> close once and verify contact -> straight up ->
   carried back level. Failed verification does not enter the carry phase.

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
obstacle. If the put-back can't be planned, the job pauses and the gripper keeps
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
- The **gripper frame is 150 mm** from the flange (the deck says 105).
  `main.py` measures this TCP offset at startup and converts from the measured
  rigid housing datum. It does not assume that the hinged fingertips stay
  at a fixed position relative to that frame.
- The camera frame is a calibrated value, `(83, -14, 18)` mm, theta -97.7.
  `preflight.py` checks broad workspace height and reach limits. To verify
  the mounting transform, compare reported world coordinates with measured
  object locations; a passing range check alone does not validate calibration.
- The table obstacle is a 200 mm box centered at z = -123, so the **table top
  is at world z = -23**; there are also wall and ceiling obstacles. The
  motion service plans around them; direct arm moves ignore them, so this
  code only moves through the motion service.
- A `pose-home` switch already stores the observe pose. `--set-home` saves
  our own `home_pose.json`; without it, objects are carried back to where
  the gripper was when the object was locked.
- **Joint range check:** the live xArm6 model already declares joint limits.
  On 2026-09-19, J6 (`gripper_rot`) read **-359.994 degrees** against a model
  range of **-359 to +359 degrees**. The rejected trajectory began at this
  out-of-range readback. `preflight.py` now lists all six joint values and
  limits, and execution checks them before startup movement or calibration.
  An out-of-range start prints `START BLOCKED` and sends no grasp/home command.
  Use the robot's operator joint controls to jog the affected joint safely
  inside its range, then restart. Do not widen the model limits or normalize
  -360 degrees to zero to conceal this mismatch.
  A motion-service `input_range_override` can narrow planner ranges, but cannot
  repair an already out-of-range starting joint. The earlier blanket claim
  that this arm had no joint limits was incorrect.

## Things to verify on real hardware

- The fixed flange-to-housing distance (`--gripper-body-from-flange-mm`).
- The full housing-to-fingertip extension range (`--finger-clearance-mm` and
  `--max-finger-extension-mm`), including the fully open descent configuration.
- The position-limited gripper close (`{"set": pos}`, 0-850 scale) and actual
  position readback; no fallback close is attempted if it cannot be verified.
- `preflight.py` passes, its projected pixels match the selected detector's
  boxes, and its world coordinates agree with measured object locations.
