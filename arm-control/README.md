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
| `webcam_gaze.py` | webcam + MediaPipe iris tracking, framing gate, 9-point calibration, smoothing, dwell timer |
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

Fill in `API_KEY`, `API_KEY_ID`, `ADDRESS`, and the service/component names at
the top of `main.py` from your Viam app, then:

```bash
python main.py
```

Every run starts with the face-framing oval and the 9-point calibration (same
as `gaze_dot.py`); `--skip-calibration` reuses the last one, and `C`
recalibrates mid-run. Keep the window where it is after calibrating: the gaze
mapping is to pixels on your physical screen. Recalibrate if you change
seating position, lighting, or move the window. `--dry-run` runs everything
except the arm and gripper.

## Machine facts (from the hackathon crash course)

- Hardware: UFactory xArm6/850 + UFactory two-finger gripper, RealSense D435
  on the arm, viam-server on a computer wired to the arm's control box.
- Components are named `arm`, `gripper`, `cam`; the camera runs with
  `sensors: ["color", "depth"]` and `align_color_depth: true`.
- Frames: arm at the world origin; gripper parented to the arm, **105 mm** out
  from its end; camera parented to the arm at (73, -40, 18) mm, orientation
  (0, 0, 1, th 90). `viam:camera-calibration:handeye` gives a measured
  camera frame if the default is off.
- Every machine has **table and wall obstacles**. The built-in motion
  service plans around them; direct arm moves ignore them. This code only
  moves through the motion service. Know where the E-stop is.
- Recommended: limit joint 5 on the motion service so the camera cable can't
  wrap around the arm. On the motion service's config, add
  `{"input_range_override": {"arm": {"5": {"min": -1.5708, "max": 1.5708}}}}`
  (radians; that's +/-90 deg).
- To find a good grasp orientation: put the arm in manual mode (arm Configure
  panel), move it by hand into a top-down grasp over the table, then run
  `preflight.py` and copy the printed orientation into
  `DEFAULT_GRASP_ORIENTATION`.

## Things to verify on real hardware

- `select_object_for_box()` matches a 2D detection to its 3D point-cloud
  object by label, falling back to list order. Check this against what
  `objects-3d` actually returns.
- The width-based gripper close uses a `do_command` whose key and units are a
  guess; it falls back to `grab()` if the command is rejected.
- `DEFAULT_GRASP_ORIENTATION` assumes a straight-down approach.
