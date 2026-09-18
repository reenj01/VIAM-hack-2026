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

Keep the window where it is after calibrating: the gaze mapping is to pixels
on your physical screen. Recalibrate if you change seating position, lighting,
or move the window.

## Things to verify on real hardware

- `select_object_for_box()` matches a 2D detection to its 3D point-cloud
  object by label, falling back to list order. Check this against what
  `objects-3d` actually returns.
- The width-based gripper close uses a `do_command` whose key and units are a
  guess; it falls back to `grab()` if the command is rejected.
- `DEFAULT_GRASP_ORIENTATION` assumes a straight-down approach.
