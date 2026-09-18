# arm-control

Connects webcam gaze tracking to the Viam-controlled arm: YOLO detections from
the robot's vision service are hit-tested against your gaze, a dwell selects
one, and the motion service + gripper pick it up.

Builds on the gaze-tracking approach from `gaze_dot.py` at the repo root
(same feature extraction and calibration math), reused here as an importable
module (`webcam_gaze.py`) instead of a standalone script, with dwell-based
selection added.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Fill in `API_KEY`, `API_KEY_ID`, `ADDRESS`, `VISION_SERVICE_NAME`,
`GRIPPER_NAME`, and `MOTION_REFERENCE_FRAME` at the top of `main.py` with the
values from your Viam app's Connect tab and machine config.

## Run

```bash
python main.py
```

First run walks through a 9-point gaze calibration against the live camera
feed window; recalibrate if you change seating position, lighting, or move
the display window.

## Notes

`find_matching_point_cloud_object()` in `main.py` matches a 2D YOLO detection
to its deprojected 3D point-cloud object by label, falling back to list
position. Verify this against your configured segmenter's actual behavior on
real hardware before trusting it for anything delicate.
