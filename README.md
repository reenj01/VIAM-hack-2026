# VIAM Hack 2026 — Webcam gaze dot

A standalone laptop-webcam prototype that calibrates estimated gaze to a red dot on screen. It does not connect to Viam or a robot.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Run

```bash
python gaze_dot.py
```

Allow camera access if macOS asks.

## Controls

- `C`: begin calibration — first shows an ID-photo-style oval frame; center your face in it and hold still for a second, then the 9-point calibration starts automatically
- `R`: recalibrate (same framing step first)
- `Q` or `Escape`: quit

## Fullscreen Viam scene selection

`viam_scene_select.py` reads one frozen RealSense image and its YOLO detection boxes after calibration. It displays the image fullscreen and highlights the detected box under the red dot. It never moves the robot or fetches full 3D point-cloud data during this display stage.

1. Copy `.env.example` to `.env` and fill in your Viam machine address, API key, API key ID, and component names.
2. Manually put the arm at its safe observe pose and make sure it stays still.
3. Run `python viam_scene_select.py`.
4. Press `C` to calibrate, `N` to take a new snapshot before calibration, and `Q` to quit.

Do not add `.env` to Git. It is ignored by default.

## Notes

Keep the preview window at the same size and position used during calibration. Recalibrate after changing seating position, lighting, or display. The framing step (orange oval = not aligned, green = aligned) exists because the 9-point calibration assumes a stable head pose and distance from the camera throughout — starting from a consistent position measurably improves calibration accuracy.
