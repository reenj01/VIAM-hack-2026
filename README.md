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

- `C`: begin calibration
- `R`: recalibrate
- `Q` or `Escape`: quit

## Notes

Keep the preview window at the same size and position used during calibration. Recalibrate after changing seating position, lighting, or display.
